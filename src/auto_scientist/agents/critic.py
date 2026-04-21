"""Critic: multi-model critique dispatcher.

Input: scientist's plan + analysis JSON + prediction history + lab notebook
+ domain knowledge.
Output: structured critique with tagged concerns and alternative hypotheses,
plus raw transcript for debugging.

In SDK mode, critics have web search and interactive prediction tree access
(MCP tool) to query specific predictions, chains, and statistics. In direct
API mode (OpenAI/Google), prediction history is provided as text in the prompt.

Critics receive the full evidence base but do not see Python code
(implementation is the Coder's domain).

Personas provide diverse critical perspectives. Each critique runs one persona;
model assignment rotates across iterations so no model is always the same role.
"""

import asyncio
import json
import logging
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from auto_core.agent_result import AgentResult
from auto_core.agents.debate_models import (
    CRITIC_OUTPUT_SCHEMA,
    Concern,
    CriticOutput,
    DebateResult,
)
from auto_core.agents.notebook_tool import (
    NOTEBOOK_SPEC,
    build_notebook_mcp_server,
    format_notebook_toc,
)
from auto_core.agents.prediction_tool import (
    PREDICTION_SPEC,
    build_prediction_mcp_server,
    format_compact_tree,
    format_full_detail,
)
from auto_core.model_config import AgentModelConfig, reasoning_to_cc_extra_args
from auto_core.models.google_client import query_google
from auto_core.models.openai_client import query_openai
from auto_core.notebook import parse_notebook_entries, read_notebook
from auto_core.retry import QueryResult, agent_retry_loop
from auto_core.sdk_backend import SDKBackend, SDKOptions, create_backend, get_backend
from auto_core.sdk_utils import (
    collect_text_from_query,
    prepare_turn_budget,
    validate_json_output,
)
from auto_core.state import PredictionRecord
from pydantic import BaseModel

from auto_scientist.prompts.critic import (
    CRITIC_USER,
    DEFAULT_CRITIC_INSTRUCTIONS,
    ITERATION_0_PERSONAS,
    PERSONAS,
    PREDICTION_PERSONAS,
    build_critic_system,
    get_model_index_for_debate,
)

logger = logging.getLogger(__name__)

# Exceptions worth retrying (transient network/rate-limit issues).
# Non-retryable errors (ValueError, TypeError, ImportError, auth errors)
# propagate immediately so the user gets a clear failure instead of a
# misleading retry-then-fail cycle.
_RETRYABLE_ERRORS = (ConnectionError, TimeoutError, OSError, RuntimeError)

CRITIC_BASE_TOOLS = ["WebSearch"]


def _build_critic_tools_and_mcp(
    prediction_history_records: list[PredictionRecord] | None,
    notebook_path: Path | None = None,
    output_dir: Path | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """Build the tools list and MCP servers dict for a critic invocation.

    Mirrors the scientist's pattern: base tools + optional prediction MCP +
    notebook MCP. MCP is only usable in SDK mode; direct API callers ignore
    mcp_servers and get inline content instead.
    """
    tools = list(CRITIC_BASE_TOOLS)
    mcp_servers: dict[str, Any] = {}
    if prediction_history_records:
        mcp_servers["predictions"] = build_prediction_mcp_server(
            prediction_history_records, output_dir=output_dir
        )
        tools.append(PREDICTION_SPEC.mcp_tool_name)
    if notebook_path is not None:
        mcp_servers["notebook"] = build_notebook_mcp_server(notebook_path, output_dir=output_dir)
        tools.append(NOTEBOOK_SPEC.mcp_tool_name)
    return tools, mcp_servers


# ---------------------------------------------------------------------------
# Low-level query helpers
# ---------------------------------------------------------------------------


async def _query_critic(
    config: AgentModelConfig,
    prompt: str,
    *,
    system_prompt: str = "",
    response_schema: type[BaseModel] | None = None,
    message_buffer: list[str] | None = None,
    allowed_tools: list[str] | None = None,
    mcp_servers: dict[str, Any] | None = None,
    backend: SDKBackend | None = None,
    resume: str | None = None,
) -> tuple[AgentResult, str | None]:
    """Dispatch a prompt to the appropriate provider and mode.

    Routes based on config.mode first:
    - mode='sdk': use the backend abstraction (Claude Code or Codex)
    - mode='api': use direct API calls (OpenAI, Google, Anthropic)

    SDK mode receives optional MCP servers for agentic prediction tree access.
    Direct API mode ignores mcp_servers (prompt text is the fallback).

    Args:
        backend: Pre-created SDK backend (from :func:`create_backend`).
            When provided, used instead of the cached singleton from
            ``get_backend()``.  Required for concurrent critics.
        resume: Session ID from a previous query to continue the
            conversation (e.g. for validation retry with correction hint).

    Returns:
        ``(AgentResult, session_id)`` where ``session_id`` can be passed
        back as ``resume`` on the next attempt.
    """
    effective_tools = allowed_tools or list(CRITIC_BASE_TOOLS)

    if config.mode == "sdk" and config.provider in ("anthropic", "openai"):
        # SDK mode: use the backend abstraction
        extra_args: dict[str, str | None] = {}
        if config.reasoning and config.reasoning.level != "off":
            extra_args.update(reasoning_to_cc_extra_args(config.reasoning))
        max_turns = 10
        budget = prepare_turn_budget(
            system_prompt, max_turns, effective_tools, provider=config.provider
        )
        effective_backend = backend or get_backend(config.provider)
        options = SDKOptions(
            model=config.model,
            system_prompt=budget.system_prompt,
            allowed_tools=budget.allowed_tools,
            max_turns=budget.max_turns,
            extra_args=extra_args,
            mcp_servers=mcp_servers or {},
            response_schema=response_schema,
            resume=resume,
        )
        text, usage, session_id = await collect_text_from_query(
            prompt, options, effective_backend, message_buffer
        )
        in_tok = (
            usage.get("input_tokens", 0)
            + usage.get("cache_creation_input_tokens", 0)
            + usage.get("cache_read_input_tokens", 0)
        )
        return AgentResult(
            text=text,
            input_tokens=in_tok,
            output_tokens=usage.get("output_tokens", 0),
            thinking_tokens=usage.get("thinking_tokens", 0),
        ), session_id

    # API mode: direct provider API calls (MCP not available)
    effective_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

    if config.provider == "openai":
        result = await query_openai(
            config.model,
            effective_prompt,
            web_search=True,
            reasoning=config.reasoning,
            response_schema=response_schema,
        )
        return result, None
    elif config.provider == "google":
        result = await query_google(
            config.model,
            effective_prompt,
            web_search=True,
            reasoning=config.reasoning,
            response_schema=response_schema,
        )
        return result, None
    elif config.provider == "anthropic":
        # Anthropic in API mode falls back to SDK (no direct API web search yet)
        extra_args_api: dict[str, str | None] = {}
        if config.reasoning and config.reasoning.level != "off":
            extra_args_api.update(reasoning_to_cc_extra_args(config.reasoning))
        max_turns = 10
        budget = prepare_turn_budget(
            system_prompt, max_turns, effective_tools, provider="anthropic"
        )
        effective_backend = backend or get_backend("anthropic")
        options = SDKOptions(
            model=config.model,
            system_prompt=budget.system_prompt,
            allowed_tools=budget.allowed_tools,
            max_turns=budget.max_turns,
            extra_args=extra_args_api,
            mcp_servers=mcp_servers or {},
            response_schema=response_schema,
            resume=resume,
        )
        text, usage, session_id = await collect_text_from_query(
            prompt, options, effective_backend, message_buffer
        )
        in_tok = (
            usage.get("input_tokens", 0)
            + usage.get("cache_creation_input_tokens", 0)
            + usage.get("cache_read_input_tokens", 0)
        )
        return AgentResult(
            text=text,
            input_tokens=in_tok,
            output_tokens=usage.get("output_tokens", 0),
            thinking_tokens=usage.get("thinking_tokens", 0),
        ), session_id
    else:
        raise ValueError(f"Unsupported mode/provider: {config.mode}/{config.provider!r}")


# ---------------------------------------------------------------------------
# Structured query + validation helpers
# ---------------------------------------------------------------------------


async def _query_critic_structured(
    config: AgentModelConfig,
    prompt: str,
    *,
    system_prompt: str = "",
    label: str = "",
    message_buffer: list[str] | None = None,
    allowed_tools: list[str] | None = None,
    mcp_servers: dict[str, Any] | None = None,
    backend: SDKBackend | None = None,
) -> tuple[CriticOutput, AgentResult]:
    """Query a critic and validate the response as structured CriticOutput.

    Returns (validated CriticOutput, raw AgentResult).
    Uses agent_retry_loop (3 attempts) with selective retryable errors.

    Args:
        backend: Pre-created SDK backend for this critic.  Passed through
            to :func:`_query_critic` and kept alive across retries so
            validation failures can resume the conversation.
    """
    last_agent_result: list[AgentResult] = [AgentResult(text="")]

    async def _query(prompt_text: str, resume_session_id: str | None) -> QueryResult:
        result, session_id = await _query_critic(
            config,
            prompt_text,
            system_prompt=system_prompt,
            response_schema=CriticOutput,
            message_buffer=message_buffer,
            allowed_tools=allowed_tools,
            mcp_servers=mcp_servers,
            backend=backend,
            resume=resume_session_id,
        )
        last_agent_result[0] = result
        return QueryResult(raw_output=result.text, session_id=session_id, usage={})

    def _validate(result: QueryResult) -> tuple[CriticOutput, AgentResult]:
        validated = validate_json_output(result.raw_output, CriticOutput, "Critic")
        return CriticOutput(**validated), last_agent_result[0]

    def _on_exhausted(
        result: QueryResult | None, error: Exception
    ) -> tuple[CriticOutput, AgentResult]:
        if result is None:
            raise error
        agent_result = last_agent_result[0]
        logger.error(
            f"{label} validation failed after retries, preserving raw text as synthetic concern"
        )
        if message_buffer is not None:
            message_buffer.append(
                f"[WARNING] {label}: critic output could not be parsed after retries. "
                "Using synthetic fallback; review raw transcript for actual content."
            )
        raw = (agent_result.text or "(empty response)")[:500]
        fallback = CriticOutput(
            concerns=[
                Concern(
                    claim=f"[SYNTHETIC - PARSE ERROR] {raw}",
                    severity="high",
                    confidence="low",
                    category="other",
                )
            ],
            alternative_hypotheses=[],
            overall_assessment=agent_result.text or "(empty response)",
        )
        return fallback, agent_result

    return await agent_retry_loop(
        query_fn=_query,
        validate_fn=_validate,
        prompt=prompt,
        agent_name=label,
        retryable_errors=_RETRYABLE_ERRORS,
        on_exhausted=_on_exhausted,
    )


# ---------------------------------------------------------------------------
# Single-critic critique
# ---------------------------------------------------------------------------


async def run_single_critic_debate(
    config: AgentModelConfig,
    plan: dict[str, Any],
    notebook_path: Path,
    domain_knowledge: str = "",
    message_buffer: list[str] | None = None,
    persona: dict[str, str] | None = None,
    analysis_json: str = "",
    goal: str = "",
    prediction_history_records: list[PredictionRecord] | None = None,
    output_dir: Path | None = None,
    pending_abductions: str = "",
) -> DebateResult:
    """Run a single critique for one persona.

    Returns a DebateResult with structured output plus raw transcript.

    Args:
        notebook_path: Path to the run's lab_notebook.xml. The critic reads
            it to build either a compact TOC (SDK mode, paired with the
            mcp__notebook__read_notebook tool) or a full inline dump
            (API mode, no MCP tool available).
        prediction_history_records: Raw PredictionRecord list. Rendered as
            the compact tree in SDK mode (paired with the
            mcp__predictions__read_predictions tool) and as the full-detail
            trajectory in API mode (no tool available). See
            :func:`auto_core.agents.prediction_tool.format_compact_tree`
            and :func:`format_full_detail`.
        output_dir: Directory for MCP data files.
    """
    persona = persona or {"name": "Generic", "system_text": ""}
    persona_name = persona["name"]
    persona_text = persona["system_text"]
    persona_instructions = persona.get("instructions", "")

    label = f"{config.provider}:{config.model}"

    has_predictions = persona_name in PREDICTION_PERSONAS
    is_sdk = config.mode == "sdk"

    # Notebook wiring: SDK-mode critics get the compact TOC + MCP tool,
    # API-mode critics get the full XML inline (no tool to call). Only
    # parse entries in SDK mode - API mode reads the file directly and
    # would discard the parsed result.
    if is_sdk:
        notebook_entries = parse_notebook_entries(notebook_path)
        notebook_content = format_notebook_toc(notebook_entries)
    else:
        notebook_content = read_notebook(notebook_path) or "(empty notebook)"

    if has_predictions:
        tools, mcp_servers = _build_critic_tools_and_mcp(
            prediction_history_records if is_sdk else None,
            notebook_path=notebook_path if is_sdk else None,
            output_dir=output_dir,
        )
        # SDK critics get compact tree + read_predictions MCP tool to drill
        # into specific entries. API critics have no tool, so they get the
        # full-detail trajectory inline (mirrors the notebook TOC-vs-XML
        # split from PR #28). Both formatters return
        # "(no prediction history yet)" for empty input, so we call them
        # unconditionally to preserve the placeholder string (the prompt
        # builder still has a belt-and-suspenders `or` fallback).
        effective_prediction_history = (
            format_compact_tree(prediction_history_records)
            if is_sdk
            else format_full_detail(prediction_history_records)
        )
    else:
        tools, mcp_servers = _build_critic_tools_and_mcp(
            None,
            notebook_path=notebook_path if is_sdk else None,
            output_dir=output_dir,
        )
        effective_prediction_history = ""

    # MCP tool references only make sense in SDK mode where the tool is wired.
    # API-mode critics get prediction history inline but no tool to call.
    has_mcp_tool = has_predictions and is_sdk
    has_notebook_tool = is_sdk

    critic_system, critic_user = _build_critic_prompt(
        plan,
        notebook_content,
        domain_knowledge,
        persona_text=persona_text,
        persona_instructions=persona_instructions,
        analysis_json=analysis_json,
        prediction_history=effective_prediction_history,
        goal=goal,
        has_predictions=has_predictions,
        has_mcp_tool=has_mcp_tool,
        has_notebook_tool=has_notebook_tool,
        provider=config.provider,
        pending_abductions=pending_abductions,
    )
    # Each critic gets its own isolated backend so parallel debates don't
    # share CodexBackend state (which would race in _ensure_client).  The
    # backend stays alive across retry attempts, enabling session resume
    # for validation corrections.  API-mode critics don't need a backend.
    backend_cm = create_backend(config.provider) if config.mode == "sdk" else nullcontext()
    async with backend_cm as critic_backend:
        critic_output, critic_result = await _query_critic_structured(
            config,
            critic_user,
            system_prompt=critic_system,
            label=f"Critic ({persona_name}, {label})",
            message_buffer=message_buffer,
            allowed_tools=tools,
            mcp_servers=mcp_servers,
            backend=critic_backend,
        )
    if message_buffer is not None:
        message_buffer.append(f"[Critic/{persona_name}] {critic_result.text}")

    return DebateResult(
        persona=persona_name,
        critic_model=label,
        critic_output=critic_output,
        raw_transcript=[{"role": "critic", "content": critic_result.text}],
        input_tokens=critic_result.input_tokens,
        output_tokens=critic_result.output_tokens,
        thinking_tokens=critic_result.thinking_tokens,
    )


# ---------------------------------------------------------------------------
# Top-level debate orchestrator
# ---------------------------------------------------------------------------

# Delay between launching non-SDK (direct API) critics to spread rate limit load
_STAGGER_DELAY_SECONDS = 2.0


async def _staggered_debate(
    *,
    delay: float,
    config: AgentModelConfig,
    plan: dict[str, Any],
    notebook_path: Path,
    domain_knowledge: str = "",
    message_buffer: list[str] | None = None,
    persona: dict[str, str] | None = None,
    analysis_json: str = "",
    goal: str = "",
    prediction_history_records: list[PredictionRecord] | None = None,
    output_dir: Path | None = None,
    pending_abductions: str = "",
) -> DebateResult:
    """Wrapper that adds a startup delay before running a critique."""
    if delay > 0:
        await asyncio.sleep(delay)
    return await run_single_critic_debate(
        config=config,
        plan=plan,
        notebook_path=notebook_path,
        domain_knowledge=domain_knowledge,
        message_buffer=message_buffer,
        persona=persona,
        analysis_json=analysis_json,
        goal=goal,
        prediction_history_records=prediction_history_records,
        output_dir=output_dir,
        pending_abductions=pending_abductions,
    )


async def run_debate(
    critic_configs: list[AgentModelConfig],
    plan: dict[str, Any],
    notebook_path: Path,
    domain_knowledge: str = "",
    message_buffer: list[str] | None = None,
    message_buffers: dict[str, list[str]] | None = None,
    iteration: int = 0,
    analysis_json: str = "",
    goal: str = "",
    prediction_history_records: list[PredictionRecord] | None = None,
    output_dir: Path | None = None,
    pending_abductions: str = "",
) -> list[DebateResult]:
    """Run parallel critiques, one per persona, with rotating model assignment.

    On iteration 0, only Methodologist and Falsification Expert run (the
    Trajectory Critic and Evidence Auditor require prior iteration history).
    On iteration 1+, all four personas run. Model assignment rotates across
    iterations regardless of persona count.

    Args:
        critic_configs: Pool of critic model configs (round-robin assigned).
        plan: Scientist's plan dict.
        notebook_path: Path to the run's lab_notebook.xml. Each critic reads
            it directly so SDK-mode critics can build a compact TOC + MCP
            tool and API-mode critics can fall back to inline XML.
        domain_knowledge: Domain-specific context.
        message_buffer: Legacy single shared buffer.
        message_buffers: Per-persona buffers keyed by persona name.
        iteration: Current iteration number (for model rotation and persona filtering).
        analysis_json: Serialized analysis JSON from the Analyst.
        goal: Investigation goal string passed through to prompt builders.
        prediction_history_records: Raw PredictionRecord list. Each critic
            picks compact-tree (SDK) or full-detail (API) rendering from
            this list internally; no pre-rendered string needed.
        output_dir: Directory for MCP data files.

    Returns:
        List of DebateResult, one per active persona.
    """
    if not critic_configs:
        return []

    active_personas = [p for p in PERSONAS if iteration > 0 or p["name"] in ITERATION_0_PERSONAS]
    random.shuffle(active_personas)

    # Track how many non-SDK critics per provider have been launched
    # to assign incremental stagger delays
    provider_launch_count: dict[str, int] = {}

    tasks = []
    for persona_index, persona in enumerate(active_personas):
        model_index = get_model_index_for_debate(
            persona_index,
            iteration,
            len(critic_configs),
        )
        config = critic_configs[model_index]
        persona_name = persona["name"]

        # Resolve buffer: per-persona dict > shared legacy buffer > None
        buf: list[str] | None
        if message_buffers is not None:
            buf = message_buffers.setdefault(persona_name, [])
        else:
            buf = message_buffer

        # SDK (Anthropic) critics launch immediately; non-SDK critics
        # get staggered delays to spread rate limit load per provider
        if config.provider != "anthropic":
            count = provider_launch_count.get(config.provider, 0)
            delay = count * _STAGGER_DELAY_SECONDS
            provider_launch_count[config.provider] = count + 1
        else:
            delay = 0.0

        tasks.append(
            _staggered_debate(
                delay=delay,
                config=config,
                plan=plan,
                notebook_path=notebook_path,
                domain_knowledge=domain_knowledge,
                message_buffer=buf,
                persona=persona,
                analysis_json=analysis_json,
                goal=goal,
                prediction_history_records=prediction_history_records,
                output_dir=output_dir,
                pending_abductions=pending_abductions,
            )
        )

    raw_results = await asyncio.gather(*tasks, return_exceptions=True)
    successful: list[DebateResult] = []
    persona_names = [p["name"] for p in active_personas]
    for persona_name, r in zip(persona_names, raw_results, strict=True):
        if isinstance(r, BaseException):
            logger.error(f"Critic debate failed for {persona_name}: {r}", exc_info=r)
        else:
            successful.append(r)
    if not successful:
        failed_msgs = [str(r) for r in raw_results if isinstance(r, BaseException)]
        raise RuntimeError(
            f"All {len(raw_results)} critic debates failed. "
            f"Check API keys and network connectivity. Errors: {failed_msgs}"
        )
    if len(successful) < len(raw_results):
        logger.warning(f"Debate: {len(successful)}/{len(raw_results)} debates succeeded")
    return successful


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def _build_critic_prompt(
    plan: dict[str, Any],
    notebook_content: str,
    domain_knowledge: str,
    persona_text: str = "",
    persona_instructions: str = "",
    analysis_json: str = "",
    prediction_history: str = "",
    goal: str = "",
    has_predictions: bool = True,
    has_mcp_tool: bool = True,
    has_notebook_tool: bool = True,
    provider: str = "anthropic",
    pending_abductions: str = "",
) -> tuple[str, str]:
    """Build the (system, user) prompt pair sent to critic models.

    persona_instructions overrides the default instructions block when provided
    (used by the Trajectory Critic which needs arc-focused instructions).

    When has_predictions is False, prediction-related text (tool references,
    history section, pipeline context) is omitted from both prompts.

    When has_mcp_tool is False (API mode), prediction history is included
    inline but tool references and "MUST call" instructions are omitted
    since no MCP tool is available.

    When has_notebook_tool is True (SDK mode), the notebook section is a
    compact Table of Contents plus mcp__notebook__read_notebook guidance.
    When False (API mode), the full notebook XML is dumped inline.

    Returns:
        (system_prompt, user_prompt) tuple.
    """
    effective_instructions = persona_instructions or DEFAULT_CRITIC_INSTRUCTIONS

    if has_predictions:
        tool_name = PREDICTION_SPEC.mcp_tool_name
        prediction_evidence_text = "prediction history (what was tested and\nthe results), "
        prediction_history_section = (
            f"\n<prediction_history>{prediction_history or '(no prediction history yet)'}"
            "</prediction_history>"
        )

        if has_mcp_tool:
            prediction_role_text = (
                f", and a {tool_name} tool to\n"
                "drill into specific predictions for full detail (evidence, diagnostics,\n"
                "implications)"
            )
            prediction_tool_guidance = (
                f"- If {tool_name} is available and you need details about a specific "
                "pred_id,\n"
                "  outcome, or prediction chain, call it rather than guessing from the\n"
                "  compact summary."
            )
            prediction_pipeline_text = (
                "\nA compact summary of the prediction history is included in the context "
                f"below.\nWhen you need more detail on a specific prediction, call the "
                f"{tool_name}\ntool rather than guessing from the summary. If you reference "
                "a specific pred_id,\na prior confirmed/refuted outcome, or a prediction "
                "chain, inspect it with\nthe tool before finalizing your critique."
            )
            prediction_task_text = (
                f"\nThe prediction tree is provided above. Call {tool_name} to "
                "look up\nspecific prediction chains or full detail when the compact summary "
                "is\ninsufficient, especially for specific pred_ids or prior outcomes."
            )
        else:
            # API mode: full-detail prediction history is inlined below.
            # The MCP tool is unavailable (direct API calls can't invoke it),
            # so the prompt must describe the payload as full evidence, not
            # a compact summary.
            prediction_role_text = ""
            prediction_tool_guidance = (
                "- Use the inline prediction history when it is relevant to your critique;\n"
                "  do not guess or invent prior outcomes."
            )
            prediction_pipeline_text = (
                "\nThe full prediction history is included inline in the context below. "
                "Each entry\nshows the prediction, diagnostic, conditional implications, "
                "observed evidence,\nand parent-child reasoning links via indentation. No "
                "tool call is needed to\nexpand entries."
            )
            prediction_task_text = (
                "\nThe full prediction history is provided above with evidence and "
                "implications.\nUse it directly to verify prediction outcomes referenced "
                "in the plan."
            )
    else:
        prediction_role_text = ""
        prediction_tool_guidance = ""
        prediction_evidence_text = ""
        prediction_pipeline_text = ""
        prediction_history_section = ""
        prediction_task_text = ""

    # Pending abductions section for critic context
    if pending_abductions:
        pending_abductions_section = (
            f"\n<pending_abductions>\n{pending_abductions}\n</pending_abductions>"
        )
        abduction_task_text = (
            "\nCheck whether the plan addresses testable consequences from "
            "prior refutation reasoning. If a consequence was neither included "
            "as a prediction (via follows_from) nor deprioritized, flag it as "
            "a dropped thread."
        )
    else:
        pending_abductions_section = ""
        abduction_task_text = ""

    # Notebook section: compact TOC + tool in SDK mode, full inline XML in API mode.
    notebook_body = notebook_content or "(empty)"
    if has_notebook_tool:
        notebook_tool_name = NOTEBOOK_SPEC.mcp_tool_name
        notebook_role_text = (
            f", and a {notebook_tool_name} tool to read full notebook entries "
            "when the Table of Contents title is not enough context"
        )
        notebook_evidence_text = "a lab notebook Table of Contents"
        notebook_pipeline_text = (
            "\nThe notebook in <context> is a Table of Contents only (version, "
            f"source, title per entry). Call {notebook_tool_name} with "
            "versions=[...], source=..., search=..., or last_n=... when you "
            "need to read the full body of an entry before finalizing a "
            "concern that references prior iteration reasoning."
        )
        notebook_tool_guidance = (
            f"\n- Call {notebook_tool_name} whenever a concern depends on the "
            "specific content of a prior notebook entry. Do not invent prior "
            "reasoning from the TOC title alone."
        )
        notebook_section = f"<notebook_toc>{notebook_body}</notebook_toc>"
    else:
        notebook_role_text = ""
        notebook_evidence_text = "the lab notebook"
        notebook_pipeline_text = ""
        notebook_tool_guidance = ""
        notebook_section = f"<notebook>{notebook_body}</notebook>"

    prompt_provider = "gpt" if provider == "openai" else "claude"
    system = build_critic_system(prompt_provider).format(
        persona_text=persona_text,
        persona_instructions=effective_instructions,
        critic_output_schema=json.dumps(CRITIC_OUTPUT_SCHEMA, indent=2),
        prediction_role_text=prediction_role_text,
        prediction_evidence_text=prediction_evidence_text,
        prediction_pipeline_text=prediction_pipeline_text,
        prediction_tool_guidance=prediction_tool_guidance,
        notebook_role_text=notebook_role_text,
        notebook_evidence_text=notebook_evidence_text,
        notebook_pipeline_text=notebook_pipeline_text,
        notebook_tool_guidance=notebook_tool_guidance,
    )

    user = CRITIC_USER.format(
        goal=goal or "(no goal specified)",
        domain_knowledge=domain_knowledge or "(none provided)",
        notebook_section=notebook_section,
        analysis_json=analysis_json or "(no analysis yet)",
        prediction_history_section=prediction_history_section,
        pending_abductions_section=pending_abductions_section,
        plan_json=json.dumps(plan, indent=2),
        prediction_task_text=prediction_task_text,
        abduction_task_text=abduction_task_text,
    )
    return system, user
