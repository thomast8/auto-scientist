"""Stop gate agents: completeness assessment, stop debate, and stop revision.

When the Scientist proposes stopping, the stop gate validates whether the
investigation goal has been thoroughly addressed before honoring the decision.
"""

import json
import logging
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from auto_scientist.agents.debate_models import (
    CRITIC_OUTPUT_SCHEMA,
    CriticOutput,
    DebateResult,
)
from auto_scientist.agents.notebook_tool import (
    NOTEBOOK_SPEC,
    build_notebook_mcp_server,
    format_notebook_toc,
)
from auto_scientist.agents.prediction_tool import (
    PREDICTION_SPEC,
    build_prediction_mcp_server,
    format_compact_tree,
    format_full_detail,
)
from auto_scientist.agents.scientist import SCIENTIST_PLAN_SCHEMA
from auto_scientist.model_config import AgentModelConfig
from auto_scientist.notebook import parse_notebook_entries, read_notebook
from auto_scientist.prompts.stop_gate import (
    ASSESSMENT_SCHEMA,
    ASSESSMENT_USER,
    STOP_CRITIC_USER,
    STOP_REVISION_USER,
    build_assessment_system,
    build_stop_critic_system,
    build_stop_revision_system,
)
from auto_scientist.retry import QueryResult, agent_retry_loop
from auto_scientist.schemas import CompletenessAssessmentOutput, ScientistPlanOutput
from auto_scientist.sdk_backend import SDKBackend, SDKOptions, create_backend, get_backend
from auto_scientist.sdk_utils import (
    collect_text_from_query,
    prepare_turn_budget,
    validate_json_output,
)
from auto_scientist.state import PredictionRecord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Completeness Assessment
# ---------------------------------------------------------------------------


async def run_completeness_assessment(
    goal: str,
    stop_reason: str,
    notebook_path: Path,
    domain_knowledge: str = "",
    prediction_history: list[PredictionRecord] | None = None,
    model: str | None = None,
    message_buffer: list[str] | None = None,
    provider: str = "anthropic",
    output_dir: Path | None = None,
    pending_abductions: str = "",
) -> dict[str, Any]:
    """Assess whether the investigation goal has been thoroughly addressed.

    Returns a structured assessment with sub-questions, coverage ratings,
    and a stop/continue recommendation.
    """
    notebook_path = Path(notebook_path)
    notebook_entries = parse_notebook_entries(notebook_path)

    # Build tools: WebSearch + notebook MCP + optional MCP for prediction drill-down
    tools: list[str] = ["WebSearch"]
    mcp_servers: dict[str, Any] = {
        "notebook": build_notebook_mcp_server(notebook_path, output_dir=output_dir),
    }
    tools.append(NOTEBOOK_SPEC.mcp_tool_name)
    if prediction_history:
        mcp_servers["predictions"] = build_prediction_mcp_server(
            prediction_history, output_dir=output_dir
        )
        tools.append(PREDICTION_SPEC.mcp_tool_name)

    abductions_section = ""
    if pending_abductions:
        abductions_section = (
            "<pending_abductions>\n"
            "Alternative explanations the Scientist raised for refuted "
            "predictions but never tested. Each unaddressed entry is an "
            "open sub-question for coverage rating.\n\n"
            f"{pending_abductions}\n"
            "</pending_abductions>\n"
        )

    user_prompt = ASSESSMENT_USER.format(
        goal=goal,
        stop_reason=stop_reason,
        domain_knowledge=domain_knowledge or "(no domain knowledge provided)",
        prediction_history=format_compact_tree(prediction_history),
        pending_abductions_section=abductions_section,
        notebook_content=format_notebook_toc(notebook_entries),
    )

    json_instruction = (
        "\n\n## Output Format\n"
        "Tool calls are allowed before the final JSON response.\n"
        'The "JSON only" rule applies only to the final assistant message.\n'
        "You MUST respond with ONLY valid JSON matching the schema below.\n"
        "No markdown fencing. No explanation. No other text.\n\n"
        f"Schema:\n{json.dumps(ASSESSMENT_SCHEMA, indent=2)}"
    )

    max_turns = 10
    prompt_provider = "gpt" if provider == "openai" else "claude"
    has_predictions = bool(prediction_history)
    assessment_system = build_assessment_system(prompt_provider, has_predictions=has_predictions)
    budget = prepare_turn_budget(
        assessment_system + json_instruction, max_turns, tools, provider=provider
    )
    backend = get_backend(provider)
    options = SDKOptions(
        system_prompt=budget.system_prompt,
        allowed_tools=budget.allowed_tools,
        max_turns=budget.max_turns,
        model=model,
        extra_args={},
        mcp_servers=mcp_servers,
        response_schema=CompletenessAssessmentOutput,
    )

    async def _query(prompt: str, resume_session_id: str | None) -> QueryResult:
        opts = replace(options, resume=resume_session_id) if resume_session_id else options
        raw, usage, sid = await collect_text_from_query(
            prompt, opts, backend, message_buffer, agent_name="Completeness Assessment"
        )
        return QueryResult(raw_output=raw, session_id=sid, usage=usage)

    def _validate(result: QueryResult) -> dict[str, Any]:
        return validate_json_output(
            result.raw_output, CompletenessAssessmentOutput, "Completeness Assessment"
        )

    return await agent_retry_loop(
        query_fn=_query,
        validate_fn=_validate,
        prompt=user_prompt,
        agent_name="Completeness Assessment",
    )


# ---------------------------------------------------------------------------
# Stop Debate
# ---------------------------------------------------------------------------


async def _query_stop_agent(
    config: AgentModelConfig,
    user_prompt: str,
    system_prompt: str,
    output_model: type[BaseModel],
    label: str,
    message_buffer: list[str] | None = None,
    allowed_tools: list[str] | None = None,
    mcp_servers: dict[str, Any] | None = None,
    backend: SDKBackend | None = None,
) -> tuple[Any, Any]:
    """Query a stop gate agent (critic or scientist) via provider-aware dispatch.

    Uses direct API clients for OpenAI/Google, Claude Code SDK for Anthropic.
    Returns (validated_model_instance, result_obj with text/token counts).

    Args:
        backend: Pre-created SDK backend for this stop-gate critic.
            Passed through to :func:`_query_critic` for subprocess isolation
            and session resume across retries.
    """
    from auto_scientist.agents.critic import _query_critic

    last_agent_result: list[Any] = []

    async def _query(prompt: str, resume_session_id: str | None) -> QueryResult:
        result, session_id = await _query_critic(
            config,
            prompt,
            system_prompt=system_prompt,
            response_schema=output_model,
            message_buffer=message_buffer,
            allowed_tools=allowed_tools,
            mcp_servers=mcp_servers,
            backend=backend,
            resume=resume_session_id,
        )
        last_agent_result.clear()
        last_agent_result.append(result)
        return QueryResult(raw_output=result.text, session_id=session_id, usage={})

    def _validate(result: QueryResult) -> tuple[Any, Any]:
        parsed = validate_json_output(result.raw_output, output_model, label)
        return output_model.model_validate(parsed), last_agent_result[0]

    return await agent_retry_loop(
        query_fn=_query,
        validate_fn=_validate,
        prompt=user_prompt,
        agent_name=label,
    )


async def run_single_stop_debate(
    config: AgentModelConfig,
    stop_reason: str,
    completeness_assessment: dict[str, Any],
    notebook_path: Path,
    domain_knowledge: str = "",
    message_buffer: list[str] | None = None,
    persona: dict[str, str] | None = None,
    analysis_json: str = "",
    goal: str = "",
    prediction_history_records: list[PredictionRecord] | None = None,
    output_dir: Path | None = None,
) -> DebateResult:
    """Run a single critic persona's challenge of the stop decision.

    The scientist responds once via run_scientist_stop_revision after
    all critics have challenged.

    Args:
        notebook_path: Path to the run's lab_notebook.xml. SDK-mode critics
            get a compact TOC + mcp__notebook__read_notebook tool; API-mode
            critics fall back to the full inline XML.
        prediction_history_records: Raw PredictionRecord list. Rendered as
            the compact tree in SDK mode (paired with the
            mcp__predictions__read_predictions tool) and as the full-detail
            trajectory in API mode (no tool available).
        output_dir: Directory for MCP data files.
    """
    from auto_scientist.agents.critic import _build_critic_tools_and_mcp

    persona = persona or {"name": "Generic", "system_text": ""}
    persona_name = persona["name"]
    persona_text = persona["system_text"]
    persona_instructions = persona.get("instructions", "")

    label = f"{config.provider}:{config.model}"
    assessment_json = json.dumps(completeness_assessment, indent=2)

    is_sdk = config.mode == "sdk"
    notebook_path = Path(notebook_path)
    if is_sdk:
        notebook_entries = parse_notebook_entries(notebook_path)
        notebook_section = f"<notebook_toc>{format_notebook_toc(notebook_entries)}</notebook_toc>"
    else:
        notebook_xml = read_notebook(notebook_path) or "(empty notebook)"
        notebook_section = f"<notebook>{notebook_xml}</notebook>"

    tools, mcp_servers = _build_critic_tools_and_mcp(
        prediction_history_records,
        notebook_path=notebook_path if is_sdk else None,
        output_dir=output_dir,
    )

    prompt_provider = "gpt" if config.provider == "openai" else "claude"
    has_predictions = bool(prediction_history_records)
    critic_system = build_stop_critic_system(
        prompt_provider,
        has_predictions=has_predictions,
        has_notebook_tool=is_sdk,
    ).format(
        persona_text=persona_text,
        persona_instructions=persona_instructions or "",
        critic_output_schema=json.dumps(CRITIC_OUTPUT_SCHEMA, indent=2),
    )
    # SDK critics get compact tree + read_predictions MCP tool to drill
    # into entries. API critics have no tool, so they get the full-detail
    # trajectory inline.
    if prediction_history_records:
        effective_prediction_history = (
            format_compact_tree(prediction_history_records)
            if is_sdk
            else format_full_detail(prediction_history_records)
        )
    else:
        effective_prediction_history = ""

    critic_user = STOP_CRITIC_USER.format(
        goal=goal,
        domain_knowledge=domain_knowledge,
        notebook_section=notebook_section,
        analysis_json=analysis_json,
        prediction_history=effective_prediction_history,
        stop_reason=stop_reason,
        completeness_assessment=assessment_json,
    )

    # Same isolation as run_single_critic_debate: each parallel stop critic
    # gets its own backend so concurrent asyncio.gather calls in the
    # orchestrator don't race through the shared CodexBackend singleton.
    backend_cm = create_backend(config.provider) if config.mode == "sdk" else nullcontext()
    async with backend_cm as stop_backend:
        critic_output, critic_result = await _query_stop_agent(
            config,
            critic_user,
            system_prompt=critic_system,
            output_model=CriticOutput,
            label=f"Stop Critic ({persona_name}, {label})",
            message_buffer=message_buffer,
            allowed_tools=tools,
            mcp_servers=mcp_servers,
            backend=stop_backend,
        )

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
# Scientist Stop Revision
# ---------------------------------------------------------------------------


async def run_scientist_stop_revision(
    stop_reason: str,
    completeness_assessment: dict[str, Any],
    concern_ledger: list[dict[str, Any]],
    analysis: dict[str, Any],
    notebook_path: Path,
    version: str,
    domain_knowledge: str = "",
    prediction_history: list[PredictionRecord] | None = None,
    model: str | None = None,
    message_buffer: list[str] | None = None,
    goal: str = "",
    provider: str = "anthropic",
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Revise the stop decision after the stop debate.

    Returns a ScientistPlanOutput-compatible dict. If should_stop is still
    true, the stop is upheld. If false, the plan contains a real experiment.
    """
    notebook_path = Path(notebook_path)
    notebook_entries = parse_notebook_entries(notebook_path)

    # Build tools: WebSearch + notebook MCP + MCP prediction tree
    tools = ["WebSearch"]
    mcp_servers: dict[str, Any] = {
        "notebook": build_notebook_mcp_server(notebook_path, output_dir=output_dir),
    }
    tools.append(NOTEBOOK_SPEC.mcp_tool_name)
    if prediction_history:
        mcp_servers["predictions"] = build_prediction_mcp_server(
            prediction_history, output_dir=output_dir
        )
        tools.append(PREDICTION_SPEC.mcp_tool_name)

    user_prompt = STOP_REVISION_USER.format(
        goal=goal or "(no goal specified)",
        domain_knowledge=domain_knowledge or "(no domain knowledge provided)",
        analysis_json=json.dumps(analysis, indent=2) if analysis else "(no analysis)",
        notebook_content=format_notebook_toc(notebook_entries),
        stop_reason=stop_reason,
        completeness_assessment=json.dumps(completeness_assessment, indent=2),
        concern_ledger=(
            json.dumps(concern_ledger, indent=2) if concern_ledger else "(no concerns raised)"
        ),
        prediction_history=format_compact_tree(prediction_history),
        version=version,
        plan_schema=json.dumps(SCIENTIST_PLAN_SCHEMA, indent=2),
    )

    json_instruction = (
        "\n\n## Output Format\n"
        "Tool calls are allowed before the final JSON response.\n"
        'The "JSON only" rule applies only to the final assistant message.\n'
        "Respond with valid JSON matching the schema below.\n"
        "No markdown fencing. No explanation. No other text.\n\n"
        f"Schema:\n{json.dumps(SCIENTIST_PLAN_SCHEMA, indent=2)}"
    )

    max_turns = 15
    prompt_provider = "gpt" if provider == "openai" else "claude"
    has_predictions = bool(prediction_history)
    stop_rev_system = build_stop_revision_system(prompt_provider, has_predictions=has_predictions)
    budget = prepare_turn_budget(
        stop_rev_system + json_instruction, max_turns, tools, provider=provider
    )
    backend = get_backend(provider)
    options = SDKOptions(
        system_prompt=budget.system_prompt,
        allowed_tools=budget.allowed_tools,
        max_turns=budget.max_turns,
        model=model,
        extra_args={},
        mcp_servers=mcp_servers,
        response_schema=ScientistPlanOutput,
    )

    async def _query(prompt: str, resume_session_id: str | None) -> QueryResult:
        opts = replace(options, resume=resume_session_id) if resume_session_id else options
        raw, usage, sid = await collect_text_from_query(
            prompt, opts, backend, message_buffer, agent_name="Stop Revision"
        )
        return QueryResult(raw_output=raw, session_id=sid, usage=usage)

    def _validate(result: QueryResult) -> dict[str, Any]:
        return validate_json_output(result.raw_output, ScientistPlanOutput, "Stop Revision")

    return await agent_retry_loop(
        query_fn=_query,
        validate_fn=_validate,
        prompt=user_prompt,
        agent_name="Stop Revision",
    )
