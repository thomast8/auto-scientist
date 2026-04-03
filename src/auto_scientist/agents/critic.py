"""Critic: multi-model critique dispatcher.

Plain API call (OpenAI/Google/Anthropic SDK), no agent tools needed.
Input: scientist's plan + analysis JSON + prediction history + lab notebook
+ domain knowledge.
Output: structured critique with tagged concerns and alternative hypotheses,
plus raw transcript for debugging.

Critics receive the full evidence base but do not see Python code
(implementation is the Coder's domain).

Personas provide diverse critical perspectives. Each critique runs one persona;
model assignment rotates across iterations so no model is always the same role.
"""

import asyncio
import json
import logging
import random
from typing import Any

from pydantic import BaseModel

from auto_scientist.agent_result import AgentResult
from auto_scientist.agents.debate_models import (
    CRITIC_OUTPUT_SCHEMA,
    Concern,
    CriticOutput,
    DebateResult,
)
from auto_scientist.model_config import AgentModelConfig, reasoning_to_cc_extra_args
from auto_scientist.models.google_client import query_google
from auto_scientist.models.openai_client import query_openai
from auto_scientist.prompts.critic import (
    CRITIC_SYSTEM_BASE,
    CRITIC_USER,
    DEFAULT_CRITIC_INSTRUCTIONS,
    ITERATION_0_PERSONAS,
    PERSONAS,
    get_model_index_for_debate,
)
from auto_scientist.retry import QueryResult, agent_retry_loop
from auto_scientist.sdk_backend import SDKOptions, get_backend
from auto_scientist.sdk_utils import (
    collect_text_from_query,
    validate_json_output,
    with_turn_budget,
)

logger = logging.getLogger(__name__)

# Exceptions worth retrying (transient network/rate-limit issues).
# Non-retryable errors (ValueError, TypeError, ImportError, auth errors)
# propagate immediately so the user gets a clear failure instead of a
# misleading retry-then-fail cycle.
_RETRYABLE_ERRORS = (ConnectionError, TimeoutError, OSError, RuntimeError)


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
) -> AgentResult:
    """Dispatch a prompt to the appropriate provider and mode.

    Routes based on config.mode first:
    - mode='sdk': use the backend abstraction (Claude Code or Codex)
    - mode='api': use direct API calls (OpenAI, Google, Anthropic)
    """
    if config.mode == "sdk" and config.provider in ("anthropic", "openai"):
        # SDK mode: use the backend abstraction
        extra_args: dict[str, str | None] = {"setting-sources": ""}
        if config.reasoning and config.reasoning.level != "off":
            extra_args.update(reasoning_to_cc_extra_args(config.reasoning))
        max_turns = 5
        allowed_tools = ["WebSearch"]
        backend = get_backend(config.provider)
        options = SDKOptions(
            model=config.model,
            system_prompt=with_turn_budget(system_prompt, max_turns, allowed_tools),
            allowed_tools=allowed_tools,
            max_turns=max_turns,
            extra_args=extra_args,
        )
        text, usage, _session_id = await collect_text_from_query(
            prompt, options, backend, message_buffer
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
        )

    # API mode: direct provider API calls
    effective_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

    if config.provider == "openai":
        return await query_openai(
            config.model,
            effective_prompt,
            web_search=True,
            reasoning=config.reasoning,
            response_schema=response_schema,
        )
    elif config.provider == "google":
        return await query_google(
            config.model,
            effective_prompt,
            web_search=True,
            reasoning=config.reasoning,
            response_schema=response_schema,
        )
    elif config.provider == "anthropic":
        # Anthropic in API mode falls back to SDK (no direct API web search yet)
        extra_args_api: dict[str, str | None] = {"setting-sources": ""}
        if config.reasoning and config.reasoning.level != "off":
            extra_args_api.update(reasoning_to_cc_extra_args(config.reasoning))
        max_turns = 5
        allowed_tools = ["WebSearch"]
        backend = get_backend("anthropic")
        options = SDKOptions(
            model=config.model,
            system_prompt=with_turn_budget(system_prompt, max_turns, allowed_tools),
            allowed_tools=allowed_tools,
            max_turns=max_turns,
            extra_args=extra_args_api,
        )
        text, usage, _session_id = await collect_text_from_query(
            prompt, options, backend, message_buffer
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
        )
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
) -> tuple[CriticOutput, AgentResult]:
    """Query a critic and validate the response as structured CriticOutput.

    Returns (validated CriticOutput, raw AgentResult).
    Uses agent_retry_loop (3 attempts) with selective retryable errors.
    """
    last_agent_result: list[AgentResult] = [AgentResult(text="")]

    async def _query(prompt_text: str, resume_session_id: str | None) -> QueryResult:
        result = await _query_critic(
            config,
            prompt_text,
            system_prompt=system_prompt,
            response_schema=CriticOutput,
            message_buffer=message_buffer,
        )
        last_agent_result[0] = result
        return QueryResult(raw_output=result.text, session_id=None, usage={})

    def _validate(result: QueryResult) -> tuple[CriticOutput, AgentResult]:
        validated = validate_json_output(result.raw_output, CriticOutput, "Critic")
        return CriticOutput(**validated), last_agent_result[0]

    def _on_exhausted(
        result: QueryResult | None, error: Exception
    ) -> tuple[CriticOutput, AgentResult]:
        # SDK/transport errors should propagate, not produce synthetic fallbacks.
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
    notebook_content: str,
    domain_knowledge: str = "",
    message_buffer: list[str] | None = None,
    persona: dict[str, str] | None = None,
    analysis_json: str = "",
    prediction_history: str = "",
    goal: str = "",
) -> DebateResult:
    """Run a single critique for one persona.

    Returns a DebateResult with structured output plus raw transcript.
    """
    persona = persona or {"name": "Generic", "system_text": ""}
    persona_name = persona["name"]
    persona_text = persona["system_text"]
    persona_instructions = persona.get("instructions", "")

    label = f"{config.provider}:{config.model}"

    critic_system, critic_user = _build_critic_prompt(
        plan,
        notebook_content,
        domain_knowledge,
        persona_text=persona_text,
        persona_instructions=persona_instructions,
        analysis_json=analysis_json,
        prediction_history=prediction_history,
        goal=goal,
    )
    critic_output, critic_result = await _query_critic_structured(
        config,
        critic_user,
        system_prompt=critic_system,
        label=f"Critic ({persona_name}, {label})",
        message_buffer=message_buffer,
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
    notebook_content: str,
    domain_knowledge: str = "",
    message_buffer: list[str] | None = None,
    persona: dict[str, str] | None = None,
    analysis_json: str = "",
    prediction_history: str = "",
    goal: str = "",
) -> DebateResult:
    """Wrapper that adds a startup delay before running a critique."""
    if delay > 0:
        await asyncio.sleep(delay)
    return await run_single_critic_debate(
        config=config,
        plan=plan,
        notebook_content=notebook_content,
        domain_knowledge=domain_knowledge,
        message_buffer=message_buffer,
        persona=persona,
        analysis_json=analysis_json,
        prediction_history=prediction_history,
        goal=goal,
    )


async def run_debate(
    critic_configs: list[AgentModelConfig],
    plan: dict[str, Any],
    notebook_content: str,
    domain_knowledge: str = "",
    message_buffer: list[str] | None = None,
    message_buffers: dict[str, list[str]] | None = None,
    iteration: int = 0,
    analysis_json: str = "",
    prediction_history: str = "",
    goal: str = "",
) -> list[DebateResult]:
    """Run parallel critiques, one per persona, with rotating model assignment.

    On iteration 0, only Methodologist and Falsification Expert run (the
    Trajectory Critic and Evidence Auditor require prior iteration history).
    On iteration 1+, all four personas run. Model assignment rotates across
    iterations regardless of persona count.

    Args:
        critic_configs: Pool of critic model configs (round-robin assigned).
        plan: Scientist's plan dict.
        notebook_content: Current lab notebook content.
        domain_knowledge: Domain-specific context.
        message_buffer: Legacy single shared buffer.
        message_buffers: Per-persona buffers keyed by persona name.
        iteration: Current iteration number (for model rotation and persona filtering).
        analysis_json: Serialized analysis JSON from the Analyst.
        prediction_history: Formatted prediction history string.
        goal: Investigation goal string passed through to prompt builders.

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
                notebook_content=notebook_content,
                domain_knowledge=domain_knowledge,
                message_buffer=buf,
                persona=persona,
                analysis_json=analysis_json,
                prediction_history=prediction_history,
                goal=goal,
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
) -> tuple[str, str]:
    """Build the (system, user) prompt pair sent to critic models.

    persona_instructions overrides the default instructions block when provided
    (used by the Trajectory Critic which needs arc-focused instructions).

    Returns:
        (system_prompt, user_prompt) tuple.
    """
    effective_instructions = persona_instructions or DEFAULT_CRITIC_INSTRUCTIONS

    system = CRITIC_SYSTEM_BASE.format(
        persona_text=persona_text,
        persona_instructions=effective_instructions,
        critic_output_schema=json.dumps(CRITIC_OUTPUT_SCHEMA, indent=2),
    )

    user = CRITIC_USER.format(
        goal=goal or "(no goal specified)",
        domain_knowledge=domain_knowledge or "(none provided)",
        notebook_content=notebook_content or "(empty)",
        analysis_json=analysis_json or "(no analysis yet)",
        prediction_history=prediction_history or "(no prediction history yet)",
        plan_json=json.dumps(plan, indent=2),
    )
    return system, user
