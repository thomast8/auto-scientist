"""Critic: multi-model critique dispatcher with Scientist debate loop.

Plain API call (OpenAI/Google/Anthropic SDK), no agent tools needed.
Input: scientist's plan + analysis JSON + prediction history + lab notebook
+ domain knowledge.
Output: structured critique with tagged concerns, alternative hypotheses,
and the scientist's structured defense, plus raw transcript for debugging.

Both the Critic and Scientist (during debate) share the full evidence base.
Neither sees Python code (implementation is the Coder's domain).

Personas provide diverse critical perspectives. Each debate runs one persona;
model assignment rotates across iterations so no model is always the same role.
"""

import asyncio
import json
import logging
import random
from typing import Any

from claude_code_sdk import ClaudeCodeOptions
from pydantic import BaseModel

from auto_scientist.agent_result import AgentResult
from auto_scientist.agents.debate_models import (
    CRITIC_OUTPUT_SCHEMA,
    SCIENTIST_DEFENSE_SCHEMA,
    Concern,
    CriticOutput,
    DebateResult,
    DebateRound,
    ScientistDefense,
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
    SCIENTIST_DEBATE_SYSTEM,
    SCIENTIST_DEBATE_USER,
    get_model_index_for_debate,
)
from auto_scientist.sdk_utils import (
    OutputValidationError,
    collect_text_from_query,
    validate_json_output,
    with_turn_budget,
)

logger = logging.getLogger(__name__)

MAX_RETRIES = 1  # 1 retry = 2 total attempts

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
    """Dispatch a prompt to the appropriate provider with web search enabled."""
    # For direct-API providers, prepend system_prompt to the user prompt
    # (these APIs accept a single prompt string).
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
        extra_args: dict[str, str | None] = {"setting-sources": ""}
        if config.reasoning and config.reasoning.level != "off":
            extra_args.update(reasoning_to_cc_extra_args(config.reasoning))
        max_turns = 5  # Allows 1-2 web searches + structured JSON response + buffer
        allowed_tools = ["WebSearch"]
        options = ClaudeCodeOptions(
            model=config.model,
            system_prompt=with_turn_budget(system_prompt, max_turns, allowed_tools),
            allowed_tools=allowed_tools,
            max_turns=max_turns,
            extra_args=extra_args,
        )
        text, usage = await collect_text_from_query(prompt, options, message_buffer)
        # SDK splits input tokens across cache buckets
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
        raise ValueError(f"Unknown critic provider: {config.provider!r}")


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
    Retries once on validation failure with a correction hint.
    """
    result = AgentResult(text="")
    correction_hint = ""

    for attempt in range(MAX_RETRIES + 1):
        effective_prompt = prompt + correction_hint
        try:
            result = await _query_critic(
                config,
                effective_prompt,
                system_prompt=system_prompt,
                response_schema=CriticOutput,
                message_buffer=message_buffer,
            )
        except _RETRYABLE_ERRORS as e:
            if attempt < MAX_RETRIES:
                logger.warning(f"{label} transient error ({e}), retrying (attempt {attempt + 1})")
                continue
            raise

        try:
            validated = validate_json_output(result.text, CriticOutput, "Critic")
            return CriticOutput(**validated), result
        except OutputValidationError as e:
            if attempt < MAX_RETRIES:
                correction_hint = f"\n\n{e.correction_prompt()}"
                logger.warning(f"{label} validation failed, retrying: {e}")
            else:
                logger.error(
                    f"{label} validation failed after retries, "
                    "preserving raw text as synthetic concern"
                )
                if message_buffer is not None:
                    message_buffer.append(
                        f"[WARNING] {label}: critic output could not be parsed after retries. "
                        "Using synthetic fallback; review raw transcript for actual content."
                    )
                raw = (result.text or "(empty response)")[:500]
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
                    overall_assessment=result.text or "(empty response)",
                )
                return fallback, result

    raise RuntimeError("Unreachable: critic structured query loop exited without return")


async def _query_scientist_structured(
    config: AgentModelConfig,
    prompt: str,
    system_prompt: str,
    *,
    label: str = "",
    message_buffer: list[str] | None = None,
) -> tuple[ScientistDefense, AgentResult]:
    """Query the scientist (direct API) and validate as structured ScientistDefense.

    Returns (validated ScientistDefense, raw AgentResult).
    Retries once on validation failure with a correction hint.
    """
    result = AgentResult(text="")
    correction_hint = ""

    for attempt in range(MAX_RETRIES + 1):
        effective_prompt = prompt + correction_hint
        try:
            result = await _query_critic(
                config,
                effective_prompt,
                system_prompt=system_prompt,
                response_schema=ScientistDefense,
                message_buffer=message_buffer,
            )
        except _RETRYABLE_ERRORS as e:
            if attempt < MAX_RETRIES:
                logger.warning(f"{label} transient error ({e}), retrying (attempt {attempt + 1})")
                continue
            raise

        try:
            validated = validate_json_output(result.text, ScientistDefense, "Scientist-debate")
            return ScientistDefense(**validated), result
        except OutputValidationError as e:
            if attempt < MAX_RETRIES:
                correction_hint = f"\n\n{e.correction_prompt()}"
                logger.warning(f"{label} validation failed, retrying: {e}")
            else:
                logger.error(f"{label} defense validation failed after retries, using raw text")
                if message_buffer is not None:
                    message_buffer.append(
                        f"[WARNING] {label}: scientist defense could not be parsed after retries. "
                        "Using raw text as fallback."
                    )
                fallback = ScientistDefense(
                    responses=[],
                    additional_points=result.text or "(empty response)",
                )
                return fallback, result

    raise RuntimeError("Unreachable: scientist defense query loop exited without return")


# ---------------------------------------------------------------------------
# Single-critic debate
# ---------------------------------------------------------------------------


async def run_single_critic_debate(
    config: AgentModelConfig,
    plan: dict[str, Any],
    notebook_content: str,
    domain_knowledge: str = "",
    max_rounds: int = 1,
    scientist_config: AgentModelConfig | None = None,
    message_buffer: list[str] | None = None,
    persona: dict[str, str] | None = None,
    analysis_json: str = "",
    prediction_history: str = "",
    goal: str = "",
) -> DebateResult:
    """Run a multi-round debate between one critic (with persona) and the scientist.

    Returns a DebateResult with structured output per round plus raw transcript.
    """
    if scientist_config is None:
        scientist_config = AgentModelConfig(model="claude-sonnet-4-6")

    persona = persona or {"name": "Generic", "system_text": ""}
    persona_name = persona["name"]
    persona_text = persona["system_text"]
    persona_instructions = persona.get("instructions", "")

    label = f"{config.provider}:{config.model}"
    raw_transcript: list[dict[str, str]] = []
    rounds: list[DebateRound] = []
    total_in = 0
    total_out = 0
    total_think = 0

    # Round 1: initial critique (no defense context)
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
        label=f"Critic ({persona_name}, {label}) round 1",
        message_buffer=message_buffer,
    )
    total_in += critic_result.input_tokens
    total_out += critic_result.output_tokens
    total_think += critic_result.thinking_tokens
    raw_transcript.append({"role": "critic", "content": critic_result.text})
    if message_buffer is not None:
        message_buffer.append(f"[Critic/{persona_name}] {critic_result.text}")

    if max_rounds <= 1:
        # Single round: no scientist defense
        rounds.append(DebateRound(critic_output=critic_output))
    else:
        # Multi-round: scientist responds, then critic critiques again
        scientist_defense = None
        for round_num in range(1, max_rounds):
            scientist_system = SCIENTIST_DEBATE_SYSTEM.format(
                scientist_defense_schema=json.dumps(SCIENTIST_DEFENSE_SCHEMA, indent=2),
            )
            scientist_user_prompt = _build_scientist_debate_user_prompt(
                plan=plan,
                notebook_content=notebook_content,
                domain_knowledge=domain_knowledge,
                critique=critic_result.text,
                critic_persona=persona_name,
                analysis_json=analysis_json,
                prediction_history=prediction_history,
                goal=goal,
            )
            scientist_defense, sci_result = await _query_scientist_structured(
                scientist_config,
                scientist_user_prompt,
                scientist_system,
                label=f"Scientist ({persona_name}) round {round_num}",
                message_buffer=message_buffer,
            )
            total_in += sci_result.input_tokens
            total_out += sci_result.output_tokens
            total_think += sci_result.thinking_tokens
            raw_transcript.append({"role": "scientist", "content": sci_result.text})
            if message_buffer is not None:
                message_buffer.append(f"[Scientist] {sci_result.text}")

            # Record the round with both critic output and scientist defense
            rounds.append(
                DebateRound(
                    critic_output=critic_output,
                    scientist_defense=scientist_defense,
                )
            )

            # Stateless: critic gets the defense but not their own prior critique
            critic_system, critic_user = _build_critic_prompt(
                plan,
                notebook_content,
                domain_knowledge,
                scientist_defense=sci_result.text,
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
                label=f"Critic ({persona_name}, {label}) round {round_num + 1}",
                message_buffer=message_buffer,
            )
            total_in += critic_result.input_tokens
            total_out += critic_result.output_tokens
            total_think += critic_result.thinking_tokens
            raw_transcript.append({"role": "critic", "content": critic_result.text})
            if message_buffer is not None:
                message_buffer.append(f"[Critic/{persona_name}] {critic_result.text}")

        # Final round's critic output (no defense after it)
        rounds.append(DebateRound(critic_output=critic_output))

    return DebateResult(
        persona=persona_name,
        critic_model=label,
        rounds=rounds,
        raw_transcript=raw_transcript,
        input_tokens=total_in,
        output_tokens=total_out,
        thinking_tokens=total_think,
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
    max_rounds: int = 1,
    scientist_config: AgentModelConfig | None = None,
    message_buffer: list[str] | None = None,
    persona: dict[str, str] | None = None,
    analysis_json: str = "",
    prediction_history: str = "",
    goal: str = "",
) -> DebateResult:
    """Wrapper that adds a startup delay before running a debate."""
    if delay > 0:
        await asyncio.sleep(delay)
    return await run_single_critic_debate(
        config=config,
        plan=plan,
        notebook_content=notebook_content,
        domain_knowledge=domain_knowledge,
        max_rounds=max_rounds,
        scientist_config=scientist_config,
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
    max_rounds: int = 1,
    scientist_config: AgentModelConfig | None = None,
    message_buffer: list[str] | None = None,
    message_buffers: dict[str, list[str]] | None = None,
    iteration: int = 0,
    analysis_json: str = "",
    prediction_history: str = "",
    goal: str = "",
) -> list[DebateResult]:
    """Run parallel debates, one per persona, with rotating model assignment.

    On iteration 0, only Methodologist and Falsification Expert run (the
    Trajectory Critic and Evidence Auditor require prior iteration history).
    On iteration 1+, all four personas run. Model assignment rotates across
    iterations regardless of persona count.

    Args:
        critic_configs: Pool of critic model configs (round-robin assigned).
        plan: Scientist's plan dict.
        notebook_content: Current lab notebook content.
        domain_knowledge: Domain-specific context.
        max_rounds: Number of critique rounds per debate (1 = single-pass).
        scientist_config: Config for the Scientist's debate responses.
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

    if scientist_config is None:
        scientist_config = AgentModelConfig(model="claude-sonnet-4-6")

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
                max_rounds=max_rounds,
                scientist_config=scientist_config,
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
    scientist_defense: str = "",
    persona_text: str = "",
    persona_instructions: str = "",
    analysis_json: str = "",
    prediction_history: str = "",
    goal: str = "",
) -> tuple[str, str]:
    """Build the (system, user) prompt pair sent to critic models.

    For round 1, scientist_defense is empty.
    For round 2+, it contains the scientist's response to the previous critique.
    The critic is not told they are "refining" anything (stateless design).

    persona_instructions overrides the default instructions block when provided
    (used by the Trajectory Critic which needs arc-focused instructions).

    Returns:
        (system_prompt, user_prompt) tuple.
    """
    defense_section = ""
    if scientist_defense:
        defense_section = f"<scientist_defense>{scientist_defense}</scientist_defense>"

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
        scientist_defense=defense_section,
    )
    return system, user


def _build_scientist_debate_user_prompt(
    plan: dict[str, Any],
    notebook_content: str,
    domain_knowledge: str,
    critique: str = "",
    critic_persona: str = "",
    analysis_json: str = "",
    prediction_history: str = "",
    goal: str = "",
) -> str:
    """Build the user prompt for the Scientist responding to a critique during debate."""
    return SCIENTIST_DEBATE_USER.format(
        goal=goal or "(no goal specified)",
        domain_knowledge=domain_knowledge or "(none provided)",
        notebook_content=notebook_content or "(empty)",
        analysis_json=analysis_json or "(no analysis yet)",
        prediction_history=prediction_history or "(no prediction history yet)",
        plan_json=json.dumps(plan, indent=2),
        critique=critique,
        critic_persona=critic_persona or "(generic critic)",
    )
