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
from typing import Any

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
from auto_scientist.model_config import AgentModelConfig
from auto_scientist.models.anthropic_client import query_anthropic
from auto_scientist.models.google_client import query_google
from auto_scientist.models.openai_client import query_openai
from auto_scientist.prompts.critic import (
    CRITIC_SYSTEM_BASE,
    CRITIC_USER,
    PERSONAS,
    SCIENTIST_DEBATE_SYSTEM,
    SCIENTIST_DEBATE_USER,
    get_model_index_for_debate,
)
from auto_scientist.sdk_utils import OutputValidationError, validate_json_output

logger = logging.getLogger(__name__)

MAX_RETRIES = 1  # 1 retry = 2 total attempts
MIN_RESPONSE_LENGTH = 50  # minimum chars for a substantive response


# ---------------------------------------------------------------------------
# Low-level query helpers
# ---------------------------------------------------------------------------


async def _query_critic(
    config: AgentModelConfig,
    prompt: str,
) -> AgentResult:
    """Dispatch a prompt to the appropriate provider with web search enabled."""
    if config.provider == "openai":
        return await query_openai(
            config.model, prompt,
            web_search=True, reasoning=config.reasoning,
        )
    elif config.provider == "google":
        return await query_google(
            config.model, prompt,
            web_search=True, reasoning=config.reasoning,
        )
    elif config.provider == "anthropic":
        return await query_anthropic(
            config.model, prompt,
            web_search=True, reasoning=config.reasoning,
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
    label: str = "",
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
            result = await _query_critic(config, effective_prompt)
        except Exception as e:
            if attempt < MAX_RETRIES:
                logger.warning(f"{label} error ({e}), retrying (attempt {attempt + 1})")
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
                logger.warning(
                    f"{label} validation failed after retries, preserving raw text as concern"
                )
                raw = (result.text or "(empty response)")[:500]
                fallback = CriticOutput(
                    concerns=[Concern(
                        claim=f"[PARSE ERROR] {raw}",
                        severity="high",
                        confidence="low",
                        category="other",
                    )],
                    alternative_hypotheses=[],
                    overall_assessment=result.text or "(empty response)",
                )
                return fallback, result

    # Should not reach here, but satisfy type checker
    return CriticOutput(
        concerns=[], alternative_hypotheses=[],
        overall_assessment="(unreachable fallback)",
    ), result


async def _query_scientist_structured(
    config: AgentModelConfig,
    prompt: str,
    system_prompt: str,
    *,
    label: str = "",
) -> tuple[ScientistDefense, AgentResult]:
    """Query the scientist (direct API) and validate as structured ScientistDefense.

    Returns (validated ScientistDefense, raw AgentResult).
    Retries once on validation failure with a correction hint.
    """
    result = AgentResult(text="")
    correction_hint = ""

    for attempt in range(MAX_RETRIES + 1):
        effective_prompt = prompt + correction_hint
        full_prompt = f"{system_prompt}\n\n{effective_prompt}"
        try:
            result = await _query_critic(config, full_prompt)
        except Exception as e:
            if attempt < MAX_RETRIES:
                logger.warning(f"{label} error ({e}), retrying (attempt {attempt + 1})")
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
                logger.warning(f"{label} validation failed after retries, using raw text")
                fallback = ScientistDefense(
                    responses=[],
                    additional_points=result.text or "(empty response)",
                )
                return fallback, result

    fallback = ScientistDefense(
        responses=[], additional_points=result.text or "(empty response)",
    )
    return fallback, result


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
) -> DebateResult:
    """Run a multi-round debate between one critic (with persona) and the scientist.

    Returns a DebateResult with structured output per round plus raw transcript.
    """
    if scientist_config is None:
        scientist_config = AgentModelConfig(model="claude-sonnet-4-6")

    persona = persona or {"name": "Generic", "system_text": ""}
    persona_name = persona["name"]
    persona_text = persona["system_text"]

    label = f"{config.provider}:{config.model}"
    raw_transcript: list[dict[str, str]] = []
    rounds: list[DebateRound] = []
    total_in = 0
    total_out = 0

    # Round 1: initial critique (no defense context)
    critic_prompt = _build_critic_prompt(
        plan, notebook_content, domain_knowledge,
        persona_text=persona_text,
        analysis_json=analysis_json,
        prediction_history=prediction_history,
    )
    critic_output, critic_result = await _query_critic_structured(
        config, critic_prompt,
        label=f"Critic ({persona_name}, {label}) round 1",
    )
    total_in += critic_result.input_tokens
    total_out += critic_result.output_tokens
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
            )
            scientist_defense, sci_result = await _query_scientist_structured(
                scientist_config, scientist_user_prompt, scientist_system,
                label=f"Scientist ({persona_name}) round {round_num}",
            )
            total_in += sci_result.input_tokens
            total_out += sci_result.output_tokens
            raw_transcript.append({"role": "scientist", "content": sci_result.text})
            if message_buffer is not None:
                message_buffer.append(f"[Scientist] {sci_result.text}")

            # Record the round with both critic output and scientist defense
            rounds.append(DebateRound(
                critic_output=critic_output,
                scientist_defense=scientist_defense,
            ))

            # Stateless: critic gets the defense but not their own prior critique
            critic_prompt = _build_critic_prompt(
                plan, notebook_content, domain_knowledge,
                scientist_defense=sci_result.text,
                persona_text=persona_text,
                analysis_json=analysis_json,
                prediction_history=prediction_history,
            )
            critic_output, critic_result = await _query_critic_structured(
                config, critic_prompt,
                label=f"Critic ({persona_name}, {label}) round {round_num + 1}",
            )
            total_in += critic_result.input_tokens
            total_out += critic_result.output_tokens
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
    )


# ---------------------------------------------------------------------------
# Top-level debate orchestrator
# ---------------------------------------------------------------------------


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
) -> list[DebateResult]:
    """Run parallel debates, one per persona, with rotating model assignment.

    Always runs 3 debates (one per persona) regardless of how many critic
    models are configured. Model assignment rotates across iterations.

    Args:
        critic_configs: Pool of critic model configs (round-robin assigned).
        plan: Scientist's plan dict.
        notebook_content: Current lab notebook content.
        domain_knowledge: Domain-specific context.
        max_rounds: Number of critique rounds per debate (1 = single-pass).
        scientist_config: Config for the Scientist's debate responses.
        message_buffer: Legacy single shared buffer.
        message_buffers: Per-persona buffers keyed by persona name.
        iteration: Current iteration number (for model rotation).
        analysis_json: Serialized analysis JSON from the Analyst.
        prediction_history: Formatted prediction history string.

    Returns:
        List of DebateResult, one per persona (always 3 unless no critics).
    """
    if not critic_configs:
        return []

    if scientist_config is None:
        scientist_config = AgentModelConfig(model="claude-sonnet-4-6")

    tasks = []
    for persona_index, persona in enumerate(PERSONAS):
        model_index = get_model_index_for_debate(
            persona_index, iteration, len(critic_configs),
        )
        config = critic_configs[model_index]
        persona_name = persona["name"]

        # Resolve buffer: per-persona dict > shared legacy buffer > None
        if message_buffers is not None:
            buf = message_buffers.setdefault(persona_name, [])
        else:
            buf = message_buffer

        tasks.append(
            run_single_critic_debate(
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
            )
        )

    raw_results = await asyncio.gather(*tasks, return_exceptions=True)
    successful: list[DebateResult] = []
    persona_names = [p["name"] for p in PERSONAS]
    for persona_name, r in zip(persona_names, raw_results, strict=True):
        if isinstance(r, BaseException):
            logger.error(f"Critic debate failed for {persona_name}: {r}", exc_info=r)
        else:
            successful.append(r)
    if len(successful) < len(raw_results):
        logger.warning(
            f"Debate: {len(successful)}/{len(raw_results)} debates succeeded"
        )
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
    analysis_json: str = "",
    prediction_history: str = "",
) -> str:
    """Build the prompt sent to critic models.

    For round 1, scientist_defense is empty.
    For round 2+, it contains the scientist's response to the previous critique.
    The critic is not told they are "refining" anything (stateless design).
    """
    defense_section = ""
    if scientist_defense:
        defense_section = (
            f"<scientist_defense>{scientist_defense}</scientist_defense>"
        )

    system = CRITIC_SYSTEM_BASE.format(
        persona_text=persona_text,
        critic_output_schema=json.dumps(CRITIC_OUTPUT_SCHEMA, indent=2),
    )

    user = CRITIC_USER.format(
        domain_knowledge=domain_knowledge or "(none provided)",
        notebook_content=notebook_content or "(empty)",
        analysis_json=analysis_json or "(no analysis yet)",
        prediction_history=prediction_history or "(no prediction history yet)",
        plan_json=json.dumps(plan, indent=2),
        scientist_defense=defense_section,
    )
    return f"{system}\n\n{user}"


def _build_scientist_debate_user_prompt(
    plan: dict[str, Any],
    notebook_content: str,
    domain_knowledge: str,
    critique: str = "",
    critic_persona: str = "",
    analysis_json: str = "",
    prediction_history: str = "",
) -> str:
    """Build the user prompt for the Scientist responding to a critique during debate."""
    return SCIENTIST_DEBATE_USER.format(
        domain_knowledge=domain_knowledge or "(none provided)",
        notebook_content=notebook_content or "(empty)",
        analysis_json=analysis_json or "(no analysis yet)",
        prediction_history=prediction_history or "(no prediction history yet)",
        plan_json=json.dumps(plan, indent=2),
        critique=critique,
        critic_persona=critic_persona or "(generic critic)",
    )


