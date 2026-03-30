"""Stop gate agents: completeness assessment, stop debate, and stop revision.

When the Scientist proposes stopping, the stop gate validates whether the
investigation goal has been thoroughly addressed before honoring the decision.
"""

import json
import logging
from pathlib import Path
from typing import Any

from claude_code_sdk import ClaudeCodeOptions
from pydantic import BaseModel

from auto_scientist.agents.debate_models import (
    CRITIC_OUTPUT_SCHEMA,
    SCIENTIST_DEFENSE_SCHEMA,
    CriticOutput,
    DebateResult,
    DebateRound,
    ScientistDefense,
)
from auto_scientist.agents.scientist import SCIENTIST_PLAN_SCHEMA
from auto_scientist.model_config import AgentModelConfig
from auto_scientist.prompts.stop_gate import (
    ASSESSMENT_SCHEMA,
    ASSESSMENT_SYSTEM,
    ASSESSMENT_USER,
    STOP_CRITIC_SYSTEM_BASE,
    STOP_CRITIC_USER,
    STOP_PERSONAS,
    STOP_REVISION_SYSTEM,
    STOP_REVISION_USER,
    STOP_SCIENTIST_DEBATE_SYSTEM,
    STOP_SCIENTIST_DEBATE_USER,
)
from auto_scientist.schemas import CompletenessAssessmentOutput, ScientistPlanOutput
from auto_scientist.sdk_utils import (
    OutputValidationError,
    collect_text_from_query,
    validate_json_output,
    with_turn_budget,
)
from auto_scientist.state import PredictionRecord

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 3


# ---------------------------------------------------------------------------
# Completeness Assessment
# ---------------------------------------------------------------------------


def _format_predictions_for_assessment(
    prediction_history: list[PredictionRecord] | None,
) -> str:
    """Format prediction history for the assessment prompt."""
    from auto_scientist.agents.scientist import _format_predictions_for_prompt

    return _format_predictions_for_prompt(prediction_history)


async def run_completeness_assessment(
    goal: str,
    stop_reason: str,
    notebook_path: Path,
    domain_knowledge: str = "",
    prediction_history: list[PredictionRecord] | None = None,
    model: str | None = None,
    message_buffer: list[str] | None = None,
) -> dict[str, Any]:
    """Assess whether the investigation goal has been thoroughly addressed.

    Returns a structured assessment with sub-questions, coverage ratings,
    and a stop/continue recommendation.
    """
    notebook_content = Path(notebook_path).read_text() if Path(notebook_path).exists() else ""

    user_prompt = ASSESSMENT_USER.format(
        goal=goal,
        stop_reason=stop_reason,
        domain_knowledge=domain_knowledge or "(no domain knowledge provided)",
        prediction_history=_format_predictions_for_assessment(prediction_history),
        notebook_content=notebook_content or "(empty notebook)",
    )

    json_instruction = (
        "\n\n## Output Format\n"
        "You MUST respond with ONLY valid JSON matching the schema below.\n"
        "No markdown fencing. No explanation. No other text.\n\n"
        f"Schema:\n{json.dumps(ASSESSMENT_SCHEMA, indent=2)}"
    )

    max_turns = 5
    options = ClaudeCodeOptions(
        system_prompt=with_turn_budget(ASSESSMENT_SYSTEM + json_instruction, max_turns, []),
        allowed_tools=[],
        max_turns=max_turns,
        model=model,
        extra_args={"setting-sources": ""},
    )

    correction_hint = ""
    for attempt in range(MAX_ATTEMPTS):
        effective_prompt = user_prompt + correction_hint

        try:
            raw, _usage = await collect_text_from_query(
                effective_prompt,
                options,
                message_buffer,
                agent_name="Completeness Assessment",
            )
        except Exception as e:
            if attempt == MAX_ATTEMPTS - 1:
                raise
            logger.warning(f"Assessment attempt {attempt + 1}: SDK error ({e}), retrying")
            continue

        try:
            return validate_json_output(
                raw, CompletenessAssessmentOutput, "Completeness Assessment"
            )
        except OutputValidationError as e:
            if attempt == MAX_ATTEMPTS - 1:
                raise
            correction_hint = f"\n\n{e.correction_prompt()}"
            logger.warning(f"Assessment attempt {attempt + 1} failed, retrying: {e}")

    raise RuntimeError("Completeness Assessment: exhausted retries")  # unreachable


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
) -> tuple[Any, Any]:
    """Query a stop gate agent (critic or scientist) via Claude Code SDK.

    Returns (validated_model_instance, result_obj with text/token counts).
    """
    from auto_scientist.model_config import reasoning_to_cc_extra_args

    extra_args: dict[str, str | None] = {"setting-sources": ""}
    extra_args.update(reasoning_to_cc_extra_args(config.reasoning))

    max_turns = 10
    tools = ["WebSearch"]
    options = ClaudeCodeOptions(
        system_prompt=with_turn_budget(system_prompt, max_turns, tools),
        allowed_tools=tools,
        max_turns=max_turns,
        model=config.model,
        extra_args=extra_args,
    )

    correction_hint = ""
    for attempt in range(MAX_ATTEMPTS):
        effective_prompt = user_prompt + correction_hint
        try:
            raw, usage = await collect_text_from_query(
                effective_prompt,
                options,
                message_buffer,
                agent_name=label,
            )
        except Exception as e:
            if attempt == MAX_ATTEMPTS - 1:
                raise
            logger.warning(f"{label} attempt {attempt + 1}: SDK error ({e}), retrying")
            continue

        try:
            parsed = validate_json_output(raw, output_model, label)
            result_obj = type(
                "Result",
                (),
                {
                    "text": raw,
                    "input_tokens": usage.get("input_tokens", 0),
                    "output_tokens": usage.get("output_tokens", 0),
                    "thinking_tokens": usage.get("thinking_tokens", 0),
                },
            )()
            return output_model.model_validate(parsed), result_obj
        except OutputValidationError as e:
            if attempt == MAX_ATTEMPTS - 1:
                raise
            correction_hint = f"\n\n{e.correction_prompt()}"
            logger.warning(f"{label} attempt {attempt + 1} failed, retrying: {e}")

    raise RuntimeError(f"{label}: exhausted retries")  # unreachable


async def run_single_stop_debate(
    config: AgentModelConfig,
    stop_reason: str,
    completeness_assessment: dict[str, Any],
    notebook_content: str,
    domain_knowledge: str = "",
    scientist_config: AgentModelConfig | None = None,
    message_buffer: list[str] | None = None,
    persona: dict[str, str] | None = None,
    analysis_json: str = "",
    prediction_history: str = "",
    goal: str = "",
) -> DebateResult:
    """Run a single-round stop debate between one critic persona and the scientist."""
    if scientist_config is None:
        scientist_config = AgentModelConfig(model="claude-sonnet-4-6")

    persona = persona or {"name": "Generic", "system_text": ""}
    persona_name = persona["name"]
    persona_text = persona["system_text"]
    persona_instructions = persona.get("instructions", "")

    label = f"{config.provider}:{config.model}"
    raw_transcript: list[dict[str, str]] = []
    total_in = 0
    total_out = 0
    total_think = 0

    assessment_json = json.dumps(completeness_assessment, indent=2)

    # Build critic prompt
    critic_system = STOP_CRITIC_SYSTEM_BASE.format(
        persona_text=persona_text,
        persona_instructions=persona_instructions or "",
        critic_output_schema=json.dumps(CRITIC_OUTPUT_SCHEMA, indent=2),
    )
    critic_user = STOP_CRITIC_USER.format(
        goal=goal,
        domain_knowledge=domain_knowledge,
        notebook_content=notebook_content,
        analysis_json=analysis_json,
        prediction_history=prediction_history,
        stop_reason=stop_reason,
        completeness_assessment=assessment_json,
    )

    # Round 1: Critic challenges the stop
    critic_output, critic_result = await _query_stop_agent(
        config,
        critic_user,
        system_prompt=critic_system,
        output_model=CriticOutput,
        label=f"Stop Critic ({persona_name}, {label})",
        message_buffer=message_buffer,
    )
    total_in += critic_result.input_tokens
    total_out += critic_result.output_tokens
    total_think += critic_result.thinking_tokens
    raw_transcript.append({"role": "critic", "content": critic_result.text})

    # Round 2: Scientist defends
    scientist_system = STOP_SCIENTIST_DEBATE_SYSTEM.format(
        scientist_defense_schema=json.dumps(SCIENTIST_DEFENSE_SCHEMA, indent=2),
    )
    scientist_user = STOP_SCIENTIST_DEBATE_USER.format(
        goal=goal,
        domain_knowledge=domain_knowledge,
        notebook_content=notebook_content,
        analysis_json=analysis_json,
        prediction_history=prediction_history,
        completeness_assessment=assessment_json,
        stop_reason=stop_reason,
        critique=critic_result.text,
        critic_persona=persona_name,
    )

    scientist_defense, sci_result = await _query_stop_agent(
        scientist_config,
        scientist_user,
        system_prompt=scientist_system,
        output_model=ScientistDefense,
        label=f"Stop Scientist ({persona_name})",
        message_buffer=message_buffer,
    )
    total_in += sci_result.input_tokens
    total_out += sci_result.output_tokens
    total_think += sci_result.thinking_tokens
    raw_transcript.append({"role": "scientist", "content": sci_result.text})

    rounds = [DebateRound(critic_output=critic_output, scientist_defense=scientist_defense)]

    return DebateResult(
        persona=persona_name,
        critic_model=label,
        rounds=rounds,
        raw_transcript=raw_transcript,
        input_tokens=total_in,
        output_tokens=total_out,
        thinking_tokens=total_think,
    )


async def run_stop_debate(
    critic_configs: list[AgentModelConfig],
    stop_reason: str,
    completeness_assessment: dict[str, Any],
    notebook_content: str,
    domain_knowledge: str = "",
    scientist_config: AgentModelConfig | None = None,
    message_buffers: dict[str, list[str]] | None = None,
    analysis_json: str = "",
    prediction_history: str = "",
    goal: str = "",
) -> list[DebateResult]:
    """Run stop debates with all stop personas in parallel.

    Each persona gets a model from critic_configs via index-based assignment.
    """
    import asyncio

    if not critic_configs:
        return []

    tasks = []
    for i, persona in enumerate(STOP_PERSONAS):
        config = critic_configs[i % len(critic_configs)]
        buffer = (message_buffers or {}).get(persona["name"])
        tasks.append(
            run_single_stop_debate(
                config=config,
                stop_reason=stop_reason,
                completeness_assessment=completeness_assessment,
                notebook_content=notebook_content,
                domain_knowledge=domain_knowledge,
                scientist_config=scientist_config,
                message_buffer=buffer,
                persona=persona,
                analysis_json=analysis_json,
                prediction_history=prediction_history,
                goal=goal,
            )
        )

    results = await asyncio.gather(*tasks, return_exceptions=True)

    debate_results = []
    for persona, result in zip(STOP_PERSONAS, results, strict=True):
        if isinstance(result, BaseException):
            logger.error(
                f"Stop debate failed for {persona['name']}: {result}",
                exc_info=result,
            )
        else:
            debate_results.append(result)

    if not debate_results:
        failed_msgs = [str(r) for r in results if isinstance(r, BaseException)]
        raise RuntimeError(
            f"All {len(results)} stop debates failed. "
            f"Check API keys and network connectivity. Errors: {failed_msgs}"
        )

    if len(debate_results) < len(results):
        lost = [
            STOP_PERSONAS[i]["name"] for i, r in enumerate(results) if isinstance(r, BaseException)
        ]
        logger.warning(
            f"Stop debate: {len(debate_results)}/{len(results)} succeeded. "
            f"Lost perspectives: {lost}"
        )

    return debate_results


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
) -> dict[str, Any]:
    """Revise the stop decision after the stop debate.

    Returns a ScientistPlanOutput-compatible dict. If should_stop is still
    true, the stop is upheld. If false, the plan contains a real experiment.
    """
    notebook_content = Path(notebook_path).read_text() if Path(notebook_path).exists() else ""

    user_prompt = STOP_REVISION_USER.format(
        goal=goal or "(no goal specified)",
        domain_knowledge=domain_knowledge or "(no domain knowledge provided)",
        analysis_json=json.dumps(analysis, indent=2) if analysis else "(no analysis)",
        notebook_content=notebook_content or "(empty notebook)",
        stop_reason=stop_reason,
        completeness_assessment=json.dumps(completeness_assessment, indent=2),
        concern_ledger=(
            json.dumps(concern_ledger, indent=2) if concern_ledger else "(no concerns raised)"
        ),
        prediction_history=_format_predictions_for_assessment(prediction_history),
        version=version,
        plan_schema=json.dumps(SCIENTIST_PLAN_SCHEMA, indent=2),
    )

    json_instruction = (
        "\n\n## Output Format\n"
        "You MUST respond with ONLY valid JSON matching the schema below.\n"
        "No markdown fencing. No explanation. No other text.\n\n"
        f"Schema:\n{json.dumps(SCIENTIST_PLAN_SCHEMA, indent=2)}"
    )

    max_turns = 5
    options = ClaudeCodeOptions(
        system_prompt=with_turn_budget(STOP_REVISION_SYSTEM + json_instruction, max_turns, []),
        allowed_tools=[],
        max_turns=max_turns,
        model=model,
        extra_args={"setting-sources": ""},
    )

    correction_hint = ""
    for attempt in range(MAX_ATTEMPTS):
        effective_prompt = user_prompt + correction_hint

        try:
            raw, _usage = await collect_text_from_query(
                effective_prompt,
                options,
                message_buffer,
                agent_name="Stop Revision",
            )
        except Exception as e:
            if attempt == MAX_ATTEMPTS - 1:
                raise
            logger.warning(f"Stop revision attempt {attempt + 1}: SDK error ({e}), retrying")
            continue

        try:
            return validate_json_output(raw, ScientistPlanOutput, "Stop Revision")
        except OutputValidationError as e:
            if attempt == MAX_ATTEMPTS - 1:
                raise
            correction_hint = f"\n\n{e.correction_prompt()}"
            logger.warning(f"Stop revision attempt {attempt + 1} failed, retrying: {e}")

    raise RuntimeError("Stop Revision: exhausted retries")  # unreachable
