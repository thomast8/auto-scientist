"""Stop gate agents: completeness assessment, stop debate, and stop revision.

When the Scientist proposes stopping, the stop gate validates whether the
investigation goal has been thoroughly addressed before honoring the decision.
"""

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from auto_scientist.agents.debate_models import (
    CRITIC_OUTPUT_SCHEMA,
    CriticOutput,
    DebateResult,
)
from auto_scientist.agents.prediction_tool import (
    PREDICTION_SPEC,
    build_prediction_mcp_server,
    format_compact_tree,
)
from auto_scientist.agents.scientist import SCIENTIST_PLAN_SCHEMA
from auto_scientist.model_config import AgentModelConfig
from auto_scientist.prompts.stop_gate import (
    ASSESSMENT_SCHEMA,
    ASSESSMENT_SYSTEM,
    ASSESSMENT_USER,
    STOP_CRITIC_SYSTEM_BASE,
    STOP_CRITIC_USER,
    STOP_REVISION_SYSTEM,
    STOP_REVISION_USER,
)
from auto_scientist.schemas import CompletenessAssessmentOutput, ScientistPlanOutput
from auto_scientist.sdk_backend import SDKOptions, get_backend
from auto_scientist.sdk_utils import (
    OutputValidationError,
    collect_text_from_query,
    prepare_turn_budget,
    validate_json_output,
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
    provider: str = "anthropic",
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Assess whether the investigation goal has been thoroughly addressed.

    Returns a structured assessment with sub-questions, coverage ratings,
    and a stop/continue recommendation.
    """
    notebook_content = Path(notebook_path).read_text() if Path(notebook_path).exists() else ""

    # Build tools: WebSearch + optional MCP for prediction drill-down
    tools: list[str] = ["WebSearch"]
    mcp_servers: dict[str, Any] = {}
    if prediction_history:
        mcp_servers["predictions"] = build_prediction_mcp_server(
            prediction_history, output_dir=output_dir
        )
        tools.append(PREDICTION_SPEC.mcp_tool_name)

    user_prompt = ASSESSMENT_USER.format(
        goal=goal,
        stop_reason=stop_reason,
        domain_knowledge=domain_knowledge or "(no domain knowledge provided)",
        prediction_history=format_compact_tree(prediction_history),
        notebook_content=notebook_content or "(empty notebook)",
    )

    json_instruction = (
        "\n\n## Output Format\n"
        "You MUST respond with ONLY valid JSON matching the schema below.\n"
        "No markdown fencing. No explanation. No other text.\n\n"
        f"Schema:\n{json.dumps(ASSESSMENT_SCHEMA, indent=2)}"
    )

    max_turns = 10
    budget = prepare_turn_budget(
        ASSESSMENT_SYSTEM + json_instruction, max_turns, tools, provider=provider
    )
    backend = get_backend(provider)
    options = SDKOptions(
        system_prompt=budget.system_prompt,
        allowed_tools=budget.allowed_tools,
        max_turns=budget.max_turns,
        model=model,
        extra_args={},
        mcp_servers=mcp_servers,
    )

    correction_hint = ""
    for attempt in range(MAX_ATTEMPTS):
        effective_prompt = user_prompt + correction_hint

        try:
            raw, _usage = await collect_text_from_query(
                effective_prompt,
                options,
                backend,
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

    raise RuntimeError("Completeness Assessment: exhausted retries")  # safety net


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
) -> tuple[Any, Any]:
    """Query a stop gate agent (critic or scientist) via provider-aware dispatch.

    Uses direct API clients for OpenAI/Google, Claude Code SDK for Anthropic.
    Returns (validated_model_instance, result_obj with text/token counts).
    """
    from auto_scientist.agents.critic import _query_critic

    correction_hint = ""
    for attempt in range(MAX_ATTEMPTS):
        effective_prompt = user_prompt + correction_hint
        try:
            result = await _query_critic(
                config,
                effective_prompt,
                system_prompt=system_prompt,
                response_schema=output_model,
                message_buffer=message_buffer,
                allowed_tools=allowed_tools,
                mcp_servers=mcp_servers,
            )
        except Exception as e:
            if attempt == MAX_ATTEMPTS - 1:
                raise
            logger.warning(f"{label} attempt {attempt + 1}: SDK error ({e}), retrying")
            continue

        try:
            parsed = validate_json_output(result.text, output_model, label)
            return output_model.model_validate(parsed), result
        except OutputValidationError as e:
            if attempt == MAX_ATTEMPTS - 1:
                raise
            correction_hint = f"\n\n{e.correction_prompt()}"
            logger.warning(f"{label} attempt {attempt + 1} failed, retrying: {e}")

    raise RuntimeError(f"{label}: exhausted retries")  # safety net


async def run_single_stop_debate(
    config: AgentModelConfig,
    stop_reason: str,
    completeness_assessment: dict[str, Any],
    notebook_content: str,
    domain_knowledge: str = "",
    message_buffer: list[str] | None = None,
    persona: dict[str, str] | None = None,
    analysis_json: str = "",
    prediction_history: str = "",
    goal: str = "",
    prediction_history_records: list[PredictionRecord] | None = None,
    output_dir: Path | None = None,
) -> DebateResult:
    """Run a single critic persona's challenge of the stop decision.

    The scientist responds once via run_scientist_stop_revision after
    all critics have challenged.

    Args:
        prediction_history: Pre-formatted text for prompt injection.
        prediction_history_records: Raw records for MCP server (SDK mode only).
        output_dir: Directory for MCP data files.
    """
    from auto_scientist.agents.critic import _build_critic_tools_and_mcp

    persona = persona or {"name": "Generic", "system_text": ""}
    persona_name = persona["name"]
    persona_text = persona["system_text"]
    persona_instructions = persona.get("instructions", "")

    label = f"{config.provider}:{config.model}"
    assessment_json = json.dumps(completeness_assessment, indent=2)

    tools, mcp_servers = _build_critic_tools_and_mcp(
        prediction_history_records, output_dir=output_dir
    )

    critic_system = STOP_CRITIC_SYSTEM_BASE.format(
        persona_text=persona_text,
        persona_instructions=persona_instructions or "",
        critic_output_schema=json.dumps(CRITIC_OUTPUT_SCHEMA, indent=2),
    )
    effective_prediction_history = (
        format_compact_tree(prediction_history_records)
        if prediction_history_records
        else prediction_history
    )

    critic_user = STOP_CRITIC_USER.format(
        goal=goal,
        domain_knowledge=domain_knowledge,
        notebook_content=notebook_content,
        analysis_json=analysis_json,
        prediction_history=effective_prediction_history,
        stop_reason=stop_reason,
        completeness_assessment=assessment_json,
    )

    critic_output, critic_result = await _query_stop_agent(
        config,
        critic_user,
        system_prompt=critic_system,
        output_model=CriticOutput,
        label=f"Stop Critic ({persona_name}, {label})",
        message_buffer=message_buffer,
        allowed_tools=tools,
        mcp_servers=mcp_servers,
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
    notebook_content = Path(notebook_path).read_text() if Path(notebook_path).exists() else ""

    # Build tools: WebSearch + MCP prediction tree
    tools = ["WebSearch"]
    mcp_servers: dict[str, Any] = {}
    if prediction_history:
        mcp_servers["predictions"] = build_prediction_mcp_server(
            prediction_history, output_dir=output_dir
        )
        tools.append(PREDICTION_SPEC.mcp_tool_name)

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
        prediction_history=format_compact_tree(prediction_history),
        version=version,
        plan_schema=json.dumps(SCIENTIST_PLAN_SCHEMA, indent=2),
    )

    json_instruction = (
        "\n\n## Output Format\n"
        "You MUST respond with ONLY valid JSON matching the schema below.\n"
        "No markdown fencing. No explanation. No other text.\n\n"
        f"Schema:\n{json.dumps(SCIENTIST_PLAN_SCHEMA, indent=2)}"
    )

    max_turns = 15
    budget = prepare_turn_budget(
        STOP_REVISION_SYSTEM + json_instruction, max_turns, tools, provider=provider
    )
    backend = get_backend(provider)
    options = SDKOptions(
        system_prompt=budget.system_prompt,
        allowed_tools=budget.allowed_tools,
        max_turns=budget.max_turns,
        model=model,
        extra_args={},
        mcp_servers=mcp_servers,
    )

    correction_hint = ""
    for attempt in range(MAX_ATTEMPTS):
        effective_prompt = user_prompt + correction_hint

        try:
            raw, _usage = await collect_text_from_query(
                effective_prompt,
                options,
                backend,
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

    raise RuntimeError("Stop Revision: exhausted retries")  # safety net
