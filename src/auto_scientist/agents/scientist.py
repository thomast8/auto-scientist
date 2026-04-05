"""Scientist agent: prompt-in, JSON-out strategic planner with web search.

Does not read Python code or data files. Receives analysis + notebook via prompt.
Has web search access to ground hypotheses in real-world knowledge.
Output: structured JSON plan with hypothesis, strategy, changes, notebook entry.
"""

import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

from auto_scientist.agents.prediction_tool import (
    PREDICTION_SPEC,
    _build_prediction_forest,
    build_prediction_mcp_server,
    format_compact_tree,
)
from auto_scientist.model_config import ReasoningConfig, reasoning_to_cc_extra_args
from auto_scientist.prompts.scientist import (
    SCIENTIST_REVISION_SYSTEM,
    SCIENTIST_REVISION_USER,
    SCIENTIST_USER,
    build_scientist_system,
)
from auto_scientist.retry import QueryResult, agent_retry_loop
from auto_scientist.schemas import ScientistPlanOutput
from auto_scientist.sdk_backend import SDKOptions, get_backend
from auto_scientist.sdk_utils import (
    collect_text_from_query,
    prepare_turn_budget,
    validate_json_output,
)
from auto_scientist.state import PredictionRecord

logger = logging.getLogger(__name__)

SCIENTIST_BASE_TOOLS = ["WebSearch"]


def _build_scientist_tools_and_mcp(
    prediction_history: list[PredictionRecord] | None,
    provider: str,
    output_dir: Path | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """Build the tools list and MCP servers dict for a Scientist invocation.

    Both Claude and Codex backends get the same stdio MCP server config.
    Claude passes it via mcp_servers; the CodexBackend writes it to
    an isolated ``$CODEX_HOME/config.toml`` automatically.
    """
    tools = list(SCIENTIST_BASE_TOOLS)
    mcp_servers: dict[str, Any] = {}
    if prediction_history:
        mcp_servers["predictions"] = build_prediction_mcp_server(
            prediction_history, output_dir=output_dir
        )
        tools.append(PREDICTION_SPEC.mcp_tool_name)
    return tools, mcp_servers


# JSON schema for structured output (injected into the prompt for LLM guidance)
SCIENTIST_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "hypothesis": {"type": "string"},
        "strategy": {"type": "string", "enum": ["incremental", "structural", "exploratory"]},
        "changes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "what": {"type": "string"},
                    "why": {"type": "string"},
                    "how": {"type": "string"},
                    "priority": {"type": "integer", "enum": [1, 2, 3]},
                },
                "required": ["what", "why", "how", "priority"],
            },
        },
        "expected_impact": {"type": "string"},
        "should_stop": {"type": "boolean"},
        "stop_reason": {"type": ["string", "null"]},
        "notebook_entry": {"type": "string"},
        "testable_predictions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "prediction": {"type": "string"},
                    "diagnostic": {"type": "string"},
                    "if_confirmed": {"type": "string"},
                    "if_refuted": {"type": "string"},
                    "follows_from": {
                        "type": ["string", "null"],
                        "description": (
                            "pred_id of a prior prediction (e.g. '0.3', '1.2'). "
                            "Must be an exact bracketed ID from the prediction "
                            "history, or null for new trajectories."
                        ),
                    },
                },
                "required": ["prediction", "diagnostic", "if_confirmed", "if_refuted"],
            },
        },
    },
    "required": [
        "hypothesis",
        "strategy",
        "changes",
        "expected_impact",
        "should_stop",
        "stop_reason",
        "notebook_entry",
    ],
}


# Re-export for backward compatibility (scripts, tests, orchestrator)
_format_compact_tree = format_compact_tree


def _format_predictions_for_prompt(
    prediction_history: list[PredictionRecord] | None,
) -> str:
    """Format prediction history as full-detail reasoning trajectories.

    Used by the stop gate assessment and compare_personas script.
    Debate critics now use the compact tree + MCP tool instead.
    Builds a forest from follows_from links and renders each tree as a
    trajectory chain showing the reasoning flow across iterations.
    """
    if not prediction_history:
        return "(no prediction history yet)"

    _by_id, children = _build_prediction_forest(prediction_history)
    visited: set[str] = set()

    def _render_record(rec: PredictionRecord, indent: int) -> list[str]:
        # Guard against circular follows_from links
        if rec.pred_id and rec.pred_id in visited:
            return []
        if rec.pred_id:
            visited.add(rec.pred_id)

        prefix = "  " * indent
        tag = rec.pred_id or f"v{rec.iteration_prescribed:02d}"
        status = rec.outcome.upper()
        lines = []
        if rec.outcome == "pending":
            lines.append(f"{prefix}[{tag}] PENDING: {rec.prediction}")
            lines.append(f"{prefix}  Diagnostic: {rec.diagnostic}")
            lines.append(f"{prefix}  If confirmed: {rec.if_confirmed}")
            lines.append(f"{prefix}  If refuted: {rec.if_refuted}")
        else:
            if rec.outcome == "confirmed":
                implication = rec.if_confirmed
            elif rec.outcome == "refuted":
                implication = rec.if_refuted
            else:
                implication = None  # inconclusive: neither implication applies
            lines.append(f"{prefix}[{tag}] {status}: {rec.prediction}")
            lines.append(f"{prefix}  Evidence: {rec.evidence}")
            if implication:
                lines.append(f"{prefix}  -> {implication}")
        # Render children (keyed by parent's pred_id)
        for child in children.get(rec.pred_id, []):
            lines.extend(_render_record(child, indent + 1))
        return lines

    trajectories = []
    for root in children[None]:
        trajectory_lines = _render_record(root, 1)
        trajectories.append("\n".join(trajectory_lines))

    return "\n\n".join(trajectories)


async def run_scientist(
    analysis: dict[str, Any],
    notebook_path: Path,
    version: str,
    domain_knowledge: str = "",
    prediction_history: list[PredictionRecord] | None = None,
    model: str | None = None,
    message_buffer: list[str] | None = None,
    goal: str = "",
    provider: str = "anthropic",
    reasoning: ReasoningConfig | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Formulate hypothesis and plan based on analysis.

    The Scientist does not read code or data files. It receives the analysis
    JSON and notebook content via prompt injection and returns a structured plan.
    Has web search access.

    Args:
        analysis: Structured analysis JSON from the Analyst.
        notebook_path: Path to the lab notebook (read for context).
        version: Version string for the new experiment (e.g., 'v01').
        domain_knowledge: Domain-specific context.
        prediction_history: Accumulated testable predictions and outcomes.
        model: Model override.
        message_buffer: Optional buffer for streaming messages.
        goal: The user's investigation goal.

    Returns:
        Structured plan dict with keys: hypothesis, strategy, changes,
        expected_impact, should_stop, stop_reason, notebook_entry.
        Optionally: testable_predictions.
    """
    notebook_path = Path(notebook_path)
    notebook_content = notebook_path.read_text() if notebook_path.exists() else ""

    user_prompt = SCIENTIST_USER.format(
        goal=goal or "(no goal specified)",
        domain_knowledge=domain_knowledge or "(no domain knowledge provided)",
        analysis_json=(
            json.dumps(analysis, indent=2) if analysis else "(no analysis yet - first iteration)"
        ),
        notebook_content=notebook_content or "(empty notebook - first iteration)",
        prediction_history=format_compact_tree(prediction_history),
        version=version,
    )

    prompt_provider = "gpt" if provider == "openai" else "claude"
    system_prompt = build_scientist_system(prompt_provider)

    json_instruction = (
        "\n\n## Output Format\n"
        "Tool calls are allowed before the final JSON response.\n"
        'The "JSON only" rule applies only to the final assistant message.\n'
        "Respond with valid JSON matching the schema below.\n"
        "No markdown fencing. No explanation. No other text.\n\n"
        f"Schema:\n{json.dumps(SCIENTIST_PLAN_SCHEMA, indent=2)}"
    )

    extra_args: dict[str, str | None] = {}
    if reasoning and reasoning.level != "off":
        extra_args.update(reasoning_to_cc_extra_args(reasoning))

    tools, mcp_servers = _build_scientist_tools_and_mcp(
        prediction_history, provider, output_dir=output_dir
    )

    max_turns = 15
    budget = prepare_turn_budget(
        system_prompt + json_instruction, max_turns, tools, provider=provider
    )
    backend = get_backend(provider)
    options = SDKOptions(
        system_prompt=budget.system_prompt,
        allowed_tools=budget.allowed_tools,
        max_turns=budget.max_turns,
        model=model,
        extra_args=extra_args,
        mcp_servers=mcp_servers,
    )

    async def _query(prompt: str, resume_session_id: str | None) -> QueryResult:
        opts = replace(options, resume=resume_session_id) if resume_session_id else options
        raw, usage, session_id = await collect_text_from_query(
            prompt, opts, backend, message_buffer, agent_name="Scientist"
        )
        return QueryResult(raw_output=raw, session_id=session_id, usage=usage)

    def _validate(result: QueryResult) -> dict[str, Any]:
        return validate_json_output(result.raw_output, ScientistPlanOutput, "Scientist")

    return await agent_retry_loop(
        query_fn=_query,
        validate_fn=_validate,
        prompt=user_prompt,
        agent_name="Scientist",
    )


async def run_scientist_revision(
    original_plan: dict[str, Any],
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
    reasoning: ReasoningConfig | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Revise the plan after a critic debate.

    Args:
        original_plan: The initial plan that was debated.
        concern_ledger: Structured list of ConcernLedgerEntry dicts.
        analysis: Structured analysis JSON from the Analyst.
        notebook_path: Path to the lab notebook.
        version: Version string.
        domain_knowledge: Domain-specific context.
        prediction_history: Accumulated testable predictions and outcomes.
        model: Model override.
        message_buffer: Optional buffer for streaming messages.
        goal: The user's investigation goal.
        reasoning: Reasoning/effort config for the model.

    Returns:
        Revised plan dict (same schema as the initial plan).
    """
    notebook_path = Path(notebook_path)
    notebook_content = notebook_path.read_text() if notebook_path.exists() else ""

    ledger_text = json.dumps(concern_ledger, indent=2) if concern_ledger else "(no concerns raised)"

    user_prompt = SCIENTIST_REVISION_USER.format(
        goal=goal or "(no goal specified)",
        domain_knowledge=domain_knowledge or "(no domain knowledge provided)",
        analysis_json=(json.dumps(analysis, indent=2) if analysis else "(no analysis)"),
        notebook_content=notebook_content or "(empty notebook)",
        original_plan=json.dumps(original_plan, indent=2),
        concern_ledger=ledger_text,
        prediction_history=format_compact_tree(prediction_history),
        version=version,
    )

    json_instruction = (
        "\n\n## Output Format\n"
        "Tool calls are allowed before the final JSON response.\n"
        'The "JSON only" rule applies only to the final assistant message.\n'
        "Respond with valid JSON matching the schema below.\n"
        "No markdown fencing. No explanation. No other text.\n\n"
        f"Schema:\n{json.dumps(SCIENTIST_PLAN_SCHEMA, indent=2)}"
    )

    extra_args: dict[str, str | None] = {}
    if reasoning and reasoning.level != "off":
        extra_args.update(reasoning_to_cc_extra_args(reasoning))

    tools, mcp_servers = _build_scientist_tools_and_mcp(
        prediction_history, provider, output_dir=output_dir
    )

    max_turns = 15
    budget = prepare_turn_budget(
        SCIENTIST_REVISION_SYSTEM + json_instruction,
        max_turns,
        tools,
        provider=provider,
    )
    backend = get_backend(provider)
    options = SDKOptions(
        system_prompt=budget.system_prompt,
        allowed_tools=budget.allowed_tools,
        max_turns=budget.max_turns,
        model=model,
        extra_args=extra_args,
        mcp_servers=mcp_servers,
    )

    async def _query(prompt: str, resume_session_id: str | None) -> QueryResult:
        opts = replace(options, resume=resume_session_id) if resume_session_id else options
        raw, usage, session_id = await collect_text_from_query(
            prompt, opts, backend, message_buffer, agent_name="Scientist revision"
        )
        return QueryResult(raw_output=raw, session_id=session_id, usage=usage)

    def _validate(result: QueryResult) -> dict[str, Any]:
        return validate_json_output(result.raw_output, ScientistPlanOutput, "Scientist revision")

    return await agent_retry_loop(
        query_fn=_query,
        validate_fn=_validate,
        prompt=user_prompt,
        agent_name="Scientist revision",
    )
