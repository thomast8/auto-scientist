"""Scientist agent: prompt-in, JSON-out strategic planner with web search.

Does not read Python code or data files. Receives analysis + notebook via prompt.
Has web search access to ground hypotheses in real-world knowledge.
Output: structured JSON plan with hypothesis, strategy, changes, notebook entry.
"""

import json
import logging
from pathlib import Path
from typing import Any

from auto_scientist.agents.prediction_tool import build_prediction_mcp_server
from auto_scientist.model_config import ReasoningConfig, reasoning_to_cc_extra_args
from auto_scientist.prompts.scientist import (
    SCIENTIST_REVISION_SYSTEM,
    SCIENTIST_REVISION_USER,
    SCIENTIST_SYSTEM,
    SCIENTIST_USER,
)
from auto_scientist.schemas import ScientistPlanOutput
from auto_scientist.sdk_backend import SDKOptions, get_backend
from auto_scientist.sdk_utils import (
    OutputValidationError,
    collect_text_from_query,
    validate_json_output,
    with_turn_budget,
)
from auto_scientist.state import PredictionRecord

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 3

SCIENTIST_BASE_TOOLS = ["WebSearch"]
SCIENTIST_MCP_TOOL = "mcp__predictions__read_predictions"


def _build_scientist_tools_and_mcp(
    prediction_history: list[PredictionRecord] | None,
    provider: str,
) -> tuple[list[str], dict[str, Any]]:
    """Build the tools list and MCP servers dict for a Scientist invocation.

    Only includes the MCP prediction tool when there are predictions to query
    and the backend supports MCP (Claude only, not Codex).
    """
    tools = list(SCIENTIST_BASE_TOOLS)
    mcp_servers: dict[str, Any] = {}
    if prediction_history and provider == "anthropic":
        mcp_servers["predictions"] = build_prediction_mcp_server(prediction_history)
        tools.append(SCIENTIST_MCP_TOOL)
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
                    "follows_from": {"type": ["string", "null"]},
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


def _get_display_text(rec: PredictionRecord) -> str:
    """Return the summary text for a prediction, falling back to truncated evidence.

    Sanitizes output to ensure single-line (collapses newlines, hard-truncates).
    """
    text = rec.summary or rec.evidence or rec.prediction
    # Collapse newlines for one-line-per-prediction guarantee
    text = " ".join(text.split())
    if len(text) > 100:
        return text[:100] + "..."
    return text


def _build_prediction_forest(
    prediction_history: list[PredictionRecord],
) -> tuple[dict[str, PredictionRecord], dict[str | None, list[PredictionRecord]]]:
    """Build parent-to-children index from follows_from links.

    Returns (by_id, children) where children[None] contains root predictions.
    """
    by_id: dict[str, PredictionRecord] = {}
    children: dict[str | None, list[PredictionRecord]] = {None: []}
    for rec in prediction_history:
        if rec.pred_id:
            by_id[rec.pred_id] = rec
    for rec in prediction_history:
        parent = rec.follows_from
        if parent and parent in by_id:
            children.setdefault(parent, []).append(rec)
        else:
            children[None].append(rec)
    return by_id, children


def _format_compact_tree(
    prediction_history: list[PredictionRecord] | None,
) -> str:
    """Format prediction history as a compact one-line-per-prediction tree.

    Each prediction gets a single line with status, summary, and implication.
    The full detail is available via the read_predictions MCP tool.
    """
    if not prediction_history:
        return "(no prediction history yet)"

    _by_id, children = _build_prediction_forest(prediction_history)
    visited: set[str] = set()

    def _render_compact(rec: PredictionRecord, indent: int) -> list[str]:
        if rec.pred_id and rec.pred_id in visited:
            return []
        if rec.pred_id:
            visited.add(rec.pred_id)

        prefix = "  " * indent
        tag = rec.pred_id or f"v{rec.iteration_prescribed:02d}"
        display = _get_display_text(rec)
        lines = []

        if rec.outcome == "pending":
            lines.append(f"{prefix}[{tag}] PENDING: {rec.prediction}")
        elif rec.outcome == "confirmed":
            lines.append(f"{prefix}[{tag}] CONFIRMED: {display} -> {rec.if_confirmed}")
        elif rec.outcome == "refuted":
            lines.append(f"{prefix}[{tag}] DEAD END: {display}")
        elif rec.outcome == "inconclusive":
            lines.append(f"{prefix}[{tag}] INCONCLUSIVE: {display}")

        for child in children.get(rec.pred_id, []):
            lines.extend(_render_compact(child, indent + 1))
        return lines

    header = "== PREDICTION TREE (use read_predictions tool for full detail) =="
    all_lines = [header]
    for root in children[None]:
        all_lines.extend(_render_compact(root, 0))

    return "\n".join(all_lines)


def _format_predictions_for_prompt(
    prediction_history: list[PredictionRecord] | None,
) -> str:
    """Format prediction history as full-detail reasoning trajectories.

    Used by the stop gate, critic debate, and compare_personas script.
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
        prediction_history=_format_compact_tree(prediction_history),
        version=version,
    )

    system_prompt = SCIENTIST_SYSTEM

    json_instruction = (
        "\n\n## Output Format\n"
        "You MUST respond with ONLY valid JSON matching the schema below.\n"
        "No markdown fencing. No explanation. No other text.\n\n"
        f"Schema:\n{json.dumps(SCIENTIST_PLAN_SCHEMA, indent=2)}"
    )

    extra_args: dict[str, str | None] = {"setting-sources": ""}
    if reasoning and reasoning.level != "off":
        extra_args.update(reasoning_to_cc_extra_args(reasoning))

    tools, mcp_servers = _build_scientist_tools_and_mcp(prediction_history, provider)

    max_turns = 15
    backend = get_backend(provider)
    options = SDKOptions(
        system_prompt=with_turn_budget(system_prompt + json_instruction, max_turns, tools),
        allowed_tools=tools,
        max_turns=max_turns,
        model=model,
        extra_args=extra_args,
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
                agent_name="Scientist",
            )
        except Exception as e:
            if attempt == MAX_ATTEMPTS - 1:
                raise
            logger.warning(f"Scientist attempt {attempt + 1}: SDK error ({e}), retrying")
            continue

        try:
            return validate_json_output(raw, ScientistPlanOutput, "Scientist")
        except OutputValidationError as e:
            if attempt == MAX_ATTEMPTS - 1:
                raise
            correction_hint = f"\n\n{e.correction_prompt()}"
            logger.warning(f"Scientist attempt {attempt + 1} failed, retrying: {e}")

    raise RuntimeError("Scientist: exhausted retries")  # unreachable


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
        prediction_history=_format_compact_tree(prediction_history),
        version=version,
    )

    json_instruction = (
        "\n\n## Output Format\n"
        "You MUST respond with ONLY valid JSON matching the schema below.\n"
        "No markdown fencing. No explanation. No other text.\n\n"
        f"Schema:\n{json.dumps(SCIENTIST_PLAN_SCHEMA, indent=2)}"
    )

    extra_args: dict[str, str | None] = {"setting-sources": ""}
    if reasoning and reasoning.level != "off":
        extra_args.update(reasoning_to_cc_extra_args(reasoning))

    tools, mcp_servers = _build_scientist_tools_and_mcp(prediction_history, provider)

    max_turns = 15
    backend = get_backend(provider)
    options = SDKOptions(
        system_prompt=with_turn_budget(
            SCIENTIST_REVISION_SYSTEM + json_instruction, max_turns, tools
        ),
        allowed_tools=tools,
        max_turns=max_turns,
        model=model,
        extra_args=extra_args,
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
                agent_name="Scientist revision",
            )
        except Exception as e:
            if attempt == MAX_ATTEMPTS - 1:
                raise
            logger.warning(f"Scientist revision attempt {attempt + 1}: SDK error ({e}), retrying")
            continue

        try:
            return validate_json_output(raw, ScientistPlanOutput, "Scientist revision")
        except OutputValidationError as e:
            if attempt == MAX_ATTEMPTS - 1:
                raise
            correction_hint = f"\n\n{e.correction_prompt()}"
            logger.warning(f"Scientist revision attempt {attempt + 1} failed, retrying: {e}")

    raise RuntimeError("Scientist revision: exhausted retries")  # unreachable
