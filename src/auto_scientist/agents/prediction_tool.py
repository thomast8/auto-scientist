"""MCP tool for querying prediction history on demand.

Provides `build_prediction_mcp_server()` which creates an stdio MCP server
config that the Claude Code CLI can connect to. The server reads predictions
from a temporary JSON file and exposes a `read_predictions` tool.

Also provides `_handle_read_predictions()` for direct use in tests.
"""

from __future__ import annotations

import contextlib
import json
import logging
from pathlib import Path
from typing import Any

from auto_scientist.agents._mcp_base import MCPToolSpec, build_mcp_server_config, register_mcp_tool
from auto_scientist.state import PredictionRecord

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool spec (registered so sdk_utils can auto-discover the description)
# ---------------------------------------------------------------------------

_READ_PREDICTIONS_DESCRIPTION = (
    "Drill into prediction history details. The compact prediction tree "
    "is already in your prompt; use this tool for full records. "
    "Use summary=true for counts, chain/pred_ids/outcome/iteration for "
    "specific predictions with full detail (evidence, diagnostics, "
    "implications). Prefer fewer targeted queries over exhaustive audits."
)

_READ_PREDICTIONS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "boolean",
            "description": (
                "Returns a count header: total predictions and "
                "breakdown by outcome (confirmed, refuted, "
                "inconclusive, pending). Use for a quick tally."
            ),
        },
        "chain": {
            "type": "string",
            "description": (
                "A prediction ID. Returns the full reasoning chain: root "
                "ancestor through this prediction to all descendants. "
                "Best for understanding why a particular investigation "
                "thread exists. E.g. chain='2.1'."
            ),
        },
        "pred_ids": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Specific prediction IDs to retrieve with full detail "
                "(evidence, diagnostics, implications). "
                "E.g. ['2.1', '3.4']."
            ),
        },
        "outcome": {
            "type": "string",
            "enum": [
                "pending",
                "refuted",
                "confirmed",
                "inconclusive",
                "active_chains",
            ],
            "description": (
                "Return all predictions with this outcome. "
                "'active_chains' returns pending predictions plus their "
                "full ancestor chains."
            ),
        },
        "iteration": {
            "type": "integer",
            "description": "Return predictions prescribed in a specific iteration.",
        },
    },
}


def _normalize_args(args: dict[str, Any]) -> dict[str, Any]:
    """Coerce common LLM type mistakes into the canonical schema.

    Fixes observed in production:
      - ``pred_ids: '["0.1"]'`` (JSON string) instead of ``["0.1"]`` (array)
      - ``pred_ids: "0.1"`` (bare string) instead of ``["0.1"]``
      - ``iteration: "1"`` (string) instead of ``1`` (int)
    """
    out = dict(args)

    # pred_ids: string -> parse as JSON array or wrap bare string
    # Also coerce elements to str (models sometimes send numeric IDs like [0.1])
    pred_ids = out.get("pred_ids")
    if isinstance(pred_ids, str):
        try:
            parsed = json.loads(pred_ids)
            if isinstance(parsed, list):
                out["pred_ids"] = [str(x) for x in parsed]
            else:
                out["pred_ids"] = [pred_ids]
        except (json.JSONDecodeError, ValueError):
            out["pred_ids"] = [pred_ids]
    elif isinstance(pred_ids, list):
        out["pred_ids"] = [str(x) for x in pred_ids]

    # iteration: numeric string -> int
    iteration = out.get("iteration")
    if isinstance(iteration, str):
        with contextlib.suppress(ValueError):
            out["iteration"] = int(iteration)

    return out


PREDICTION_SPEC = MCPToolSpec(
    server_name="predictions",
    tool_name="read_predictions",
    description=_READ_PREDICTIONS_DESCRIPTION,
    input_schema=_READ_PREDICTIONS_SCHEMA,
    deferred_description=(
        "mcp__predictions__read_predictions("
        "summary?, chain?, pred_ids?, outcome?, iteration?) "
        "- Drill into prediction details. Tree is already in your prompt."
    ),
)

register_mcp_tool(PREDICTION_SPEC)


# ---------------------------------------------------------------------------
# Compact tree (canonical implementation, inlined in all agent prompts)
# ---------------------------------------------------------------------------


def _get_display_text(rec: PredictionRecord) -> str:
    """Return the summary text for a prediction, falling back to truncated evidence.

    Sanitizes output to ensure single-line (collapses newlines, hard-truncates).
    """
    text = rec.summary or rec.evidence or rec.prediction
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


def format_compact_tree(
    prediction_history: list[PredictionRecord] | None,
) -> str:
    """Format prediction history as a compact one-line-per-prediction tree.

    Each prediction gets a single line with status, summary, and implication.
    Parent-child chains are shown via indentation.
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
            lines.append(f"{prefix}[{tag}] REFUTED: {display} -> {rec.if_refuted}")
        elif rec.outcome == "inconclusive":
            lines.append(f"{prefix}[{tag}] INCONCLUSIVE: {display}")

        for child in children.get(rec.pred_id, []):
            lines.extend(_render_compact(child, indent + 1))
        return lines

    header = "== PREDICTION TREE =="
    all_lines = [header]
    for root in children[None]:
        all_lines.extend(_render_compact(root, 0))

    return "\n".join(all_lines)


# ---------------------------------------------------------------------------
# Full-detail formatting (used by detail queries and _handle_read_predictions)
# ---------------------------------------------------------------------------


def _format_record_detail(rec: PredictionRecord) -> str:
    """Format a single PredictionRecord as full-detail text."""
    tag = rec.pred_id or f"v{rec.iteration_prescribed:02d}"
    status = rec.outcome.upper()
    eval_info = f"prescribed iter {rec.iteration_prescribed}"
    if rec.iteration_evaluated is not None:
        eval_info += f", evaluated iter {rec.iteration_evaluated}"

    lines = [f"[{tag}] {status} ({eval_info})"]
    lines.append(f"  Prediction: {rec.prediction}")
    lines.append(f"  Diagnostic: {rec.diagnostic}")
    lines.append(f"  If confirmed: {rec.if_confirmed}")
    lines.append(f"  If refuted: {rec.if_refuted}")
    if rec.evidence:
        lines.append(f"  Evidence: {rec.evidence}")
    if rec.follows_from:
        lines.append(f"  Follows from: [{rec.follows_from}]")
    return "\n".join(lines)


def _get_ancestor_ids(pred_id: str, by_id: dict[str, PredictionRecord]) -> set[str]:
    """Walk follows_from links upward, collecting ancestor pred_ids."""
    ancestors: set[str] = set()
    current = pred_id
    while current in by_id:
        rec = by_id[current]
        parent = rec.follows_from
        if not parent or parent in ancestors:
            break
        ancestors.add(parent)
        current = parent
    return ancestors


def _get_descendant_ids(pred_id: str, all_preds: list[PredictionRecord]) -> set[str]:
    """Walk follows_from links downward, collecting all descendant pred_ids."""
    descendants: set[str] = set()
    frontier = {pred_id}
    while frontier:
        current = frontier.pop()
        for rec in all_preds:
            if rec.follows_from == current and rec.pred_id and rec.pred_id not in descendants:
                descendants.add(rec.pred_id)
                frontier.add(rec.pred_id)
    return descendants


def _get_full_chain_ids(
    pred_id: str,
    by_id: dict[str, PredictionRecord],
    all_preds: list[PredictionRecord],
) -> set[str]:
    """Get the full chain (ancestors + self + descendants) for a prediction."""
    chain = {pred_id}
    chain |= _get_ancestor_ids(pred_id, by_id)
    chain |= _get_descendant_ids(pred_id, all_preds)
    return chain


def _build_status_response(
    prediction_history: list[PredictionRecord],
) -> dict[str, Any]:
    """Build a counts-only status summary (no tree, since tree is inline in prompt)."""
    by_status: dict[str, int] = {}
    for rec in prediction_history:
        by_status[rec.outcome] = by_status.get(rec.outcome, 0) + 1

    lines = [f"Total: {len(prediction_history)} predictions"]
    for status in ["confirmed", "refuted", "inconclusive", "pending"]:
        count = by_status.get(status, 0)
        if count:
            lines.append(f"  {status}: {count}")

    return {"content": [{"type": "text", "text": "\n".join(lines)}]}


# ---------------------------------------------------------------------------
# Direct handler (for unit tests - same logic as the MCP server subprocess)
# ---------------------------------------------------------------------------


async def _handle_read_predictions(
    prediction_history: list[PredictionRecord],
    args: dict[str, Any],
) -> dict[str, Any]:
    """Core handler logic for the read_predictions tool."""
    args = _normalize_args(args)

    if not prediction_history:
        return {"content": [{"type": "text", "text": "No predictions in history yet."}]}

    by_id = {r.pred_id: r for r in prediction_history if r.pred_id}
    available = ", ".join(sorted(by_id.keys()))

    # Summary mode: counts only (tree is already in the prompt)
    if args.get("summary"):
        return _build_status_response(prediction_history)

    pred_ids = args.get("pred_ids")
    chain_id = args.get("chain")
    outcome_value = args.get("outcome")
    iteration = args.get("iteration")

    # Require at least one query parameter
    if not pred_ids and not chain_id and not outcome_value and iteration is None:
        return {
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Please specify a query: summary, pred_ids, chain, "
                        f"outcome, or iteration. Available IDs: {available}"
                    ),
                }
            ]
        }

    selected: list[PredictionRecord] = []

    if pred_ids:
        missing = [pid for pid in pred_ids if pid not in by_id]
        for pid in pred_ids:
            if pid in by_id:
                selected.append(by_id[pid])
        if not selected:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Not found: {', '.join(pred_ids)}. Available IDs: {available}",
                    }
                ]
            }
        if missing:
            missing_note = f"Note: IDs not found: {', '.join(missing)}\n\n"
            formatted = missing_note + "\n\n".join(_format_record_detail(r) for r in selected)
            return {"content": [{"type": "text", "text": formatted}]}

    elif chain_id:
        if chain_id not in by_id:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Not found: {chain_id}. Available IDs: {available}",
                    }
                ]
            }
        chain_ids = _get_full_chain_ids(chain_id, by_id, prediction_history)
        # Return in chronological order (by iteration prescribed)
        selected = sorted(
            [r for r in prediction_history if r.pred_id in chain_ids],
            key=lambda r: (r.iteration_prescribed, r.pred_id),
        )

    elif outcome_value == "active_chains":
        pending = [r for r in prediction_history if r.outcome == "pending"]
        active_ids: set[str] = set()
        id_less_pending: list[PredictionRecord] = []
        for rec in pending:
            if rec.pred_id:
                active_ids.add(rec.pred_id)
                active_ids |= _get_ancestor_ids(rec.pred_id, by_id)
            else:
                id_less_pending.append(rec)
        selected = [r for r in prediction_history if r.pred_id in active_ids]
        selected.extend(id_less_pending)

    elif outcome_value:
        selected = [r for r in prediction_history if r.outcome == outcome_value]

    elif iteration is not None:
        selected = [r for r in prediction_history if r.iteration_prescribed == iteration]

    if not selected:
        return {"content": [{"type": "text", "text": "No predictions match the query."}]}

    formatted = "\n\n".join(_format_record_detail(r) for r in selected)
    return {"content": [{"type": "text", "text": formatted}]}


# ---------------------------------------------------------------------------
# Server config builder
# ---------------------------------------------------------------------------


def build_prediction_mcp_server(
    prediction_history: list[PredictionRecord],
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Create an stdio MCP server config for the read_predictions tool.

    Delegates to the shared ``build_mcp_server_config()`` for serialization
    and stdio config generation.

    Returns a dict suitable for ``SDKOptions.mcp_servers`` values.
    """
    server_script = Path(__file__).parent / "_prediction_mcp_server.py"
    return build_mcp_server_config(
        data_dicts=[r.model_dump() for r in prediction_history],
        server_script=server_script,
        output_dir=output_dir,
        filename="predictions.json",
    )
