"""MCP tool for querying prediction history on demand.

Provides `build_prediction_mcp_server()` which creates an stdio MCP server
config that the Claude Code CLI can connect to. The server reads predictions
from a temporary JSON file and exposes a `read_predictions` tool.

Also provides `_handle_read_predictions()` for direct use in tests.
"""

from __future__ import annotations

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
    "Query the prediction history for detail not shown in the compact tree. "
    "Start with stats=true to see counts by status/iteration, then use "
    "targeted queries (chain, pred_ids, filter) to inspect specifics. "
    "Each call loads results into your context, so prefer fewer targeted "
    "queries over exhaustive audits."
)

_READ_PREDICTIONS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "stats": {
            "type": "boolean",
            "description": (
                "Returns counts by status and iteration, plus a one-line "
                "summary per prediction. Use this first to orient, then "
                "drill into specifics with other parameters."
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
        "filter": {
            "type": "string",
            "enum": [
                "pending",
                "refuted",
                "confirmed",
                "inconclusive",
                "active_chains",
            ],
            "description": (
                "Return all predictions with this status. "
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

PREDICTION_SPEC = MCPToolSpec(
    server_name="predictions",
    tool_name="read_predictions",
    description=_READ_PREDICTIONS_DESCRIPTION,
    input_schema=_READ_PREDICTIONS_SCHEMA,
    deferred_description=(
        "mcp__predictions__read_predictions("
        "stats?, chain?, pred_ids?, filter?, iteration?) "
        "- Query prediction history for detail beyond the compact tree."
    ),
)

register_mcp_tool(PREDICTION_SPEC)


# ---------------------------------------------------------------------------
# Formatting and traversal helpers (used by _handle_read_predictions)
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


def _build_stats_response(
    prediction_history: list[PredictionRecord],
) -> dict[str, Any]:
    """Build a compact stats summary of prediction counts."""
    by_status: dict[str, int] = {}
    by_iter: dict[int, list[str]] = {}
    for rec in prediction_history:
        by_status[rec.outcome] = by_status.get(rec.outcome, 0) + 1
        it = rec.iteration_prescribed
        by_iter.setdefault(it, []).append(f"[{rec.pred_id}] {rec.outcome.upper()}")

    lines = [f"Total: {len(prediction_history)} predictions"]
    lines.append("")
    lines.append("By status:")
    for status in ["confirmed", "refuted", "inconclusive", "pending"]:
        count = by_status.get(status, 0)
        if count:
            lines.append(f"  {status}: {count}")

    lines.append("")
    lines.append("By iteration:")
    for it in sorted(by_iter):
        preds = by_iter[it]
        lines.append(f"  iter {it}: {', '.join(preds)}")

    return {"content": [{"type": "text", "text": "\n".join(lines)}]}


# ---------------------------------------------------------------------------
# Direct handler (for unit tests - same logic as the MCP server subprocess)
# ---------------------------------------------------------------------------


async def _handle_read_predictions(
    prediction_history: list[PredictionRecord],
    args: dict[str, Any],
) -> dict[str, Any]:
    """Core handler logic for the read_predictions tool."""
    if not prediction_history:
        return {"content": [{"type": "text", "text": "No predictions in history yet."}]}

    by_id = {r.pred_id: r for r in prediction_history if r.pred_id}
    available = ", ".join(sorted(by_id.keys()))

    # Stats mode: return counts overview, no detail
    if args.get("stats"):
        return _build_stats_response(prediction_history)

    pred_ids = args.get("pred_ids")
    chain_id = args.get("chain")
    status_filter = args.get("filter")
    iteration = args.get("iteration")

    # Require at least one query parameter
    if not pred_ids and not chain_id and not status_filter and iteration is None:
        return {
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Please specify a query: stats, pred_ids, chain, "
                        f"filter, or iteration. Available IDs: {available}"
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

    elif status_filter == "active_chains":
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

    elif status_filter:
        selected = [r for r in prediction_history if r.outcome == status_filter]

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
