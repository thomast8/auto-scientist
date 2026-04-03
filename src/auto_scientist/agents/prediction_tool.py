"""In-process MCP tool for querying prediction history on demand.

Provides `build_prediction_mcp_server()` which creates an MCP server with a
`read_predictions` tool that the Scientist agent can call to inspect specific
predictions without having the full history dumped into its prompt.
"""

from __future__ import annotations

from typing import Any

from claude_code_sdk import McpSdkServerConfig, create_sdk_mcp_server, tool

from auto_scientist.state import PredictionRecord

# JSON Schema for the tool input (full schema, not simple type mapping,
# because we need optional params and enums).
_READ_PREDICTIONS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "pred_ids": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Specific prediction IDs to retrieve, e.g. ['2.1', '3.4']. "
                "Use the bracketed IDs from the prediction tree."
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
                "Filter predictions by status. 'active_chains' returns pending "
                "predictions plus their full ancestor chains."
            ),
        },
        "iteration": {
            "type": "integer",
            "description": "Return predictions prescribed in a specific iteration.",
        },
    },
}


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


async def _handle_read_predictions(
    prediction_history: list[PredictionRecord],
    args: dict[str, Any],
) -> dict[str, Any]:
    """Core handler logic for the read_predictions tool."""
    if not prediction_history:
        return {"content": [{"type": "text", "text": "No predictions in history yet."}]}

    by_id = {r.pred_id: r for r in prediction_history if r.pred_id}

    pred_ids = args.get("pred_ids")
    status_filter = args.get("filter")
    iteration = args.get("iteration")

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
                        "text": (
                            f"Not found: {', '.join(pred_ids)}. "
                            f"Available IDs: {', '.join(sorted(by_id.keys()))}"
                        ),
                    }
                ]
            }
        if missing:
            # Prepend a note about missing IDs so the Scientist knows
            missing_note = f"Note: IDs not found: {', '.join(missing)}\n\n"
            formatted = missing_note + "\n\n".join(_format_record_detail(r) for r in selected)
            return {"content": [{"type": "text", "text": formatted}]}

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

    else:
        selected = list(prediction_history)

    if not selected:
        return {"content": [{"type": "text", "text": "No predictions match the query."}]}

    formatted = "\n\n".join(_format_record_detail(r) for r in selected)
    return {"content": [{"type": "text", "text": formatted}]}


def build_prediction_mcp_server(
    prediction_history: list[PredictionRecord],
) -> McpSdkServerConfig:
    """Create an in-process MCP server with a read_predictions tool.

    The tool captures ``prediction_history`` by reference, so mutations
    by the orchestrator are visible in real-time during an agent's turn.
    """

    @tool(
        "read_predictions",
        "Read full detail for specific predictions from the prediction history. "
        "Use this to inspect evidence, diagnostics, and implications for any "
        "prediction shown in the compact tree summary.",
        _READ_PREDICTIONS_SCHEMA,
    )
    async def read_predictions(args: dict[str, Any]) -> dict[str, Any]:
        return await _handle_read_predictions(prediction_history, args)

    return create_sdk_mcp_server(
        name="predictions",
        version="1.0.0",
        tools=[read_predictions],
    )
