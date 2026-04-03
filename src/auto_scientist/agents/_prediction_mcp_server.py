#!/usr/bin/env python3
"""Stdio MCP server for querying prediction history.

Launched as a subprocess by `build_prediction_mcp_server()`. Reads predictions
from a JSON file passed as the first CLI argument and exposes a
`read_predictions` tool over stdio.

Usage (by the Claude Code CLI, not directly):
    python3 _prediction_mcp_server.py /path/to/predictions.json
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Prediction-specific formatting and traversal
# ---------------------------------------------------------------------------


def _format_record(rec: dict[str, Any]) -> str:
    tag = rec.get("pred_id") or f"v{rec.get('iteration_prescribed', 0):02d}"
    status = rec.get("outcome", "pending").upper()
    eval_info = f"prescribed iter {rec.get('iteration_prescribed', '?')}"
    if rec.get("iteration_evaluated") is not None:
        eval_info += f", evaluated iter {rec['iteration_evaluated']}"

    lines = [f"[{tag}] {status} ({eval_info})"]
    lines.append(f"  Prediction: {rec.get('prediction', '')}")
    lines.append(f"  Diagnostic: {rec.get('diagnostic', '')}")
    lines.append(f"  If confirmed: {rec.get('if_confirmed', '')}")
    lines.append(f"  If refuted: {rec.get('if_refuted', '')}")
    if rec.get("evidence"):
        lines.append(f"  Evidence: {rec['evidence']}")
    if rec.get("follows_from"):
        lines.append(f"  Follows from: [{rec['follows_from']}]")
    return "\n".join(lines)


def _get_ancestors(pred_id: str, by_id: dict[str, dict]) -> set[str]:
    ancestors: set[str] = set()
    current = pred_id
    while current in by_id:
        parent = by_id[current].get("follows_from")
        if not parent or parent in ancestors:
            break
        ancestors.add(parent)
        current = parent
    return ancestors


def _get_descendants(pred_id: str, predictions: list[dict]) -> set[str]:
    descendants: set[str] = set()
    frontier = {pred_id}
    while frontier:
        current = frontier.pop()
        for rec in predictions:
            child_id = rec.get("pred_id", "")
            if rec.get("follows_from") == current and child_id and child_id not in descendants:
                descendants.add(child_id)
                frontier.add(child_id)
    return descendants


def _get_full_chain(pred_id: str, by_id: dict[str, dict], predictions: list[dict]) -> set[str]:
    chain = {pred_id}
    chain |= _get_ancestors(pred_id, by_id)
    chain |= _get_descendants(pred_id, predictions)
    return chain


def _build_stats(predictions: list[dict]) -> str:
    by_status: dict[str, int] = {}
    by_iter: dict[int, list[str]] = {}
    for rec in predictions:
        outcome = rec.get("outcome", "pending")
        by_status[outcome] = by_status.get(outcome, 0) + 1
        it = rec.get("iteration_prescribed", 0)
        pid = rec.get("pred_id", "?")
        by_iter.setdefault(it, []).append(f"[{pid}] {outcome.upper()}")

    lines = [f"Total: {len(predictions)} predictions", ""]
    lines.append("By status:")
    for status in ["confirmed", "refuted", "inconclusive", "pending"]:
        count = by_status.get(status, 0)
        if count:
            lines.append(f"  {status}: {count}")
    lines.append("")
    lines.append("By iteration:")
    for it in sorted(by_iter):
        lines.append(f"  iter {it}: {', '.join(by_iter[it])}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Query handler (the only prediction-specific logic)
# ---------------------------------------------------------------------------


def _query(predictions: list[dict[str, Any]], args: dict[str, Any]) -> str:
    if not predictions:
        return "No predictions in history yet."

    if args.get("stats"):
        return _build_stats(predictions)

    by_id = {r["pred_id"]: r for r in predictions if r.get("pred_id")}
    available = ", ".join(sorted(by_id.keys()))

    pred_ids = args.get("pred_ids")
    chain_id = args.get("chain")
    status_filter = args.get("filter")
    iteration = args.get("iteration")

    if not pred_ids and not chain_id and not status_filter and iteration is None:
        return (
            "Please specify a query: stats, pred_ids, chain, filter, "
            f"or iteration. Available IDs: {available}"
        )

    selected: list[dict] = []

    if pred_ids:
        missing = [pid for pid in pred_ids if pid not in by_id]
        selected = [by_id[pid] for pid in pred_ids if pid in by_id]
        if not selected:
            return f"Not found: {', '.join(pred_ids)}. Available IDs: {available}"
        if missing:
            note = f"Note: IDs not found: {', '.join(missing)}\n\n"
            return note + "\n\n".join(_format_record(r) for r in selected)

    elif chain_id:
        if chain_id not in by_id:
            return f"Not found: {chain_id}. Available IDs: {available}"
        chain_ids = _get_full_chain(chain_id, by_id, predictions)
        selected = sorted(
            [r for r in predictions if r.get("pred_id") in chain_ids],
            key=lambda r: (r.get("iteration_prescribed", 0), r.get("pred_id", "")),
        )

    elif status_filter == "active_chains":
        pending = [r for r in predictions if r.get("outcome") == "pending"]
        active_ids: set[str] = set()
        id_less = []
        for rec in pending:
            pid = rec.get("pred_id", "")
            if pid:
                active_ids.add(pid)
                active_ids |= _get_ancestors(pid, by_id)
            else:
                id_less.append(rec)
        selected = [r for r in predictions if r.get("pred_id") in active_ids]
        selected.extend(id_less)

    elif status_filter:
        selected = [r for r in predictions if r.get("outcome") == status_filter]

    elif iteration is not None:
        selected = [r for r in predictions if r.get("iteration_prescribed") == iteration]

    if not selected:
        return "No predictions match the query."

    return "\n\n".join(_format_record(r) for r in selected)


# ---------------------------------------------------------------------------
# Entry point - all boilerplate handled by _mcp_base
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from auto_scientist.agents._mcp_base import MCPToolSpec, run_mcp_server_main

    _SPEC = MCPToolSpec(
        server_name="predictions",
        tool_name="read_predictions",
        description=(
            "Query the prediction history for detail not shown "
            "in the compact tree. Start with stats=true to see "
            "counts by status/iteration, then use targeted "
            "queries (chain, pred_ids, filter) to inspect "
            "specifics. Each call loads results into your "
            "context, so prefer fewer targeted queries over "
            "exhaustive audits."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "stats": {
                    "type": "boolean",
                    "description": (
                        "Returns counts by status and iteration, "
                        "plus a one-line summary per prediction. "
                        "Use this first to orient, then drill "
                        "into specifics with other parameters."
                    ),
                },
                "chain": {
                    "type": "string",
                    "description": (
                        "A prediction ID. Returns the full "
                        "reasoning chain: root ancestor through "
                        "this prediction to all descendants. "
                        "Best for understanding why a particular "
                        "investigation thread exists."
                    ),
                },
                "pred_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Specific prediction IDs to retrieve "
                        "with full detail (evidence, diagnostics, "
                        "implications)."
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
                        "'active_chains' returns pending "
                        "predictions plus their full ancestor "
                        "chains."
                    ),
                },
                "iteration": {
                    "type": "integer",
                    "description": ("Return predictions prescribed in a specific iteration."),
                },
            },
        },
        deferred_description=(
            "mcp__predictions__read_predictions("
            "stats?, chain?, pred_ids?, filter?, iteration?) "
            "- Query prediction history for detail beyond the compact tree."
        ),
    )

    run_mcp_server_main("predictions", _SPEC, _query)
