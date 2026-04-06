#!/usr/bin/env python3
"""Stdio MCP server for querying prediction history.

Launched as a subprocess by `build_prediction_mcp_server()`. Reads predictions
from a JSON file passed as the first CLI argument and exposes a
`read_predictions` tool over stdio.

Usage (by the Claude Code CLI, not directly):
    python3 _prediction_mcp_server.py /path/to/predictions.json

NOTE: Query helpers (_format_record, _get_ancestors, _build_compact_tree, etc.)
are intentionally duplicated from prediction_tool.py. This module runs as an
isolated subprocess and must not import framework code (Pydantic, persistence,
etc.). Keep both files in sync when changing query semantics.
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


def _get_display_text(rec: dict) -> str:
    """Return summary text, falling back to truncated evidence."""
    text = rec.get("summary") or rec.get("evidence") or rec.get("prediction", "")
    text = " ".join(text.split())
    if len(text) > 100:
        return text[:100] + "..."
    return text


def _build_compact_tree(predictions: list[dict]) -> str:
    """Build compact one-line-per-prediction tree from raw dicts."""
    by_id: dict[str, dict] = {}
    children: dict[str | None, list[dict]] = {None: []}
    for rec in predictions:
        pid = rec.get("pred_id")
        if pid:
            by_id[pid] = rec
    for rec in predictions:
        parent = rec.get("follows_from")
        if parent and parent in by_id:
            children.setdefault(parent, []).append(rec)
        else:
            children[None].append(rec)

    visited: set[str] = set()

    def _render(rec: dict, indent: int) -> list[str]:
        pid = rec.get("pred_id")
        if pid and pid in visited:
            return []
        if pid:
            visited.add(pid)

        prefix = "  " * indent
        tag = pid or f"v{rec.get('iteration_prescribed', 0):02d}"
        display = _get_display_text(rec)
        outcome = rec.get("outcome", "pending")
        lines = []

        if outcome == "pending":
            lines.append(f"{prefix}[{tag}] PENDING: {rec.get('prediction', '')}")
        elif outcome == "confirmed":
            lines.append(f"{prefix}[{tag}] CONFIRMED: {display} -> {rec.get('if_confirmed', '')}")
        elif outcome == "refuted":
            lines.append(f"{prefix}[{tag}] REFUTED: {display} -> {rec.get('if_refuted', '')}")
        elif outcome == "inconclusive":
            lines.append(f"{prefix}[{tag}] INCONCLUSIVE: {display}")

        for child in children.get(pid, []):
            lines.extend(_render(child, indent + 1))
        return lines

    header = "== PREDICTION TREE (call mcp__predictions__read_predictions for full detail) =="
    all_lines = [header]
    for root in children[None]:
        all_lines.extend(_render(root, 0))
    return "\n".join(all_lines)


def _build_status(predictions: list[dict]) -> str:
    """Build a counts-only status summary (no tree, since tree is inline in prompt)."""
    by_status: dict[str, int] = {}
    for rec in predictions:
        outcome = rec.get("outcome", "pending")
        by_status[outcome] = by_status.get(outcome, 0) + 1

    lines = [f"Total: {len(predictions)} predictions"]
    for status in ["confirmed", "refuted", "inconclusive", "pending"]:
        count = by_status.get(status, 0)
        if count:
            lines.append(f"  {status}: {count}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Query handler (the only prediction-specific logic)
# ---------------------------------------------------------------------------


def _normalize_args(args: dict[str, Any]) -> dict[str, Any]:
    """Coerce common LLM type mistakes into the canonical schema.

    Duplicated from prediction_tool.py to avoid importing framework code
    in this standalone subprocess script.
    """
    import json as _json

    out = dict(args)

    pred_ids = out.get("pred_ids")
    if isinstance(pred_ids, str):
        try:
            parsed = _json.loads(pred_ids)
            if isinstance(parsed, list):
                out["pred_ids"] = [str(x) for x in parsed]
            else:
                out["pred_ids"] = [pred_ids]
        except (ValueError, _json.JSONDecodeError):
            out["pred_ids"] = [pred_ids]
    elif isinstance(pred_ids, list):
        out["pred_ids"] = [str(x) for x in pred_ids]

    import contextlib as _contextlib

    iteration = out.get("iteration")
    if isinstance(iteration, str):
        with _contextlib.suppress(ValueError):
            out["iteration"] = int(iteration)

    return out


def _query(predictions: list[dict[str, Any]], args: dict[str, Any]) -> str:
    args = _normalize_args(args)

    if not predictions:
        return "No predictions in history yet."

    if args.get("summary"):
        return _build_status(predictions)

    by_id = {r["pred_id"]: r for r in predictions if r.get("pred_id")}
    available = ", ".join(sorted(by_id.keys()))

    pred_ids = args.get("pred_ids")
    chain_id = args.get("chain")
    outcome_value = args.get("outcome")
    iteration = args.get("iteration")

    if not pred_ids and not chain_id and not outcome_value and iteration is None:
        return (
            "Please specify a query: summary, pred_ids, chain, outcome, "
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

    elif outcome_value == "active_chains":
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

    elif outcome_value:
        selected = [r for r in predictions if r.get("outcome") == outcome_value]

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
            "Read full evidence, diagnostics, and conditional outcomes "
            "for specific predictions. The compact tree in your prompt "
            "shows status and short summaries only. Call this tool when "
            "you need to verify a claim, inspect why a prediction was "
            "confirmed or refuted, check the diagnostic used, or trace "
            "a reasoning chain. Use summary=true for counts, chain/ "
            "pred_ids/outcome/iteration for detailed records."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "summary": {
                    "type": "boolean",
                    "description": (
                        "Returns a count header: total "
                        "predictions and breakdown by outcome "
                        "(confirmed, refuted, inconclusive, "
                        "pending). Use for a quick tally."
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
            "summary?, chain?, pred_ids?, outcome?, iteration?) "
            "- Read evidence, diagnostics, outcomes for predictions. "
            "Tree in prompt shows summaries only."
        ),
    )

    run_mcp_server_main("predictions", _SPEC, _query)
