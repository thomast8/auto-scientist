"""Pure query logic for the prediction MCP tool.

Stdlib only. This module is imported both by the framework-side wrapper in
``prediction_tool.py`` and by the stdio subprocess script
``_prediction_mcp_server.py``. See ``notebook_query.py`` for the rationale
behind extracting query helpers into a shared zero-dep module.

Inputs are plain ``list[dict[str, Any]]`` of pre-serialized prediction
records (``PredictionRecord.model_dump()`` on the framework side, the same
JSON shape on the subprocess side). Output of :func:`query` is a ``str``
ready to be wrapped in MCP TextContent.
"""

from __future__ import annotations

import contextlib
import json as _json
from typing import Any

# ---------------------------------------------------------------------------
# LLM-arg normalization
# ---------------------------------------------------------------------------


def normalize_args(args: dict[str, Any]) -> dict[str, Any]:
    """Coerce common LLM type mistakes into the canonical schema.

    Fixes observed in production:

    * ``pred_ids: '["0.1"]'`` (JSON string) instead of ``["0.1"]`` (array).
    * ``pred_ids: "0.1"`` (bare string) instead of ``["0.1"]``.
    * ``iteration: "1"`` (string) instead of ``1`` (int).
    """
    out = dict(args)

    pred_ids = out.get("pred_ids")
    if isinstance(pred_ids, str):
        try:
            parsed = _json.loads(pred_ids)
            if isinstance(parsed, list):
                out["pred_ids"] = [str(x) for x in parsed]
            else:
                out["pred_ids"] = [pred_ids]
        except (_json.JSONDecodeError, ValueError):
            out["pred_ids"] = [pred_ids]
    elif isinstance(pred_ids, list):
        out["pred_ids"] = [str(x) for x in pred_ids]

    iteration = out.get("iteration")
    if isinstance(iteration, str):
        with contextlib.suppress(ValueError):
            out["iteration"] = int(iteration)

    return out


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def format_record(rec: dict[str, Any]) -> str:
    """Format a single prediction record dict as full-detail text."""
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


def build_status(predictions: list[dict[str, Any]]) -> str:
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
# Tree traversal helpers
# ---------------------------------------------------------------------------


def get_ancestors(pred_id: str, by_id: dict[str, dict[str, Any]]) -> set[str]:
    """Walk follows_from links upward, collecting ancestor pred_ids."""
    ancestors: set[str] = set()
    current = pred_id
    while current in by_id:
        parent = by_id[current].get("follows_from")
        if not parent or parent in ancestors:
            break
        ancestors.add(parent)
        current = parent
    return ancestors


def get_descendants(pred_id: str, predictions: list[dict[str, Any]]) -> set[str]:
    """Walk follows_from links downward, collecting all descendant pred_ids."""
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


def get_full_chain(
    pred_id: str,
    by_id: dict[str, dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> set[str]:
    """Get the full chain (ancestors + self + descendants) for a prediction."""
    chain = {pred_id}
    chain |= get_ancestors(pred_id, by_id)
    chain |= get_descendants(pred_id, predictions)
    return chain


# ---------------------------------------------------------------------------
# Query handler
# ---------------------------------------------------------------------------


def query(predictions: list[dict[str, Any]], args: dict[str, Any]) -> str:
    """Core handler logic for the read_predictions tool.

    Returns a plain ``str``. Both the framework-side ``_handle_read_predictions``
    (which wraps it in MCP content shape) and the subprocess
    ``run_mcp_server_main`` (which wraps it in TextContent) call this.
    """
    args = normalize_args(args)

    if not predictions:
        return "No predictions in history yet."

    if args.get("summary"):
        return build_status(predictions)

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

    selected: list[dict[str, Any]] = []

    if pred_ids:
        missing = [pid for pid in pred_ids if pid not in by_id]
        selected = [by_id[pid] for pid in pred_ids if pid in by_id]
        if not selected:
            return f"Not found: {', '.join(pred_ids)}. Available IDs: {available}"
        if missing:
            note = f"Note: IDs not found: {', '.join(missing)}\n\n"
            return note + "\n\n".join(format_record(r) for r in selected)

    elif chain_id:
        if chain_id not in by_id:
            return f"Not found: {chain_id}. Available IDs: {available}"
        chain_ids = get_full_chain(chain_id, by_id, predictions)
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
                active_ids |= get_ancestors(pid, by_id)
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

    return "\n\n".join(format_record(r) for r in selected)
