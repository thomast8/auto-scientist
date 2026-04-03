"""MCP tool for querying prediction history on demand.

Provides `build_prediction_mcp_server()` which creates an stdio MCP server
config that the Claude Code CLI can connect to. The server reads predictions
from a temporary JSON file and exposes a `read_predictions` tool.

Also provides `_handle_read_predictions()` for direct use in tests.
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

from auto_scientist.state import PredictionRecord

logger = logging.getLogger(__name__)

# JSON Schema for the tool input.
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
        "chain": {
            "type": "string",
            "description": (
                "A prediction ID. Returns the full chain from root ancestor "
                "to all descendants for that prediction. E.g. chain='2.1' "
                "returns the root, 2.1, and any children of 2.1."
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


async def _handle_read_predictions(
    prediction_history: list[PredictionRecord],
    args: dict[str, Any],
) -> dict[str, Any]:
    """Core handler logic for the read_predictions tool."""
    if not prediction_history:
        return {"content": [{"type": "text", "text": "No predictions in history yet."}]}

    by_id = {r.pred_id: r for r in prediction_history if r.pred_id}
    available = ", ".join(sorted(by_id.keys()))

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
                        "Please specify a query: pred_ids, chain, filter, or iteration. "
                        f"Available IDs: {available}"
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


def build_prediction_mcp_server(
    prediction_history: list[PredictionRecord],
) -> dict[str, Any]:
    """Create an stdio MCP server config for the read_predictions tool.

    Writes predictions to a temp JSON file and returns a stdio server config
    that launches a Python subprocess serving them via the ``mcp`` library.
    The Claude Code CLI connects to this server via stdin/stdout.

    Returns a dict suitable for ``ClaudeCodeOptions.mcp_servers``.
    """
    # Write predictions to a temp file the subprocess can read
    predictions_data = [r.model_dump() for r in prediction_history]
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        prefix="predictions_",
        delete=False,
    ) as tmp:
        json.dump(predictions_data, tmp)
    logger.debug(f"Wrote {len(prediction_history)} predictions to {tmp.name}")

    # Return stdio server config pointing to the MCP server script
    server_script = str(Path(__file__).parent / "_prediction_mcp_server.py")
    return {
        "type": "stdio",
        "command": "python3",
        "args": [server_script, tmp.name],
    }


def write_codex_mcp_config(
    prediction_history: list[PredictionRecord],
    cwd: Path,
) -> None:
    """Write a .codex/config.toml with the predictions MCP server.

    Codex reads MCP config from ``<cwd>/.codex/config.toml`` at startup.
    This writes the predictions MCP server there so Codex agents get the
    same ``read_predictions`` tool that Claude agents get via mcp_servers.
    """
    predictions_data = [r.model_dump() for r in prediction_history]
    predictions_path = cwd / ".codex" / "predictions.json"
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.write_text(json.dumps(predictions_data))

    server_script = str(Path(__file__).parent / "_prediction_mcp_server.py")

    # Codex config.toml format for stdio MCP servers
    config_path = cwd / ".codex" / "config.toml"

    # Read existing config if present, preserve non-MCP sections
    existing = config_path.read_text() if config_path.exists() else ""
    lines = existing.splitlines()
    # Remove any existing predictions MCP section
    filtered: list[str] = []
    skip = False
    for line in lines:
        if line.strip() == "[mcp_servers.predictions]":
            skip = True
            continue
        if skip and line.strip().startswith("["):
            skip = False
        if not skip:
            filtered.append(line)

    # Append predictions MCP server
    filtered.append("")
    filtered.append("[mcp_servers.predictions]")
    filtered.append('command = "python3"')
    filtered.append(f'args = ["{server_script}", "{predictions_path}"]')
    filtered.append("")

    config_path.write_text("\n".join(filtered))
    logger.debug(f"Wrote Codex MCP config to {config_path}")
