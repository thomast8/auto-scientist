#!/usr/bin/env python3
"""Stdio MCP server for querying prediction history.

Launched as a subprocess by `build_prediction_mcp_server()`. Reads predictions
from a JSON file passed as the first CLI argument and exposes a
`read_predictions` tool over stdio.

Usage (by the Claude Code CLI, not directly):
    python3 _prediction_mcp_server.py /path/to/predictions.json
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool


def _load_predictions(path: str) -> list[dict[str, Any]]:
    with open(path) as f:
        result: list[dict[str, Any]] = json.load(f)
    return result


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


def _query(predictions: list[dict], args: dict[str, Any]) -> str:
    if not predictions:
        return "No predictions in history yet."

    by_id = {r["pred_id"]: r for r in predictions if r.get("pred_id")}

    pred_ids = args.get("pred_ids")
    status_filter = args.get("filter")
    iteration = args.get("iteration")

    selected: list[dict] = []

    if pred_ids:
        missing = [pid for pid in pred_ids if pid not in by_id]
        selected = [by_id[pid] for pid in pred_ids if pid in by_id]
        if not selected:
            available = ", ".join(sorted(by_id.keys()))
            return f"Not found: {', '.join(pred_ids)}. Available IDs: {available}"
        if missing:
            note = f"Note: IDs not found: {', '.join(missing)}\n\n"
            return note + "\n\n".join(_format_record(r) for r in selected)

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

    else:
        selected = list(predictions)

    if not selected:
        return "No predictions match the query."

    return "\n\n".join(_format_record(r) for r in selected)


async def main():
    if len(sys.argv) < 2:
        print("Usage: _prediction_mcp_server.py <predictions.json>", file=sys.stderr)
        sys.exit(1)

    predictions = _load_predictions(sys.argv[1])

    server = Server("predictions")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="read_predictions",
                description=(
                    "Read full detail for specific predictions from the "
                    "prediction history. Use this to inspect evidence, "
                    "diagnostics, and implications for any prediction shown "
                    "in the compact tree summary."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "pred_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Prediction IDs to retrieve",
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
                            "description": "Filter by status",
                        },
                        "iteration": {
                            "type": "integer",
                            "description": "Predictions from this iteration",
                        },
                    },
                },
            )
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any] | None) -> list[TextContent]:
        if name != "read_predictions":
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
        result = _query(predictions, arguments or {})
        return [TextContent(type="text", text=result)]

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
