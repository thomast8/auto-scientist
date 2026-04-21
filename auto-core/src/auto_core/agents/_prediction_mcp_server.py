#!/usr/bin/env python3
"""Stdio MCP server for querying prediction history.

Launched as a subprocess by ``build_prediction_mcp_server()``. Reads
predictions from a JSON file passed as the first CLI argument and exposes
a ``read_predictions`` tool over stdio.

Usage (by the Claude Code CLI, not directly):
    python3 _prediction_mcp_server.py /path/to/predictions.json

The actual query logic lives in ``prediction_query`` (a stdlib-only module
that the framework-side wrapper in ``prediction_tool.py`` also imports).
This file is intentionally tiny: it only declares the MCPToolSpec and plumbs
the shared ``query`` function into ``run_mcp_server_main``. There is no
duplicated query/format logic to keep in sync.
"""

from __future__ import annotations

if __name__ == "__main__":
    from auto_core.agents._mcp_base import MCPToolSpec, run_mcp_server_main
    from auto_core.agents.prediction_query import query

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

    run_mcp_server_main("predictions", _SPEC, query)
