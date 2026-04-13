#!/usr/bin/env python3
"""Stdio MCP server for querying lab notebook entries.

Launched as a subprocess by ``build_notebook_mcp_server()``. Reads notebook
entries from a JSON file passed as the first CLI argument and exposes a
``read_notebook`` tool over stdio.

Usage (by the Claude Code CLI, not directly):
    python3 _notebook_mcp_server.py /path/to/notebook_entries.json

The actual query logic lives in ``notebook_query`` (a stdlib-only module
that the framework-side wrapper in ``notebook_tool.py`` also imports). This
file is intentionally tiny: it only declares the MCPToolSpec and plumbs the
shared ``query`` function into ``run_mcp_server_main``. There is no
duplicated query/format logic to keep in sync.
"""

from __future__ import annotations

if __name__ == "__main__":
    from auto_scientist.agents._mcp_base import MCPToolSpec, run_mcp_server_main
    from auto_scientist.agents.notebook_query import query

    _SPEC = MCPToolSpec(
        server_name="notebook",
        tool_name="read_notebook",
        description=(
            "Read full entries from the lab notebook. The Table of Contents "
            "in your prompt shows only the version, source, and title of "
            "each entry. Call this tool to read the narrative body when you "
            "need the full context of a prior iteration's reasoning, "
            "results, or debate outcomes. Use summary=true for counts, "
            "versions/source/search/last_n for detailed entries."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "summary": {
                    "type": "boolean",
                    "description": (
                        "Returns a count header: total entries and "
                        "breakdown by source (scientist, stop_gate, "
                        "revision, ingestor). Use for a quick tally."
                    ),
                },
                "versions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Specific version strings to retrieve "
                        "(e.g. ['v01', 'v03']). A single version can "
                        "match multiple entries."
                    ),
                },
                "source": {
                    "type": "string",
                    "enum": [
                        "scientist",
                        "revision",
                        "stop_gate",
                        "stop_revision",
                        "ingestor",
                    ],
                    "description": ("Return all entries from a specific source."),
                },
                "search": {
                    "type": "string",
                    "description": (
                        "Case-insensitive whitespace-tokenized search "
                        "across entry titles and content. The query is "
                        "split on whitespace and every token must appear "
                        "(as a substring) in the title or body. Multi-word "
                        "queries are AND, not phrase."
                    ),
                },
                "last_n": {
                    "type": "integer",
                    "description": ("Return the most recent N entries."),
                },
            },
        },
        deferred_description=(
            "mcp__notebook__read_notebook("
            "summary?, versions?, source?, search?, last_n?) "
            "- Read full notebook entries. TOC in prompt shows titles only."
        ),
    )

    run_mcp_server_main("notebook", _SPEC, query)
