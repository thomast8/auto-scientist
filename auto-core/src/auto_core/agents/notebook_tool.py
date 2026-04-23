"""Framework-side wrappers for the notebook MCP tool.

This module owns the pieces that need real framework dependencies:

* The ``MCPToolSpec`` (registered with ``_mcp_base`` so sdk_utils can
  auto-discover the tool description).
* The ``format_notebook_toc()`` formatter used at prompt-build time by
  every agent that injects the compact Table of Contents.
* The ``build_notebook_mcp_server()`` factory that parses
  ``lab_notebook.xml`` and delegates to ``build_mcp_server_config`` for
  the stdio handshake.
* A thin ``_handle_read_notebook()`` shim that wraps the shared query
  function in MCP content shape so unit tests can call it directly.

The actual query/format logic lives in :mod:`auto_core.agents.notebook_query`,
a stdlib-only module that the subprocess script
``_notebook_mcp_server.py`` also imports. There is no duplicated query
code to keep in sync.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from auto_core.agents._mcp_base import (
    MCPToolSpec,
    build_mcp_server_config,
    register_mcp_tool,
)
from auto_core.agents.notebook_query import normalize_args as _normalize_args
from auto_core.agents.notebook_query import query as _query
from auto_core.notebook import parse_notebook_entries

logger = logging.getLogger(__name__)

# Re-export so existing tests/scripts can keep importing _normalize_args
# from notebook_tool.py.
__all__ = [
    "NOTEBOOK_SPEC",
    "build_notebook_mcp_server",
    "format_notebook_toc",
    "_handle_read_notebook",
    "_normalize_args",
]


# ---------------------------------------------------------------------------
# Tool spec (registered so sdk_utils can auto-discover the description)
# ---------------------------------------------------------------------------

_READ_NOTEBOOK_DESCRIPTION = (
    "Read full entries from the lab notebook. The Table of Contents in your "
    "prompt shows only the version, source, and title of each entry. Call "
    "this tool to read the narrative body when you need the full context of "
    "a prior iteration's reasoning, results, or debate outcomes. Use "
    "summary=true for counts, versions/source/search/last_n for detailed "
    "entries."
)

_READ_NOTEBOOK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "boolean",
            "description": (
                "Returns a count header: total entries and breakdown by "
                "source (scientist, stop_gate, revision, ingestor). Use for "
                "a quick tally."
            ),
        },
        "versions": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Specific version strings to retrieve (e.g. ['v01', 'v03']). "
                "A single version can match multiple entries (scientist + "
                "revision + stop_gate all share the same version)."
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
            "description": (
                "Return all entries from a specific source. 'scientist' for "
                "plan entries, 'revision' for post-debate revised plans, "
                "'stop_gate' for stop-gate outcomes, 'stop_revision' for "
                "post-stop-debate revised plans, 'ingestor' for initial "
                "domain setup."
            ),
        },
        "search": {
            "type": "string",
            "description": (
                "Case-insensitive whitespace-tokenized search across entry "
                "titles and content. The query is split on whitespace and "
                "every resulting token must appear (as a substring) in the "
                "title or body for the entry to match. Multi-word queries "
                "are AND, not phrase: 'repeated cross validation' matches "
                "an entry containing all three words in any order."
            ),
        },
        "last_n": {
            "type": "integer",
            "description": (
                "Return the most recent N entries (in the order they were "
                "written). Useful for quickly re-reading recent context."
            ),
        },
    },
}

NOTEBOOK_SPEC = MCPToolSpec(
    server_name="notebook",
    tool_name="read_notebook",
    description=_READ_NOTEBOOK_DESCRIPTION,
    input_schema=_READ_NOTEBOOK_SCHEMA,
    deferred_description=(
        "mcp__notebook__read_notebook("
        "summary?, versions?, source?, search?, last_n?) "
        "- Read full notebook entries. TOC in prompt shows titles only."
    ),
)

register_mcp_tool(NOTEBOOK_SPEC)


# ---------------------------------------------------------------------------
# Compact TOC (canonical implementation, inlined in all agent prompts)
# ---------------------------------------------------------------------------


def format_notebook_toc(entries: list[dict[str, str]] | None) -> str:
    """Format notebook entries as a compact one-line-per-entry Table of Contents.

    Each entry renders as ``[<version> <source>] <title>``. Parent-child
    structure is not meaningful for notebooks (unlike prediction chains), so
    this is a flat list preserving file order.
    """
    if not entries:
        return "(no notebook entries yet)"

    header = "== NOTEBOOK TOC (mcp__notebook__read_notebook for full entries) =="
    lines = [header]
    for entry in entries:
        version = entry.get("version", "")
        source = entry.get("source", "")
        title = entry.get("title") or "(untitled)"
        lines.append(f"[{version} {source}] {title}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Direct handler (for unit tests). Wraps the shared query() in MCP shape.
# ---------------------------------------------------------------------------


async def _handle_read_notebook(
    entries: list[dict[str, str]],
    args: dict[str, Any],
) -> dict[str, Any]:
    """Wrap :func:`notebook_query.query` in the MCP content envelope.

    Used by ``tests/test_notebook_tool.py``. Production traffic goes through
    the subprocess via ``_notebook_mcp_server.py``, which calls the same
    underlying ``query()`` function but wraps results in TextContent via
    ``_mcp_base.run_mcp_server_main``.
    """
    text = _query(entries, args)
    return {"content": [{"type": "text", "text": text}]}


# ---------------------------------------------------------------------------
# Server config builder
# ---------------------------------------------------------------------------


def build_notebook_mcp_server(
    notebook_path: Path,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Create an stdio MCP server config for the read_notebook tool.

    Parses the notebook file into entry dicts and delegates to the shared
    ``build_mcp_server_config()`` for serialization and stdio config
    generation.

    Returns a dict suitable for ``SDKOptions.mcp_servers`` values.
    """
    entries = parse_notebook_entries(notebook_path)
    server_script = Path(__file__).parent / "_notebook_mcp_server.py"
    return build_mcp_server_config(
        data_dicts=entries,
        server_script=server_script,
        output_dir=output_dir,
        filename="notebook_entries.json",
    )
