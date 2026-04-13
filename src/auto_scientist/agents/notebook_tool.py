"""MCP tool for querying lab notebook entries on demand.

Provides ``build_notebook_mcp_server()`` which creates an stdio MCP server
config that the Claude Code CLI (or Codex) can connect to. The server reads
notebook entries from a temporary JSON file and exposes a ``read_notebook``
tool.

Also provides ``_handle_read_notebook()`` for direct use in tests.

NOTE: Query helpers (``_format_entry_detail``, ``_query`` etc.) are
intentionally duplicated in ``_notebook_mcp_server.py``, which runs as an
isolated subprocess and avoids importing the framework's heavy modules
(notebook.py, state.py, Pydantic schemas). Keep both files in sync when
changing query semantics.
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Any

from auto_scientist.agents._mcp_base import (
    MCPToolSpec,
    build_mcp_server_config,
    register_mcp_tool,
)
from auto_scientist.notebook import parse_notebook_entries

logger = logging.getLogger(__name__)

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
                "Case-insensitive substring search across entry titles and "
                "content. Returns every entry whose title or body contains "
                "the substring."
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


def _normalize_args(args: dict[str, Any]) -> dict[str, Any]:
    """Coerce common LLM type mistakes into the canonical schema.

    Fixes observed in production for the prediction tool (same patterns apply
    here):
      - ``versions: '["v01"]'`` (JSON string) instead of ``["v01"]``.
      - ``versions: "v01"`` (bare string) instead of ``["v01"]``.
      - ``last_n: "3"`` (string) instead of ``3``.
    """
    import json as _json

    out = dict(args)

    versions = out.get("versions")
    if isinstance(versions, str):
        try:
            parsed = _json.loads(versions)
            if isinstance(parsed, list):
                out["versions"] = [str(x) for x in parsed]
            else:
                out["versions"] = [versions]
        except (ValueError, _json.JSONDecodeError):
            out["versions"] = [versions]
    elif isinstance(versions, list):
        out["versions"] = [str(x) for x in versions]

    last_n = out.get("last_n")
    if isinstance(last_n, str):
        with contextlib.suppress(ValueError):
            out["last_n"] = int(last_n)

    return out


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
# Full-detail formatting (used by detail queries and _handle_read_notebook)
# ---------------------------------------------------------------------------


def _format_entry_detail(entry: dict[str, str]) -> str:
    """Format a single notebook entry as full-detail text."""
    version = entry.get("version", "")
    source = entry.get("source", "")
    title = entry.get("title") or "(untitled)"
    content = entry.get("content") or "(no body)"
    return f"[{version} {source}] {title}\n{content}"


_KNOWN_SOURCES = ("ingestor", "scientist", "revision", "stop_gate", "stop_revision")


def _build_status_response(entries: list[dict[str, str]]) -> dict[str, Any]:
    """Build a counts-only status summary (no TOC, since TOC is inline in prompt)."""
    by_source: dict[str, int] = {}
    for entry in entries:
        src = entry.get("source", "")
        by_source[src] = by_source.get(src, 0) + 1

    lines = [f"Total: {len(entries)} notebook entries"]
    for src in _KNOWN_SOURCES:
        count = by_source.get(src, 0)
        if count:
            lines.append(f"  {src}: {count}")
    # Include any other source types not in the canonical list
    for src, count in sorted(by_source.items()):
        if src not in _KNOWN_SOURCES and count:
            lines.append(f"  {src}: {count}")

    return {"content": [{"type": "text", "text": "\n".join(lines)}]}


# ---------------------------------------------------------------------------
# Direct handler (for unit tests - same logic as the MCP server subprocess)
# ---------------------------------------------------------------------------


async def _handle_read_notebook(
    entries: list[dict[str, str]],
    args: dict[str, Any],
) -> dict[str, Any]:
    """Core handler logic for the read_notebook tool."""
    args = _normalize_args(args)

    if not entries:
        return {"content": [{"type": "text", "text": "No notebook entries yet."}]}

    available_versions = sorted({e.get("version", "") for e in entries if e.get("version")})
    available = ", ".join(available_versions)

    if args.get("summary"):
        return _build_status_response(entries)

    versions = args.get("versions")
    source_value = args.get("source")
    search = args.get("search")
    last_n = args.get("last_n")

    if not versions and not source_value and not search and last_n is None:
        return {
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Please specify a query: summary, versions, source, "
                        f"search, or last_n. Available versions: {available}"
                    ),
                }
            ]
        }

    selected: list[dict[str, str]] = []

    if versions:
        missing = [v for v in versions if not any(e.get("version") == v for e in entries)]
        for v in versions:
            for entry in entries:
                if entry.get("version") == v:
                    selected.append(entry)
        if not selected:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Not found: {', '.join(versions)}. Available versions: {available}"
                        ),
                    }
                ]
            }
        if missing:
            missing_note = f"Note: versions not found: {', '.join(missing)}\n\n"
            formatted = missing_note + "\n\n".join(_format_entry_detail(e) for e in selected)
            return {"content": [{"type": "text", "text": formatted}]}

    elif source_value:
        selected = [e for e in entries if e.get("source") == source_value]

    elif search:
        needle = search.lower()
        selected = [
            e
            for e in entries
            if needle in (e.get("title") or "").lower()
            or needle in (e.get("content") or "").lower()
        ]

    elif last_n is not None:
        if last_n <= 0:
            return {"content": [{"type": "text", "text": "last_n must be a positive integer."}]}
        selected = entries[-last_n:]

    if not selected:
        return {"content": [{"type": "text", "text": "No entries match the query."}]}

    formatted = "\n\n".join(_format_entry_detail(e) for e in selected)
    return {"content": [{"type": "text", "text": formatted}]}


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
