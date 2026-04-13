#!/usr/bin/env python3
"""Stdio MCP server for querying lab notebook entries.

Launched as a subprocess by ``build_notebook_mcp_server()``. Reads notebook
entries from a JSON file passed as the first CLI argument and exposes a
``read_notebook`` tool over stdio.

Usage (by the Claude Code CLI, not directly):
    python3 _notebook_mcp_server.py /path/to/notebook_entries.json

NOTE: Query helpers (``_format_entry``, ``_query``, etc.) are intentionally
duplicated from ``notebook_tool.py``. This module runs as an isolated
subprocess and must not import framework code (Pydantic, persistence, etc.).
Keep both files in sync when changing query semantics.
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Notebook-specific formatting
# ---------------------------------------------------------------------------


def _format_entry(entry: dict[str, Any]) -> str:
    version = entry.get("version", "")
    source = entry.get("source", "")
    title = entry.get("title") or "(untitled)"
    content = entry.get("content") or "(no body)"
    return f"[{version} {source}] {title}\n{content}"


def _build_status(entries: list[dict[str, Any]]) -> str:
    """Build a counts-only status summary (no TOC, since TOC is inline in prompt)."""
    by_source: dict[str, int] = {}
    for entry in entries:
        src = entry.get("source", "")
        by_source[src] = by_source.get(src, 0) + 1

    lines = [f"Total: {len(entries)} notebook entries"]
    for src in ["ingestor", "scientist", "revision", "stop_gate"]:
        count = by_source.get(src, 0)
        if count:
            lines.append(f"  {src}: {count}")
    for src, count in sorted(by_source.items()):
        if src not in {"ingestor", "scientist", "revision", "stop_gate"} and count:
            lines.append(f"  {src}: {count}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Query handler (the only notebook-specific logic)
# ---------------------------------------------------------------------------


def _normalize_args(args: dict[str, Any]) -> dict[str, Any]:
    """Coerce common LLM type mistakes into the canonical schema.

    Duplicated from notebook_tool.py to avoid importing framework code
    in this standalone subprocess script.
    """
    import contextlib as _contextlib
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
        with _contextlib.suppress(ValueError):
            out["last_n"] = int(last_n)

    return out


def _query(entries: list[dict[str, Any]], args: dict[str, Any]) -> str:
    args = _normalize_args(args)

    if not entries:
        return "No notebook entries yet."

    available_versions = sorted({e.get("version", "") for e in entries if e.get("version")})
    available = ", ".join(available_versions)

    if args.get("summary"):
        return _build_status(entries)

    versions = args.get("versions")
    source_value = args.get("source")
    search = args.get("search")
    last_n = args.get("last_n")

    if not versions and not source_value and not search and last_n is None:
        return (
            "Please specify a query: summary, versions, source, search, "
            f"or last_n. Available versions: {available}"
        )

    selected: list[dict] = []

    if versions:
        missing = [v for v in versions if not any(e.get("version") == v for e in entries)]
        for v in versions:
            for entry in entries:
                if entry.get("version") == v:
                    selected.append(entry)
        if not selected:
            return f"Not found: {', '.join(versions)}. Available versions: {available}"
        if missing:
            note = f"Note: versions not found: {', '.join(missing)}\n\n"
            return note + "\n\n".join(_format_entry(e) for e in selected)

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
            return "last_n must be a positive integer."
        selected = entries[-last_n:]

    if not selected:
        return "No entries match the query."

    return "\n\n".join(_format_entry(e) for e in selected)


# ---------------------------------------------------------------------------
# Entry point - all boilerplate handled by _mcp_base
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from auto_scientist.agents._mcp_base import MCPToolSpec, run_mcp_server_main

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
                    "enum": ["scientist", "stop_gate", "revision", "ingestor"],
                    "description": ("Return all entries from a specific source."),
                },
                "search": {
                    "type": "string",
                    "description": (
                        "Case-insensitive substring search across entry titles and content."
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

    run_mcp_server_main("notebook", _SPEC, _query)
