"""Pure query logic for the notebook MCP tool.

Stdlib only. This module is imported both by the framework-side wrapper in
``notebook_tool.py`` and by the stdio subprocess script
``_notebook_mcp_server.py``. Keeping the query helpers here (rather than
duplicating them in the subprocess script) eliminates a class of drift bugs
that the original PR #19 design left open: every change to query semantics
would otherwise need to be applied in two places, and only the framework
side has unit-test coverage.

The module deliberately avoids importing anything from ``auto_scientist.*``
beyond the stdlib so that the subprocess script can ``import`` it without
pulling in heavy framework dependencies (Pydantic, notebook XML parsing,
state.py).

Inputs are plain ``list[dict[str, Any]]`` of pre-parsed notebook entries
with keys ``version``, ``source``, ``title``, ``content``. Output of
:func:`query` is a ``str`` ready to be wrapped in MCP TextContent.
"""

from __future__ import annotations

import contextlib
import json as _json
from typing import Any

KNOWN_SOURCES: tuple[str, ...] = (
    "ingestor",
    "scientist",
    "revision",
    "stop_gate",
    "stop_revision",
)


# ---------------------------------------------------------------------------
# LLM-arg normalization
# ---------------------------------------------------------------------------


def normalize_args(args: dict[str, Any]) -> dict[str, Any]:
    """Coerce common LLM type mistakes into the canonical schema.

    Fixes observed in production for the prediction tool (same patterns
    apply to the notebook tool):

    * ``versions: '["v01"]'`` (JSON string) instead of ``["v01"]``.
    * ``versions: "v01"`` (bare string) instead of ``["v01"]``.
    * ``last_n: "3"`` (string) instead of ``3``.
    """
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


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def format_entry(entry: dict[str, Any]) -> str:
    """Format a single notebook entry as full-detail text."""
    version = entry.get("version", "")
    source = entry.get("source", "")
    title = entry.get("title") or "(untitled)"
    content = entry.get("content") or "(no body)"
    return f"[{version} {source}] {title}\n{content}"


def build_status(entries: list[dict[str, Any]]) -> str:
    """Build a counts-only status summary (no TOC, since TOC is inline in prompt)."""
    by_source: dict[str, int] = {}
    for entry in entries:
        src = entry.get("source", "")
        by_source[src] = by_source.get(src, 0) + 1

    lines = [f"Total: {len(entries)} notebook entries"]
    for src in KNOWN_SOURCES:
        count = by_source.get(src, 0)
        if count:
            lines.append(f"  {src}: {count}")
    for src, count in sorted(by_source.items()):
        if src not in KNOWN_SOURCES and count:
            lines.append(f"  {src}: {count}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Query handler
# ---------------------------------------------------------------------------


def query(entries: list[dict[str, Any]], args: dict[str, Any]) -> str:
    """Core handler logic for the read_notebook tool.

    Returns a plain ``str``. Both the framework-side ``_handle_read_notebook``
    (which wraps it in MCP content shape) and the subprocess
    ``run_mcp_server_main`` (which wraps it in TextContent) call this.
    """
    args = normalize_args(args)

    if not entries:
        return "No notebook entries yet."

    available_versions = sorted({e.get("version", "") for e in entries if e.get("version")})
    available = ", ".join(available_versions)

    if args.get("summary"):
        return build_status(entries)

    versions = args.get("versions")
    source_value = args.get("source")
    search = args.get("search")
    last_n = args.get("last_n")

    if not versions and not source_value and not search and last_n is None:
        return (
            "Please specify a query: summary, versions, source, search, "
            f"or last_n. Available versions: {available}"
        )

    selected: list[dict[str, Any]] = []

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
            return note + "\n\n".join(format_entry(e) for e in selected)

    elif source_value:
        selected = [e for e in entries if e.get("source") == source_value]

    elif search:
        # Whitespace-tokenized AND search. Splitting on whitespace turns
        # 'repeated cross validation' into three independent substring
        # constraints, which is the closest cheap approximation of how a
        # human would expect a search box to work without reaching for a
        # real tokenizer or embedding model.
        tokens = [t for t in search.lower().split() if t]
        if not tokens:
            return "Empty search query."
        selected = [
            e
            for e in entries
            if all(
                tok in (e.get("title") or "").lower() or tok in (e.get("content") or "").lower()
                for tok in tokens
            )
        ]

    elif last_n is not None:
        if last_n <= 0:
            return "last_n must be a positive integer."
        selected = entries[-last_n:]

    if not selected:
        return "No entries match the query."

    return "\n\n".join(format_entry(e) for e in selected)
