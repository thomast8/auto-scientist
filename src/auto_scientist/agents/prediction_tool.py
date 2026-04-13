"""Framework-side wrappers for the prediction MCP tool.

Owns the pieces that need real framework dependencies:

* The ``MCPToolSpec`` (registered with ``_mcp_base`` so sdk_utils can
  auto-discover the tool description).
* The Pydantic-based ``format_compact_tree()`` formatter used at
  prompt-build time by the Scientist, Critic, and Stop Gate to inject
  the compact prediction tree into agent prompts.
* The ``build_prediction_mcp_server()`` factory.
* A thin ``_handle_read_predictions()`` shim that wraps the shared query
  function in MCP content shape so unit tests can call it directly.

The actual query/format logic lives in
:mod:`auto_scientist.agents.prediction_query`, a stdlib-only module that
the subprocess script ``_prediction_mcp_server.py`` also imports. There
is no duplicated query code to keep in sync.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from auto_scientist.agents._mcp_base import MCPToolSpec, build_mcp_server_config, register_mcp_tool
from auto_scientist.agents.prediction_query import normalize_args as _normalize_args
from auto_scientist.agents.prediction_query import query as _query
from auto_scientist.state import PredictionRecord

logger = logging.getLogger(__name__)

# Re-export so existing tests/scripts can keep importing _normalize_args
# and the inline-prompt formatters from prediction_tool.py.
__all__ = [
    "PREDICTION_SPEC",
    "build_prediction_mcp_server",
    "format_compact_tree",
    "_handle_read_predictions",
    "_normalize_args",
    "_build_prediction_forest",
]


# ---------------------------------------------------------------------------
# Tool spec (registered so sdk_utils can auto-discover the description)
# ---------------------------------------------------------------------------

_READ_PREDICTIONS_DESCRIPTION = (
    "Read full evidence, diagnostics, and conditional outcomes for specific "
    "predictions. The compact tree in your prompt shows status and short "
    "summaries only. Call this tool when you need to verify a claim, inspect "
    "why a prediction was confirmed or refuted, check the diagnostic used, "
    "or trace a reasoning chain. Use summary=true for counts, chain/pred_ids/"
    "outcome/iteration for detailed records."
)

_READ_PREDICTIONS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "boolean",
            "description": (
                "Returns a count header: total predictions and "
                "breakdown by outcome (confirmed, refuted, "
                "inconclusive, pending). Use for a quick tally."
            ),
        },
        "chain": {
            "type": "string",
            "description": (
                "A prediction ID. Returns the full reasoning chain: root "
                "ancestor through this prediction to all descendants. "
                "Best for understanding why a particular investigation "
                "thread exists. E.g. chain='2.1'."
            ),
        },
        "pred_ids": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Specific prediction IDs to retrieve with full detail "
                "(evidence, diagnostics, implications). "
                "E.g. ['2.1', '3.4']."
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
                "'active_chains' returns pending predictions plus their "
                "full ancestor chains."
            ),
        },
        "iteration": {
            "type": "integer",
            "description": "Return predictions prescribed in a specific iteration.",
        },
    },
}

PREDICTION_SPEC = MCPToolSpec(
    server_name="predictions",
    tool_name="read_predictions",
    description=_READ_PREDICTIONS_DESCRIPTION,
    input_schema=_READ_PREDICTIONS_SCHEMA,
    deferred_description=(
        "mcp__predictions__read_predictions("
        "summary?, chain?, pred_ids?, outcome?, iteration?) "
        "- Read evidence, diagnostics, outcomes for predictions. "
        "Tree in prompt shows summaries only."
    ),
)

register_mcp_tool(PREDICTION_SPEC)


# ---------------------------------------------------------------------------
# Compact tree (canonical implementation, inlined in all agent prompts)
# ---------------------------------------------------------------------------
#
# The compact tree formatter is intentionally Pydantic-based because it is
# called at prompt-build time on a live ``list[PredictionRecord]`` from
# ExperimentState. The framework-side caller already has the typed objects;
# round-tripping them through model_dump() just to call a dict-based helper
# would be wasted work and would lose Pydantic validation. The MCP tool's
# query path is the only one that needs the dict-based helpers, and those
# live in prediction_query.py.


def _get_display_text(rec: PredictionRecord, max_chars: int = 60) -> str:
    """Return the summary text for a prediction, falling back to truncated evidence.

    Sanitizes output to ensure single-line (collapses newlines, truncates at
    word boundary to avoid cutting mid-concept).
    """
    text = rec.summary or rec.evidence or rec.prediction
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_space = truncated.rfind(" ")
    if last_space > max_chars // 2:
        truncated = truncated[:last_space]
    return f"{truncated}..."


def _build_prediction_forest(
    prediction_history: list[PredictionRecord],
) -> tuple[dict[str, PredictionRecord], dict[str | None, list[PredictionRecord]]]:
    """Build parent-to-children index from follows_from links.

    Returns (by_id, children) where children[None] contains root predictions.
    """
    by_id: dict[str, PredictionRecord] = {}
    children: dict[str | None, list[PredictionRecord]] = {None: []}
    for rec in prediction_history:
        if rec.pred_id:
            by_id[rec.pred_id] = rec
    for rec in prediction_history:
        parent = rec.follows_from
        if parent and parent in by_id:
            children.setdefault(parent, []).append(rec)
        else:
            children[None].append(rec)
    return by_id, children


def format_compact_tree(
    prediction_history: list[PredictionRecord] | None,
) -> str:
    """Format prediction history as a compact one-line-per-prediction tree.

    Each prediction gets a single line with status, summary, and implication.
    Parent-child chains are shown via indentation.
    """
    if not prediction_history:
        return "(no prediction history yet)"

    _by_id, children = _build_prediction_forest(prediction_history)
    visited: set[str] = set()

    def _render_compact(rec: PredictionRecord, indent: int) -> list[str]:
        if rec.pred_id and rec.pred_id in visited:
            return []
        if rec.pred_id:
            visited.add(rec.pred_id)

        prefix = "  " * indent
        tag = rec.pred_id or f"v{rec.iteration_prescribed:02d}"
        display = _get_display_text(rec)
        lines = []

        if rec.outcome == "pending":
            lines.append(f"{prefix}[{tag}] PENDING: {rec.prediction}")
        elif rec.outcome == "confirmed":
            lines.append(f"{prefix}[{tag}] CONFIRMED: {display} -> {rec.if_confirmed}")
        elif rec.outcome == "refuted":
            lines.append(f"{prefix}[{tag}] REFUTED: {display} -> {rec.if_refuted}")
        elif rec.outcome == "inconclusive":
            lines.append(f"{prefix}[{tag}] INCONCLUSIVE: {display}")

        for child in children.get(rec.pred_id, []):
            lines.extend(_render_compact(child, indent + 1))
        return lines

    header = "== PREDICTION TREE (mcp__predictions__read_predictions for details) =="
    all_lines = [header]
    for root in children[None]:
        all_lines.extend(_render_compact(root, 0))

    return "\n".join(all_lines)


# ---------------------------------------------------------------------------
# Direct handler (for unit tests). Wraps the shared query() in MCP shape.
# ---------------------------------------------------------------------------


async def _handle_read_predictions(
    prediction_history: list[PredictionRecord],
    args: dict[str, Any],
) -> dict[str, Any]:
    """Wrap :func:`prediction_query.query` in the MCP content envelope.

    Used by ``tests/test_prediction_tool.py``. Production traffic goes through
    the subprocess via ``_prediction_mcp_server.py``, which calls the same
    underlying ``query()`` function but wraps results in TextContent via
    ``_mcp_base.run_mcp_server_main``.

    Converts the typed ``list[PredictionRecord]`` to the dict shape the
    pure-stdlib ``query()`` expects.
    """
    records = [r.model_dump() for r in prediction_history]
    text = _query(records, args)
    return {"content": [{"type": "text", "text": text}]}


# ---------------------------------------------------------------------------
# Server config builder
# ---------------------------------------------------------------------------


def build_prediction_mcp_server(
    prediction_history: list[PredictionRecord],
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Create an stdio MCP server config for the read_predictions tool.

    Delegates to the shared ``build_mcp_server_config()`` for serialization
    and stdio config generation.

    Returns a dict suitable for ``SDKOptions.mcp_servers`` values.
    """
    server_script = Path(__file__).parent / "_prediction_mcp_server.py"
    return build_mcp_server_config(
        data_dicts=[r.model_dump() for r in prediction_history],
        server_script=server_script,
        output_dir=output_dir,
        filename="predictions.json",
    )
