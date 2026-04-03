"""Utilities for working with SDK backends (Claude Code SDK and Codex SDK).

Provides output validation, retry utilities, and helper functions that work
with the unified SDKMessage type from sdk_backend.py.

The monkey-patch for unknown message types now lives in sdk_backend.py
(ClaudeBackend-specific).
"""

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel, ValidationError

from auto_scientist.sdk_backend import SDKBackend

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def append_block_to_buffer(block: Any, buffer: list[str]) -> None:
    """Append a content block's text to a message buffer.

    Handles TextBlock (text), ToolUseBlock (tool name + truncated input),
    and ToolResultBlock (truncated output). Silently skips unknown types.
    Also logs each block to debug.log for post-mortem analysis.

    Works with both Claude Code SDK block types (via attribute checks)
    and any future block types that follow the same interface.
    """
    if hasattr(block, "text") and not hasattr(block, "name"):
        # TextBlock-like: has .text but not .name
        buffer.append(block.text)
        preview = block.text[:300].replace("\n", " ")
        logger.debug(f"[text] {preview}")
    elif hasattr(block, "name") and hasattr(block, "input"):
        # ToolUseBlock-like: has .name and .input
        input_str = str(block.input)
        if len(input_str) > 200:
            input_str = input_str[:200] + "..."
        entry = f"[Tool: {block.name}] {input_str}"
        buffer.append(entry)
        logger.debug(entry)
    elif hasattr(block, "thinking"):
        # ThinkingBlock-like: has .thinking attribute (not .text)
        thinking_preview = block.thinking[:300].replace("\n", " ")
        entry = f"[Thinking] {thinking_preview}"
        buffer.append(entry)
        logger.debug(f"[thinking] {thinking_preview}")
    elif hasattr(block, "content") and hasattr(block, "is_error"):
        # ToolResultBlock-like: has .content and .is_error
        content = str(block.content) if block.content else ""
        if len(content) > 200:
            content = content[:200] + "..."
        prefix = "[Error] " if block.is_error else "[Result] "
        entry = f"{prefix}{content}"
        buffer.append(entry)
        logger.debug(entry)


async def safe_query(
    prompt: str,
    options: Any,
    backend: SDKBackend,
) -> AsyncIterator[Any]:
    """Wrap a backend query, filtering out None messages.

    Yields SDKMessage objects from the backend.
    """
    logger.debug(
        f"SDK query start: model={options.model}, "
        f"max_turns={options.max_turns}, "
        f"prompt_len={len(prompt)}"
    )
    async for msg in backend.query(prompt=prompt, options=options):
        if msg is not None:
            yield msg


# ---------------------------------------------------------------------------
# Output validation and retry
# ---------------------------------------------------------------------------


class OutputValidationError(Exception):
    """Raised when an agent's output fails JSON parsing or schema validation."""

    def __init__(
        self,
        raw_output: str,
        validation_error: Exception,
        agent_name: str,
    ) -> None:
        self.raw_output = raw_output
        self.validation_error = validation_error
        self.agent_name = agent_name
        super().__init__(f"{agent_name} output validation failed: {validation_error}")

    def correction_prompt(self) -> str:
        """Format a correction hint for the LLM to fix its output."""
        truncated = self.raw_output[:500]
        if len(self.raw_output) > 500:
            truncated += "..."
        return (
            "<validation_error>\n"
            f"Your previous output could not be parsed. Error: {self.validation_error}\n"
            f"Raw output (first 500 chars): {truncated}\n"
            "Please output ONLY valid JSON matching the schema. No markdown fencing.\n"
            "</validation_error>"
        )


def _strip_markdown_fencing(raw: str) -> str:
    """Remove markdown code fences and leading prose from a string.

    Handles several messy-output patterns:
    - ```json ... ``` fencing
    - Leading prose before the JSON object/array
    - Multiple fenced blocks (extracts the first)
    """
    raw = raw.strip()

    # Strip ``` fences (possibly with language tag)
    if raw.startswith("```"):
        lines = raw.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        raw = "\n".join(lines).strip()

    # If the string doesn't start with { or [, try to find the first JSON
    # object/array start. Models sometimes prepend prose like "Here is my output:"
    if raw and raw[0] not in "{[":
        for i, ch in enumerate(raw):
            if ch in "{[":
                discarded = raw[:i]
                logger.warning(
                    f"Stripped {len(discarded)} chars of leading prose before JSON: "
                    f"{discarded[:100]!r}"
                )
                raw = raw[i:]
                break

    return raw


def _find_json_object(text: str, *, require_dict: bool = False) -> tuple[Any, int, int] | None:
    """Scan *text* for the first valid JSON object or array.

    Tries ``raw_decode`` at every ``{`` (and ``[`` unless *require_dict*)
    position until one succeeds.
    Returns ``(parsed, start_idx, end_idx)`` or ``None``.

    When *require_dict* is True only ``{`` positions are tried.  This avoids
    false-positive matches on JSON arrays like ``[0]`` embedded in Python
    expressions (e.g. ``rows[0]``) that appear in Codex shell-command output
    before the actual JSON object.
    """
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch == "{" or (ch == "[" and not require_dict):
            try:
                parsed, length = decoder.raw_decode(text, i)
                return parsed, i, i + length
            except json.JSONDecodeError:
                continue
    return None


def validate_json_output(
    raw: str,
    model_cls: type[BaseModel],
    agent_name: str,
) -> dict[str, Any]:
    """Parse and validate a raw JSON string against a Pydantic model.

    Returns model_dump() dict on success.
    Raises OutputValidationError on JSON parse or schema validation failure.

    Uses a two-phase approach:
    1. Fast path: strip markdown fencing, then raw_decode from position 0.
    2. Slow path (on failure): scan every ``{``/``[`` with raw_decode until
       one parses. This handles Codex output where shell commands containing
       braces appear before the actual JSON.
    """
    cleaned = _strip_markdown_fencing(raw)

    # Fast path: try raw_decode from the start (works for clean output)
    parsed = None
    try:
        decoded, end_idx = json.JSONDecoder().raw_decode(cleaned)
        # BaseModel always maps to a JSON object (dict).  If the fast path
        # decoded something else (e.g. ``[0]`` from Python indexing in Codex
        # shell output), reject it and fall through to the slow path.
        if isinstance(decoded, dict):
            parsed = decoded
            trailing = cleaned[end_idx:].strip()
            if trailing:
                logger.warning(
                    f"{agent_name}: raw_decode ignored {len(trailing)} chars of trailing content: "
                    f"{trailing[:200]!r}"
                )
        else:
            logger.warning(
                f"{agent_name}: fast-path decoded {type(decoded).__name__}, "
                f"expected dict - trying slow path"
            )
    except json.JSONDecodeError:
        pass

    if parsed is None:
        # Slow path: scan for the first valid JSON object.
        # Use require_dict=True since Pydantic BaseModel always maps to a
        # dict, skipping false-positive array matches from shell/Python code.
        result = _find_json_object(cleaned, require_dict=True)
        if result is not None:
            parsed, start_idx, _end_idx = result
            logger.warning(
                f"{agent_name}: found JSON at position {start_idx} after "
                f"skipping non-JSON content: {cleaned[:start_idx][:200]!r}"
            )

    if parsed is None:
        raise OutputValidationError(
            raw_output=raw,
            validation_error=json.JSONDecodeError(
                "No valid JSON object found in output", raw[:500], 0
            ),
            agent_name=agent_name,
        )

    try:
        validated = model_cls.model_validate(parsed)
    except ValidationError as e:
        raise OutputValidationError(
            raw_output=raw,
            validation_error=e,
            agent_name=agent_name,
        ) from e

    return validated.model_dump()


# Descriptions for deferred tools (not loaded by default in Claude Code).
# Including these in the system prompt lets the model call them directly
# without wasting a turn on ToolSearch.
_DEFERRED_TOOL_DESCRIPTIONS: dict[str, str] = {
    "WebSearch": (
        "WebSearch(query: str) - Search the web. "
        "Required param: query (string, min 2 chars). "
        "Optional: allowed_domains (list[str]), blocked_domains (list[str])."
    ),
    "AskUserQuestion": ("AskUserQuestion(question: str) - Ask the user a clarifying question."),
    "mcp__predictions__read_predictions": (
        "mcp__predictions__read_predictions(pred_ids?: list[str], filter?: str, iteration?: int) "
        "- Query prediction history for full detail. "
        "pred_ids: specific IDs like ['2.1','3.4']. "
        "filter: 'pending'|'refuted'|'confirmed'|'inconclusive'|'active_chains'. "
        "iteration: predictions from a specific iteration. "
        "No args returns all predictions."
    ),
}


def with_turn_budget(system_prompt: str, max_turns: int, tools: list[str] | None = None) -> str:
    """Append turn budget and available tool descriptions to a system prompt.

    Tells the model how many turns it has so it can plan tool usage
    accordingly instead of spiraling into unbounded research loops.
    Lists available tools so the model can call them directly without
    wasting a turn on ToolSearch.
    """
    parts = [system_prompt]

    if tools:
        tool_lines = []
        for tool in tools:
            if tool in _DEFERRED_TOOL_DESCRIPTIONS:
                tool_lines.append(f"  - {_DEFERRED_TOOL_DESCRIPTIONS[tool]}")
            else:
                tool_lines.append(f"  - {tool}")
        tool_block = "\n".join(tool_lines)
        parts.append(
            f"\n<available_tools>\n"
            f"Your available tools (call directly, do NOT use ToolSearch):\n"
            f"{tool_block}\n"
            f"</available_tools>"
        )

    parts.append(
        f"\n<turn_budget>You have a budget of {max_turns} turns for this task. "
        f"Each tool use consumes one turn. Plan your tool usage carefully "
        f"and produce your final output within this budget.</turn_budget>"
    )

    return "".join(parts)


async def collect_text_from_query(
    prompt: str,
    options: Any,
    backend: SDKBackend,
    message_buffer: list[str] | None = None,
    agent_name: str = "Agent",
) -> tuple[str, dict[str, Any]]:
    """Run an SDK query and collect the text response.

    Returns:
        (text, usage) tuple. Usage dict contains token counts and cost info.
        Also sets ``last_usage`` on the function object so the orchestrator
        can read it after sequential agent phases (coder, ingestor).
    """
    _message_buffer = message_buffer
    result_text = ""
    assistant_texts: list[str] = []
    usage: dict[str, Any] = {}
    has_streaming = False

    async for message in backend.query(prompt=prompt, options=options):
        if message.type == "result":
            if message.result:
                result_text = message.result
            usage = message.usage
        elif message.type == "stream":
            # Partial content deltas (Claude backend with include_partial_messages).
            # Populate only the message_buffer so the summarizer sees real-time
            # progress.  Final text is still collected from the complete
            # AssistantMessage below.
            has_streaming = True
            if _message_buffer is not None:
                for block in message.content_blocks:
                    append_block_to_buffer(block, _message_buffer)
        elif message.type == "assistant":
            for block in message.content_blocks:
                is_text = hasattr(block, "text") and not hasattr(block, "name")
                is_thinking = hasattr(block, "thinking")

                # Always collect complete text for final output extraction.
                if is_text:
                    assistant_texts.append(block.text)

                if _message_buffer is not None:
                    # When streaming delivered text/thinking via deltas,
                    # skip them here to avoid double-counting in the buffer.
                    # Tool use/result blocks are NOT streamed, so always add them.
                    if has_streaming and (is_text or is_thinking):
                        continue
                    append_block_to_buffer(block, _message_buffer)

    raw = result_text if result_text else "\n".join(assistant_texts)

    if not raw:
        raise RuntimeError(f"{agent_name} agent returned no output")

    # Also store on the function for callers that don't use the return value
    # (orchestrator's _apply_sdk_usage reads this after sequential agent phases)
    collect_text_from_query.last_usage = usage  # type: ignore[attr-defined]

    return raw, usage


# Initialize the usage attribute
collect_text_from_query.last_usage = {}  # type: ignore[attr-defined]


# ── Report structure validation ─────────────────────────────────────────────

# Expected heading keywords for fuzzy matching (case-insensitive substring)
_EXPECTED_HEADINGS = [
    "executive summary",
    "problem statement",
    "methodology",
    "journey",
    "best approach",
    "results",
    "insights",
    "limitations",
    "future work",
    "version comparison",
]


def validate_report_structure(text: str) -> list[str]:
    """Validate that a report has the expected section structure.

    Returns a list of issue strings (empty = valid).
    Checks for: required headings, non-empty sections, markdown table in
    Version Comparison section.
    """
    issues: list[str] = []
    lines = text.split("\n")

    # Find all ## headings and their line indices
    heading_indices: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        if line.startswith("## "):
            heading_indices.append((i, line[3:].strip()))

    # Check for expected headings (case-insensitive substring match)
    found_headings = [h.lower() for _, h in heading_indices]
    for expected in _EXPECTED_HEADINGS:
        if not any(expected in h for h in found_headings):
            issues.append(f"Missing section: {expected}")

    # Check each section has non-empty content
    for idx, (line_num, heading) in enumerate(heading_indices):
        if idx + 1 < len(heading_indices):
            next_line_num = heading_indices[idx + 1][0]
        else:
            next_line_num = len(lines)
        section_content = "\n".join(lines[line_num + 1 : next_line_num]).strip()
        if not section_content:
            issues.append(f"Empty section: {heading}")

    # Check Version Comparison Table has a markdown table
    for line_num, heading in heading_indices:
        if "version comparison" in heading.lower():
            # Find content until next heading or end
            end = len(lines)
            for future_num, _ in heading_indices:
                if future_num > line_num:
                    end = future_num
                    break
            section = "\n".join(lines[line_num + 1 : end])
            if "|" not in section:
                issues.append("Version Comparison Table section missing markdown table")
            break

    return issues
