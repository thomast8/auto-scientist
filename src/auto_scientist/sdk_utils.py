"""Utilities for working with claude_code_sdk.

Monkey-patches the SDK's message parser at import time so that unknown message
types (e.g., rate_limit_event) are silently skipped instead of crashing the
stream. This keeps the async generator in client.py alive through events the
SDK doesn't yet handle.

Also provides output validation and retry utilities for agent output parsing.
"""

import json
import logging
import os
from collections.abc import AsyncIterator
from typing import Any

import claude_code_sdk._internal.client as _client_mod
import claude_code_sdk._internal.message_parser as _parser_mod
from claude_code_sdk import (
    AssistantMessage,
    ClaudeCodeOptions,
    Message,
    ResultMessage,
    TextBlock,
    query,
)
from claude_code_sdk._errors import MessageParseError
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Monkey-patch: make parse_message return None for unknown types
# ---------------------------------------------------------------------------
_original_parse_message = _parser_mod.parse_message


def _tolerant_parse_message(data: dict[str, Any]) -> Message | None:
    """parse_message wrapper that returns None for unknown message types."""
    try:
        return _original_parse_message(data)
    except MessageParseError as exc:
        if "Unknown message type" in str(exc):
            msg_type = data.get("type", "<missing>")
            logger.debug(f"Skipping unknown SDK message type: {msg_type}")
            return None
        raise


# Patch both the module-level function and the reference imported by client.py
_parser_mod.parse_message = _tolerant_parse_message  # type: ignore[assignment]
_client_mod.parse_message = _tolerant_parse_message  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------------
def append_block_to_buffer(block: Any, buffer: list[str]) -> None:
    """Append a content block's text to a message buffer.

    Handles TextBlock (text), ToolUseBlock (tool name + truncated input),
    and ToolResultBlock (truncated output). Silently skips unknown types.
    Also logs each block to debug.log for post-mortem analysis.
    """
    from claude_code_sdk import TextBlock, ToolResultBlock, ToolUseBlock

    if isinstance(block, TextBlock):
        buffer.append(block.text)
        preview = block.text[:300].replace("\n", " ")
        logger.debug(f"[text] {preview}")
    elif isinstance(block, ToolUseBlock):
        input_str = str(block.input)
        if len(input_str) > 200:
            input_str = input_str[:200] + "..."
        entry = f"[Tool: {block.name}] {input_str}"
        buffer.append(entry)
        logger.debug(entry)
    elif isinstance(block, ToolResultBlock):
        content = str(block.content) if block.content else ""
        if len(content) > 200:
            content = content[:200] + "..."
        prefix = "[Error] " if block.is_error else "[Result] "
        entry = f"{prefix}{content}"
        buffer.append(entry)
        logger.debug(entry)


async def safe_query(prompt: str, options: ClaudeCodeOptions) -> AsyncIterator[Message]:
    """Wrap claude_code_sdk.query, filtering out None (unknown message types).

    Strips ANTHROPIC_API_KEY from the subprocess environment so SDK agents use
    the Claude Code subscription (Max plan) instead of direct API billing.  The
    key is still available in the parent process for direct Anthropic client
    calls (e.g. anthropic_client.py).
    """
    if "ANTHROPIC_API_KEY" not in options.env and os.environ.get("ANTHROPIC_API_KEY"):
        logger.info(
            "Stripping ANTHROPIC_API_KEY from SDK subprocess env "
            "(using Claude Code subscription instead of direct API billing)"
        )
        options = ClaudeCodeOptions(
            **{
                field: getattr(options, field)
                for field in options.__dataclass_fields__
                if field != "env"
            },
            env={**options.env, "ANTHROPIC_API_KEY": ""},
        )
    logger.debug(
        f"SDK query start: model={options.model}, "
        f"max_turns={options.max_turns}, "
        f"prompt_len={len(prompt)}"
    )
    async for msg in query(prompt=prompt, options=options):
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


def validate_json_output(
    raw: str,
    model_cls: type[BaseModel],
    agent_name: str,
) -> dict[str, Any]:
    """Parse and validate a raw JSON string against a Pydantic model.

    Returns model_dump() dict on success.
    Raises OutputValidationError on JSON parse or schema validation failure.

    Uses raw_decode() to extract the first JSON object, tolerating
    trailing text that models sometimes append after the JSON.
    """
    cleaned = _strip_markdown_fencing(raw)
    try:
        parsed, end_idx = json.JSONDecoder().raw_decode(cleaned)
        trailing = cleaned[end_idx:].strip()
        if trailing:
            logger.warning(
                f"{agent_name}: raw_decode ignored {len(trailing)} chars of trailing content: "
                f"{trailing[:200]!r}"
            )
    except json.JSONDecodeError as e:
        raise OutputValidationError(
            raw_output=raw,
            validation_error=e,
            agent_name=agent_name,
        ) from e

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
    options: ClaudeCodeOptions,
    message_buffer: list[str] | None = None,
    agent_name: str = "Agent",
) -> tuple[str, dict[str, Any]]:
    """Run an SDK query and collect the text response.

    Prefers ResultMessage.result; falls back to concatenated AssistantMessage
    TextBlocks. Raises RuntimeError if no text is produced.

    Returns:
        (text, usage) tuple. Usage dict contains token counts and cost info.
        Also sets ``last_usage`` on the function object so the orchestrator
        can read it after sequential agent phases (coder, ingestor).
    """
    result_text = ""
    assistant_texts: list[str] = []
    usage: dict[str, Any] = {}

    async for message in query(prompt=prompt, options=options):
        if isinstance(message, ResultMessage):
            if message.result:
                result_text = message.result
            usage = getattr(message, "usage", None) or {}
            usage["num_turns"] = getattr(message, "num_turns", 0)
            usage["total_cost_usd"] = getattr(message, "total_cost_usd", None)
        elif isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    assistant_texts.append(block.text)
                if message_buffer is not None:
                    append_block_to_buffer(block, message_buffer)

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
