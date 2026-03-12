"""Utilities for working with claude_code_sdk.

Monkey-patches the SDK's message parser at import time so that unknown message
types (e.g., rate_limit_event) are silently skipped instead of crashing the
stream. This keeps the async generator in client.py alive through events the
SDK doesn't yet handle.

Also provides output validation and retry utilities for agent output parsing.
"""

import json
import logging
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


async def safe_query(
    prompt: str, options: ClaudeCodeOptions
) -> AsyncIterator[Message]:
    """Wrap claude_code_sdk.query, filtering out None (unknown message types)."""
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
        super().__init__(
            f"{agent_name} output validation failed: {validation_error}"
        )

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
    """Remove markdown code fences from a string."""
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        raw = "\n".join(lines)
    return raw


def validate_json_output(
    raw: str,
    model_cls: type[BaseModel],
    agent_name: str,
) -> dict[str, Any]:
    """Parse and validate a raw JSON string against a Pydantic model.

    Returns model_dump() dict on success.
    Raises OutputValidationError on JSON parse or schema validation failure.
    """
    cleaned = _strip_markdown_fencing(raw)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise OutputValidationError(
            raw_output=raw, validation_error=e, agent_name=agent_name,
        ) from e

    try:
        validated = model_cls.model_validate(parsed)
    except ValidationError as e:
        raise OutputValidationError(
            raw_output=raw, validation_error=e, agent_name=agent_name,
        ) from e

    return validated.model_dump()


async def collect_text_from_query(
    prompt: str,
    options: ClaudeCodeOptions,
    message_buffer: list[str] | None = None,
    agent_name: str = "Agent",
) -> str:
    """Run an SDK query and collect the text response.

    Prefers ResultMessage.result; falls back to concatenated AssistantMessage
    TextBlocks. Raises RuntimeError if no text is produced.

    This extracts the common pattern shared by Analyst, Scientist, and
    Scientist Revision agents.
    """
    result_text = ""
    assistant_texts: list[str] = []

    async for message in query(prompt=prompt, options=options):
        if isinstance(message, ResultMessage):
            if message.result:
                result_text = message.result
        elif isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    assistant_texts.append(block.text)
                if message_buffer is not None:
                    append_block_to_buffer(block, message_buffer)

    raw = result_text if result_text else "\n".join(assistant_texts)

    if not raw:
        raise RuntimeError(f"{agent_name} agent returned no output")

    return raw
