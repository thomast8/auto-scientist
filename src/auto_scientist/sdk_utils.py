"""Utilities for working with claude_code_sdk.

Monkey-patches the SDK's message parser at import time so that unknown message
types (e.g., rate_limit_event) are silently skipped instead of crashing the
stream. This keeps the async generator in client.py alive through events the
SDK doesn't yet handle.
"""

import logging
from collections.abc import AsyncIterator
from typing import Any

import claude_code_sdk._internal.client as _client_mod
import claude_code_sdk._internal.message_parser as _parser_mod
from claude_code_sdk import ClaudeCodeOptions, Message, query
from claude_code_sdk._errors import MessageParseError

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
    """
    from claude_code_sdk import TextBlock, ToolResultBlock, ToolUseBlock

    if isinstance(block, TextBlock):
        buffer.append(block.text)
    elif isinstance(block, ToolUseBlock):
        input_str = str(block.input)
        if len(input_str) > 200:
            input_str = input_str[:200] + "..."
        buffer.append(f"[Tool: {block.name}] {input_str}")
    elif isinstance(block, ToolResultBlock):
        content = str(block.content) if block.content else ""
        if len(content) > 200:
            content = content[:200] + "..."
        prefix = "[Error] " if block.is_error else "[Result] "
        buffer.append(f"{prefix}{content}")


async def safe_query(
    prompt: str, options: ClaudeCodeOptions
) -> AsyncIterator[Message]:
    """Wrap claude_code_sdk.query, filtering out None (unknown message types)."""
    async for msg in query(prompt=prompt, options=options):
        if msg is not None:
            yield msg
