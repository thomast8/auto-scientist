"""Anthropic API wrapper with optional streaming and reasoning support."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from anthropic import AsyncAnthropic

if TYPE_CHECKING:
    from auto_scientist.model_config import ReasoningConfig

logger = logging.getLogger(__name__)

ANTHROPIC_BUDGET_DEFAULTS: dict[str, int] = {
    "minimal": 1024,
    "low": 2048,
    "medium": 4096,
    "high": 16384,
    "max": 32768,
}


async def query_anthropic(
    model: str,
    prompt: str,
    *,
    web_search: bool = False,
    reasoning: ReasoningConfig | None = None,
    on_token: Callable[[str], None] | None = None,
) -> str:
    """Send a prompt to an Anthropic model and return the response text.

    Args:
        model: Model name (e.g., 'claude-sonnet-4-6').
        prompt: The full prompt to send.
        web_search: Enable the web_search server tool.
        reasoning: Optional reasoning config for extended thinking.
        on_token: Optional callback invoked with each text delta for live streaming.

    Returns:
        The model's text response.
    """
    logger.debug(f"Anthropic call: model={model}, prompt_len={len(prompt)}, ws={web_search}")
    client = AsyncAnthropic()

    kwargs: dict = {
        "model": model,
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": prompt}],
    }

    if web_search:
        kwargs["tools"] = [{"type": "web_search_20250305", "name": "web_search"}]

    if reasoning is not None and reasoning.level not in ("default", "off"):
        budget = reasoning.budget or ANTHROPIC_BUDGET_DEFAULTS[reasoning.level]
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}
        kwargs["max_tokens"] = max(kwargs["max_tokens"], budget + 4096)

    if on_token is not None:
        parts: list[str] = []
        async with client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                on_token(text)
                parts.append(text)
        result = "".join(parts)
        logger.debug(f"Anthropic response: {len(result)} chars")
        return result

    response = await client.messages.create(**kwargs)

    # Extract text blocks (skip tool_use/web_search result blocks)
    text_parts = []
    for block in response.content:
        if hasattr(block, "text"):
            text_parts.append(block.text)
    result = "\n".join(text_parts) if text_parts else ""
    logger.debug(f"Anthropic response: {len(result)} chars")
    return result
