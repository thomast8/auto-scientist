"""Anthropic API wrapper with optional streaming, reasoning, and structured output."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from anthropic import AsyncAnthropic
from pydantic import BaseModel

from auto_scientist.agent_result import AgentResult

if TYPE_CHECKING:
    from auto_scientist.images import ImageData
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
    system_prompt: str | None = None,
    response_schema: type[BaseModel] | None = None,
    images: list[ImageData] | None = None,
) -> AgentResult:
    """Send a prompt to an Anthropic model and return the response.

    Args:
        model: Model name (e.g., 'claude-sonnet-4-6').
        prompt: The full prompt to send.
        web_search: Enable the web_search server tool.
        reasoning: Optional reasoning config for extended thinking.
        on_token: Optional callback invoked with each text delta for live streaming.

    Returns:
        AgentResult with text and token counts (zero for streaming).
    """
    logger.debug(f"Anthropic call: model={model}, prompt_len={len(prompt)}, ws={web_search}")
    client = AsyncAnthropic()

    # Build user message content (multimodal when images provided)
    user_content: str | list[dict[str, Any]]
    if images:
        content_parts: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for img in images:
            content_parts.append(
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": img.media_type, "data": img.data},
                }
            )
        user_content = content_parts
    else:
        user_content = prompt

    kwargs: dict = {
        "model": model,
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": user_content}],
    }

    if system_prompt:
        kwargs["system"] = system_prompt

    if web_search:
        kwargs.setdefault("tools", [])
        kwargs["tools"].append({"type": "web_search_20250305", "name": "web_search"})

    if response_schema is not None:
        schema = response_schema.model_json_schema()
        kwargs.setdefault("tools", [])
        kwargs["tools"].append(
            {
                "name": "submit_response",
                "description": "Submit your structured response.",
                "input_schema": schema,
            }
        )
        # Only force tool_choice when web_search is not enabled;
        # when both are active, let the model choose freely between tools
        if not web_search:
            kwargs["tool_choice"] = {"type": "tool", "name": "submit_response"}

    if reasoning is not None and reasoning.level != "off":
        budget = reasoning.budget or ANTHROPIC_BUDGET_DEFAULTS.get(reasoning.level)
        if budget is None:
            valid = ", ".join(ANTHROPIC_BUDGET_DEFAULTS.keys())
            raise ValueError(
                f"Unknown Anthropic reasoning level: {reasoning.level!r}. Valid levels: {valid}"
            )
        kwargs["thinking"] = {"type": "adaptive", "budget_tokens": budget}
        kwargs["max_tokens"] = max(kwargs["max_tokens"], budget + 4096)

    if on_token is not None:
        if response_schema is not None:
            raise ValueError(
                "Streaming (on_token) and response_schema cannot be used together. "
                "Structured output requires the non-streaming path."
            )
        parts: list[str] = []
        async with client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                on_token(text)
                parts.append(text)
        result = "".join(parts)
        logger.debug(f"Anthropic response: {len(result)} chars")
        return AgentResult(text=result)

    response = await client.messages.create(**kwargs)
    usage = getattr(response, "usage", None)
    in_tok = getattr(usage, "input_tokens", 0) or 0
    out_tok = getattr(usage, "output_tokens", 0) or 0

    # When using structured output via tool_use, extract the submit_response tool input
    if response_schema is not None:
        for block in response.content:
            if (
                getattr(block, "type", None) == "tool_use"
                and getattr(block, "name", None) == "submit_response"
            ):
                result = json.dumps(block.input)
                logger.debug(f"Anthropic response (structured): {len(result)} chars")
                return AgentResult(text=result, input_tokens=in_tok, output_tokens=out_tok)
            elif getattr(block, "type", None) == "tool_use":
                logger.debug(f"Skipping non-submit tool_use block: {getattr(block, 'name', '?')}")
        # Structured output requested but no submit_response block found
        block_info = [
            f"{getattr(b, 'type', type(b).__name__)}({getattr(b, 'name', '')})"
            for b in response.content
        ]
        logger.error(
            f"Anthropic response contained no submit_response tool_use block "
            f"despite response_schema={response_schema.__name__}. "
            f"Content blocks: {block_info}"
        )
        raise RuntimeError(
            f"Anthropic structured output failed: model did not call "
            f"submit_response tool for schema {response_schema.__name__}"
        )

    # Extract text blocks (skip tool_use/web_search result blocks)
    text_parts = []
    for block in response.content:
        if hasattr(block, "text"):
            text_parts.append(block.text)
    result = "\n".join(text_parts) if text_parts else ""
    logger.debug(f"Anthropic response: {len(result)} chars, {in_tok} in / {out_tok} out tokens")
    return AgentResult(text=result, input_tokens=in_tok, output_tokens=out_tok)
