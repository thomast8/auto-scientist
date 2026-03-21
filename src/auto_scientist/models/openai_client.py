"""OpenAI API wrapper with optional streaming and reasoning support."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from openai import AsyncOpenAI

if TYPE_CHECKING:
    from auto_scientist.model_config import ReasoningConfig

logger = logging.getLogger(__name__)

OPENAI_EFFORT_MAP: dict[str, str] = {
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "max": "high",
    "adaptive": "medium",
}


async def query_openai(
    model: str,
    prompt: str,
    *,
    web_search: bool = False,
    reasoning: ReasoningConfig | None = None,
    on_token: Callable[[str], None] | None = None,
    max_tokens: int = 4096,
) -> str:
    """Send a prompt to an OpenAI model and return the response text.

    Args:
        model: Model name (e.g., 'gpt-4o').
        prompt: The full prompt to send.
        web_search: Enable web search via the Responses API.
        reasoning: Optional reasoning config for reasoning effort.
        on_token: Optional callback invoked with each text delta for live streaming.

    Returns:
        The model's text response.
    """
    logger.debug(f"OpenAI call: model={model}, prompt_len={len(prompt)}, web_search={web_search}")
    client = AsyncOpenAI()

    # Resolve reasoning effort
    effort: str | None = None
    if reasoning is not None and reasoning.level != "off":
        effort = OPENAI_EFFORT_MAP.get(reasoning.level)

    if web_search:
        resp_kwargs: dict = {
            "model": model,
            "input": prompt,
            "tools": [{"type": "web_search_preview"}],
        }
        if effort:
            resp_kwargs["reasoning"] = {"effort": effort}

        if on_token is not None:
            parts: list[str] = []
            async for event in await client.responses.create(**resp_kwargs, stream=True):
                if event.type == "response.output_text.delta":
                    on_token(event.delta)
                    parts.append(event.delta)
            result = "".join(parts)
            logger.debug(f"OpenAI response: {len(result)} chars")
            return result

        response = await client.responses.create(**resp_kwargs)
        result = response.output_text or ""
        logger.debug(f"OpenAI response: {len(result)} chars")
        return result

    chat_kwargs: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    if effort:
        chat_kwargs["reasoning_effort"] = effort
        chat_kwargs["max_completion_tokens"] = max_tokens
    else:
        chat_kwargs["max_tokens"] = max_tokens

    if on_token is not None:
        parts = []
        async for chunk in await client.chat.completions.create(**chat_kwargs, stream=True):
            delta = chunk.choices[0].delta.content
            if delta:
                on_token(delta)
                parts.append(delta)
        result = "".join(parts)
        logger.debug(f"OpenAI response: {len(result)} chars")
        return result

    response = await client.chat.completions.create(**chat_kwargs)
    result = response.choices[0].message.content or ""
    logger.debug(f"OpenAI response: {len(result)} chars")
    return result
