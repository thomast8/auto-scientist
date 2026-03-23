"""OpenAI API wrapper with optional streaming, reasoning, and structured output."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from openai import AsyncOpenAI
from pydantic import BaseModel

if TYPE_CHECKING:
    from auto_scientist.images import ImageData
    from auto_scientist.model_config import ReasoningConfig

logger = logging.getLogger(__name__)

OPENAI_EFFORT_MAP: dict[str, str] = {
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "max": "xhigh",
}


async def query_openai(
    model: str,
    prompt: str,
    *,
    web_search: bool = False,
    reasoning: ReasoningConfig | None = None,
    on_token: Callable[[str], None] | None = None,
    max_tokens: int = 4096,
    system_prompt: str | None = None,
    response_schema: type[BaseModel] | None = None,
    images: list[ImageData] | None = None,
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
    if reasoning is not None and reasoning.level not in ("default", "off"):
        effort = OPENAI_EFFORT_MAP.get(reasoning.level)

    if web_search:
        # Build input (multimodal when images provided)
        if images:
            resp_content: list[dict] = [{"type": "input_text", "text": prompt}]
            for img in images:
                resp_content.append({
                    "type": "input_image",
                    "image_url": f"data:{img.media_type};base64,{img.data}",
                })
            resp_input: str | list[dict] = [{"role": "user", "content": resp_content}]
        else:
            resp_input = prompt

        resp_kwargs: dict = {
            "model": model,
            "input": resp_input,
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

    # Build user message content (multimodal when images provided)
    if images:
        user_content: str | list[dict] = [{"type": "text", "text": prompt}]
        for img in images:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{img.media_type};base64,{img.data}"},
            })
    else:
        user_content = prompt

    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})

    chat_kwargs: dict = {
        "model": model,
        "messages": messages,
    }

    if response_schema is not None:
        chat_kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "response",
                "schema": response_schema.model_json_schema(),
            },
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
