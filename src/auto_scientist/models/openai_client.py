"""OpenAI API wrapper with optional streaming."""

import logging
from collections.abc import Callable

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


async def query_openai(
    model: str,
    prompt: str,
    *,
    web_search: bool = False,
    on_token: Callable[[str], None] | None = None,
    max_tokens: int = 4096,
) -> str:
    """Send a prompt to an OpenAI model and return the response text.

    Args:
        model: Model name (e.g., 'gpt-4o').
        prompt: The full prompt to send.
        web_search: Enable web search via the Responses API.
        on_token: Optional callback invoked with each text delta for live streaming.

    Returns:
        The model's text response.
    """
    logger.debug(f"OpenAI call: model={model}, prompt_len={len(prompt)}, web_search={web_search}")
    client = AsyncOpenAI()

    if web_search:
        if on_token is not None:
            parts: list[str] = []
            async for event in await client.responses.create(
                model=model,
                input=prompt,
                tools=[{"type": "web_search_preview"}],
                stream=True,
            ):
                if event.type == "response.output_text.delta":
                    on_token(event.delta)
                    parts.append(event.delta)
            result = "".join(parts)
            logger.debug(f"OpenAI response: {len(result)} chars")
            return result

        response = await client.responses.create(
            model=model,
            input=prompt,
            tools=[{"type": "web_search_preview"}],
        )
        result = response.output_text or ""
        logger.debug(f"OpenAI response: {len(result)} chars")
        return result

    if on_token is not None:
        parts = []
        async for chunk in await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            stream=True,
        ):
            delta = chunk.choices[0].delta.content
            if delta:
                on_token(delta)
                parts.append(delta)
        result = "".join(parts)
        logger.debug(f"OpenAI response: {len(result)} chars")
        return result

    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    result = response.choices[0].message.content or ""
    logger.debug(f"OpenAI response: {len(result)} chars")
    return result
