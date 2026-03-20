"""OpenAI API wrapper with optional streaming."""

from collections.abc import Callable

from openai import AsyncOpenAI


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
            return "".join(parts)

        response = await client.responses.create(
            model=model,
            input=prompt,
            tools=[{"type": "web_search_preview"}],
        )
        return response.output_text or ""

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
        return "".join(parts)

    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""
