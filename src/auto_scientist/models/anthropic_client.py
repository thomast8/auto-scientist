"""Anthropic API wrapper with optional streaming."""

from collections.abc import Callable

from anthropic import AsyncAnthropic


async def query_anthropic(
    model: str,
    prompt: str,
    *,
    web_search: bool = False,
    on_token: Callable[[str], None] | None = None,
) -> str:
    """Send a prompt to an Anthropic model and return the response text.

    Args:
        model: Model name (e.g., 'claude-sonnet-4-6').
        prompt: The full prompt to send.
        web_search: Enable the web_search server tool.
        on_token: Optional callback invoked with each text delta for live streaming.

    Returns:
        The model's text response.
    """
    client = AsyncAnthropic()

    kwargs: dict = {
        "model": model,
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": prompt}],
    }

    if web_search:
        kwargs["tools"] = [{"type": "web_search_20250305", "name": "web_search"}]

    if on_token is not None:
        parts: list[str] = []
        async with client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                on_token(text)
                parts.append(text)
        return "".join(parts)

    response = await client.messages.create(**kwargs)

    # Extract text blocks (skip tool_use/web_search result blocks)
    text_parts = []
    for block in response.content:
        if hasattr(block, "text"):
            text_parts.append(block.text)
    return "\n".join(text_parts) if text_parts else ""
