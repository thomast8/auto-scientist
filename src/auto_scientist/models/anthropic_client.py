"""Anthropic API wrapper for the Critic."""

from anthropic import AsyncAnthropic


async def query_anthropic(model: str, prompt: str, *, web_search: bool = False) -> str:
    """Send a prompt to an Anthropic model and return the response text.

    Args:
        model: Model name (e.g., 'claude-sonnet-4-6').
        prompt: The full prompt to send.
        web_search: Enable the web_search server tool.

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

    response = await client.messages.create(**kwargs)

    # Extract text blocks (skip tool_use/web_search result blocks)
    text_parts = []
    for block in response.content:
        if hasattr(block, "text"):
            text_parts.append(block.text)
    return "\n".join(text_parts) if text_parts else ""
