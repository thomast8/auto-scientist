"""Anthropic API wrapper for the Critic."""

from anthropic import AsyncAnthropic


async def query_anthropic(model: str, prompt: str) -> str:
    """Send a prompt to an Anthropic model and return the response text.

    Args:
        model: Model name (e.g., 'claude-sonnet-4-6').
        prompt: The full prompt to send.

    Returns:
        The model's text response.
    """
    client = AsyncAnthropic()
    response = await client.messages.create(
        model=model,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text if response.content else ""
