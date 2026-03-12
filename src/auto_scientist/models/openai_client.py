"""OpenAI API wrapper for the Critic."""

from openai import AsyncOpenAI


async def query_openai(model: str, prompt: str) -> str:
    """Send a prompt to an OpenAI model and return the response text.

    Args:
        model: Model name (e.g., 'gpt-4o').
        prompt: The full prompt to send.

    Returns:
        The model's text response.
    """
    client = AsyncOpenAI()
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4096,
    )
    return response.choices[0].message.content or ""
