"""OpenAI API wrapper for the Critic."""

from openai import AsyncOpenAI


async def query_openai(model: str, prompt: str, *, web_search: bool = False) -> str:
    """Send a prompt to an OpenAI model and return the response text.

    Args:
        model: Model name (e.g., 'gpt-4o').
        prompt: The full prompt to send.
        web_search: Enable web search via the Responses API.

    Returns:
        The model's text response.
    """
    client = AsyncOpenAI()

    if web_search:
        # Use the Responses API which supports web_search_preview
        response = await client.responses.create(
            model=model,
            input=prompt,
            tools=[{"type": "web_search_preview"}],
        )
        return response.output_text or ""

    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4096,
    )
    return response.choices[0].message.content or ""
