"""Google AI wrapper with optional streaming."""

from collections.abc import Callable

from google import genai
from google.genai import types


async def query_google(
    model: str,
    prompt: str,
    *,
    web_search: bool = False,
    on_token: Callable[[str], None] | None = None,
) -> str:
    """Send a prompt to a Google AI model and return the response text.

    Args:
        model: Model name (e.g., 'gemini-2.5-pro').
        prompt: The full prompt to send.
        web_search: Enable Google Search grounding.
        on_token: Optional callback invoked with each text delta for live streaming.

    Returns:
        The model's text response.
    """
    client = genai.Client()

    config = None
    if web_search:
        config = types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())]
        )

    if on_token is not None:
        parts: list[str] = []
        async for chunk in await client.aio.models.generate_content_stream(
            model=model,
            contents=prompt,
            config=config,
        ):
            text = chunk.text
            if text:
                on_token(text)
                parts.append(text)
        return "".join(parts)

    response = await client.aio.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )
    return response.text or ""
