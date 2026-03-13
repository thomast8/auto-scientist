"""Google AI wrapper for the Critic."""

from google import genai
from google.genai import types


async def query_google(model: str, prompt: str, *, web_search: bool = False) -> str:
    """Send a prompt to a Google AI model and return the response text.

    Args:
        model: Model name (e.g., 'gemini-2.5-pro').
        prompt: The full prompt to send.
        web_search: Enable Google Search grounding.

    Returns:
        The model's text response.
    """
    client = genai.Client()

    config = None
    if web_search:
        config = types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())]
        )

    response = await client.aio.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )
    return response.text or ""
