"""Google AI wrapper for the Critic."""

from google import genai


async def query_google(model: str, prompt: str) -> str:
    """Send a prompt to a Google AI model and return the response text.

    Args:
        model: Model name (e.g., 'gemini-2.5-pro').
        prompt: The full prompt to send.

    Returns:
        The model's text response.
    """
    client = genai.Client()
    response = await client.aio.models.generate_content(
        model=model,
        contents=prompt,
    )
    return response.text or ""
