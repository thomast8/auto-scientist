"""Google AI wrapper with optional streaming and reasoning support."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from google import genai
from google.genai import types

if TYPE_CHECKING:
    from auto_scientist.model_config import ReasoningConfig

logger = logging.getLogger(__name__)

GOOGLE_BUDGET_DEFAULTS: dict[str, int] = {
    "off": 0,
    "minimal": 128,
    "low": 1024,
    "medium": 4096,
    "high": 16384,
    "max": 32768,
    "adaptive": -1,
}

GOOGLE_LEVEL_MAP: dict[str, str] = {
    "minimal": "MINIMAL",
    "low": "LOW",
    "medium": "MEDIUM",
    "high": "HIGH",
    "max": "HIGH",
    "adaptive": "HIGH",
}


def _is_gemini_25(model: str) -> bool:
    """Check if a model is a Gemini 2.5 (budget-based thinking)."""
    return "2.5" in model


async def query_google(
    model: str,
    prompt: str,
    *,
    web_search: bool = False,
    reasoning: ReasoningConfig | None = None,
    on_token: Callable[[str], None] | None = None,
) -> str:
    """Send a prompt to a Google AI model and return the response text.

    Args:
        model: Model name (e.g., 'gemini-2.5-pro').
        prompt: The full prompt to send.
        web_search: Enable Google Search grounding.
        reasoning: Optional reasoning config for thinking control.
        on_token: Optional callback invoked with each text delta for live streaming.

    Returns:
        The model's text response.
    """
    logger.debug(f"Google call: model={model}, prompt_len={len(prompt)}, web_search={web_search}")
    client = genai.Client()

    # Build thinking config
    thinking_config = None
    if reasoning is not None and reasoning.level != "off":
        if _is_gemini_25(model):
            budget = reasoning.budget or GOOGLE_BUDGET_DEFAULTS.get(reasoning.level, 4096)
            thinking_config = types.ThinkingConfig(thinking_budget=budget)
        else:
            level_str = GOOGLE_LEVEL_MAP.get(reasoning.level, "HIGH")
            thinking_config = types.ThinkingConfig(thinking_level=level_str)
    elif reasoning is not None and reasoning.level == "off" and _is_gemini_25(model):
        thinking_config = types.ThinkingConfig(thinking_budget=0)

    # Build config
    config_kwargs: dict = {}
    if web_search:
        config_kwargs["tools"] = [types.Tool(google_search=types.GoogleSearch())]
    if thinking_config is not None:
        config_kwargs["thinking_config"] = thinking_config

    config = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

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
        result = "".join(parts)
        logger.debug(f"Google response: {len(result)} chars")
        return result

    response = await client.aio.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )
    result = response.text or ""
    logger.debug(f"Google response: {len(result)} chars")
    return result
