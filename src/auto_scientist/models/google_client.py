"""Google AI wrapper with optional streaming, reasoning, and structured output."""

from __future__ import annotations

import base64
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from google import genai
from google.genai import types
from pydantic import BaseModel

from auto_scientist.agent_result import AgentResult

if TYPE_CHECKING:
    from auto_scientist.images import ImageData
    from auto_scientist.model_config import ReasoningConfig

logger = logging.getLogger(__name__)

GOOGLE_BUDGET_DEFAULTS: dict[str, int] = {
    "off": 0,
    "minimal": 128,
    "low": 1024,
    "medium": 4096,
    "high": 16384,
    "max": 32768,
}

GOOGLE_LEVEL_MAP: dict[str, types.ThinkingLevel] = {
    "minimal": types.ThinkingLevel.MINIMAL,
    "low": types.ThinkingLevel.LOW,
    "medium": types.ThinkingLevel.MEDIUM,
    "high": types.ThinkingLevel.HIGH,
    "max": types.ThinkingLevel.HIGH,
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
    system_prompt: str | None = None,
    response_schema: type[BaseModel] | None = None,
    images: list[ImageData] | None = None,
) -> AgentResult:
    """Send a prompt to a Google AI model and return the response.

    Args:
        model: Model name (e.g., 'gemini-2.5-pro').
        prompt: The full prompt to send.
        web_search: Enable Google Search grounding.
        reasoning: Optional reasoning config for thinking control.
        on_token: Optional callback invoked with each text delta for live streaming.

    Returns:
        AgentResult with text and token counts (zero for streaming).
    """
    logger.debug(f"Google call: model={model}, prompt_len={len(prompt)}, web_search={web_search}")
    client = genai.Client()

    # Build thinking config
    thinking_config = None
    if reasoning is not None and reasoning.level != "off":
        if _is_gemini_25(model):
            budget = reasoning.budget or GOOGLE_BUDGET_DEFAULTS.get(reasoning.level)
            if budget is None:
                valid = ", ".join(GOOGLE_BUDGET_DEFAULTS.keys())
                raise ValueError(
                    f"Unknown Google reasoning level: {reasoning.level!r}. Valid levels: {valid}"
                )
            thinking_config = types.ThinkingConfig(thinking_budget=budget)
        else:
            level_str = GOOGLE_LEVEL_MAP.get(reasoning.level)
            if level_str is None:
                valid = ", ".join(GOOGLE_LEVEL_MAP.keys())
                raise ValueError(
                    f"Unknown Google reasoning level: {reasoning.level!r}. Valid levels: {valid}"
                )
            thinking_config = types.ThinkingConfig(thinking_level=level_str)
    elif reasoning is not None and reasoning.level == "off" and _is_gemini_25(model):
        thinking_config = types.ThinkingConfig(thinking_budget=0)

    # Build config
    config_kwargs: dict = {}
    if web_search:
        config_kwargs["tools"] = [types.Tool(google_search=types.GoogleSearch())]
    if thinking_config is not None:
        config_kwargs["thinking_config"] = thinking_config
    if system_prompt:
        config_kwargs["system_instruction"] = system_prompt
    if response_schema is not None:
        config_kwargs["response_mime_type"] = "application/json"
        config_kwargs["response_schema"] = response_schema.model_json_schema()

    config = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

    # Build contents (multimodal when images provided)
    contents: str | list = prompt
    if images:
        contents = [prompt]
        for img in images:
            contents.append(
                types.Part.from_bytes(data=base64.b64decode(img.data), mime_type=img.media_type)
            )

    if on_token is not None:
        parts: list[str] = []
        async for chunk in await client.aio.models.generate_content_stream(
            model=model,
            contents=contents,
            config=config,
        ):
            text = chunk.text
            if text:
                on_token(text)
                parts.append(text)
        result = "".join(parts)
        logger.debug(f"Google response: {len(result)} chars")
        return AgentResult(text=result)

    response = await client.aio.models.generate_content(
        model=model,
        contents=contents,
        config=config,
    )
    result = response.text or ""
    usage = getattr(response, "usage_metadata", None)
    in_tok = getattr(usage, "prompt_token_count", 0) or 0
    out_tok = getattr(usage, "candidates_token_count", 0) or 0
    logger.debug(f"Google response: {len(result)} chars, {in_tok} in / {out_tok} out tokens")
    return AgentResult(text=result, input_tokens=in_tok, output_tokens=out_tok)
