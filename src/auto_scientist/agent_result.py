"""Lightweight result type returned by model clients."""

from dataclasses import dataclass


@dataclass
class AgentResult:
    """Response from a model client, including optional token usage metadata.

    Direct API clients (OpenAI, Google, Anthropic) populate token counts from
    non-streaming responses. Streaming responses and SDK-based agents return
    zero tokens (the information is unavailable).
    """

    text: str
    input_tokens: int = 0
    output_tokens: int = 0
