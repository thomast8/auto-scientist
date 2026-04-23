"""Pydantic models for structured debate output.

These models define the structured JSON output for critics. The orchestrator
uses these to build a concern ledger that the revision scientist consumes.

All models use extra="ignore" so unexpected fields from LLM output don't
cause validation failures (same convention as schemas.py).
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

ConcernCategory = Literal[
    "methodology",
    "trajectory",
    "falsification",
    "consistency",
    "criteria",
    "other",
]


class Concern(BaseModel):
    """A single concern raised by a critic."""

    model_config = ConfigDict(extra="ignore")

    claim: str
    severity: Literal["high", "medium", "low"]
    confidence: Literal["high", "medium", "low"]
    category: ConcernCategory


class CriticOutput(BaseModel):
    """Structured output from a critic's evaluation of a plan."""

    model_config = ConfigDict(extra="ignore")

    concerns: list[Concern]
    alternative_hypotheses: list[str]
    overall_assessment: str


class ConcernLedgerEntry(BaseModel):
    """A single entry in the concern ledger passed to the revision scientist.

    Combines a critic's concern with metadata (persona, model).
    """

    model_config = ConfigDict(extra="ignore")

    claim: str
    severity: Literal["high", "medium", "low"]
    confidence: Literal["high", "medium", "low"]
    category: ConcernCategory
    persona: str
    critic_model: str


class DebateResult(BaseModel):
    """Complete result of a single persona's critique."""

    model_config = ConfigDict(extra="ignore")

    persona: str
    critic_model: str
    critic_output: CriticOutput
    raw_transcript: list[dict[str, str]]
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0


# JSON schema dict for prompt injection (mirrors the Pydantic model).
CRITIC_OUTPUT_SCHEMA: dict[str, Any] = {
    "concerns": [
        {
            "claim": "<specific concern about the plan>",
            "severity": "high | medium | low",
            "confidence": "high | medium | low",
            "category": "methodology | trajectory | falsification | consistency | criteria | other",
        }
    ],
    "alternative_hypotheses": ["<alternative hypothesis the scientist has not considered>"],
    "overall_assessment": "<brief overall assessment of the plan>",
}
