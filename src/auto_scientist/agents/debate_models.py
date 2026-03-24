"""Pydantic models for structured debate output.

These models define the structured JSON output for critics and scientist-in-debate,
replacing free-text prose with validated, typed data. The orchestrator uses these
to build a concern ledger that the revision scientist consumes.

All models use extra="ignore" so unexpected fields from LLM output don't
cause validation failures (same convention as schemas.py).
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class Concern(BaseModel):
    """A single concern raised by a critic."""

    model_config = ConfigDict(extra="ignore")

    claim: str
    severity: Literal["high", "medium", "low"]
    confidence: Literal["high", "medium", "low"]
    category: Literal["methodology", "novelty", "feasibility", "criteria", "other"]


class CriticOutput(BaseModel):
    """Structured output from a critic's evaluation of a plan."""

    model_config = ConfigDict(extra="ignore")

    concerns: list[Concern]
    alternative_hypotheses: list[str]
    overall_assessment: str


class DefenseResponse(BaseModel):
    """Scientist's response to a single critic concern."""

    model_config = ConfigDict(extra="ignore")

    concern: str
    verdict: Literal["accepted", "rejected", "partially_accepted"]
    reasoning: str


class ScientistDefense(BaseModel):
    """Structured output from the scientist defending against a critique."""

    model_config = ConfigDict(extra="ignore")

    responses: list[DefenseResponse]
    additional_points: str = ""


class ConcernLedgerEntry(BaseModel):
    """A single entry in the concern ledger passed to the revision scientist.

    Combines a critic's concern with metadata (persona, model) and the
    scientist's defense verdict (if a multi-round debate occurred).
    """

    model_config = ConfigDict(extra="ignore")

    claim: str
    severity: Literal["high", "medium", "low"]
    confidence: Literal["high", "medium", "low"]
    category: Literal["methodology", "novelty", "feasibility", "criteria", "other"]
    persona: str
    critic_model: str
    scientist_verdict: Literal["accepted", "rejected", "partially_accepted"] | None = None
    scientist_reasoning: str | None = None


class DebateRound(BaseModel):
    """One round of a critic-scientist debate."""

    model_config = ConfigDict(extra="ignore")

    critic_output: CriticOutput
    scientist_defense: ScientistDefense | None = None


class DebateResult(BaseModel):
    """Complete result of a single persona's debate."""

    model_config = ConfigDict(extra="ignore")

    persona: str
    critic_model: str
    rounds: list[DebateRound]
    raw_transcript: list[dict[str, str]]
    input_tokens: int = 0
    output_tokens: int = 0


# JSON schema dicts for prompt injection (mirrors the Pydantic models).
CRITIC_OUTPUT_SCHEMA: dict[str, Any] = {
    "concerns": [
        {
            "claim": "<specific concern about the plan>",
            "severity": "high | medium | low",
            "confidence": "high | medium | low",
            "category": "methodology | novelty | feasibility | criteria | other",
        }
    ],
    "alternative_hypotheses": ["<alternative hypothesis the scientist has not considered>"],
    "overall_assessment": "<brief overall assessment of the plan>",
}

SCIENTIST_DEFENSE_SCHEMA: dict[str, Any] = {
    "responses": [
        {
            "concern": "<the concern being addressed>",
            "verdict": "accepted | rejected | partially_accepted",
            "reasoning": "<why you accept, reject, or partially accept this concern>",
        }
    ],
    "additional_points": "<any points not tied to a specific concern>",
}
