"""Pydantic output schemas for agent output validation.

These models validate the JSON output of the Analyst and Scientist agents
at runtime. They mirror the JSON schema dicts in the agent modules
(ANALYST_SCHEMA, SCIENTIST_PLAN_SCHEMA) which remain as prompt-injected
guidance for the LLM.

All models use extra="ignore" so unexpected fields from the LLM don't
cause validation failures.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class PredictionOutcome(BaseModel):
    """An outcome for a testable prediction, extracted by the Analyst."""

    model_config = ConfigDict(extra="ignore")

    pred_id: str = ""
    prediction: str
    outcome: Literal["confirmed", "refuted", "inconclusive"]
    evidence: str


class AnalystOutput(BaseModel):
    """Validated output from the Analyst agent."""

    model_config = ConfigDict(extra="ignore")

    key_metrics: dict[str, float]
    improvements: list[str]
    regressions: list[str]
    observations: list[str]
    prediction_outcomes: list[PredictionOutcome] = Field(default_factory=list)
    domain_knowledge: str = ""
    data_summary: dict[str, Any] | None = None


class PlanChange(BaseModel):
    """A single change entry in the Scientist's plan."""

    model_config = ConfigDict(extra="ignore")

    what: str
    why: str
    how: str
    priority: int


class CoderRunResult(BaseModel):
    """Validated schema for run_result.json written by the Coder's experiment script."""

    model_config = ConfigDict(extra="ignore")

    success: bool
    return_code: int = -1
    timed_out: bool = False
    error: str | None = None
    attempts: int = 1


class TestablePrediction(BaseModel):
    """A testable prediction from the Scientist's plan output."""

    model_config = ConfigDict(extra="ignore")

    prediction: str
    diagnostic: str
    if_confirmed: str
    if_refuted: str
    follows_from: str | None = None


class ScientistPlanOutput(BaseModel):
    """Validated output from the Scientist agent (plan and revision)."""

    model_config = ConfigDict(extra="ignore")

    hypothesis: str
    strategy: Literal["incremental", "structural", "exploratory"]
    changes: list[PlanChange]
    expected_impact: str
    should_stop: bool
    stop_reason: str | None
    notebook_entry: str
    testable_predictions: list[TestablePrediction] = Field(default_factory=list)
