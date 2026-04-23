"""Science-specific Pydantic output schemas for auto-scientist.

The shared / generic types (PredictionOutcome, PlanChange, HypothesisPrediction,
RefutationReasoning, DeprioritizedAbduction, RunResult, CompletenessAssessmentOutput,
SubQuestionAssessment) live in `auto_core.schemas`; they are re-exported here so
existing call sites can keep `from auto_scientist.schemas import *`.

What stays here: `AnalystOutput`, `ScientistPlanOutput`, `KeyMetric`,
`DataDiagnostic` - the scientific-investigation output shapes.
"""

from typing import Literal

# Re-export shared pieces so existing imports keep working.
from auto_core.schemas import (  # noqa: F401
    CoderRunResult,
    CompletenessAssessmentOutput,
    DeprioritizedAbduction,
    HypothesisPrediction,
    PlanChange,
    PredictionOutcome,
    RefutationReasoning,
    RunResult,
    SubQuestionAssessment,
)
from pydantic import BaseModel, ConfigDict, Field


class KeyMetric(BaseModel):
    """A single named numeric metric from experiment results."""

    model_config = ConfigDict(extra="ignore")

    name: str
    value: float


class DataDiagnostic(BaseModel):
    """A cross-cutting structural pattern noticed by the Analyst."""

    model_config = ConfigDict(extra="ignore")

    variables: list[str]
    pattern: str
    evidence: str


class AnalystOutput(BaseModel):
    """Validated output from the Analyst agent."""

    model_config = ConfigDict(extra="ignore")

    key_metrics: list[KeyMetric]
    improvements: list[str]
    regressions: list[str]
    observations: list[str]
    prediction_outcomes: list[PredictionOutcome] = Field(default_factory=list)
    data_diagnostics: list[DataDiagnostic] = Field(default_factory=list)
    domain_knowledge: str = ""
    data_summary: str | None = None


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
    testable_predictions: list[HypothesisPrediction] = Field(default_factory=list)
    refutation_reasoning: list[RefutationReasoning] = Field(default_factory=list)
    deprioritized_abductions: list[DeprioritizedAbduction] = Field(default_factory=list)
