"""Shared Pydantic output schemas for validating agent JSON output.

Contains the generic types used by the orchestrator and by any app on the
shared runtime: `PredictionOutcome`, `PredictionOutcomeReport`,
`PlanChange`, `HypothesisPrediction` (reproduction recipe in review),
`RefutationReasoning`, `DeprioritizedAbduction`, `RunResult`, and the
`CompletenessAssessmentOutput` family used by the stop gate.

App-specific output schemas (e.g. `AnalystOutput`, `ScientistPlanOutput` for
auto-scientist) stay in the app package and import the generic pieces from
here.

All models use `extra="ignore"` so unexpected fields from the LLM don't
cause validation failures.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PredictionOutcome(BaseModel):
    """Observation of how a prior testable prediction resolved."""

    model_config = ConfigDict(extra="ignore")

    pred_id: str = ""
    prediction: str
    outcome: Literal["confirmed", "refuted", "inconclusive"]
    evidence: str
    summary: str = ""


class PlanChange(BaseModel):
    """A single change entry in a planner's plan output."""

    model_config = ConfigDict(extra="ignore")

    what: str
    why: str
    how: str
    priority: int


class HypothesisPrediction(BaseModel):
    """A testable prediction produced by the planner.

    In auto-scientist this is a scientific hypothesis with a diagnostic. In
    auto-reviewer it is a suspected bug plus a reproduction recipe; the fields
    carry the same shape with app-level wording baked into the prompts.
    """

    model_config = ConfigDict(extra="ignore")

    prediction: str
    diagnostic: str
    if_confirmed: str
    if_refuted: str
    follows_from: str | None = None


class RefutationReasoning(BaseModel):
    """Abductive reasoning about why a prior prediction was refuted."""

    model_config = ConfigDict(extra="ignore")

    refuted_pred_id: str
    assumptions_violated: str
    alternative_explanation: str
    testable_consequence: str


class DeprioritizedAbduction(BaseModel):
    """An explicit decision to not pursue a prior abduction's testable consequence."""

    model_config = ConfigDict(extra="ignore")

    refuted_pred_id: str
    reason: str


class RunResult(BaseModel):
    """Validated schema for run_result.json written by the implementer's script."""

    model_config = ConfigDict(extra="ignore")

    success: bool
    return_code: int = -1
    timed_out: bool = False
    error: str | None = None
    attempts: int = 1


# Legacy alias: auto-scientist historically spelled this `CoderRunResult`.
CoderRunResult = RunResult


class SubQuestionAssessment(BaseModel):
    """Assessment of a single sub-question extracted from the investigation goal."""

    model_config = ConfigDict(extra="ignore")

    question: str = Field(min_length=1)
    coverage: Literal["thorough", "shallow", "unexplored"]
    evidence: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)


class CompletenessAssessmentOutput(BaseModel):
    """Validated output from the completeness-assessment agent (stop gate)."""

    model_config = ConfigDict(extra="ignore")

    sub_questions: list[SubQuestionAssessment] = Field(min_length=1)
    overall_coverage: Literal["thorough", "partial", "incomplete"]
    recommendation: Literal["stop", "continue"]

    @model_validator(mode="after")
    def recommendation_matches_coverage(self) -> "CompletenessAssessmentOutput":
        """Coerce recommendation to 'continue' if coverage is not thorough."""
        if self.recommendation == "stop" and self.overall_coverage != "thorough":
            self.recommendation = "continue"
        return self
