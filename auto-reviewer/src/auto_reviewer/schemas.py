"""Auto-reviewer output schemas.

Generic types (PredictionOutcome, PlanChange, HypothesisPrediction,
RefutationReasoning, DeprioritizedAbduction, RunResult, CompletenessAssessmentOutput,
SubQuestionAssessment) come from `auto_core.schemas` and are re-exported.

Review-specific shapes: `SurveyorOutput` (reviewer analogue of AnalystOutput),
`HunterPlanOutput` (reviewer analogue of ScientistPlanOutput), `ProberRunResult`
(alias of RunResult).
"""

from typing import Literal

from auto_core.schemas import (  # noqa: F401
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

# Review prober writes run_result.json with the same shape.
ProberRunResult = RunResult


class TouchedSymbol(BaseModel):
    """A symbol modified by the PR + surveyor's read of its blast radius."""

    model_config = ConfigDict(extra="ignore")

    name: str
    file: str
    kind: Literal["function", "method", "class", "module", "other"] = "other"


class Suspicion(BaseModel):
    """A diff-level pattern the Surveyor surfaces on iteration 0.

    `summary` describes the pattern as an observation ("A and B touch
    shared state on the same path"), never as a bug claim ("there is a
    race in A and B") - the Hunter hypothesizes and the Prober confirms.
    `severity` is notability: how much the pattern warrants a probe,
    independent of whether it turns out to be a bug.
    """

    model_config = ConfigDict(extra="ignore")

    summary: str
    evidence: str  # diff hunk / call-site pattern that triggered the observation
    severity: Literal["low", "medium", "high"] = "medium"


class SurveyorOutput(BaseModel):
    """Validated output from the Surveyor agent (the review analogue of AnalystOutput).

    Iteration 0: reads diff + touched files, populates `suspicions`.
    Iteration N>0: reads probe results + lab notebook, populates
    `prediction_outcomes` (each suspected bug's resolution from the probe).
    """

    model_config = ConfigDict(extra="ignore")

    suspicions: list[Suspicion] = Field(default_factory=list)
    touched_symbols: list[TouchedSymbol] = Field(default_factory=list)
    observations: list[str] = Field(default_factory=list)
    prediction_outcomes: list[PredictionOutcome] = Field(default_factory=list)
    repo_knowledge: str = ""
    diff_summary: str | None = None


class ReproductionRecipe(BaseModel):
    """A planned reproduction recipe for a suspected bug.

    Kept as a distinct type from `HypothesisPrediction` for clarity even
    though the storage shape is identical (see auto_core/schemas.py).
    """

    model_config = ConfigDict(extra="ignore")

    prediction: str  # "the bug is X"
    diagnostic: str  # "reproduce by Y"
    if_confirmed: str
    if_refuted: str
    follows_from: str | None = None


class HunterPlanOutput(BaseModel):
    """Validated output from the Hunter agent (reviewer analogue of ScientistPlanOutput)."""

    model_config = ConfigDict(extra="ignore")

    hypothesis: str  # one-line summary of the bug being chased this iteration
    strategy: Literal["incremental", "structural", "exploratory"]
    changes: list[PlanChange]  # what the Prober should do
    expected_impact: str
    should_stop: bool
    stop_reason: str | None
    notebook_entry: str
    testable_predictions: list[HypothesisPrediction] = Field(default_factory=list)
    refutation_reasoning: list[RefutationReasoning] = Field(default_factory=list)
    deprioritized_abductions: list[DeprioritizedAbduction] = Field(default_factory=list)
