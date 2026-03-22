"""Pydantic output schemas for agent output validation.

These models validate the JSON output of the Analyst and Scientist agents
at runtime. They mirror the JSON schema dicts in the agent modules
(ANALYST_SCHEMA, SCIENTIST_PLAN_SCHEMA) which remain as prompt-injected
guidance for the LLM.

All models use extra="ignore" so unexpected fields from the LLM don't
cause validation failures.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class CriterionResult(BaseModel):
    """A single criterion evaluation result from the Analyst."""

    model_config = ConfigDict(extra="ignore")

    name: str
    measured_value: str | int | float | None
    target: str
    status: Literal["pass", "fail", "unable_to_measure"]


class IterationCriterionResult(BaseModel):
    """Per-iteration criterion result from the Analyst."""

    model_config = ConfigDict(extra="ignore")

    name: str
    status: Literal["pass", "fail"]
    measured_value: str


class AnalystOutput(BaseModel):
    """Validated output from the Analyst agent."""

    model_config = ConfigDict(extra="ignore")

    success_score: int | None = None
    criteria_results: list[CriterionResult]
    key_metrics: dict[str, float]
    improvements: list[str]
    regressions: list[str]
    observations: list[str]
    iteration_criteria_results: list[IterationCriterionResult]
    domain_knowledge: str = ""
    data_summary: dict[str, Any] | None = None


class PlanChange(BaseModel):
    """A single change entry in the Scientist's plan."""

    model_config = ConfigDict(extra="ignore")

    what: str
    why: str
    how: str
    priority: int


class CriterionDefinition(BaseModel):
    """A criterion definition from the Scientist's plan output."""

    model_config = ConfigDict(extra="ignore")

    name: str
    description: str
    metric_key: str
    condition: str


class CriteriaRevisionOutput(BaseModel):
    """Criteria revision block from the Scientist's plan."""

    model_config = ConfigDict(extra="ignore")

    changes: str
    revised_criteria: list[CriterionDefinition]


class CoderRunResult(BaseModel):
    """Validated schema for run_result.json written by the Coder's experiment script."""

    model_config = ConfigDict(extra="ignore")

    success: bool
    return_code: int = -1
    timed_out: bool = False
    error: str | None = None
    attempts: int = 1


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
    success_criteria: list[CriterionDefinition]
    top_level_criteria: list[CriterionDefinition] | None = None
    criteria_revision: CriteriaRevisionOutput | None = None
