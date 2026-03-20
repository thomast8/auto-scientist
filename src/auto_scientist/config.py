"""Domain configuration schema."""

from pydantic import BaseModel, Field


class SuccessCriterion(BaseModel):
    """A single success criterion for evaluating experiment results."""

    name: str
    description: str
    metric_key: str
    target_min: float | None = None
    target_max: float | None = None
    required: bool = True


class DomainConfig(BaseModel):
    """Operational configuration for a specific scientific domain.

    Contains only runtime/infrastructure settings. Scientific concerns
    (success_criteria, domain_knowledge) live in ExperimentState.
    """

    name: str
    description: str
    data_paths: list[str]
    run_command: str = "uv run python -u {script_path}"
    run_cwd: str = "."
    run_timeout_minutes: int = 120
    version_prefix: str = "v"
    protected_paths: list[str] = Field(default_factory=list)
    experiment_dependencies: list[str] = Field(default_factory=list)
