"""Domain configuration schema."""

from pydantic import BaseModel, Field, field_validator


class DomainConfig(BaseModel):
    """Operational configuration for a specific scientific domain.

    Contains only runtime/infrastructure settings. Scientific concerns
    (domain_knowledge, prediction_history) live in ExperimentState.
    """

    name: str
    description: str
    data_paths: list[str]
    run_command: str = "uv run {script_path}"
    run_cwd: str = "."

    @field_validator("data_paths", mode="before")
    @classmethod
    def coerce_data_paths(cls, v: object) -> list[str]:
        if isinstance(v, dict):
            return list(v.values())
        return v

    @field_validator("run_command")
    @classmethod
    def run_command_must_contain_placeholder(cls, v: str) -> str:
        if "{script_path}" not in v:
            raise ValueError(
                f"run_command must contain '{{script_path}}' placeholder, got: {v}"
            )
        return v
    run_timeout_minutes: int = 120
    version_prefix: str = "v"
    protected_paths: list[str] = Field(default_factory=list)
