"""Unified YAML experiment configuration.

A single YAML file captures data path, goal, iteration settings, preset,
and optional per-agent model overrides. Replaces long multiline CLI commands.
"""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from auto_scientist.model_config import BUILTIN_PRESETS, AgentModelConfig


class ExperimentModelsConfig(BaseModel):
    """Per-agent model overrides layered on top of a preset.

    When critics is non-empty, it replaces the entire preset critics list
    (not appended to it).
    """

    model_config = ConfigDict(extra="forbid")

    analyst: AgentModelConfig | None = None
    scientist: AgentModelConfig | None = None
    coder: AgentModelConfig | None = None
    ingestor: AgentModelConfig | None = None
    report: AgentModelConfig | None = None
    summarizer: AgentModelConfig | None = None
    critics: list[AgentModelConfig] = []


class ExperimentConfig(BaseModel):
    """Unified experiment configuration loaded from YAML."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    # Required
    data: str = Field(min_length=1)
    goal: str = Field(min_length=1)

    # Optional with defaults
    max_iterations: int = Field(default=20, ge=1)
    preset: str = "default"
    debate_rounds: int = Field(default=1, ge=0)
    output_dir: str = "experiments/runs"
    schedule: str | None = None
    interactive: bool = False
    stream: bool = True
    verbose: bool = False
    summaries: bool = True

    # Optional per-agent model overrides
    models: ExperimentModelsConfig | None = None

    @field_validator("preset")
    @classmethod
    def _validate_preset(cls, v: str) -> str:
        if v not in BUILTIN_PRESETS:
            raise ValueError(
                f"Unknown preset: {v!r}. Available: {list(BUILTIN_PRESETS)}"
            )
        return v

    @field_validator("models", mode="before")
    @classmethod
    def _parse_models(cls, v):
        if isinstance(v, dict):
            return ExperimentModelsConfig.model_validate(v)
        return v

    def resolve_data_path(self, yaml_dir: Path) -> Path:
        """Resolve data path relative to the YAML file's directory."""
        data_path = Path(self.data)
        if data_path.is_absolute():
            return data_path
        return yaml_dir / data_path

    @classmethod
    def from_yaml(cls, path: Path) -> "ExperimentConfig":
        """Load an ExperimentConfig from a YAML file."""
        try:
            with open(path) as f:
                raw = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {path}: {e}") from e

        if raw is None:
            raise ValueError(f"Empty config file: {path}")
        if not isinstance(raw, dict):
            raise ValueError(
                f"Expected a YAML mapping in {path}, got {type(raw).__name__}"
            )

        try:
            return cls.model_validate(raw)
        except ValidationError as e:
            raise ValueError(f"Invalid experiment config in {path}:\n{e}") from e

    def to_yaml(self, path: Path) -> None:
        """Write this config to a YAML file, omitting None and default values."""
        data = self.model_dump(exclude_none=True, exclude_defaults=True)
        # Always include required fields
        data["data"] = self.data
        data["goal"] = self.goal
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
