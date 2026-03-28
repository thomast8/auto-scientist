"""Per-agent model and reasoning configuration.

Supports TOML config files, built-in presets, and a unified reasoning
abstraction that maps to Anthropic/OpenAI/Google native APIs.
"""

from __future__ import annotations

import logging
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal

if TYPE_CHECKING:
    from auto_scientist.experiment_config import ExperimentConfig

from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)


class ReasoningConfig(BaseModel):
    """Unified reasoning config that maps to any provider's native API."""

    level: Literal["off", "minimal", "low", "medium", "high", "max"] = "off"
    budget: int | None = None

    @field_validator("level", mode="before")
    @classmethod
    def _migrate_default_level(cls, v):
        """Accept legacy 'default' and migrate to 'off'."""
        if v == "default":
            logger.warning("ReasoningConfig level='default' is deprecated, use 'off' instead")
            return "off"
        return v


class AgentModelConfig(BaseModel):
    """Configuration for a single agent's model and reasoning."""

    provider: Literal["anthropic", "openai", "google"] = "anthropic"
    model: str = Field(min_length=1)
    reasoning: ReasoningConfig = ReasoningConfig()

    @field_validator("reasoning", mode="before")
    @classmethod
    def _parse_reasoning_shorthand(cls, v):
        """Accept a plain string like 'high' as shorthand for ReasoningConfig(level='high')."""
        if isinstance(v, str):
            return ReasoningConfig(level=v)
        return v


BUILTIN_PRESETS: dict[str, dict] = {
    # Smoke tests, cheapest possible, quality not important
    "turbo": {
        "defaults": {"model": "claude-haiku-4-5-20251001", "reasoning": "off"},
        "summarizer": {"provider": "openai", "model": "gpt-5.4-nano", "reasoning": "off"},
        "critics": [
            {"provider": "google", "model": "gemini-3-flash-preview", "reasoning": "off"},
            {"provider": "google", "model": "gemini-3-flash-preview", "reasoning": "off"},
        ],
    },
    # Quick but competent: scientist upgraded so plans are usable
    "fast": {
        "defaults": {"model": "claude-haiku-4-5-20251001", "reasoning": "low"},
        "scientist": {"model": "claude-sonnet-4-6", "reasoning": "low"},
        "summarizer": {"provider": "openai", "model": "gpt-5.4-nano", "reasoning": "off"},
        "critics": [
            {"provider": "openai", "model": "gpt-5.4-mini", "reasoning": "low"},
            {"provider": "anthropic", "model": "claude-haiku-4-5-20251001", "reasoning": "off"},
        ],
    },
    # Balanced quality/cost
    "default": {
        "defaults": {"model": "claude-sonnet-4-6", "reasoning": "medium"},
        "scientist": {"model": "claude-opus-4-6", "reasoning": "medium"},
        "summarizer": {"provider": "openai", "model": "gpt-5.4-nano", "reasoning": "off"},
        "critics": [
            {"provider": "openai", "model": "gpt-5.4", "reasoning": "medium"},
            {"provider": "anthropic", "model": "claude-sonnet-4-6", "reasoning": "medium"},
        ],
    },
    # High quality: analyst upgraded to Opus for deeper observations
    "high": {
        "defaults": {"model": "claude-sonnet-4-6", "reasoning": "high"},
        "analyst": {"model": "claude-opus-4-6", "reasoning": "medium"},
        "scientist": {"model": "claude-opus-4-6", "reasoning": "high"},
        "summarizer": {"provider": "openai", "model": "gpt-5.4-nano", "reasoning": "off"},
        "critics": [
            {"provider": "openai", "model": "gpt-5.4", "reasoning": "high"},
            {"provider": "anthropic", "model": "claude-sonnet-4-6", "reasoning": "high"},
        ],
    },
    # Best quality, but coder/ingestor/report stay on Sonnet (they're high-volume)
    "max": {
        "defaults": {"model": "claude-opus-4-6", "reasoning": "high"},
        "scientist": {"model": "claude-opus-4-6", "reasoning": "max"},
        "coder": {"model": "claude-sonnet-4-6", "reasoning": "high"},
        "ingestor": {"model": "claude-sonnet-4-6", "reasoning": "high"},
        "report": {"model": "claude-sonnet-4-6", "reasoning": "high"},
        "summarizer": {"provider": "openai", "model": "gpt-5.4-mini", "reasoning": "off"},
        "critics": [
            {"provider": "openai", "model": "gpt-5.4", "reasoning": "max"},
            {"provider": "anthropic", "model": "claude-sonnet-4-6", "reasoning": "max"},
        ],
    },
}
BUILTIN_PRESETS["medium"] = BUILTIN_PRESETS["default"]


CC_EFFORT_MAP: dict[str, str] = {
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "max": "max",
}


def reasoning_to_cc_extra_args(reasoning: ReasoningConfig) -> dict[str, str | None]:
    """Convert a ReasoningConfig to Claude Code CLI extra_args.

    Returns empty dict for 'off' (no thinking).
    """
    effort = CC_EFFORT_MAP.get(reasoning.level)
    if effort:
        return {"--effort": effort}
    return {}


class ModelConfig(BaseModel):
    """Top-level model configuration loaded from TOML or presets."""

    model_config = ConfigDict(validate_assignment=True)

    defaults: AgentModelConfig
    analyst: AgentModelConfig | None = None
    scientist: AgentModelConfig | None = None
    coder: AgentModelConfig | None = None
    ingestor: AgentModelConfig | None = None
    report: AgentModelConfig | None = None
    summarizer: AgentModelConfig | None = None
    critics: list[AgentModelConfig] = []

    _AGENT_FIELDS: ClassVar[set[str]] = {
        "analyst",
        "scientist",
        "coder",
        "ingestor",
        "report",
        "summarizer",
    }

    def resolve(self, agent: str) -> AgentModelConfig:
        """Return effective config: agent-specific if set, otherwise defaults.

        Raises ValueError for 'critics' (use model_config.critics directly).
        """
        if agent == "critics":
            raise ValueError("Use model_config.critics directly for critic configs")
        if agent not in self._AGENT_FIELDS:
            raise ValueError(f"Unknown agent: {agent!r}")
        override: AgentModelConfig | None = getattr(self, agent, None)
        if override is not None:
            return override
        return self.defaults

    @classmethod
    def from_experiment_config(cls, exp_config: ExperimentConfig) -> ModelConfig:
        """Build a ModelConfig from an ExperimentConfig.

        Loads the preset, then layers per-agent model overrides from the
        YAML models block on top. summaries=False always nullifies the summarizer.
        """
        mc = cls.builtin_preset(exp_config.preset)

        if exp_config.models is not None:
            overrides = exp_config.models
            for field in cls._AGENT_FIELDS:
                agent_override = getattr(overrides, field, None)
                if agent_override is not None:
                    setattr(mc, field, agent_override)
            if overrides.critics:
                mc.critics = list(overrides.critics)

        if not exp_config.summaries:
            mc.summarizer = None

        return mc

    @classmethod
    def builtin_preset(cls, name: str) -> ModelConfig:
        """Return a built-in preset by name."""
        if name not in BUILTIN_PRESETS:
            raise ValueError(f"Unknown preset: {name!r}. Available: {list(BUILTIN_PRESETS)}")
        return cls._from_dict(BUILTIN_PRESETS[name])

    @classmethod
    def from_toml(cls, path: Path) -> ModelConfig:
        """Load config from a TOML file."""
        with open(path, "rb") as f:
            raw = tomllib.load(f)
        return cls._from_dict(raw)

    @classmethod
    def _from_dict(cls, raw: dict) -> ModelConfig:
        """Build a ModelConfig from a raw dict (from TOML or preset)."""
        kwargs: dict = {}
        kwargs["defaults"] = AgentModelConfig.model_validate(raw["defaults"])

        agents = raw.get("agents", {})
        # Top-level agent keys (from presets) merge with agents block (from TOML)
        for field in cls._AGENT_FIELDS:
            agent_data = agents.get(field) or raw.get(field)
            if agent_data:
                kwargs[field] = AgentModelConfig.model_validate(agent_data)

        # Critics: TOML uses singular 'critic' array, presets may use 'critics'
        critic_data = agents.get("critic", []) or raw.get("critics", [])
        if critic_data:
            kwargs["critics"] = [AgentModelConfig.model_validate(c) for c in critic_data]

        return cls(**kwargs)
