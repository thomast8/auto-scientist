"""Per-agent model and reasoning configuration.

Supports TOML config files, built-in presets, and a unified reasoning
abstraction that maps to Anthropic/OpenAI/Google native APIs.
"""

import tomllib
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import BaseModel, field_validator


class ReasoningConfig(BaseModel):
    """Unified reasoning config that maps to any provider's native API."""

    level: Literal["default", "off", "minimal", "low", "medium", "high", "max"] = "default"
    budget: int | None = None


class AgentModelConfig(BaseModel):
    """Configuration for a single agent's model and reasoning."""

    provider: Literal["anthropic", "openai", "google"] = "anthropic"
    model: str
    reasoning: ReasoningConfig = ReasoningConfig()

    @field_validator("reasoning", mode="before")
    @classmethod
    def _parse_reasoning_shorthand(cls, v):
        """Accept a plain string like 'high' as shorthand for ReasoningConfig(level='high')."""
        if isinstance(v, str):
            return ReasoningConfig(level=v)
        return v


BUILTIN_PRESETS: dict[str, dict] = {
    "default": {
        "defaults": {"model": "claude-sonnet-4-6"},
        "analyst": {"model": "claude-opus-4-6"},
        "scientist": {"model": "claude-opus-4-6"},
        "summarizer": {"provider": "openai", "model": "gpt-5.4-nano"},
        "critics": [
            {"provider": "google", "model": "gemini-3.1-pro-preview"},
            {"provider": "openai", "model": "gpt-5.4"},
        ],
    },
    "fast": {
        "defaults": {"model": "claude-haiku-4-5-20251001"},
        "summarizer": {"provider": "openai", "model": "gpt-5.4-nano"},
        "critics": [
            {"provider": "google", "model": "gemini-3.1-flash-lite-preview"},
            {"provider": "openai", "model": "gpt-5.4-nano"},
        ],
    },
}


CC_EFFORT_MAP: dict[str, str] = {
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "max": "max",
}


def reasoning_to_cc_extra_args(reasoning: ReasoningConfig) -> dict[str, str | None]:
    """Convert a ReasoningConfig to Claude Code CLI extra_args.

    Returns empty dict for 'default' and 'off' (let CC decide / no thinking).
    """
    effort = CC_EFFORT_MAP.get(reasoning.level)
    if effort:
        return {"--effort": effort}
    return {}


class ModelConfig(BaseModel):
    """Top-level model configuration loaded from TOML or presets."""

    defaults: AgentModelConfig
    analyst: AgentModelConfig | None = None
    scientist: AgentModelConfig | None = None
    coder: AgentModelConfig | None = None
    ingestor: AgentModelConfig | None = None
    report: AgentModelConfig | None = None
    summarizer: AgentModelConfig | None = None
    critics: list[AgentModelConfig] = []

    _AGENT_FIELDS: ClassVar[set[str]] = {
        "analyst", "scientist", "coder", "ingestor", "report", "summarizer",
    }

    def resolve(self, agent: str) -> AgentModelConfig:
        """Return effective config: agent-specific if set, otherwise defaults.

        Raises ValueError for 'critics' (use model_config.critics directly).
        """
        if agent == "critics":
            raise ValueError("Use model_config.critics directly for critic configs")
        if agent not in self._AGENT_FIELDS:
            raise ValueError(f"Unknown agent: {agent!r}")
        override = getattr(self, agent, None)
        if override is not None:
            return override
        return self.defaults

    @classmethod
    def builtin_preset(cls, name: str) -> "ModelConfig":
        """Return a built-in preset by name."""
        if name not in BUILTIN_PRESETS:
            raise ValueError(f"Unknown preset: {name!r}. Available: {list(BUILTIN_PRESETS)}")
        return cls._from_dict(BUILTIN_PRESETS[name])

    @classmethod
    def from_toml(cls, path: Path) -> "ModelConfig":
        """Load config from a TOML file."""
        with open(path, "rb") as f:
            raw = tomllib.load(f)
        return cls._from_dict(raw)

    @classmethod
    def _from_dict(cls, raw: dict) -> "ModelConfig":
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
