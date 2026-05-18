"""Per-agent model and reasoning configuration.

Supports TOML config files, built-in presets, and a unified reasoning
abstraction that maps to Anthropic/OpenAI/Google native APIs.
"""

from __future__ import annotations

import logging
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal, cast

if TYPE_CHECKING:
    from auto_scientist.experiment_config import ExperimentConfig

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

ProviderName = Literal["anthropic", "openai", "google"]
_PROVIDER_NAMES: frozenset[str] = frozenset(("anthropic", "openai", "google"))


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

    model_config = ConfigDict(validate_assignment=True)

    provider: ProviderName = "openai"
    model: str = Field(min_length=1)
    reasoning: ReasoningConfig = ReasoningConfig()
    mode: Literal["sdk", "api"] = "sdk"

    @model_validator(mode="before")
    @classmethod
    def _infer_provider_from_model_name(cls, values):
        """Keep old config files working when the model name names a provider."""
        if not isinstance(values, dict) or values.get("provider") is not None:
            return values
        model = values.get("model")
        if isinstance(model, str):
            inferred_provider: str | None = None
            if model.startswith("claude-"):
                inferred_provider = "anthropic"
            elif model.startswith("gemini-"):
                inferred_provider = "google"
            if inferred_provider is not None:
                updated = dict(values)
                updated["provider"] = inferred_provider
                return updated
        return values

    @field_validator("reasoning", mode="before")
    @classmethod
    def _parse_reasoning_shorthand(cls, v):
        """Accept a plain string like 'high' as shorthand for ReasoningConfig(level='high')."""
        if isinstance(v, str):
            return ReasoningConfig(level=v)
        return v


_ANTHROPIC_COMPAT_PRESETS: dict[str, dict] = {
    # Smoke tests: latest fast Anthropic model + GPT-5.5 family sidecars.
    "turbo": {
        "defaults": {"model": "claude-haiku-4-5-20251001", "reasoning": "off"},
        "summarizer": {
            "provider": "openai",
            "model": "gpt-5.5-nano",
            "reasoning": "off",
            "mode": "api",
        },
        "critics": [
            {"provider": "anthropic", "model": "claude-haiku-4-5-20251001", "reasoning": "off"},
            {"provider": "openai", "model": "gpt-5.5", "reasoning": "off"},
        ],
    },
    # Quick but competent: latest fast defaults, stronger planner.
    "fast": {
        "defaults": {"model": "claude-haiku-4-5-20251001", "reasoning": "low"},
        "scientist": {"model": "claude-sonnet-4-6", "reasoning": "low"},
        "summarizer": {
            "provider": "openai",
            "model": "gpt-5.5-nano",
            "reasoning": "off",
            "mode": "api",
        },
        "critics": [
            {"provider": "openai", "model": "gpt-5.5-mini", "reasoning": "low"},
            {"provider": "anthropic", "model": "claude-haiku-4-5-20251001", "reasoning": "low"},
        ],
    },
    # Balanced quality using latest Claude family models.
    "default": {
        "defaults": {"model": "claude-sonnet-4-6", "reasoning": "medium"},
        "scientist": {"model": "claude-opus-4-7", "reasoning": "medium"},
        "summarizer": {
            "provider": "openai",
            "model": "gpt-5.5-nano",
            "reasoning": "off",
            "mode": "api",
        },
        "critics": [
            {"provider": "openai", "model": "gpt-5.5-mini", "reasoning": "medium"},
            {"provider": "anthropic", "model": "claude-sonnet-4-6", "reasoning": "medium"},
        ],
    },
    # High quality: latest Opus where deeper reasoning pays off.
    "high": {
        "defaults": {"model": "claude-sonnet-4-6", "reasoning": "high"},
        "analyst": {"model": "claude-opus-4-7", "reasoning": "medium"},
        "scientist": {"model": "claude-opus-4-7", "reasoning": "high"},
        "assessor": {"model": "claude-opus-4-7", "reasoning": "medium"},
        "summarizer": {
            "provider": "openai",
            "model": "gpt-5.5-nano",
            "reasoning": "off",
            "mode": "api",
        },
        "critics": [
            {"provider": "openai", "model": "gpt-5.5", "reasoning": "high"},
            {"provider": "anthropic", "model": "claude-opus-4-7", "reasoning": "high"},
        ],
    },
    # Best quality across every core agent.
    "max": {
        "defaults": {"model": "claude-opus-4-7", "reasoning": "high"},
        "scientist": {"model": "claude-opus-4-7", "reasoning": "max"},
        "coder": {"model": "claude-sonnet-4-6", "reasoning": "high"},
        "ingestor": {"model": "claude-sonnet-4-6", "reasoning": "high"},
        "report": {"model": "claude-sonnet-4-6", "reasoning": "high"},
        "summarizer": {
            "provider": "openai",
            "model": "gpt-5.5-nano",
            "reasoning": "off",
            "mode": "api",
        },
        "critics": [
            {"provider": "openai", "model": "gpt-5.5", "reasoning": "max"},
            {"provider": "anthropic", "model": "claude-opus-4-7", "reasoning": "max"},
        ],
    },
}

# Model mapping: Anthropic -> OpenAI equivalents.
# Built-in OpenAI presets stay on the GPT-5.5 model family. SDK agents use
# the full tool-capable model; lightweight API sidecars can use mini/nano.
_OPENAI_PRESETS: dict[str, dict] = {
    "turbo-openai": {
        "defaults": {
            "provider": "openai",
            "model": "gpt-5.5",
            "reasoning": "off",
        },
        "summarizer": {
            "provider": "openai",
            "model": "gpt-5.5-nano",
            "reasoning": "off",
            "mode": "api",
        },
        "critics": [
            {"provider": "openai", "model": "gpt-5.5", "reasoning": "off"},
            {"provider": "openai", "model": "gpt-5.5", "reasoning": "off"},
        ],
    },
    "fast-openai": {
        "defaults": {
            "provider": "openai",
            "model": "gpt-5.5",
            "reasoning": "low",
        },
        "scientist": {"provider": "openai", "model": "gpt-5.5", "reasoning": "low"},
        "summarizer": {
            "provider": "openai",
            "model": "gpt-5.5-nano",
            "reasoning": "off",
            "mode": "api",
        },
        "critics": [
            {"provider": "openai", "model": "gpt-5.5", "reasoning": "low"},
            {"provider": "openai", "model": "gpt-5.5", "reasoning": "low"},
        ],
    },
    "default-openai": {
        "defaults": {
            "provider": "openai",
            "model": "gpt-5.5",
            "reasoning": "medium",
        },
        "scientist": {"provider": "openai", "model": "gpt-5.5", "reasoning": "medium"},
        "summarizer": {
            "provider": "openai",
            "model": "gpt-5.5-nano",
            "reasoning": "off",
            "mode": "api",
        },
        "critics": [
            {"provider": "openai", "model": "gpt-5.5", "reasoning": "medium"},
            {"provider": "openai", "model": "gpt-5.5", "reasoning": "medium"},
        ],
    },
    "high-openai": {
        "defaults": {
            "provider": "openai",
            "model": "gpt-5.5",
            "reasoning": "high",
        },
        "analyst": {"provider": "openai", "model": "gpt-5.5", "reasoning": "medium"},
        "scientist": {"provider": "openai", "model": "gpt-5.5", "reasoning": "high"},
        "assessor": {"provider": "openai", "model": "gpt-5.5", "reasoning": "medium"},
        "summarizer": {
            "provider": "openai",
            "model": "gpt-5.5-nano",
            "reasoning": "off",
            "mode": "api",
        },
        "critics": [
            {"provider": "openai", "model": "gpt-5.5", "reasoning": "high"},
            {"provider": "openai", "model": "gpt-5.5", "reasoning": "high"},
        ],
    },
    "max-openai": {
        "defaults": {
            "provider": "openai",
            "model": "gpt-5.5",
            "reasoning": "high",
        },
        "scientist": {"provider": "openai", "model": "gpt-5.5", "reasoning": "max"},
        "coder": {"provider": "openai", "model": "gpt-5.5", "reasoning": "high"},
        "ingestor": {"provider": "openai", "model": "gpt-5.5", "reasoning": "high"},
        "report": {"provider": "openai", "model": "gpt-5.5", "reasoning": "high"},
        "summarizer": {
            "provider": "openai",
            "model": "gpt-5.5-nano",
            "reasoning": "off",
            "mode": "api",
        },
        "critics": [
            {"provider": "openai", "model": "gpt-5.5", "reasoning": "max"},
            {"provider": "openai", "model": "gpt-5.5", "reasoning": "max"},
        ],
    },
}
_OPENAI_PRESETS["medium-openai"] = _OPENAI_PRESETS["default-openai"]

_ANTHROPIC_PRESETS = {f"{name}-anthropic": cfg for name, cfg in _ANTHROPIC_COMPAT_PRESETS.items()}
_OPENAI_BASE_PRESETS = {name.removesuffix("-openai"): cfg for name, cfg in _OPENAI_PRESETS.items()}

BUILTIN_PRESETS = {
    **_OPENAI_BASE_PRESETS,
    **_OPENAI_PRESETS,
    **_ANTHROPIC_PRESETS,
}
BUILTIN_PRESETS["medium"] = BUILTIN_PRESETS["default"]
BUILTIN_PRESETS["medium-openai"] = BUILTIN_PRESETS["default-openai"]
BUILTIN_PRESETS["medium-anthropic"] = BUILTIN_PRESETS["default-anthropic"]


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
        return {"effort": effort}
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
    assessor: AgentModelConfig | None = None
    critics: list[AgentModelConfig] = []

    # ModelConfig field names known to `.resolve()`. Populated at app
    # bootstrap via `install_agent_fields` (which `auto_core.roles.install`
    # calls through from the app's RoleRegistry). Defaults to the
    # auto-scientist set so tests that import ModelConfig without first
    # installing a registry keep working.
    _AGENT_FIELDS: ClassVar[set[str]] = {
        "analyst",
        "scientist",
        "coder",
        "ingestor",
        "report",
        "summarizer",
        "assessor",
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

        If exp_config.provider is set, explicit provider selection wins over
        any suffix already present on the preset name.
        """
        provider = getattr(exp_config, "provider", None)
        mc = cls.builtin_preset_for_provider(exp_config.preset, provider)

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
    def builtin_preset_for_provider(
        cls,
        name: str = "default",
        provider: str | None = None,
    ) -> ModelConfig:
        """Return a built-in preset after applying an explicit provider.

        Unsuffixed preset names are OpenAI defaults. If a caller supplies a
        provider explicitly, that choice wins over any suffix already present
        on ``name``. For example, ``default-anthropic`` with provider
        ``openai`` resolves to ``default``.
        """
        normalized_provider = cls._normalize_provider(provider)
        preset_name = cls._preset_name_for_provider(name, normalized_provider)
        mc = cls.builtin_preset(preset_name)
        if (
            normalized_provider
            and mc.defaults.provider != normalized_provider
            and not preset_name.endswith(f"-{normalized_provider}")
        ):
            mc.defaults = mc.defaults.model_copy(update={"provider": normalized_provider})
        return mc

    @staticmethod
    def _normalize_provider(provider: str | None) -> ProviderName | None:
        if provider is None:
            return None
        if provider not in _PROVIDER_NAMES:
            available = sorted(_PROVIDER_NAMES)
            raise ValueError(f"Unknown provider: {provider!r}. Available: {available}")
        return cast("ProviderName", provider)

    @staticmethod
    def _preset_name_for_provider(
        name: str,
        provider: ProviderName | None,
    ) -> str:
        if provider is None:
            return name
        base = name
        for suffix_provider in ("openai", "anthropic"):
            suffix = f"-{suffix_provider}"
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        if provider == "openai" and base in BUILTIN_PRESETS:
            return base
        variant = f"{base}-{provider}"
        if variant in BUILTIN_PRESETS:
            return variant
        return base if base in BUILTIN_PRESETS else name

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


def install_agent_fields(fields: frozenset[str] | set[str]) -> None:
    """Override `ModelConfig._AGENT_FIELDS` from a RoleRegistry.

    `auto_core.roles.install` calls this at app bootstrap. Accepts an empty
    set as a no-op (i.e. keep the default scientist field set) so apps that
    only need the legacy field names can pass `frozenset()`.
    """
    if not fields:
        return
    ModelConfig._AGENT_FIELDS = set(fields)
