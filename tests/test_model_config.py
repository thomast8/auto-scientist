"""Tests for per-agent model and reasoning configuration."""

import pytest

from auto_scientist.model_config import (
    AgentModelConfig,
    ModelConfig,
    ReasoningConfig,
)


class TestReasoningConfig:
    def test_defaults_to_adaptive(self):
        rc = ReasoningConfig()
        assert rc.level == "adaptive"
        assert rc.budget is None

    def test_explicit_level(self):
        rc = ReasoningConfig(level="high")
        assert rc.level == "high"

    def test_with_budget(self):
        rc = ReasoningConfig(level="high", budget=8192)
        assert rc.level == "high"
        assert rc.budget == 8192

    def test_invalid_level_raises(self):
        with pytest.raises(Exception):
            ReasoningConfig(level="turbo")


class TestAgentModelConfig:
    def test_defaults_provider_to_anthropic(self):
        cfg = AgentModelConfig(model="claude-sonnet-4-6")
        assert cfg.provider == "anthropic"
        assert cfg.reasoning.level == "adaptive"

    def test_openai_provider(self):
        cfg = AgentModelConfig(provider="openai", model="o4-mini")
        assert cfg.provider == "openai"

    def test_reasoning_string_shorthand(self):
        """A plain string like 'high' should parse as ReasoningConfig(level='high')."""
        cfg = AgentModelConfig.model_validate(
            {"model": "claude-sonnet-4-6", "reasoning": "high"}
        )
        assert cfg.reasoning.level == "high"
        assert cfg.reasoning.budget is None

    def test_reasoning_dict(self):
        cfg = AgentModelConfig.model_validate(
            {"model": "claude-sonnet-4-6", "reasoning": {"level": "high", "budget": 4096}}
        )
        assert cfg.reasoning.level == "high"
        assert cfg.reasoning.budget == 4096

    def test_reasoning_object(self):
        cfg = AgentModelConfig(
            model="claude-sonnet-4-6",
            reasoning=ReasoningConfig(level="medium"),
        )
        assert cfg.reasoning.level == "medium"


class TestModelConfigResolve:
    def test_resolve_returns_defaults_when_no_override(self):
        mc = ModelConfig(defaults=AgentModelConfig(model="claude-sonnet-4-6"))
        cfg = mc.resolve("analyst")
        assert cfg.model == "claude-sonnet-4-6"

    def test_resolve_returns_override_when_set(self):
        mc = ModelConfig(
            defaults=AgentModelConfig(model="claude-sonnet-4-6"),
            scientist=AgentModelConfig(model="claude-opus-4-6"),
        )
        cfg = mc.resolve("scientist")
        assert cfg.model == "claude-opus-4-6"

    def test_resolve_critics_raises(self):
        mc = ModelConfig(defaults=AgentModelConfig(model="claude-sonnet-4-6"))
        with pytest.raises(ValueError, match="critics"):
            mc.resolve("critics")

    def test_resolve_unknown_agent_raises(self):
        mc = ModelConfig(defaults=AgentModelConfig(model="claude-sonnet-4-6"))
        with pytest.raises(ValueError, match="Unknown agent"):
            mc.resolve("batman")

    def test_resolve_all_agent_fields(self):
        mc = ModelConfig(defaults=AgentModelConfig(model="claude-sonnet-4-6"))
        for agent in ["analyst", "scientist", "coder", "ingestor", "report", "summarizer"]:
            cfg = mc.resolve(agent)
            assert cfg.model == "claude-sonnet-4-6"


class TestBuiltinPresets:
    def test_default_preset(self):
        mc = ModelConfig.builtin_preset("default")
        assert mc.defaults.model == "claude-sonnet-4-6"
        assert mc.summarizer is not None
        assert mc.summarizer.provider == "openai"
        assert mc.summarizer.model == "gpt-4o-mini"

    def test_fast_preset(self):
        mc = ModelConfig.builtin_preset("fast")
        assert mc.defaults.model == "claude-haiku-4-5"
        assert mc.summarizer is not None
        assert mc.summarizer.provider == "openai"
        assert mc.summarizer.model == "gpt-5.4-nano"
        assert mc.summarizer.reasoning.level == "off"

    def test_fast_preset_all_agents_use_haiku(self):
        mc = ModelConfig.builtin_preset("fast")
        for agent in ["analyst", "scientist", "coder", "ingestor", "report"]:
            cfg = mc.resolve(agent)
            assert cfg.model == "claude-haiku-4-5"

    def test_nonexistent_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            ModelConfig.builtin_preset("turbo")

    def test_default_preset_has_no_critics(self):
        mc = ModelConfig.builtin_preset("default")
        assert mc.critics == []
