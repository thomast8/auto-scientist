"""Tests for per-agent model and reasoning configuration."""

import logging

import pytest

from auto_scientist.model_config import (
    AgentModelConfig,
    ModelConfig,
    ReasoningConfig,
)


class TestReasoningConfig:
    def test_defaults_to_off(self):
        rc = ReasoningConfig()
        assert rc.level == "off"
        assert rc.budget is None

    def test_legacy_default_migrates_to_off(self, caplog):
        with caplog.at_level(logging.WARNING, logger="auto_scientist.model_config"):
            rc = ReasoningConfig(level="default")
        assert rc.level == "off"
        assert any("deprecated" in r.message for r in caplog.records)

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
        assert cfg.reasoning.level == "off"

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
        assert mc.defaults.reasoning.level == "medium"
        assert mc.resolve("analyst").model == "claude-sonnet-4-6"
        assert mc.resolve("analyst").reasoning.level == "medium"
        assert mc.resolve("scientist").model == "claude-opus-4-6"
        assert mc.resolve("scientist").reasoning.level == "medium"
        assert mc.resolve("coder").model == "claude-sonnet-4-6"
        assert mc.resolve("ingestor").model == "claude-sonnet-4-6"
        assert mc.resolve("report").model == "claude-sonnet-4-6"
        assert mc.summarizer is not None
        assert mc.summarizer.provider == "openai"
        assert mc.summarizer.model == "gpt-5.4-nano"
        assert mc.summarizer.reasoning.level == "off"

    def test_fast_preset(self):
        mc = ModelConfig.builtin_preset("fast")
        assert mc.defaults.model == "claude-haiku-4-5-20251001"
        assert mc.defaults.reasoning.level == "off"
        assert mc.summarizer is not None
        assert mc.summarizer.provider == "openai"
        assert mc.summarizer.model == "gpt-5.4-nano"
        assert mc.summarizer.reasoning.level == "off"

    def test_fast_preset_all_agents_use_haiku_with_off_reasoning(self):
        mc = ModelConfig.builtin_preset("fast")
        for agent in ["analyst", "scientist", "coder", "ingestor", "report"]:
            cfg = mc.resolve(agent)
            assert cfg.model == "claude-haiku-4-5-20251001"
            assert cfg.reasoning.level == "off"

    def test_medium_is_alias_for_default(self):
        default = ModelConfig.builtin_preset("default")
        medium = ModelConfig.builtin_preset("medium")
        assert default.defaults.model == medium.defaults.model
        assert default.defaults.reasoning.level == medium.defaults.reasoning.level
        assert default.resolve("scientist").model == medium.resolve("scientist").model

    def test_nonexistent_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            ModelConfig.builtin_preset("turbo")

    def test_default_preset_has_critics(self):
        mc = ModelConfig.builtin_preset("default")
        assert len(mc.critics) == 2
        assert mc.critics[0].provider == "google"
        assert mc.critics[0].model == "gemini-3.1-pro-preview"
        assert mc.critics[1].provider == "openai"
        assert mc.critics[1].model == "gpt-5.4"


class TestFromToml:
    def test_minimal_config(self, tmp_path):
        toml_file = tmp_path / "models.toml"
        toml_file.write_text('[defaults]\nmodel = "claude-sonnet-4-6"\n')
        mc = ModelConfig.from_toml(toml_file)
        assert mc.defaults.model == "claude-sonnet-4-6"
        assert mc.scientist is None
        assert mc.critics == []

    def test_full_config_round_trip(self, tmp_path):
        toml_file = tmp_path / "models.toml"
        toml_file.write_text("""\
[defaults]
model = "claude-sonnet-4-6"
reasoning = "high"

[agents.scientist]
model = "claude-opus-4-6"
reasoning = "high"

[agents.summarizer]
provider = "openai"
model = "gpt-4o-mini"
""")
        mc = ModelConfig.from_toml(toml_file)
        assert mc.defaults.model == "claude-sonnet-4-6"
        assert mc.defaults.reasoning.level == "high"
        assert mc.scientist is not None
        assert mc.scientist.model == "claude-opus-4-6"
        assert mc.scientist.reasoning.level == "high"
        assert mc.summarizer is not None
        assert mc.summarizer.provider == "openai"

    def test_critic_array(self, tmp_path):
        toml_file = tmp_path / "models.toml"
        toml_file.write_text("""\
[defaults]
model = "claude-sonnet-4-6"

[[agents.critic]]
provider = "openai"
model = "o4-mini"
reasoning = "high"

[[agents.critic]]
provider = "google"
model = "gemini-2.5-pro"
""")
        mc = ModelConfig.from_toml(toml_file)
        assert len(mc.critics) == 2
        assert mc.critics[0].provider == "openai"
        assert mc.critics[0].model == "o4-mini"
        assert mc.critics[0].reasoning.level == "high"
        assert mc.critics[1].provider == "google"

    def test_reasoning_budget_in_toml(self, tmp_path):
        toml_file = tmp_path / "models.toml"
        toml_file.write_text("""\
[defaults]
model = "claude-sonnet-4-6"

[agents.analyst]
model = "claude-sonnet-4-6"
reasoning = { level = "high", budget = 4096 }
""")
        mc = ModelConfig.from_toml(toml_file)
        assert mc.analyst is not None
        assert mc.analyst.reasoning.level == "high"
        assert mc.analyst.reasoning.budget == 4096

    def test_missing_agents_resolve_to_defaults(self, tmp_path):
        toml_file = tmp_path / "models.toml"
        toml_file.write_text("""\
[defaults]
model = "claude-haiku-4-5"
reasoning = "low"

[agents.scientist]
model = "claude-opus-4-6"
""")
        mc = ModelConfig.from_toml(toml_file)
        # Analyst not in TOML, should resolve to defaults
        cfg = mc.resolve("analyst")
        assert cfg.model == "claude-haiku-4-5"
        assert cfg.reasoning.level == "low"
        # Scientist has override
        cfg = mc.resolve("scientist")
        assert cfg.model == "claude-opus-4-6"


class TestAgentFieldSync:
    def test_experiment_models_fields_match_model_config_agent_fields(self):
        """ExperimentModelsConfig fields should stay in sync with ModelConfig._AGENT_FIELDS."""
        from auto_scientist.experiment_config import ExperimentModelsConfig

        emc_fields = set(ExperimentModelsConfig.model_fields) - {"critics"}
        assert emc_fields == ModelConfig._AGENT_FIELDS


class TestFromExperimentConfig:
    def test_preset_only(self):
        from auto_scientist.experiment_config import ExperimentConfig

        exp = ExperimentConfig(data="data.csv", goal="test", preset="fast")
        mc = ModelConfig.from_experiment_config(exp)
        assert mc.defaults.model == "claude-haiku-4-5-20251001"

    def test_preset_with_agent_override(self):
        from auto_scientist.experiment_config import ExperimentConfig

        exp = ExperimentConfig(
            data="data.csv",
            goal="test",
            preset="fast",
            models={
                "scientist": {"model": "claude-opus-4-6", "reasoning": "high"},
            },
        )
        mc = ModelConfig.from_experiment_config(exp)
        # Scientist overridden
        assert mc.scientist.model == "claude-opus-4-6"
        assert mc.scientist.reasoning.level == "high"
        # Other agents still use fast preset defaults
        assert mc.defaults.model == "claude-haiku-4-5-20251001"

    def test_summaries_false_wins(self):
        from auto_scientist.experiment_config import ExperimentConfig

        exp = ExperimentConfig(
            data="data.csv",
            goal="test",
            summaries=False,
            models={
                "summarizer": {"provider": "openai", "model": "gpt-5.4-nano"},
            },
        )
        mc = ModelConfig.from_experiment_config(exp)
        assert mc.summarizer is None

    def test_critics_override_replaces_preset(self):
        from auto_scientist.experiment_config import ExperimentConfig

        exp = ExperimentConfig(
            data="data.csv",
            goal="test",
            preset="default",
            models={
                "critics": [
                    {"provider": "openai", "model": "gpt-5.4", "reasoning": "high"},
                ],
            },
        )
        mc = ModelConfig.from_experiment_config(exp)
        # YAML critics replace preset critics entirely
        assert len(mc.critics) == 1
        assert mc.critics[0].model == "gpt-5.4"
        assert mc.critics[0].reasoning.level == "high"

    def test_no_models_uses_preset_as_is(self):
        from auto_scientist.experiment_config import ExperimentConfig

        exp = ExperimentConfig(data="data.csv", goal="test", preset="high")
        mc = ModelConfig.from_experiment_config(exp)
        assert mc.resolve("scientist").model == "claude-opus-4-6"
        assert mc.resolve("scientist").reasoning.level == "high"
        assert len(mc.critics) == 2
