"""Tests for the unified YAML experiment configuration."""

import pytest
import yaml

from auto_scientist.experiment_config import ExperimentConfig


class TestExperimentConfigRequired:
    def test_data_and_goal_required(self):
        cfg = ExperimentConfig(data="data.csv", goal="Find patterns")
        assert cfg.data == "data.csv"
        assert cfg.goal == "Find patterns"

    def test_missing_data_raises(self):
        with pytest.raises(Exception):
            ExperimentConfig(goal="Find patterns")

    def test_missing_goal_raises(self):
        with pytest.raises(Exception):
            ExperimentConfig(data="data.csv")


class TestExperimentConfigDefaults:
    def test_default_values(self):
        cfg = ExperimentConfig(data="data.csv", goal="test")
        assert cfg.max_iterations == 20
        assert cfg.preset == "default"
        assert cfg.debate_rounds == 1
        assert cfg.output_dir == "experiments"
        assert cfg.schedule is None
        assert cfg.interactive is False
        assert cfg.stream is True
        assert cfg.verbose is False
        assert cfg.summaries is True
        assert cfg.models is None

    def test_extra_fields_forbidden(self):
        with pytest.raises(Exception):
            ExperimentConfig(data="data.csv", goal="test", unknown_field="bad")


class TestExperimentConfigPresetValidation:
    @pytest.mark.parametrize("preset", ["fast", "default", "medium", "high", "max"])
    def test_valid_presets(self, preset):
        cfg = ExperimentConfig(data="data.csv", goal="test", preset=preset)
        assert cfg.preset == preset

    def test_invalid_preset_raises(self):
        with pytest.raises(Exception):
            ExperimentConfig(data="data.csv", goal="test", preset="turbo")


class TestExperimentConfigModels:
    def test_models_with_agent_override(self):
        cfg = ExperimentConfig(
            data="data.csv",
            goal="test",
            models={
                "scientist": {"model": "claude-opus-4-6", "reasoning": "high"},
            },
        )
        assert cfg.models is not None
        assert cfg.models.scientist is not None
        assert cfg.models.scientist.model == "claude-opus-4-6"

    def test_models_with_critics(self):
        cfg = ExperimentConfig(
            data="data.csv",
            goal="test",
            models={
                "critics": [
                    {"provider": "google", "model": "gemini-3.1-pro-preview"},
                    {"provider": "openai", "model": "gpt-5.4"},
                ],
            },
        )
        assert cfg.models is not None
        assert len(cfg.models.critics) == 2
        assert cfg.models.critics[0].provider == "google"


class TestFromYaml:
    def test_minimal_yaml(self, tmp_path):
        yaml_file = tmp_path / "experiment.yaml"
        yaml_file.write_text("data: data.csv\ngoal: Find patterns\n")

        cfg = ExperimentConfig.from_yaml(yaml_file)
        assert cfg.data == "data.csv"
        assert cfg.goal == "Find patterns"
        assert cfg.preset == "default"

    def test_full_yaml(self, tmp_path):
        content = {
            "data": "seed/data/spo2.db",
            "goal": "Model SpO2 dynamics",
            "max_iterations": 10,
            "preset": "high",
            "debate_rounds": 2,
            "output_dir": "my_experiments",
            "models": {
                "scientist": {"model": "claude-opus-4-6", "reasoning": "high"},
            },
        }
        yaml_file = tmp_path / "experiment.yaml"
        yaml_file.write_text(yaml.dump(content))

        cfg = ExperimentConfig.from_yaml(yaml_file)
        assert cfg.data == "seed/data/spo2.db"
        assert cfg.goal == "Model SpO2 dynamics"
        assert cfg.max_iterations == 10
        assert cfg.preset == "high"
        assert cfg.debate_rounds == 2
        assert cfg.output_dir == "my_experiments"
        assert cfg.models.scientist.model == "claude-opus-4-6"

    def test_yml_extension(self, tmp_path):
        yaml_file = tmp_path / "experiment.yml"
        yaml_file.write_text("data: data.csv\ngoal: test\n")

        cfg = ExperimentConfig.from_yaml(yaml_file)
        assert cfg.data == "data.csv"


class TestResolveDataPath:
    def test_relative_path_resolved_from_yaml_dir(self, tmp_path):
        cfg = ExperimentConfig(data="seed/data/test.csv", goal="test")
        resolved = cfg.resolve_data_path(tmp_path)
        assert resolved == tmp_path / "seed/data/test.csv"

    def test_absolute_path_unchanged(self, tmp_path):
        abs_path = "/absolute/path/data.csv"
        cfg = ExperimentConfig(data=abs_path, goal="test")
        resolved = cfg.resolve_data_path(tmp_path)
        assert resolved == Path(abs_path)


class TestToYaml:
    def test_round_trip(self, tmp_path):
        cfg = ExperimentConfig(
            data="data.csv",
            goal="Find patterns in the data",
            max_iterations=10,
            preset="fast",
        )
        out_path = tmp_path / "output.yaml"
        cfg.to_yaml(out_path)

        loaded = ExperimentConfig.from_yaml(out_path)
        assert loaded.data == cfg.data
        assert loaded.goal == cfg.goal
        assert loaded.max_iterations == cfg.max_iterations
        assert loaded.preset == cfg.preset

    def test_omits_none_and_default_values(self, tmp_path):
        cfg = ExperimentConfig(data="data.csv", goal="test")
        out_path = tmp_path / "output.yaml"
        cfg.to_yaml(out_path)

        raw = yaml.safe_load(out_path.read_text())
        assert "data" in raw
        assert "goal" in raw
        # None fields should not appear
        assert "schedule" not in raw
        assert "models" not in raw

    def test_models_survive_round_trip(self, tmp_path):
        cfg = ExperimentConfig(
            data="data.csv",
            goal="test",
            models={
                "scientist": {"model": "claude-opus-4-6"},
                "critics": [{"provider": "openai", "model": "gpt-5.4"}],
            },
        )
        out_path = tmp_path / "output.yaml"
        cfg.to_yaml(out_path)

        loaded = ExperimentConfig.from_yaml(out_path)
        assert loaded.models.scientist.model == "claude-opus-4-6"
        assert len(loaded.models.critics) == 1
        assert loaded.models.critics[0].model == "gpt-5.4"


from pathlib import Path
