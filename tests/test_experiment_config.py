"""Tests for the unified YAML experiment configuration."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from auto_scientist.experiment_config import ExperimentConfig


class TestExperimentConfigRequired:
    def test_data_and_goal_required(self):
        cfg = ExperimentConfig(data="data.csv", goal="Find patterns")
        assert cfg.data == "data.csv"
        assert cfg.goal == "Find patterns"

    def test_missing_data_raises(self):
        with pytest.raises(ValidationError):
            ExperimentConfig(goal="Find patterns")

    def test_missing_goal_raises(self):
        with pytest.raises(ValidationError):
            ExperimentConfig(data="data.csv")


class TestExperimentConfigDefaults:
    def test_default_values(self):
        cfg = ExperimentConfig(data="data.csv", goal="test")
        assert cfg.max_iterations == 20
        assert cfg.preset == "default"
        assert cfg.output_dir == "experiments/runs"
        assert cfg.schedule is None
        assert cfg.interactive is False
        assert cfg.verbose is False
        assert cfg.summaries is True
        assert cfg.models is None

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            ExperimentConfig(data="data.csv", goal="test", unknown_field="bad")


class TestExperimentConfigDifficulty:
    def test_default_is_none(self):
        cfg = ExperimentConfig(data="data.csv", goal="test")
        assert cfg.difficulty is None

    @pytest.mark.parametrize("difficulty", ["easy", "medium", "hard", "expert"])
    def test_valid_difficulty_accepted(self, difficulty):
        cfg = ExperimentConfig(data="data.csv", goal="test", difficulty=difficulty)
        assert cfg.difficulty == difficulty

    @pytest.mark.parametrize("bad", ["super", "EASY", "", "normal"])
    def test_invalid_difficulty_rejected(self, bad):
        with pytest.raises(ValidationError):
            ExperimentConfig(data="data.csv", goal="test", difficulty=bad)

    def test_to_yaml_includes_difficulty_when_set(self, tmp_path):
        cfg = ExperimentConfig(data="data.csv", goal="test", difficulty="hard")
        out = tmp_path / "out.yaml"
        cfg.to_yaml(out)
        raw = yaml.safe_load(out.read_text())
        assert raw["difficulty"] == "hard"

    def test_to_yaml_omits_difficulty_when_none(self, tmp_path):
        cfg = ExperimentConfig(data="data.csv", goal="test")
        out = tmp_path / "out.yaml"
        cfg.to_yaml(out)
        raw = yaml.safe_load(out.read_text())
        assert "difficulty" not in raw

    def test_round_trip_with_difficulty(self, tmp_path):
        cfg = ExperimentConfig(data="data.csv", goal="test", difficulty="expert")
        out = tmp_path / "out.yaml"
        cfg.to_yaml(out)
        loaded = ExperimentConfig.from_yaml(out)
        assert loaded.difficulty == "expert"


class TestExperimentConfigPresetValidation:
    @pytest.mark.parametrize("preset", ["turbo", "fast", "default", "medium", "high", "max"])
    def test_valid_presets(self, preset):
        cfg = ExperimentConfig(data="data.csv", goal="test", preset=preset)
        assert cfg.preset == preset

    def test_invalid_preset_raises(self):
        with pytest.raises(ValidationError):
            ExperimentConfig(data="data.csv", goal="test", preset="nonexistent")


class TestExperimentConfigBounds:
    def test_max_iterations_must_be_positive(self):
        with pytest.raises(ValidationError):
            ExperimentConfig(data="data.csv", goal="test", max_iterations=0)

    def test_max_iterations_negative_raises(self):
        with pytest.raises(ValidationError):
            ExperimentConfig(data="data.csv", goal="test", max_iterations=-1)

    def test_empty_data_raises(self):
        with pytest.raises(ValidationError):
            ExperimentConfig(data="", goal="test")

    def test_empty_goal_raises(self):
        with pytest.raises(ValidationError):
            ExperimentConfig(data="data.csv", goal="")

    def test_validate_assignment_catches_bad_preset(self):
        cfg = ExperimentConfig(data="data.csv", goal="test")
        with pytest.raises(ValidationError):
            cfg.preset = "nonexistent"


class TestFromYamlErrors:
    def test_empty_yaml_file(self, tmp_path):
        yaml_file = tmp_path / "empty.yaml"
        yaml_file.write_text("")
        with pytest.raises(ValueError, match="Empty config"):
            ExperimentConfig.from_yaml(yaml_file)

    def test_non_dict_yaml(self, tmp_path):
        yaml_file = tmp_path / "list.yaml"
        yaml_file.write_text("- item1\n- item2\n")
        with pytest.raises(ValueError, match="Expected a YAML mapping"):
            ExperimentConfig.from_yaml(yaml_file)

    def test_malformed_yaml(self, tmp_path):
        yaml_file = tmp_path / "bad.yaml"
        yaml_file.write_text("data: [\n")
        with pytest.raises(ValueError, match="Invalid YAML"):
            ExperimentConfig.from_yaml(yaml_file)

    def test_invalid_schema_yaml(self, tmp_path):
        yaml_file = tmp_path / "bad_schema.yaml"
        yaml_file.write_text(yaml.dump({"data": "x", "goal": "y", "unknown_field": "z"}))
        with pytest.raises(ValueError, match="Invalid experiment config"):
            ExperimentConfig.from_yaml(yaml_file)


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


# Root of the project (tests/ is one level down from repo root)
PROJECT_ROOT = Path(__file__).parent.parent


class TestDomainExperimentYamls:
    @pytest.mark.parametrize(
        "domain",
        [
            "toy_function",
            "spo2",
            "alien_minerals",
            "alloy_design",
            "water_treatment",
            "example_template",
        ],
    )
    def test_domain_yaml_valid_schema(self, domain):
        yaml_path = PROJECT_ROOT / "domains" / domain / "experiment.yaml"
        assert yaml_path.exists(), f"Missing experiment.yaml for domain {domain}"

        cfg = ExperimentConfig.from_yaml(yaml_path)
        assert cfg.data
        assert cfg.goal

    @pytest.mark.parametrize(
        "domain",
        [
            "toy_function",
            "spo2",
            "alien_minerals",
            "alloy_design",
            "water_treatment",
        ],
    )
    def test_domain_yaml_data_path_exists(self, domain):
        yaml_path = PROJECT_ROOT / "domains" / domain / "experiment.yaml"
        cfg = ExperimentConfig.from_yaml(yaml_path)
        resolved = cfg.resolve_data_path(yaml_path.parent)
        assert resolved.exists(), f"Data path {resolved} does not exist for domain {domain}"
