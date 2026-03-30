"""Tests for the CLI entry point."""

from unittest.mock import MagicMock, patch

import yaml
from click.testing import CliRunner

from auto_scientist.cli import _next_output_dir, cli
from auto_scientist.experiment_config import ExperimentConfig
from auto_scientist.model_config import ModelConfig
from auto_scientist.state import ExperimentState


class TestStatusCommand:
    def test_displays_state_info(self, tmp_path):
        state = ExperimentState(
            domain="auto",
            goal="test",
            phase="iteration",
            iteration=5,
        )
        state_path = tmp_path / "state.json"
        state.save(state_path)

        runner = CliRunner()
        result = runner.invoke(cli, ["status", "--state", str(state_path)])

        assert result.exit_code == 0
        assert "auto" in result.output
        assert "iteration" in result.output
        assert "5" in result.output
        assert "5" in result.output


class TestRunCommand:
    @patch("auto_scientist.cli.PipelineApp")
    @patch("auto_scientist.cli.Orchestrator")
    def test_required_options(self, mock_orch, mock_app_cls, tmp_path):
        data_file = tmp_path / "data.csv"
        data_file.write_text("a,b\n1,2\n")

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "run",
                "--data",
                str(data_file),
                "--goal",
                "test goal",
            ],
        )

        assert result.exit_code == 0
        mock_orch.assert_called_once()
        call_kwargs = mock_orch.call_args.kwargs
        assert call_kwargs["state"].goal == "test goal"
        # Default preset should be used
        mc = call_kwargs["model_config"]
        assert mc.defaults.model == "claude-sonnet-4-6"
        mock_app_cls.assert_called_once()
        mock_app_cls.return_value.run.assert_called_once()

    def test_missing_data_fails(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "--goal", "test"])
        assert result.exit_code != 0

    def test_missing_goal_fails(self, tmp_path):
        data_file = tmp_path / "data.csv"
        data_file.write_text("a,b\n1,2\n")

        runner = CliRunner()
        result = runner.invoke(cli, ["run", "--data", str(data_file)])
        assert result.exit_code != 0


class TestRunCommandPresets:
    @patch("auto_scientist.cli.PipelineApp")
    @patch("auto_scientist.cli.Orchestrator")
    def test_fast_preset(self, mock_orch, mock_app_cls, tmp_path):
        data_file = tmp_path / "data.csv"
        data_file.write_text("a,b\n1,2\n")

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "run",
                "--data",
                str(data_file),
                "--goal",
                "test",
                "--preset",
                "fast",
            ],
        )

        assert result.exit_code == 0
        mc = mock_orch.call_args.kwargs["model_config"]
        assert mc.defaults.model == "claude-haiku-4-5-20251001"

    @patch("auto_scientist.cli.PipelineApp")
    @patch("auto_scientist.cli.Orchestrator")
    def test_config_file(self, mock_orch, mock_app_cls, tmp_path):
        data_file = tmp_path / "data.csv"
        data_file.write_text("a,b\n1,2\n")
        config_file = tmp_path / "models.toml"
        config_file.write_text('[defaults]\nmodel = "claude-opus-4-6"\n')

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "run",
                "--data",
                str(data_file),
                "--goal",
                "test",
                "--config",
                str(config_file),
            ],
        )

        assert result.exit_code == 0
        mc = mock_orch.call_args.kwargs["model_config"]
        assert mc.defaults.model == "claude-opus-4-6"

    @patch("auto_scientist.cli.PipelineApp")
    @patch("auto_scientist.cli.Orchestrator")
    def test_config_and_preset_mutually_exclusive(self, mock_orch, mock_app_cls, tmp_path):
        data_file = tmp_path / "data.csv"
        data_file.write_text("a,b\n1,2\n")
        config_file = tmp_path / "models.toml"
        config_file.write_text('[defaults]\nmodel = "claude-sonnet-4-6"\n')

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "run",
                "--data",
                str(data_file),
                "--goal",
                "test",
                "--config",
                str(config_file),
                "--preset",
                "fast",
            ],
        )

        assert result.exit_code != 0

    @patch("auto_scientist.cli.PipelineApp")
    @patch("auto_scientist.cli.Orchestrator")
    def test_no_summaries_disables_summarizer(self, mock_orch, mock_app_cls, tmp_path):
        data_file = tmp_path / "data.csv"
        data_file.write_text("a,b\n1,2\n")

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "run",
                "--data",
                str(data_file),
                "--goal",
                "test",
                "--no-summaries",
            ],
        )

        assert result.exit_code == 0
        mc = mock_orch.call_args.kwargs["model_config"]
        assert mc.summarizer is None

    @patch("auto_scientist.cli.PipelineApp")
    @patch("auto_scientist.cli.Orchestrator")
    def test_default_has_summarizer(self, mock_orch, mock_app_cls, tmp_path):
        data_file = tmp_path / "data.csv"
        data_file.write_text("a,b\n1,2\n")

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "run",
                "--data",
                str(data_file),
                "--goal",
                "test",
            ],
        )

        assert result.exit_code == 0
        mc = mock_orch.call_args.kwargs["model_config"]
        assert mc.summarizer is not None
        assert mc.summarizer.model == "gpt-5.4-nano"


class TestNextOutputDir:
    def test_returns_base_when_no_state_exists(self, tmp_path):
        base = tmp_path / "experiments"
        base.mkdir()
        assert _next_output_dir(base) == base

    def test_returns_base_when_dir_does_not_exist(self, tmp_path):
        base = tmp_path / "experiments"
        assert _next_output_dir(base) == base

    def test_increments_when_state_exists(self, tmp_path):
        base = tmp_path / "experiments"
        base.mkdir()
        (base / "state.json").write_text("{}")
        assert _next_output_dir(base) == tmp_path / "experiments_001"

    def test_skips_existing_numbered_dirs(self, tmp_path):
        base = tmp_path / "experiments"
        base.mkdir()
        (base / "state.json").write_text("{}")

        d1 = tmp_path / "experiments_001"
        d1.mkdir()
        (d1 / "state.json").write_text("{}")

        assert _next_output_dir(base) == tmp_path / "experiments_002"

    def test_three_sequential_increments(self, tmp_path):
        base = tmp_path / "experiments"
        base.mkdir()
        (base / "state.json").write_text("{}")

        d1 = tmp_path / "experiments_001"
        d1.mkdir()
        (d1 / "state.json").write_text("{}")

        d2 = tmp_path / "experiments_002"
        d2.mkdir()
        (d2 / "state.json").write_text("{}")

        assert _next_output_dir(base) == tmp_path / "experiments_003"


class TestRunCommandOptions:
    @patch("auto_scientist.cli.PipelineApp")
    @patch("auto_scientist.cli.Orchestrator")
    def test_passes_max_iterations(self, mock_orch, mock_app_cls, tmp_path):
        data_file = tmp_path / "data.csv"
        data_file.write_text("a,b\n1,2\n")

        runner = CliRunner()
        runner.invoke(
            cli,
            [
                "run",
                "--data",
                str(data_file),
                "--goal",
                "test",
                "--max-iterations",
                "10",
            ],
        )
        call_kwargs = mock_orch.call_args.kwargs
        assert call_kwargs["max_iterations"] == 10

    @patch("auto_scientist.cli.PipelineApp")
    @patch("auto_scientist.cli.Orchestrator")
    def test_passes_schedule(self, mock_orch, mock_app_cls, tmp_path):
        data_file = tmp_path / "data.csv"
        data_file.write_text("a,b\n1,2\n")

        runner = CliRunner()
        runner.invoke(
            cli,
            [
                "run",
                "--data",
                str(data_file),
                "--goal",
                "test",
                "--schedule",
                "22:00-06:00",
            ],
        )
        call_kwargs = mock_orch.call_args.kwargs
        assert call_kwargs["state"].schedule == "22:00-06:00"

    @patch("auto_scientist.cli.PipelineApp")
    @patch("auto_scientist.cli.Orchestrator")
    def test_passes_debate_rounds(self, mock_orch, mock_app_cls, tmp_path):
        data_file = tmp_path / "data.csv"
        data_file.write_text("a,b\n1,2\n")

        runner = CliRunner()
        runner.invoke(
            cli,
            [
                "run",
                "--data",
                str(data_file),
                "--goal",
                "test",
                "--debate-rounds",
                "3",
            ],
        )
        call_kwargs = mock_orch.call_args.kwargs
        assert call_kwargs["debate_rounds"] == 3


class TestResumeCommand:
    @patch("auto_scientist.cli.PipelineApp")
    @patch("auto_scientist.cli.Orchestrator")
    def test_loads_state_and_creates_orchestrator(self, mock_orch, mock_app_cls, tmp_path):
        state = ExperimentState(domain="auto", goal="g", phase="iteration")
        state_path = tmp_path / "state.json"
        state.save(state_path)

        runner = CliRunner()
        result = runner.invoke(cli, ["resume", "--state", str(state_path)])

        assert result.exit_code == 0
        mock_orch.assert_called_once()
        call_kwargs = mock_orch.call_args.kwargs
        assert call_kwargs["state"].domain == "auto"
        # Should use default preset when no saved config
        mc = call_kwargs["model_config"]
        assert mc.defaults.model == "claude-sonnet-4-6"

    @patch("auto_scientist.cli.PipelineApp")
    @patch("auto_scientist.cli.Orchestrator")
    def test_resume_loads_saved_config(self, mock_orch, mock_app_cls, tmp_path):
        state = ExperimentState(domain="auto", goal="g", phase="iteration")
        state_path = tmp_path / "state.json"
        state.save(state_path)

        # Save a model config as if a previous run wrote it
        mc = ModelConfig.builtin_preset("fast")
        (tmp_path / "model_config.json").write_text(mc.model_dump_json(indent=2))

        runner = CliRunner()
        result = runner.invoke(cli, ["resume", "--state", str(state_path)])

        assert result.exit_code == 0
        loaded_mc = mock_orch.call_args.kwargs["model_config"]
        assert loaded_mc.defaults.model == "claude-haiku-4-5-20251001"


class TestYamlConfig:
    @patch("auto_scientist.cli.PipelineApp")
    @patch("auto_scientist.cli.Orchestrator")
    def test_yaml_config_basic(self, mock_orch, mock_app_cls, tmp_path):
        data_file = tmp_path / "data.csv"
        data_file.write_text("a,b\n1,2\n")
        yaml_file = tmp_path / "experiment.yaml"
        yaml_file.write_text(
            yaml.dump(
                {
                    "data": str(data_file),
                    "goal": "yaml goal",
                    "max_iterations": 10,
                    "preset": "fast",
                }
            )
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["run", "-c", str(yaml_file)])

        assert result.exit_code == 0, result.output
        kw = mock_orch.call_args.kwargs
        assert kw["state"].goal == "yaml goal"
        assert kw["max_iterations"] == 10
        assert kw["model_config"].defaults.model == "claude-haiku-4-5-20251001"

    @patch("auto_scientist.cli.PipelineApp")
    @patch("auto_scientist.cli.Orchestrator")
    def test_yaml_data_path_resolved_relative_to_yaml(self, mock_orch, mock_app_cls, tmp_path):
        """Data path in YAML is relative to the YAML file, not CWD."""
        sub = tmp_path / "domain"
        sub.mkdir()
        data_file = sub / "data.csv"
        data_file.write_text("a,b\n1,2\n")
        yaml_file = sub / "experiment.yaml"
        yaml_file.write_text(yaml.dump({"data": "data.csv", "goal": "test"}))

        runner = CliRunner()
        result = runner.invoke(cli, ["run", "-c", str(yaml_file)])

        assert result.exit_code == 0, result.output
        kw = mock_orch.call_args.kwargs
        assert kw["data_path"] == data_file

    @patch("auto_scientist.cli.PipelineApp")
    @patch("auto_scientist.cli.Orchestrator")
    def test_yaml_with_preset_override(self, mock_orch, mock_app_cls, tmp_path):
        """YAML + --preset is allowed; --preset overrides the YAML's preset."""
        data_file = tmp_path / "data.csv"
        data_file.write_text("a,b\n1,2\n")
        yaml_file = tmp_path / "experiment.yaml"
        yaml_file.write_text(
            yaml.dump(
                {
                    "data": str(data_file),
                    "goal": "test",
                    "preset": "default",
                }
            )
        )

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "run",
                "-c",
                str(yaml_file),
                "--preset",
                "fast",
            ],
        )

        assert result.exit_code == 0, result.output
        mc = mock_orch.call_args.kwargs["model_config"]
        assert mc.defaults.model == "claude-haiku-4-5-20251001"

    @patch("auto_scientist.cli.PipelineApp")
    @patch("auto_scientist.cli.Orchestrator")
    def test_cli_flag_overrides_yaml_value(self, mock_orch, mock_app_cls, tmp_path):
        """CLI --max-iterations overrides YAML max_iterations."""
        data_file = tmp_path / "data.csv"
        data_file.write_text("a,b\n1,2\n")
        yaml_file = tmp_path / "experiment.yaml"
        yaml_file.write_text(
            yaml.dump(
                {
                    "data": str(data_file),
                    "goal": "test",
                    "max_iterations": 10,
                }
            )
        )

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "run",
                "-c",
                str(yaml_file),
                "--max-iterations",
                "30",
            ],
        )

        assert result.exit_code == 0, result.output
        assert mock_orch.call_args.kwargs["max_iterations"] == 30

    @patch("auto_scientist.cli.PipelineApp")
    @patch("auto_scientist.cli.Orchestrator")
    def test_toml_and_preset_still_mutually_exclusive(self, mock_orch, mock_app_cls, tmp_path):
        """TOML config + --preset should still fail."""
        data_file = tmp_path / "data.csv"
        data_file.write_text("a,b\n1,2\n")
        config_file = tmp_path / "models.toml"
        config_file.write_text('[defaults]\nmodel = "claude-sonnet-4-6"\n')

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "run",
                "--data",
                str(data_file),
                "--goal",
                "test",
                "--config",
                str(config_file),
                "--preset",
                "fast",
            ],
        )

        assert result.exit_code != 0

    @patch("auto_scientist.cli.PipelineApp")
    @patch("auto_scientist.cli.Orchestrator")
    def test_yaml_with_model_overrides(self, mock_orch, mock_app_cls, tmp_path):
        data_file = tmp_path / "data.csv"
        data_file.write_text("a,b\n1,2\n")
        yaml_file = tmp_path / "experiment.yaml"
        yaml_file.write_text(
            yaml.dump(
                {
                    "data": str(data_file),
                    "goal": "test",
                    "preset": "fast",
                    "models": {
                        "scientist": {"model": "claude-opus-4-6", "reasoning": "high"},
                    },
                }
            )
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["run", "-c", str(yaml_file)])

        assert result.exit_code == 0, result.output
        mc = mock_orch.call_args.kwargs["model_config"]
        # Fast preset defaults
        assert mc.defaults.model == "claude-haiku-4-5-20251001"
        # But scientist overridden by YAML
        assert mc.scientist.model == "claude-opus-4-6"

    @patch("auto_scientist.cli.PipelineApp")
    @patch("auto_scientist.cli.Orchestrator")
    def test_short_alias_c(self, mock_orch, mock_app_cls, tmp_path):
        """The -c alias works for both YAML and TOML."""
        data_file = tmp_path / "data.csv"
        data_file.write_text("a,b\n1,2\n")
        config_file = tmp_path / "models.toml"
        config_file.write_text('[defaults]\nmodel = "claude-opus-4-6"\n')

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "run",
                "--data",
                str(data_file),
                "--goal",
                "test",
                "-c",
                str(config_file),
            ],
        )

        assert result.exit_code == 0, result.output
        mc = mock_orch.call_args.kwargs["model_config"]
        assert mc.defaults.model == "claude-opus-4-6"


class TestBareCommand:
    @patch("auto_scientist.cli.LaunchApp")
    def test_bare_command_launches_tui(self, mock_launch_cls):
        """Bare `auto-scientist` (no subcommand) should launch LaunchApp."""
        mock_app = MagicMock()
        mock_app.run.return_value = None
        mock_app.result_config = None
        mock_launch_cls.return_value = mock_app

        runner = CliRunner()
        result = runner.invoke(cli, [])

        assert result.exit_code == 0
        mock_launch_cls.assert_called_once_with(prefill=None)
        mock_app.run.assert_called_once()

    @patch("auto_scientist.cli.PipelineApp")
    @patch("auto_scientist.cli.Orchestrator")
    @patch("auto_scientist.cli.LaunchApp")
    def test_bare_command_runs_experiment_on_config(
        self, mock_launch_cls, mock_orch, mock_pipeline_cls, tmp_path
    ):
        """When LaunchApp returns a config, the experiment should run."""
        data_file = tmp_path / "data.csv"
        data_file.write_text("a,b\n1,2\n")

        cfg = ExperimentConfig(data=str(data_file), goal="tui goal")
        mock_app = MagicMock()
        mock_app.run.return_value = cfg
        mock_app.result_config = cfg
        mock_launch_cls.return_value = mock_app

        runner = CliRunner()
        result = runner.invoke(cli, [])

        assert result.exit_code == 0, result.output
        mock_orch.assert_called_once()
        assert mock_orch.call_args.kwargs["state"].goal == "tui goal"

    @patch("auto_scientist.cli.PipelineApp")
    @patch("auto_scientist.cli.Orchestrator")
    def test_subcommand_still_works(self, mock_orch, mock_app_cls, tmp_path):
        """Explicit `run` subcommand should bypass the TUI."""
        data_file = tmp_path / "data.csv"
        data_file.write_text("a,b\n1,2\n")

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "run",
                "--data",
                str(data_file),
                "--goal",
                "direct run",
            ],
        )

        assert result.exit_code == 0
        assert mock_orch.call_args.kwargs["state"].goal == "direct run"
