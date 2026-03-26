"""Tests for the CLI entry point."""

from unittest.mock import patch

from click.testing import CliRunner

from auto_scientist.cli import cli
from auto_scientist.experiment_store import next_output_dir
from auto_scientist.model_config import ModelConfig
from auto_scientist.state import ExperimentState


class TestBareInvocation:
    @patch("auto_scientist.cli.PipelineApp")
    def test_no_subcommand_launches_ui(self, mock_app_cls):
        runner = CliRunner()
        result = runner.invoke(cli, [])
        assert result.exit_code == 0
        mock_app_cls.assert_called_once_with()
        mock_app_cls.return_value.run.assert_called_once()

    @patch("auto_scientist.cli.PipelineApp")
    def test_ui_subcommand_launches_ui(self, mock_app_cls):
        runner = CliRunner()
        result = runner.invoke(cli, ["ui"])
        assert result.exit_code == 0
        mock_app_cls.assert_called_once_with()
        mock_app_cls.return_value.run.assert_called_once()


class TestStatusCommand:
    def test_displays_state_info(self, tmp_path):
        state = ExperimentState(
            domain="auto", goal="test", phase="iteration",
            iteration=5, best_version="v03", best_score=75,
        )
        state_path = tmp_path / "state.json"
        state.save(state_path)

        runner = CliRunner()
        result = runner.invoke(cli, ["status", "--state", str(state_path)])

        assert result.exit_code == 0
        assert "auto" in result.output
        assert "iteration" in result.output
        assert "5" in result.output
        assert "v03" in result.output
        assert "75" in result.output


class TestRunCommand:
    @patch("auto_scientist.cli.PipelineApp")
    @patch("auto_scientist.cli.Orchestrator")
    def test_required_options(self, mock_orch, mock_app_cls, tmp_path):
        data_file = tmp_path / "data.csv"
        data_file.write_text("a,b\n1,2\n")

        runner = CliRunner()
        result = runner.invoke(cli, [
            "run", "--data", str(data_file), "--goal", "test goal",
        ])

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
        result = runner.invoke(cli, [
            "run", "--data", str(data_file), "--goal", "test",
            "--preset", "fast",
        ])

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
        result = runner.invoke(cli, [
            "run", "--data", str(data_file), "--goal", "test",
            "--config", str(config_file),
        ])

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
        result = runner.invoke(cli, [
            "run", "--data", str(data_file), "--goal", "test",
            "--config", str(config_file), "--preset", "fast",
        ])

        assert result.exit_code != 0

    @patch("auto_scientist.cli.PipelineApp")
    @patch("auto_scientist.cli.Orchestrator")
    def test_no_summaries_disables_summarizer(self, mock_orch, mock_app_cls, tmp_path):
        data_file = tmp_path / "data.csv"
        data_file.write_text("a,b\n1,2\n")

        runner = CliRunner()
        result = runner.invoke(cli, [
            "run", "--data", str(data_file), "--goal", "test",
            "--no-summaries",
        ])

        assert result.exit_code == 0
        mc = mock_orch.call_args.kwargs["model_config"]
        assert mc.summarizer is None

    @patch("auto_scientist.cli.PipelineApp")
    @patch("auto_scientist.cli.Orchestrator")
    def test_default_has_summarizer(self, mock_orch, mock_app_cls, tmp_path):
        data_file = tmp_path / "data.csv"
        data_file.write_text("a,b\n1,2\n")

        runner = CliRunner()
        result = runner.invoke(cli, [
            "run", "--data", str(data_file), "--goal", "test",
        ])

        assert result.exit_code == 0
        mc = mock_orch.call_args.kwargs["model_config"]
        assert mc.summarizer is not None
        assert mc.summarizer.model == "gpt-5.4-nano"


class TestNextOutputDir:
    def test_returns_base_when_no_state_exists(self, tmp_path):
        base = tmp_path / "experiments"
        base.mkdir()
        assert next_output_dir(base) == base

    def test_returns_base_when_dir_does_not_exist(self, tmp_path):
        base = tmp_path / "experiments"
        assert next_output_dir(base) == base

    def test_increments_when_state_exists(self, tmp_path):
        base = tmp_path / "experiments"
        base.mkdir()
        (base / "state.json").write_text("{}")
        assert next_output_dir(base) == tmp_path / "experiments_001"

    def test_skips_existing_numbered_dirs(self, tmp_path):
        base = tmp_path / "experiments"
        base.mkdir()
        (base / "state.json").write_text("{}")

        d1 = tmp_path / "experiments_001"
        d1.mkdir()
        (d1 / "state.json").write_text("{}")

        assert next_output_dir(base) == tmp_path / "experiments_002"

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

        assert next_output_dir(base) == tmp_path / "experiments_003"


class TestRunCommandOptions:
    @patch("auto_scientist.cli.PipelineApp")
    @patch("auto_scientist.cli.Orchestrator")
    def test_passes_max_iterations(self, mock_orch, mock_app_cls, tmp_path):
        data_file = tmp_path / "data.csv"
        data_file.write_text("a,b\n1,2\n")

        runner = CliRunner()
        runner.invoke(cli, [
            "run", "--data", str(data_file), "--goal", "test",
            "--max-iterations", "10",
        ])
        call_kwargs = mock_orch.call_args.kwargs
        assert call_kwargs["max_iterations"] == 10

    @patch("auto_scientist.cli.PipelineApp")
    @patch("auto_scientist.cli.Orchestrator")
    def test_passes_schedule(self, mock_orch, mock_app_cls, tmp_path):
        data_file = tmp_path / "data.csv"
        data_file.write_text("a,b\n1,2\n")

        runner = CliRunner()
        runner.invoke(cli, [
            "run", "--data", str(data_file), "--goal", "test",
            "--schedule", "22:00-06:00",
        ])
        call_kwargs = mock_orch.call_args.kwargs
        assert call_kwargs["state"].schedule == "22:00-06:00"

    @patch("auto_scientist.cli.PipelineApp")
    @patch("auto_scientist.cli.Orchestrator")
    def test_passes_debate_rounds(self, mock_orch, mock_app_cls, tmp_path):
        data_file = tmp_path / "data.csv"
        data_file.write_text("a,b\n1,2\n")

        runner = CliRunner()
        runner.invoke(cli, [
            "run", "--data", str(data_file), "--goal", "test",
            "--debate-rounds", "3",
        ])
        call_kwargs = mock_orch.call_args.kwargs
        assert call_kwargs["debate_rounds"] == 3

    @patch("auto_scientist.cli.PipelineApp")
    @patch("auto_scientist.cli.Orchestrator")
    def test_no_stream_flag(self, mock_orch, mock_app_cls, tmp_path):
        data_file = tmp_path / "data.csv"
        data_file.write_text("a,b\n1,2\n")

        runner = CliRunner()
        runner.invoke(cli, [
            "run", "--data", str(data_file), "--goal", "test",
            "--no-stream",
        ])
        call_kwargs = mock_orch.call_args.kwargs
        assert call_kwargs["stream"] is False


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
