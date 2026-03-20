"""Tests for the CLI entry point."""

from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from auto_scientist.cli import _next_output_dir, cli
from auto_scientist.state import ExperimentState


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
    @patch("auto_scientist.cli.asyncio.run")
    @patch("auto_scientist.cli.Orchestrator")
    def test_required_options(self, mock_orch, mock_async_run, tmp_path):
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
        assert "config" not in call_kwargs
        mock_async_run.assert_called_once()

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

    @patch("auto_scientist.cli.asyncio.run")
    @patch("auto_scientist.cli.Orchestrator")
    def test_run_command_prints_message_on_increment(self, mock_orch, mock_async_run, tmp_path):
        base = tmp_path / "experiments"
        base.mkdir()
        (base / "state.json").write_text("{}")

        data_file = tmp_path / "data.csv"
        data_file.write_text("a,b\n1,2\n")

        runner = CliRunner()
        result = runner.invoke(cli, [
            "run", "--data", str(data_file), "--goal", "test",
            "--output-dir", str(base),
        ])

        assert result.exit_code == 0
        assert "Previous run detected" in result.output
        call_kwargs = mock_orch.call_args.kwargs
        assert call_kwargs["output_dir"] == tmp_path / "experiments_001"


class TestResumeCommand:
    @patch("auto_scientist.cli.asyncio.run")
    @patch("auto_scientist.cli.Orchestrator")
    def test_loads_state_and_creates_orchestrator(self, mock_orch, mock_async_run, tmp_path):
        state = ExperimentState(domain="auto", goal="g", phase="iteration")
        state_path = tmp_path / "state.json"
        state.save(state_path)

        runner = CliRunner()
        result = runner.invoke(cli, ["resume", "--state", str(state_path)])

        assert result.exit_code == 0
        mock_orch.assert_called_once()
        call_kwargs = mock_orch.call_args.kwargs
        assert call_kwargs["state"].domain == "auto"
        assert "config" not in call_kwargs
