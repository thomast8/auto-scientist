"""Tests for the CLI entry point."""

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from auto_scientist.cli import cli, load_domain_config
from auto_scientist.config import DomainConfig
from auto_scientist.state import ExperimentState


class TestLoadDomainConfig:
    @patch("auto_scientist.cli.importlib.import_module")
    def test_loads_config_and_knowledge(self, mock_import):
        mock_config_mod = MagicMock()
        mock_config_mod.TEST_CONFIG = DomainConfig(
            name="test", description="Test", data_paths=[],
        )
        mock_prompts_mod = MagicMock()
        mock_prompts_mod.TEST_DOMAIN_KNOWLEDGE = "domain knowledge text"

        def import_side_effect(name):
            if name == "domains.test.config":
                return mock_config_mod
            if name == "domains.test.prompts":
                return mock_prompts_mod
            raise ModuleNotFoundError(name)

        mock_import.side_effect = import_side_effect

        config = load_domain_config("test")

        assert config.name == "test"
        assert config.domain_knowledge == "domain knowledge text"

    @patch("auto_scientist.cli.importlib.import_module")
    def test_missing_prompts_module_uses_empty_knowledge(self, mock_import):
        mock_config_mod = MagicMock()
        mock_config_mod.TEST_CONFIG = DomainConfig(
            name="test", description="Test", data_paths=[],
        )

        def import_side_effect(name):
            if name == "domains.test.config":
                return mock_config_mod
            raise ModuleNotFoundError(name)

        mock_import.side_effect = import_side_effect

        config = load_domain_config("test")

        assert config.domain_knowledge == ""


class TestStatusCommand:
    def test_displays_state_info(self, tmp_path):
        state = ExperimentState(
            domain="spo2", goal="test", phase="iteration",
            iteration=5, best_version="v03", best_score=75,
        )
        state_path = tmp_path / "state.json"
        state.save(state_path)

        runner = CliRunner()
        result = runner.invoke(cli, ["status", "--state", str(state_path)])

        assert result.exit_code == 0
        assert "spo2" in result.output
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


class TestResumeCommand:
    @patch("auto_scientist.cli.asyncio.run")
    @patch("auto_scientist.cli.Orchestrator")
    def test_loads_state_and_creates_orchestrator(self, mock_orch, mock_async_run, tmp_path):
        # Use "auto" domain to skip load_domain_config
        state = ExperimentState(domain="auto", goal="g", phase="iteration")
        state_path = tmp_path / "state.json"
        state.save(state_path)

        runner = CliRunner()
        result = runner.invoke(cli, ["resume", "--state", str(state_path)])

        assert result.exit_code == 0
        mock_orch.assert_called_once()
        call_kwargs = mock_orch.call_args.kwargs
        assert call_kwargs["state"].domain == "auto"
