"""Tests for the CLI entry point."""

from unittest.mock import MagicMock, patch

import yaml
from click.testing import CliRunner

from auto_scientist.cli import _detect_retry_agent, _next_output_dir, _resolve_source, cli
from auto_scientist.experiment_config import ExperimentConfig
from auto_scientist.model_config import ModelConfig
from auto_scientist.state import ExperimentState


class TestStatusCommand:
    def test_displays_state_info(self, tmp_path):
        state = ExperimentState(
            domain="auto",
            goal="test goal",
            phase="iteration",
            iteration=5,
        )
        state_path = tmp_path / "state.json"
        state.save(state_path)

        runner = CliRunner()
        result = runner.invoke(cli, ["status", "--state", str(state_path)])

        assert result.exit_code == 0
        assert "Domain:     auto" in result.output
        assert "Goal:       test goal" in result.output
        assert "Phase:      iteration" in result.output
        assert "Iteration:  5" in result.output

    def test_shows_run_dir_and_data_path(self, tmp_path):
        state = ExperimentState(
            domain="auto",
            goal="test",
            phase="iteration",
            iteration=1,
            data_path="/some/data/dir",
        )
        state.save(tmp_path / "state.json")

        runner = CliRunner()
        result = runner.invoke(cli, ["status", "--from", str(tmp_path)])

        assert result.exit_code == 0
        assert f"Run dir:    {tmp_path}" in result.output
        assert "Data:       /some/data/dir" in result.output

    def test_data_path_hidden_when_absent(self, tmp_path):
        state = ExperimentState(domain="auto", goal="test", phase="iteration", iteration=1)
        state.save(tmp_path / "state.json")

        runner = CliRunner()
        result = runner.invoke(cli, ["status", "--from", str(tmp_path)])

        assert result.exit_code == 0
        assert "Data:" not in result.output

    def test_long_goal_truncated(self, tmp_path):
        long_goal = "x" * 100
        state = ExperimentState(domain="auto", goal=long_goal, phase="iteration", iteration=1)
        state.save(tmp_path / "state.json")

        runner = CliRunner()
        result = runner.invoke(cli, ["status", "--from", str(tmp_path)])

        assert result.exit_code == 0
        assert "x" * 72 + "..." in result.output

    def test_no_versions_line(self, tmp_path):
        """Status should not show a confusing 'Versions' count."""
        state = ExperimentState(domain="auto", goal="test", phase="iteration", iteration=2)
        state.save(tmp_path / "state.json")

        runner = CliRunner()
        result = runner.invoke(cli, ["status", "--from", str(tmp_path)])

        assert result.exit_code == 0
        assert "Versions" not in result.output

    def test_run_status_summary(self, tmp_path):
        """Shows completed/failed counts from version entries."""
        from auto_scientist.state import VersionEntry

        state = ExperimentState(domain="auto", goal="test", phase="iteration", iteration=3)
        state.versions = [
            VersionEntry(version="v00", iteration=0, script_path="s", status="completed"),
            VersionEntry(version="v01", iteration=1, script_path="s", status="completed"),
            VersionEntry(version="v02", iteration=2, script_path="s", status="failed"),
        ]
        state.save(tmp_path / "state.json")

        runner = CliRunner()
        result = runner.invoke(cli, ["status", "--from", str(tmp_path)])

        assert result.exit_code == 0
        assert "Runs:" in result.output
        assert "2 completed" in result.output
        assert "1 failed" in result.output

    def test_dead_ends_hidden_when_zero(self, tmp_path):
        state = ExperimentState(domain="auto", goal="test", phase="iteration", iteration=1)
        state.save(tmp_path / "state.json")

        runner = CliRunner()
        result = runner.invoke(cli, ["status", "--from", str(tmp_path)])

        assert result.exit_code == 0
        assert "Dead ends" not in result.output

    def test_dead_ends_shown_when_present(self, tmp_path):
        state = ExperimentState(
            domain="auto",
            goal="test",
            phase="iteration",
            iteration=2,
            dead_ends=["Tried polynomial fit, R^2 < 0.5"],
        )
        state.save(tmp_path / "state.json")

        runner = CliRunner()
        result = runner.invoke(cli, ["status", "--from", str(tmp_path)])

        assert result.exit_code == 0
        assert "Dead ends:  1" in result.output

    def test_iterations_on_disk(self, tmp_path):
        """Shows per-iteration agent artifacts."""
        state = ExperimentState(domain="auto", goal="test", phase="iteration", iteration=2)
        state.save(tmp_path / "state.json")

        v00 = tmp_path / "v00"
        v00.mkdir()
        (v00 / "analysis.json").write_text("{}")
        (v00 / "plan.json").write_text('{"hypothesis": "explore data"}')
        (v00 / "debate.json").write_text("{}")
        (v00 / "experiment.py").write_text("")

        v01 = tmp_path / "v01"
        v01.mkdir()
        (v01 / "analysis.json").write_text("{}")
        (v01 / "plan.json").write_text('{"hypothesis": "test quadratic"}')

        runner = CliRunner()
        result = runner.invoke(cli, ["status", "--from", str(tmp_path)])

        assert result.exit_code == 0
        assert "analyst, scientist, debate, coder" in result.output
        assert "analyst, scientist" in result.output

    def test_stop_reason_shown(self, tmp_path):
        import json

        state = ExperimentState(domain="auto", goal="test", phase="iteration", iteration=1)
        state.save(tmp_path / "state.json")

        v00 = tmp_path / "v00"
        v00.mkdir()
        (v00 / "analysis.json").write_text("{}")
        (v00 / "plan.json").write_text(
            json.dumps(
                {
                    "hypothesis": "Done",
                    "should_stop": True,
                    "stop_reason": "All criteria met",
                }
            )
        )

        runner = CliRunner()
        result = runner.invoke(cli, ["status", "--from", str(tmp_path)])

        assert result.exit_code == 0
        assert "Stop requested: All criteria met" in result.output

    def test_resume_suggests_next_unrun_agent(self, tmp_path):
        """Resume example should suggest the first agent that hasn't run yet."""
        state = ExperimentState(domain="auto", goal="test", phase="iteration", iteration=2)
        state.save(tmp_path / "state.json")

        v01 = tmp_path / "v01"
        v01.mkdir()
        (v01 / "analysis.json").write_text("{}")
        (v01 / "plan.json").write_text('{"hypothesis": "h"}')

        runner = CliRunner()
        result = runner.invoke(cli, ["status", "--from", str(tmp_path)])

        assert result.exit_code == 0
        assert "--from-agent debate" in result.output

    def test_status_shows_stop_gate_steps(self, tmp_path):
        """Status shows assessment, stop_debate, stop_revision when present."""
        state = ExperimentState(domain="auto", goal="test", phase="iteration", iteration=1)
        state.save(tmp_path / "state.json")

        v00 = tmp_path / "v00"
        v00.mkdir()
        (v00 / "analysis.json").write_text("{}")
        (v00 / "plan.json").write_text('{"should_stop": true}')
        (v00 / "completeness_assessment.json").write_text("{}")
        (v00 / "stop_debate.json").write_text("{}")
        (v00 / "stop_revision_plan.json").write_text("{}")
        (v00 / "debate.json").write_text("{}")

        runner = CliRunner()
        result = runner.invoke(cli, ["status", "--from", str(tmp_path)])

        assert result.exit_code == 0
        assert "assessment" in result.output
        assert "stop_debate" in result.output
        assert "stop_revision" in result.output

    def test_status_shows_revision_step(self, tmp_path):
        """Status shows revision when revision_plan.json exists."""
        state = ExperimentState(domain="auto", goal="test", phase="iteration", iteration=1)
        state.save(tmp_path / "state.json")

        v00 = tmp_path / "v00"
        v00.mkdir()
        (v00 / "analysis.json").write_text("{}")
        (v00 / "plan.json").write_text("{}")
        (v00 / "debate.json").write_text("{}")
        (v00 / "revision_plan.json").write_text("{}")
        (v00 / "experiment.py").write_text("")

        runner = CliRunner()
        result = runner.invoke(cli, ["status", "--from", str(tmp_path)])

        assert result.exit_code == 0
        assert "revision" in result.output
        assert "coder" in result.output

    def test_resume_suggests_revision_when_debate_done(self, tmp_path):
        """When debate exists but revision_plan.json doesn't, suggest revision."""
        state = ExperimentState(domain="auto", goal="test", phase="iteration", iteration=1)
        state.save(tmp_path / "state.json")

        v00 = tmp_path / "v00"
        v00.mkdir()
        (v00 / "analysis.json").write_text("{}")
        (v00 / "plan.json").write_text("{}")
        (v00 / "debate.json").write_text("{}")

        runner = CliRunner()
        result = runner.invoke(cli, ["status", "--from", str(tmp_path)])

        assert result.exit_code == 0
        assert "--from-agent revision" in result.output


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

    @patch("auto_scientist.cli.PipelineApp")
    @patch("auto_scientist.cli.Orchestrator")
    def test_run_persists_max_iterations_on_state(self, mock_orch, mock_app_cls, tmp_path):
        data_file = tmp_path / "data.csv"
        data_file.write_text("a,b\n1,2\n")

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["run", "--data", str(data_file), "--goal", "g", "--max-iterations", "7"],
        )

        assert result.exit_code == 0
        assert mock_orch.call_args.kwargs["state"].max_iterations == 7

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

    @patch("auto_scientist.cli.PipelineApp")
    @patch("auto_scientist.cli.Orchestrator")
    def test_resume_uses_saved_max_iterations(self, mock_orch, mock_app_cls, tmp_path):
        state = ExperimentState(domain="auto", goal="g", phase="iteration", max_iterations=3)
        state.save(tmp_path / "state.json")

        runner = CliRunner()
        result = runner.invoke(cli, ["resume", "--state", str(tmp_path)])

        assert result.exit_code == 0
        assert mock_orch.call_args.kwargs["max_iterations"] == 3
        assert "restored from original run" in result.output

    @patch("auto_scientist.cli.PipelineApp")
    @patch("auto_scientist.cli.Orchestrator")
    def test_resume_explicit_max_iterations_overrides_saved(
        self, mock_orch, mock_app_cls, tmp_path
    ):
        state = ExperimentState(domain="auto", goal="g", phase="iteration", max_iterations=3)
        state.save(tmp_path / "state.json")

        runner = CliRunner()
        result = runner.invoke(cli, ["resume", "--state", str(tmp_path), "--max-iterations", "10"])

        assert result.exit_code == 0
        assert mock_orch.call_args.kwargs["max_iterations"] == 10
        assert "from --max-iterations" in result.output

    @patch("auto_scientist.cli.PipelineApp")
    @patch("auto_scientist.cli.Orchestrator")
    def test_resume_old_state_without_max_iterations_uses_default(
        self, mock_orch, mock_app_cls, tmp_path
    ):
        state = ExperimentState(domain="auto", goal="g", phase="iteration")
        state.save(tmp_path / "state.json")

        runner = CliRunner()
        result = runner.invoke(cli, ["resume", "--state", str(tmp_path)])

        assert result.exit_code == 0
        assert mock_orch.call_args.kwargs["max_iterations"] == 20
        assert "old format" in result.output

    @patch("auto_scientist.cli.PipelineApp")
    @patch("auto_scientist.cli.Orchestrator")
    def test_resume_completed_fork_respects_saved_max(self, mock_orch, mock_app_cls, tmp_path):
        """Forking a completed run should respect the saved max_iterations."""
        state = ExperimentState(
            domain="auto", goal="g", phase="stopped", iteration=3, max_iterations=3
        )
        state.save(tmp_path / "state.json")
        # rewind_run needs a v0 dir to exist
        (tmp_path / "v0").mkdir()

        runner = CliRunner()
        result = runner.invoke(cli, ["resume", "--state", str(tmp_path), "--fork"])

        assert result.exit_code == 0
        assert mock_orch.call_args.kwargs["max_iterations"] == 3
        assert "restored from original run" in result.output

    def test_resume_old_state_past_default_requires_explicit(self, tmp_path):
        """Old state at iteration >= 20 must require explicit --max-iterations."""
        state = ExperimentState(domain="auto", goal="g", phase="iteration", iteration=25)
        state.save(tmp_path / "state.json")

        runner = CliRunner()
        result = runner.invoke(cli, ["resume", "--state", str(tmp_path)])

        assert result.exit_code != 0
        assert "--max-iterations" in result.output

    @patch("auto_scientist.cli.PipelineApp")
    @patch("auto_scientist.cli.Orchestrator")
    def test_resume_persists_max_iterations_to_disk(self, mock_orch, mock_app_cls, tmp_path):
        """Resolved max_iterations should be written to state.json on disk."""
        state = ExperimentState(domain="auto", goal="g", phase="iteration", max_iterations=3)
        state.save(tmp_path / "state.json")

        runner = CliRunner()
        result = runner.invoke(cli, ["resume", "--state", str(tmp_path)])

        assert result.exit_code == 0
        reloaded = ExperimentState.load(tmp_path / "state.json")
        assert reloaded.max_iterations == 3

    @patch("auto_scientist.cli.PipelineApp")
    @patch("auto_scientist.cli.Orchestrator")
    def test_resume_retries_failed_iteration_in_place(self, mock_orch, mock_app_cls, tmp_path):
        """Naked resume of a stopped run with a failed last version should retry from coder."""
        from auto_scientist.state import VersionEntry

        state = ExperimentState(
            domain="auto",
            goal="g",
            phase="stopped",
            iteration=1,
            max_iterations=5,
            consecutive_failures=1,
        )
        state.versions = [
            VersionEntry(
                version="v00",
                iteration=0,
                script_path=str(tmp_path / "v00" / "experiment.py"),
                results_path=None,
                status="failed",
            ),
        ]
        state.save(tmp_path / "state.json")
        # Create version directory with artifacts from earlier agents
        v00 = tmp_path / "v00"
        v00.mkdir()
        (v00 / "analysis.json").write_text('{"data_summary": "test"}')
        (v00 / "plan.json").write_text('{"hypothesis": "test"}')
        (v00 / "debate.json").write_text('{"original_plan": {}, "debate_results": []}')
        (v00 / "revision_plan.json").write_text('{"strategy": "test"}')
        (v00 / "experiment.py").write_text("# failed script")

        runner = CliRunner()
        result = runner.invoke(cli, ["resume", "--state", str(tmp_path)])

        assert result.exit_code == 0, result.output
        assert "Retrying" in result.output
        assert "coder" in result.output
        # State should be rewound to iteration 0 with coder as resume point
        reloaded = ExperimentState.load(tmp_path / "state.json")
        assert reloaded.phase == "iteration"
        assert reloaded.iteration == 0
        assert reloaded.consecutive_failures == 0
        # Orchestrator should receive skip_to_agent="coder"
        assert mock_orch.call_args.kwargs["skip_to_agent"] == "coder"
        # Earlier agents' artifacts should be preserved
        assert (v00 / "analysis.json").exists()
        assert (v00 / "plan.json").exists()
        # Coder artifacts should be cleaned up
        assert not (v00 / "experiment.py").exists()

    @patch("auto_scientist.cli.PipelineApp")
    @patch("auto_scientist.cli.Orchestrator")
    def test_resume_retries_from_earliest_missing_agent(self, mock_orch, mock_app_cls, tmp_path):
        """When only analyst completed, retry should resume from scientist."""
        from auto_scientist.state import VersionEntry

        state = ExperimentState(
            domain="auto",
            goal="g",
            phase="stopped",
            iteration=1,
            max_iterations=5,
            consecutive_failures=1,
        )
        state.versions = [
            VersionEntry(
                version="v00",
                iteration=0,
                script_path=str(tmp_path / "v00" / "experiment.py"),
                results_path=None,
                status="failed",
            ),
        ]
        state.save(tmp_path / "state.json")
        v00 = tmp_path / "v00"
        v00.mkdir()
        # Only analyst artifact exists
        (v00 / "analysis.json").write_text('{"data_summary": "test"}')

        runner = CliRunner()
        result = runner.invoke(cli, ["resume", "--state", str(tmp_path)])

        assert result.exit_code == 0, result.output
        assert "scientist" in result.output
        assert mock_orch.call_args.kwargs["skip_to_agent"] == "scientist"

    def test_resume_completed_run_still_blocked(self, tmp_path):
        """Naked resume of a run that completed normally should still require --fork."""
        from auto_scientist.state import VersionEntry

        state = ExperimentState(
            domain="auto",
            goal="g",
            phase="stopped",
            iteration=3,
            max_iterations=5,
            consecutive_failures=0,
        )
        state.versions = [
            VersionEntry(
                version="v00",
                iteration=0,
                script_path="s",
                results_path="r",
                status="completed",
            ),
            VersionEntry(
                version="v01",
                iteration=1,
                script_path="s",
                results_path="r",
                status="completed",
            ),
            VersionEntry(
                version="v02",
                iteration=2,
                script_path="s",
                results_path="r",
                status="completed",
            ),
        ]
        state.save(tmp_path / "state.json")

        runner = CliRunner()
        result = runner.invoke(cli, ["resume", "--state", str(tmp_path)])

        assert result.exit_code != 0
        assert "--fork" in result.output


class TestDetectRetryAgent:
    def test_all_agents_completed_returns_coder(self, tmp_path):
        (tmp_path / "analysis.json").touch()
        (tmp_path / "plan.json").touch()
        (tmp_path / "debate.json").touch()
        (tmp_path / "revision_plan.json").touch()
        (tmp_path / "experiment.py").touch()
        assert _detect_retry_agent(tmp_path) == "coder"

    def test_only_analyst_returns_scientist(self, tmp_path):
        (tmp_path / "analysis.json").touch()
        assert _detect_retry_agent(tmp_path) == "scientist"

    def test_through_debate_returns_revision(self, tmp_path):
        (tmp_path / "analysis.json").touch()
        (tmp_path / "plan.json").touch()
        (tmp_path / "debate.json").touch()
        assert _detect_retry_agent(tmp_path) == "revision"

    def test_empty_dir_returns_none(self, tmp_path):
        assert _detect_retry_agent(tmp_path) is None


class TestResolveSource:
    def test_directory_with_state_json(self, tmp_path):
        state = ExperimentState(domain="auto", goal="g", phase="iteration")
        (tmp_path / "state.json").write_text(state.model_dump_json())

        run_dir, loaded = _resolve_source(str(tmp_path))
        assert run_dir == tmp_path
        assert loaded.domain == "auto"

    def test_direct_state_json_path(self, tmp_path):
        state = ExperimentState(domain="auto", goal="g", phase="iteration")
        state_path = tmp_path / "state.json"
        state_path.write_text(state.model_dump_json())

        run_dir, loaded = _resolve_source(str(state_path))
        assert run_dir == tmp_path
        assert loaded.goal == "g"

    def test_missing_state_json_raises(self, tmp_path):
        import click
        import pytest

        with pytest.raises(click.UsageError, match="No state.json found"):
            _resolve_source(str(tmp_path))


class TestResumeFlagValidation:
    """Tests for --fork-required guards that protect against data destruction."""

    def _make_run(self, tmp_path):
        state = ExperimentState(domain="auto", goal="g", phase="iteration")
        state.save(tmp_path / "state.json")
        return tmp_path

    def test_from_iteration_without_fork_rejected(self, tmp_path):
        self._make_run(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["resume", "--from", str(tmp_path), "--from-iteration", "1"])
        assert result.exit_code != 0
        assert "--from-iteration requires --fork" in result.output

    def test_from_agent_without_fork_rejected(self, tmp_path):
        self._make_run(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            cli, ["resume", "--from", str(tmp_path), "--from-agent", "scientist"]
        )
        assert result.exit_code != 0
        assert "--from-agent requires --fork" in result.output

    def test_output_dir_without_fork_rejected(self, tmp_path):
        self._make_run(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["resume", "--from", str(tmp_path), "--output-dir", "/tmp/out"])
        assert result.exit_code != 0
        assert "--output-dir requires --fork" in result.output

    def test_completed_run_without_fork_rejected(self, tmp_path):
        state = ExperimentState(domain="auto", goal="g", phase="stopped")
        state.save(tmp_path / "state.json")

        runner = CliRunner()
        result = runner.invoke(cli, ["resume", "--from", str(tmp_path)])
        assert result.exit_code != 0
        assert "--fork" in result.output

    def test_report_phase_without_fork_rejected(self, tmp_path):
        state = ExperimentState(domain="auto", goal="g", phase="report")
        state.save(tmp_path / "state.json")

        runner = CliRunner()
        result = runner.invoke(cli, ["resume", "--from", str(tmp_path)])
        assert result.exit_code != 0
        assert "--fork" in result.output

    def test_from_alias_works(self, tmp_path):
        """The --from alias resolves identically to --state."""
        state = ExperimentState(domain="auto", goal="g", phase="iteration")
        state.save(tmp_path / "state.json")

        runner = CliRunner()
        with (
            patch("auto_scientist.cli.PipelineApp"),
            patch("auto_scientist.cli.Orchestrator") as mock_orch,
        ):
            result = runner.invoke(cli, ["resume", "--from", str(tmp_path)])

        assert result.exit_code == 0
        assert mock_orch.call_args.kwargs["state"].domain == "auto"


class TestForkDestination:
    """Fork default destination should be adjacent to source, not CWD-relative."""

    @patch("auto_scientist.cli.PipelineApp")
    @patch("auto_scientist.cli.Orchestrator")
    def test_fork_places_output_adjacent_to_source(self, mock_orch, mock_app_cls, tmp_path):
        """When --output-dir is not given, fork goes next to the source run."""
        source_dir = tmp_path / "remote" / "experiments" / "runs" / "my_run"
        source_dir.mkdir(parents=True)
        state = ExperimentState(domain="auto", goal="g", phase="iteration", iteration=1)
        state.save(source_dir / "state.json")

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["resume", "--from", str(source_dir), "--fork", "--from-iteration", "0"],
        )

        assert result.exit_code == 0
        # Fork should be in the same parent directory as the source
        fork_dir = mock_orch.call_args.kwargs["output_dir"]
        assert fork_dir.parent == source_dir.parent
        assert fork_dir.name == "my_run_001"


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
