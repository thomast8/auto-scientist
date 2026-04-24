from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import auto_reviewer.cli as reviewer_cli
import pytest
from auto_core.state import RunState
from click.testing import CliRunner


class TestRunOrchestrator:
    @patch("auto_reviewer.cli.install_child_cleanup_handlers")
    @patch("auto_reviewer.cli.PipelineApp")
    def test_installs_shared_cleanup_handlers(self, mock_app_cls, mock_install):
        orchestrator = MagicMock()

        reviewer_cli._run_orchestrator(orchestrator, "Interrupted.")

        mock_install.assert_called_once()
        mock_app_cls.assert_called_once_with(orchestrator)
        mock_app_cls.return_value.run.assert_called_once()

    @patch("auto_reviewer.cli.install_child_cleanup_handlers")
    @patch("auto_reviewer.cli.click.echo")
    @patch("auto_reviewer.cli.PipelineApp")
    def test_keyboard_interrupt_exits_130(self, mock_app_cls, mock_echo, mock_install):
        mock_app_cls.return_value.run.side_effect = KeyboardInterrupt

        with pytest.raises(SystemExit) as exc:
            reviewer_cli._run_orchestrator(MagicMock(), "Interrupted again.")

        assert exc.value.code == 130
        mock_install.assert_called_once()
        mock_echo.assert_called_once_with("Interrupted again.")


class TestReviewCommand:
    @patch("auto_reviewer.cli._run_orchestrator")
    @patch("auto_reviewer.cli.Orchestrator")
    def test_review_routes_through_cleanup_wrapper(self, mock_orch, mock_run, tmp_path):
        # pre_resolve runs before the orchestrator; point --cwd at a
        # throwaway non-git dir so no real clone happens in the unit
        # test path.
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        runner = CliRunner()
        result = runner.invoke(
            reviewer_cli.cli,
            [
                "review",
                "review my current branch",
                "--cwd",
                str(scratch),
                "--output-dir",
                str(tmp_path / "review_workspace"),
            ],
        )

        assert result.exit_code == 0
        mock_orch.assert_called_once()
        mock_run.assert_called_once_with(
            mock_orch.return_value,
            (
                "Interrupted. State is persisted at state.json; "
                "use `auto-reviewer resume` to continue."
            ),
        )


class TestResumeCommand:
    @patch("auto_core.resume.rewind_run")
    @patch("auto_reviewer.cli._run_orchestrator")
    @patch("auto_reviewer.cli.Orchestrator")
    def test_resume_routes_through_cleanup_wrapper(
        self,
        mock_orch,
        mock_run,
        mock_rewind,
        tmp_path,
    ):
        run_dir = tmp_path / "review_workspace"
        run_dir.mkdir()
        state = RunState(
            domain="owner/repo#1",
            goal="review the PR",
            phase="ingestion",
            data_path=str(tmp_path),
        )
        state.save(run_dir / "state.json")
        mock_rewind.return_value = SimpleNamespace(state=state, restored_panels=[])

        runner = CliRunner()
        result = runner.invoke(reviewer_cli.cli, ["resume", str(run_dir)])

        assert result.exit_code == 0
        mock_orch.assert_called_once()
        mock_run.assert_called_once_with(
            mock_orch.return_value,
            "Interrupted again. Re-run `auto-reviewer resume` to continue.",
        )
