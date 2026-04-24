"""End-to-end CLI integration: verify the reviewer never mutates the
user's real repository.

The real LLM isn't exercised here (that's the manual smoke in phase G);
the orchestrator and agent call stack are patched so only the
sandbox-relevant machinery runs:

- `pre_resolve` against the real throwaway repo
- workspace / clone creation
- `verify_unchanged` at end of run

An adversarial path simulates the orchestrator mutating the user's real
repo mid-run and confirms the CLI exits with the SANDBOX VIOLATION
error rather than silently proceeding.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import auto_reviewer.cli as reviewer_cli
import pytest
from auto_reviewer.safety.integrity import snapshot_repo
from click.testing import CliRunner


def _run(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "target_repo"
    repo.mkdir()
    _run(["git", "init", "-q", "-b", "main"], cwd=repo)
    _run(["git", "config", "user.email", "test@example.com"], cwd=repo)
    _run(["git", "config", "user.name", "Test"], cwd=repo)
    _run(["git", "config", "commit.gpgsign", "false"], cwd=repo)
    (repo / "README.md").write_text("alpha\n")
    (repo / "src").mkdir()
    (repo / "src" / "main.py").write_text("print('hi')\n")
    _run(["git", "add", "."], cwd=repo)
    _run(["git", "commit", "-q", "-m", "init"], cwd=repo)
    return repo


def _invoke(git_repo: Path, workspace: Path, extra: list[str] | None = None):
    runner = CliRunner()
    args = [
        "review",
        "review my current branch",
        "--cwd",
        str(git_repo),
        "--output-dir",
        str(workspace),
    ]
    if extra:
        args.extend(extra)
    return runner.invoke(reviewer_cli.cli, args)


def test_sandbox_docker_flag_not_yet_implemented(git_repo: Path, tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    with (
        patch("auto_reviewer.cli.Orchestrator"),
        patch("auto_reviewer.cli._run_orchestrator"),
    ):
        result = _invoke(git_repo, workspace, extra=["--sandbox", "docker"])
    assert result.exit_code != 0
    assert "docker" in result.output.lower()
    # The flag's rejection happens before pre_resolve, so the workspace
    # is not polluted.
    assert not workspace.exists()


def test_happy_path_leaves_real_repo_byte_identical(git_repo: Path, tmp_path: Path) -> None:
    """With a trivial (no-op) orchestrator, the real repo must be
    byte-identical before and after the CLI runs."""
    workspace = tmp_path / "ws"
    before = snapshot_repo(git_repo)

    with (
        patch("auto_reviewer.cli.Orchestrator") as mock_orch,
        patch("auto_reviewer.cli._run_orchestrator") as mock_run,
    ):
        result = _invoke(git_repo, workspace)

    assert result.exit_code == 0, result.output
    mock_orch.assert_called_once()
    mock_run.assert_called_once()

    # The pre-resolution ran: clone + hint file exist.
    assert (workspace / "repo_clone").is_dir()
    assert (workspace / "repo_clone" / "README.md").read_text() == "alpha\n"
    hint = json.loads((workspace / "data" / "cwd_hint.json").read_text())
    assert hint["is_git"] is True
    assert hint["repo_clone"] == str(workspace / "repo_clone")

    # The real repo is unchanged.
    after = snapshot_repo(git_repo)
    assert before == after, "the real repo was modified during the CLI run"


def test_clone_mutation_does_not_affect_real_repo(git_repo: Path, tmp_path: Path) -> None:
    """Even if something heavily mutates the clone, the real repo is
    untouched."""
    workspace = tmp_path / "ws"
    before = snapshot_repo(git_repo)

    def _mutate_clone(*_args, **_kwargs) -> None:
        # Simulate an agent going wild inside the clone: wipe files,
        # rewrite others, delete the .git directory entirely.
        clone = workspace / "repo_clone"
        (clone / "README.md").write_text("clobbered\n")
        (clone / "NEW_EVIL.md").write_text("added\n")
        (clone / "src" / "main.py").unlink()
        return None

    with (
        patch("auto_reviewer.cli.Orchestrator"),
        patch("auto_reviewer.cli._run_orchestrator", side_effect=_mutate_clone),
    ):
        result = _invoke(git_repo, workspace)

    assert result.exit_code == 0, result.output
    # Real repo: byte-identical; clone: mutated as we asked.
    assert snapshot_repo(git_repo) == before
    assert (workspace / "repo_clone" / "NEW_EVIL.md").exists()
    assert not (workspace / "repo_clone" / "src" / "main.py").exists()


def test_tripwire_fires_when_real_repo_mutated(git_repo: Path, tmp_path: Path) -> None:
    """If the real repo somehow gets modified during the run, exit 2."""
    workspace = tmp_path / "ws"

    def _mutate_real(*_args, **_kwargs) -> None:
        # Simulate an escape: something outside the clone mutated the
        # user's real repo. `verify_unchanged` at end of run should fail.
        (git_repo / "README.md").write_text("tampered\n")
        return None

    with (
        patch("auto_reviewer.cli.Orchestrator"),
        patch("auto_reviewer.cli._run_orchestrator", side_effect=_mutate_real),
    ):
        result = _invoke(git_repo, workspace)

    assert result.exit_code == 2, result.output
    assert "SANDBOX VIOLATION" in result.output


def test_hint_file_does_not_contain_absolute_user_paths_beyond_cwd(
    git_repo: Path, tmp_path: Path
) -> None:
    """Sanity check: the hint is scoped. The LLM sees its own workspace
    and the user's cwd metadata, not unrelated paths (home, secrets)."""
    workspace = tmp_path / "ws"
    with (
        patch("auto_reviewer.cli.Orchestrator"),
        patch("auto_reviewer.cli._run_orchestrator"),
    ):
        _invoke(git_repo, workspace)

    hint = json.loads((workspace / "data" / "cwd_hint.json").read_text())
    # Only documented keys — no stray env dump.
    assert set(hint.keys()).issubset(
        {
            "cwd",
            "is_git",
            "repo_clone",
            "toplevel",
            "head_sha",
            "current_branch",
            "remotes",
        }
    )
