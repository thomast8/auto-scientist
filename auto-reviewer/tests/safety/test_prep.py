"""Tests for pre-Intake resolution.

`pre_resolve` is the one place the orchestrator touches the user's real
repository. These tests confirm:

- It fingerprints the real cwd (via verify_unchanged round-trip).
- It clones the real repo into `<workspace>/repo_clone/` when the cwd
  is a git repo — without disturbing the original.
- It writes `cwd_hint.json` with the metadata Intake needs so the LLM
  never has to touch the user's filesystem path itself.
- Running the full flow against a throwaway repo leaves that repo
  byte-identical.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from auto_reviewer.prep import PreResolved, pre_resolve
from auto_reviewer.safety.integrity import verify_unchanged


def _run(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "throwaway"
    repo.mkdir()
    _run(["git", "init", "-q", "-b", "main"], cwd=repo)
    _run(["git", "config", "user.email", "test@example.com"], cwd=repo)
    _run(["git", "config", "user.name", "Test"], cwd=repo)
    _run(["git", "config", "commit.gpgsign", "false"], cwd=repo)
    _run(["git", "remote", "add", "origin", "https://example.com/foo/bar.git"], cwd=repo)
    (repo / "README.md").write_text("# hello\n")
    (repo / "src").mkdir()
    (repo / "src" / "main.py").write_text("print('hi')\n")
    _run(["git", "add", "."], cwd=repo)
    _run(["git", "commit", "-q", "-m", "init"], cwd=repo)
    return repo


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    return tmp_path / "ws"


def test_clone_into_workspace(git_repo: Path, workspace: Path) -> None:
    resolved = pre_resolve(git_repo, workspace)
    assert isinstance(resolved, PreResolved)
    assert resolved.repo_clone == workspace / "repo_clone"
    assert resolved.repo_clone.exists()
    assert (resolved.repo_clone / "README.md").read_text() == "# hello\n"
    assert (resolved.repo_clone / "src" / "main.py").exists()


def test_clone_is_independent_of_original(git_repo: Path, workspace: Path) -> None:
    """Modifying the clone must not change the original."""
    resolved = pre_resolve(git_repo, workspace)
    before = (git_repo / "README.md").read_text()
    assert resolved.repo_clone is not None
    (resolved.repo_clone / "README.md").write_text("# tampered\n")
    after = (git_repo / "README.md").read_text()
    assert before == after
    # And the integrity check agrees.
    verify_unchanged(resolved.fingerprint)


def test_cwd_hint_contains_metadata(git_repo: Path, workspace: Path) -> None:
    resolved = pre_resolve(git_repo, workspace)
    hint = json.loads(resolved.hint_path.read_text())
    assert hint["is_git"] is True
    assert hint["current_branch"] == "main"
    assert hint["head_sha"]  # non-empty SHA
    assert hint["repo_clone"] == str(workspace / "repo_clone")
    remotes = hint["remotes"]
    assert any(r["name"] == "origin" for r in remotes)


def test_non_git_cwd_produces_no_clone(tmp_path: Path, workspace: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "x").write_text("y")
    resolved = pre_resolve(plain, workspace)
    assert resolved.repo_clone is None
    hint = json.loads(resolved.hint_path.read_text())
    assert hint["is_git"] is False
    assert "repo_clone" not in hint


def test_full_round_trip_leaves_real_repo_untouched(git_repo: Path, workspace: Path) -> None:
    """Simulate a full run where the clone gets heavily modified."""
    resolved = pre_resolve(git_repo, workspace)
    # Treat the clone like an LLM would — write files, delete things, etc.
    assert resolved.repo_clone is not None
    (resolved.repo_clone / "NEW.md").write_text("added by LLM\n")
    (resolved.repo_clone / "README.md").write_text("mutated\n")
    (resolved.repo_clone / "src" / "main.py").unlink()
    # Real repo is still byte-identical.
    verify_unchanged(resolved.fingerprint)


def test_existing_clone_refuses_to_overwrite(git_repo: Path, workspace: Path) -> None:
    workspace.mkdir(parents=True)
    (workspace / "repo_clone").mkdir()
    with pytest.raises(RuntimeError, match="already exists"):
        pre_resolve(git_repo, workspace)
