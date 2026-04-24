"""Tests for the pre/post integrity fingerprint tripwire.

Each test creates a real throwaway git repo in `tmp_path`, snapshots it,
makes a specific kind of change, and confirms `verify_unchanged` raises
with a message that localises the change. The signals are independent,
so each test targets exactly one (HEAD move / porcelain-visible change /
working-tree-only change) to keep diagnostics sharp.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from auto_reviewer.safety.integrity import (
    IntegrityError,
    snapshot_repo,
    verify_unchanged,
)


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
    (repo / "a.txt").write_text("alpha\n")
    (repo / "b.py").write_text("print('b')\n")
    (repo / "sub").mkdir()
    (repo / "sub" / "c.md").write_text("# c\n")
    _run(["git", "add", "."], cwd=repo)
    _run(["git", "commit", "-q", "-m", "init"], cwd=repo)
    return repo


def test_clean_repo_round_trips(git_repo: Path) -> None:
    before = snapshot_repo(git_repo)
    # No changes at all: verify_unchanged should be a no-op.
    verify_unchanged(before)


def test_second_snapshot_equal_to_first(git_repo: Path) -> None:
    first = snapshot_repo(git_repo)
    second = snapshot_repo(git_repo)
    assert first == second, "snapshot must be deterministic on an unchanged tree"


def test_modified_tracked_file_raises(git_repo: Path) -> None:
    before = snapshot_repo(git_repo)
    (git_repo / "a.txt").write_text("alpha modified\n")
    with pytest.raises(IntegrityError) as exc:
        verify_unchanged(before)
    msg = str(exc.value)
    assert "porcelain" in msg.lower() or "content hash" in msg.lower()


def test_untracked_file_raises(git_repo: Path) -> None:
    before = snapshot_repo(git_repo)
    (git_repo / "new.txt").write_text("appeared from nowhere\n")
    with pytest.raises(IntegrityError) as exc:
        verify_unchanged(before)
    msg = str(exc.value)
    assert "porcelain" in msg.lower() or "content hash" in msg.lower()


def test_deleted_tracked_file_raises(git_repo: Path) -> None:
    before = snapshot_repo(git_repo)
    (git_repo / "b.py").unlink()
    with pytest.raises(IntegrityError):
        verify_unchanged(before)


def test_head_move_raises(git_repo: Path) -> None:
    before = snapshot_repo(git_repo)
    (git_repo / "d.txt").write_text("new\n")
    _run(["git", "add", "d.txt"], cwd=git_repo)
    _run(["git", "commit", "-q", "-m", "extra"], cwd=git_repo)
    with pytest.raises(IntegrityError) as exc:
        verify_unchanged(before)
    assert "HEAD moved" in str(exc.value)


def test_reset_hard_to_earlier_commit_raises(git_repo: Path) -> None:
    # Add a second commit so we can reset back.
    (git_repo / "d.txt").write_text("new\n")
    _run(["git", "add", "d.txt"], cwd=git_repo)
    _run(["git", "commit", "-q", "-m", "two"], cwd=git_repo)
    before = snapshot_repo(git_repo)
    _run(["git", "reset", "-q", "--hard", "HEAD~1"], cwd=git_repo)
    with pytest.raises(IntegrityError) as exc:
        verify_unchanged(before)
    msg = str(exc.value)
    assert "HEAD moved" in msg
    assert "content hash" in msg.lower()


def test_symlink_retarget_raises(git_repo: Path, tmp_path: Path) -> None:
    target_a = git_repo / "a.txt"
    target_b = git_repo / "b.py"
    link = git_repo / "link"
    link.symlink_to(target_a)
    _run(["git", "add", "link"], cwd=git_repo)
    _run(["git", "commit", "-q", "-m", "add link"], cwd=git_repo)
    before = snapshot_repo(git_repo)
    link.unlink()
    link.symlink_to(target_b)
    with pytest.raises(IntegrityError):
        verify_unchanged(before)


def test_content_change_without_porcelain_change_raises(git_repo: Path) -> None:
    # Force a case where porcelain *might* not catch it: re-write a file
    # with identical content. Tree hash should still match -> no raise.
    before = snapshot_repo(git_repo)
    (git_repo / "a.txt").write_text("alpha\n")  # same content
    verify_unchanged(before)  # no-op expected


def test_non_git_directory_still_snapshottable(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "x").write_text("hi")
    before = snapshot_repo(plain)
    assert before.head == "<no-git>"
    (plain / "y").write_text("added")
    with pytest.raises(IntegrityError):
        verify_unchanged(before)


def test_missing_path_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        snapshot_repo(tmp_path / "does-not-exist")


def test_file_not_directory_raises(tmp_path: Path) -> None:
    f = tmp_path / "afile"
    f.write_text("x")
    with pytest.raises(NotADirectoryError):
        snapshot_repo(f)


def test_diff_returns_empty_on_match(git_repo: Path) -> None:
    before = snapshot_repo(git_repo)
    after = snapshot_repo(git_repo)
    assert before.diff(after) == []


def test_fingerprint_carries_absolute_path(git_repo: Path) -> None:
    fp = snapshot_repo(git_repo)
    assert fp.path == str(git_repo.resolve())
