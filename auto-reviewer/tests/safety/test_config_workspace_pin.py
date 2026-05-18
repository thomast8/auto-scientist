"""Tests for `ReviewConfig.require_inside_workspace`.

The LLM-written ReviewConfig.repo_path is the one place a downstream
agent's `run_cwd` gets decided. Pinning it to inside the workspace
closes the "Intake wrote a path to the user's real repo" escape.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from auto_reviewer.config import ReviewConfig


def _make_config(repo_path: Path, run_cwd: str | None = None) -> ReviewConfig:
    return ReviewConfig(
        name="pr-1",
        description="",
        repo_path=str(repo_path),
        run_cwd=run_cwd or ".",
    )


def test_accepts_path_inside_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    clone = workspace / "repo_clone"
    clone.mkdir(parents=True)
    cfg = _make_config(clone)
    cfg.require_inside_workspace(workspace)  # must not raise


def test_accepts_nested_path_inside_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    deep = workspace / "repo_clone" / "sub" / "dir"
    deep.mkdir(parents=True)
    cfg = _make_config(deep)
    cfg.require_inside_workspace(workspace)


def test_accepts_run_cwd_inside_repo(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    clone = workspace / "repo_clone"
    package = clone / "packages" / "api"
    package.mkdir(parents=True)
    cfg = _make_config(clone, run_cwd=str(package))
    cfg.require_inside_workspace(workspace)


def test_accepts_relative_run_cwd_inside_repo(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    clone = workspace / "repo_clone"
    (clone / "packages" / "api").mkdir(parents=True)
    cfg = _make_config(clone, run_cwd="packages/api")
    cfg.require_inside_workspace(workspace)


def test_rejects_path_outside_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    outside = tmp_path / "real_repo"
    outside.mkdir()
    cfg = _make_config(outside)
    with pytest.raises(ValueError, match="not inside workspace"):
        cfg.require_inside_workspace(workspace)


def test_rejects_sibling_directory(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    sibling = tmp_path / "ws_sibling"
    sibling.mkdir()
    cfg = _make_config(sibling)
    with pytest.raises(ValueError, match="not inside workspace"):
        cfg.require_inside_workspace(workspace)


def test_rejects_root_path(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    cfg = _make_config(Path("/"))
    with pytest.raises(ValueError, match="not inside workspace"):
        cfg.require_inside_workspace(workspace)


def test_rejects_run_cwd_outside_repo(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    clone = workspace / "repo_clone"
    clone.mkdir(parents=True)
    outside = tmp_path / "real_repo"
    outside.mkdir()
    cfg = _make_config(clone, run_cwd=str(outside))
    with pytest.raises(ValueError, match="run_cwd .* not inside repo_path"):
        cfg.require_inside_workspace(workspace)


def test_rejects_relative_run_cwd_escaping_repo(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    clone = workspace / "repo_clone"
    clone.mkdir(parents=True)
    cfg = _make_config(clone, run_cwd="../sibling")
    with pytest.raises(ValueError, match="run_cwd .* not inside repo_path"):
        cfg.require_inside_workspace(workspace)
