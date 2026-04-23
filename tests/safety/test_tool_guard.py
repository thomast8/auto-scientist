"""Unit tests for the workspace guard.

The guard is the Claude-path enforcement layer of the reviewer sandbox,
and the policy mirror for the Codex prompt. These tests exercise each
mode's allowed and denied surface with concrete tool inputs that match
what the SDK actually passes in.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from auto_core.safety.tool_guard import (
    Decision,
    make_workspace_guard,
)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "review_workspace"
    ws.mkdir()
    (ws / "data").mkdir()
    return ws


@pytest.fixture
def repo_clone(workspace: Path) -> Path:
    clone = workspace / "repo_clone"
    clone.mkdir()
    (clone / ".auto_reviewer_probes").mkdir()
    return clone


@pytest.fixture
def outside_path(tmp_path: Path) -> Path:
    outside = tmp_path / "real_repo_pretend"
    outside.mkdir()
    (outside / "precious.py").write_text("x = 1")
    return outside


# ---------------------------------------------------------------------------
# Construction invariants
# ---------------------------------------------------------------------------


def test_repo_clone_must_be_inside_workspace(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    elsewhere = tmp_path / "somewhere_else"
    elsewhere.mkdir()
    with pytest.raises(ValueError, match="not inside workspace"):
        make_workspace_guard(ws, elsewhere, mode="probe")


# ---------------------------------------------------------------------------
# Read-only mode
# ---------------------------------------------------------------------------


def test_read_only_allows_read_glob_grep(workspace: Path, repo_clone: Path) -> None:
    guard = make_workspace_guard(workspace, repo_clone, mode="read_only")
    assert guard("Read", {"file_path": "/etc/passwd"}).allowed is True
    assert guard("Glob", {"pattern": "**/*.py"}).allowed is True
    assert guard("Grep", {"pattern": "foo"}).allowed is True


def test_read_only_denies_write_and_bash(workspace: Path, repo_clone: Path) -> None:
    guard = make_workspace_guard(workspace, repo_clone, mode="read_only")
    for tool in ("Write", "Edit", "Bash", "NotebookEdit"):
        d = guard(tool, {"file_path": str(workspace / "x"), "command": "echo hi"})
        assert not d.allowed, f"{tool} should be denied in read_only mode"


# ---------------------------------------------------------------------------
# Write / Edit path checks
# ---------------------------------------------------------------------------


def test_write_inside_workspace_allowed(workspace: Path, repo_clone: Path) -> None:
    guard = make_workspace_guard(workspace, repo_clone, mode="intake")
    d = guard("Write", {"file_path": str(workspace / "data" / "notes.md")})
    assert d.allowed, d.reason


def test_write_outside_workspace_denied(
    workspace: Path, repo_clone: Path, outside_path: Path
) -> None:
    guard = make_workspace_guard(workspace, repo_clone, mode="intake")
    d = guard("Write", {"file_path": str(outside_path / "evil.py")})
    assert not d.allowed
    assert "outside the review workspace" in d.reason


def test_write_with_tilde_expansion_denied(workspace: Path, repo_clone: Path) -> None:
    guard = make_workspace_guard(workspace, repo_clone, mode="intake")
    d = guard("Write", {"file_path": "~/.bashrc"})
    assert not d.allowed


def test_write_missing_path_field_denied(workspace: Path, repo_clone: Path) -> None:
    guard = make_workspace_guard(workspace, repo_clone, mode="intake")
    d = guard("Write", {"content": "x"})
    assert not d.allowed
    assert "no file_path" in d.reason


# ---------------------------------------------------------------------------
# Probe mode: repo_clone writes limited to .auto_reviewer_probes/
# ---------------------------------------------------------------------------


def test_probe_may_write_probes_dir(workspace: Path, repo_clone: Path) -> None:
    guard = make_workspace_guard(workspace, repo_clone, mode="probe")
    d = guard(
        "Write",
        {"file_path": str(repo_clone / ".auto_reviewer_probes" / "shim.py")},
    )
    assert d.allowed, d.reason


def test_probe_may_not_write_target_source(workspace: Path, repo_clone: Path) -> None:
    guard = make_workspace_guard(workspace, repo_clone, mode="probe")
    (repo_clone / "src").mkdir()
    d = guard("Edit", {"file_path": str(repo_clone / "src" / "prod.py")})
    assert not d.allowed
    assert ".auto_reviewer_probes" in d.reason


def test_probe_workspace_writes_outside_clone_allowed(workspace: Path, repo_clone: Path) -> None:
    guard = make_workspace_guard(workspace, repo_clone, mode="probe")
    d = guard("Write", {"file_path": str(workspace / "v01" / "run_result.json")})
    assert d.allowed, d.reason


# ---------------------------------------------------------------------------
# Bash destructive-verb deny list
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        "sudo rm -rf /",
        "chmod 777 /tmp",
        "dd if=/dev/zero of=/tmp/foo",
        "mkfs.ext4 /dev/sda1",
        "shutdown now",
        "launchctl unload -w /Library/LaunchDaemons/foo.plist",
        "systemctl stop ssh",
    ],
)
def test_bash_destructive_verbs_denied(command: str, workspace: Path, repo_clone: Path) -> None:
    guard = make_workspace_guard(workspace, repo_clone, mode="probe")
    d = guard("Bash", {"command": command})
    assert not d.allowed, f"should deny: {command}"


def test_bash_rm_rf_denied(workspace: Path, repo_clone: Path) -> None:
    guard = make_workspace_guard(workspace, repo_clone, mode="probe")
    for cmd in (
        f"rm -rf {workspace}/data",
        "rm -rf /",
        f"rm -rf {workspace / 'inside'}",  # even inside: blanket ban on recursive rm
        "rm -r .",
        "rm -fr ../foo",
    ):
        d = guard("Bash", {"command": cmd})
        assert not d.allowed, f"should deny: {cmd}"
        assert "recursive" in d.reason.lower()


def test_bash_rm_single_file_inside_allowed(workspace: Path, repo_clone: Path) -> None:
    guard = make_workspace_guard(workspace, repo_clone, mode="probe")
    d = guard("Bash", {"command": f"rm {workspace}/stale.txt"})
    assert d.allowed, d.reason


def test_bash_rm_single_file_outside_denied(
    workspace: Path, repo_clone: Path, outside_path: Path
) -> None:
    guard = make_workspace_guard(workspace, repo_clone, mode="probe")
    d = guard("Bash", {"command": f"rm {outside_path}/precious.py"})
    assert not d.allowed


# ---------------------------------------------------------------------------
# Bash: absolute path arguments must live inside workspace
# ---------------------------------------------------------------------------


def test_bash_abs_path_outside_denied(
    workspace: Path, repo_clone: Path, outside_path: Path
) -> None:
    guard = make_workspace_guard(workspace, repo_clone, mode="probe")
    d = guard("Bash", {"command": f"cat {outside_path}/precious.py"})
    assert not d.allowed
    assert "outside workspace" in d.reason


def test_bash_abs_path_inside_allowed(workspace: Path, repo_clone: Path) -> None:
    guard = make_workspace_guard(workspace, repo_clone, mode="probe")
    d = guard("Bash", {"command": f"cat {workspace}/data/notes.md"})
    assert d.allowed, d.reason


def test_bash_relative_path_allowed(workspace: Path, repo_clone: Path) -> None:
    guard = make_workspace_guard(workspace, repo_clone, mode="probe")
    d = guard("Bash", {"command": "cat data/notes.md"})
    assert d.allowed, d.reason


def test_bash_redirect_outside_denied(
    workspace: Path, repo_clone: Path, outside_path: Path
) -> None:
    guard = make_workspace_guard(workspace, repo_clone, mode="probe")
    d = guard("Bash", {"command": f"echo hi > {outside_path}/note.txt"})
    assert not d.allowed
    assert "redirection" in d.reason.lower()


# ---------------------------------------------------------------------------
# Bash: git subcommand policy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        "git push origin main",
        "git commit -m 'oops'",
        "git reset --hard HEAD",
        "git clean -fdx",
        "git rebase -i HEAD~5",
        "git checkout main",
        "git branch -D old",
        "git remote add evil http://evil.example",
    ],
)
def test_bash_forbidden_git_subcommand_denied(
    command: str, workspace: Path, repo_clone: Path
) -> None:
    guard = make_workspace_guard(workspace, repo_clone, mode="probe")
    d = guard("Bash", {"command": command})
    assert not d.allowed, f"should deny: {command}"
    assert "forbidden-subcommand" in d.reason


@pytest.mark.parametrize(
    "command",
    [
        "git log --oneline -10",
        "git show HEAD:src/foo.py",
        "git diff base...head -- src/",
        "git rev-parse HEAD",
        "git ls-files",
        "git status",
        "git blame src/foo.py",
    ],
)
def test_bash_readonly_git_subcommand_allowed(
    command: str, workspace: Path, repo_clone: Path
) -> None:
    guard = make_workspace_guard(workspace, repo_clone, mode="intake")
    d = guard("Bash", {"command": command})
    assert d.allowed, f"should allow: {command} ({d.reason})"


def test_bash_git_c_target_outside_workspace_denied(
    workspace: Path, repo_clone: Path, outside_path: Path
) -> None:
    guard = make_workspace_guard(workspace, repo_clone, mode="intake")
    d = guard("Bash", {"command": f"git -C {outside_path} log"})
    assert not d.allowed
    assert "outside" in d.reason


def test_bash_git_c_target_inside_workspace_allowed(workspace: Path, repo_clone: Path) -> None:
    guard = make_workspace_guard(workspace, repo_clone, mode="intake")
    d = guard("Bash", {"command": f"git -C {repo_clone} log --oneline"})
    assert d.allowed, d.reason


def test_bash_git_clone_into_workspace_allowed(workspace: Path, repo_clone: Path) -> None:
    guard = make_workspace_guard(workspace, repo_clone, mode="intake")
    d = guard(
        "Bash",
        {"command": f"git clone https://github.com/foo/bar.git {workspace}/repo_clone"},
    )
    assert d.allowed, d.reason


def test_bash_git_clone_outside_workspace_denied(
    workspace: Path, repo_clone: Path, outside_path: Path
) -> None:
    guard = make_workspace_guard(workspace, repo_clone, mode="intake")
    d = guard(
        "Bash",
        {"command": f"git clone https://github.com/foo/bar.git {outside_path}/sneak"},
    )
    assert not d.allowed


def test_bash_git_clone_without_destination_denied(workspace: Path, repo_clone: Path) -> None:
    # `git clone <url>` without an explicit dest clones into cwd — we
    # require a bounded destination so the clone lands where we expect.
    guard = make_workspace_guard(workspace, repo_clone, mode="intake")
    d = guard("Bash", {"command": "git clone https://github.com/foo/bar.git"})
    assert not d.allowed
    assert "bounded destination" in d.reason


# ---------------------------------------------------------------------------
# Bash: gh subcommand policy
# ---------------------------------------------------------------------------


def test_bash_gh_pr_view_allowed(workspace: Path, repo_clone: Path) -> None:
    guard = make_workspace_guard(workspace, repo_clone, mode="intake")
    assert guard("Bash", {"command": "gh pr view 123 --json title,body"}).allowed
    assert guard("Bash", {"command": "gh pr diff 123"}).allowed


def test_bash_gh_pr_merge_denied(workspace: Path, repo_clone: Path) -> None:
    guard = make_workspace_guard(workspace, repo_clone, mode="intake")
    d = guard("Bash", {"command": "gh pr merge 123"})
    assert not d.allowed


def test_bash_gh_issue_denied(workspace: Path, repo_clone: Path) -> None:
    guard = make_workspace_guard(workspace, repo_clone, mode="intake")
    d = guard("Bash", {"command": "gh issue close 5"})
    assert not d.allowed


def test_bash_gh_api_get_allowed(workspace: Path, repo_clone: Path) -> None:
    guard = make_workspace_guard(workspace, repo_clone, mode="intake")
    d = guard("Bash", {"command": "gh api repos/foo/bar/pulls/1"})
    assert d.allowed


def test_bash_gh_api_post_denied(workspace: Path, repo_clone: Path) -> None:
    guard = make_workspace_guard(workspace, repo_clone, mode="intake")
    d = guard("Bash", {"command": "gh api -X POST repos/foo/bar/issues"})
    assert not d.allowed


# ---------------------------------------------------------------------------
# Bash: chained commands must each pass
# ---------------------------------------------------------------------------


def test_bash_chained_destructive_segment_denied(workspace: Path, repo_clone: Path) -> None:
    guard = make_workspace_guard(workspace, repo_clone, mode="probe")
    # Innocuous first command, destructive second.
    d = guard("Bash", {"command": "echo hi && rm -rf /"})
    assert not d.allowed


def test_bash_pipeline_destructive_segment_denied(workspace: Path, repo_clone: Path) -> None:
    guard = make_workspace_guard(workspace, repo_clone, mode="probe")
    d = guard("Bash", {"command": "cat foo | sudo tee /etc/passwd"})
    assert not d.allowed


def test_bash_env_prefix_allowed(workspace: Path, repo_clone: Path) -> None:
    guard = make_workspace_guard(workspace, repo_clone, mode="probe")
    d = guard("Bash", {"command": "PYTHONPATH=. python3 -m pytest"})
    assert d.allowed, d.reason


# ---------------------------------------------------------------------------
# Unknown tool rejection (defense-in-depth against tool-surface expansion)
# ---------------------------------------------------------------------------


def test_unknown_tool_denied(workspace: Path, repo_clone: Path) -> None:
    guard = make_workspace_guard(workspace, repo_clone, mode="intake")
    d = guard("SuperExec", {"command": "anything"})
    assert not d.allowed
    assert "not on the reviewer allowlist" in d.reason


def test_mcp_tool_name_strips_prefix(workspace: Path, repo_clone: Path) -> None:
    # SDK sometimes prefixes MCP tool names; the stripped name is what we
    # check against the allowlist.
    guard = make_workspace_guard(workspace, repo_clone, mode="intake")
    d = guard("mcp__some_server__Read", {"file_path": "/etc/passwd"})
    assert d.allowed


# ---------------------------------------------------------------------------
# Decision convenience constructors
# ---------------------------------------------------------------------------


def test_decision_helpers() -> None:
    assert Decision.allow().allowed is True
    assert Decision.deny("nope").allowed is False
    assert Decision.deny("nope").reason == "nope"
