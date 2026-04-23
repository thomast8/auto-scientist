"""Phase C wire-up tests.

Exercises the machinery between `SDKOptions.pre_tool_use_hook`, the
Claude `can_use_tool` adapter, and the Codex cwd/workspace assertion —
without spawning any SDK subprocess (which would require real
credentials). The real-SDK smoke check lives in the integration test
suite.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from auto_core.safety.tool_guard import make_workspace_guard
from auto_core.sdk_backend import (
    CodexBackend,
    SDKOptions,
    _make_claude_can_use_tool,
)
from claude_code_sdk import (
    PermissionResultAllow,
    PermissionResultDeny,
    ToolPermissionContext,
)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    ws.mkdir()
    return ws


@pytest.fixture
def repo_clone(workspace: Path) -> Path:
    clone = workspace / "repo_clone"
    clone.mkdir()
    return clone


# ---------------------------------------------------------------------------
# Claude adapter: maps guard Decisions to SDK PermissionResult types
# ---------------------------------------------------------------------------


async def test_claude_adapter_allows_read_everywhere(workspace: Path, repo_clone: Path) -> None:
    guard = make_workspace_guard(workspace, repo_clone, mode="intake")
    adapter = _make_claude_can_use_tool(guard)
    result = await adapter("Read", {"file_path": "/etc/passwd"}, ToolPermissionContext())
    assert isinstance(result, PermissionResultAllow)


async def test_claude_adapter_denies_write_outside(
    tmp_path: Path, workspace: Path, repo_clone: Path
) -> None:
    guard = make_workspace_guard(workspace, repo_clone, mode="intake")
    adapter = _make_claude_can_use_tool(guard)
    elsewhere = tmp_path / "not_in_workspace"
    elsewhere.mkdir()
    result = await adapter(
        "Write",
        {"file_path": str(elsewhere / "evil.py")},
        ToolPermissionContext(),
    )
    assert isinstance(result, PermissionResultDeny)
    assert "outside the review workspace" in result.message
    # interrupt=False so the model sees the error and can course-correct
    # rather than the whole query dying.
    assert result.interrupt is False


async def test_claude_adapter_denies_destructive_bash(workspace: Path, repo_clone: Path) -> None:
    guard = make_workspace_guard(workspace, repo_clone, mode="probe")
    adapter = _make_claude_can_use_tool(guard)
    result = await adapter("Bash", {"command": "rm -rf /"}, ToolPermissionContext())
    assert isinstance(result, PermissionResultDeny)
    assert "recursive" in result.message.lower()


# ---------------------------------------------------------------------------
# Codex wire-up: cwd must match the guard's workspace
# ---------------------------------------------------------------------------


async def test_codex_rejects_missing_cwd_when_hook_present(
    workspace: Path, repo_clone: Path
) -> None:
    guard = make_workspace_guard(workspace, repo_clone, mode="probe")
    options = SDKOptions(
        system_prompt="x",
        allowed_tools=("Read", "Bash"),
        max_turns=5,
        cwd=None,
        pre_tool_use_hook=guard,
    )
    backend = CodexBackend()
    with pytest.raises(RuntimeError, match="options.cwd is None"):
        await backend._ensure_client(options, model=None)


async def test_codex_rejects_cwd_outside_workspace(
    tmp_path: Path, workspace: Path, repo_clone: Path
) -> None:
    guard = make_workspace_guard(workspace, repo_clone, mode="probe")
    wrong_cwd = tmp_path / "some_other_dir"
    wrong_cwd.mkdir()
    options = SDKOptions(
        system_prompt="x",
        allowed_tools=("Read", "Bash"),
        max_turns=5,
        cwd=wrong_cwd,
        pre_tool_use_hook=guard,
    )
    backend = CodexBackend()
    with pytest.raises(RuntimeError, match="does not match the workspace"):
        await backend._ensure_client(options, model=None)


# ---------------------------------------------------------------------------
# SDKOptions default back-compat: existing callers don't pass a hook
# ---------------------------------------------------------------------------


def test_sdk_options_hook_defaults_to_none() -> None:
    opts = SDKOptions(system_prompt="x", allowed_tools=("Read",), max_turns=1)
    assert opts.pre_tool_use_hook is None
