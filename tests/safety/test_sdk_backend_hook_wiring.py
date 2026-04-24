"""Phase C wire-up tests.

Exercises the machinery between `SDKOptions.pre_tool_use_hook`, the
Claude PreToolUse-hook adapter, and the Codex cwd/workspace assertion —
without spawning any SDK subprocess (which would require real
credentials). The real-SDK smoke check lives in the integration test
suite.

Why PreToolUse hooks and not `can_use_tool`: the SDK skips
`can_use_tool` entirely for any tool listed in `allowed_tools`. For a
workspace guard whose whole point is catching escapes, that bypass is
unacceptable. PreToolUse hooks run on every tool call regardless of
allowlist — they are the right primitive.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from auto_core.safety.tool_guard import make_workspace_guard
from auto_core.sdk_backend import (
    CodexBackend,
    SDKOptions,
    _make_claude_pretooluse_hook,
)
from claude_code_sdk import HookContext


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


def _hook_input(tool_name: str, tool_input: dict) -> dict:
    """Shape matching the SDK's PreToolUse hook input contract."""
    return {
        "hook_event_name": "PreToolUse",
        "tool_name": tool_name,
        "tool_input": tool_input,
        "session_id": "s1",
        "cwd": "/tmp",
    }


# ---------------------------------------------------------------------------
# Claude PreToolUse adapter: translates guard Decisions into HookJSONOutput
# ---------------------------------------------------------------------------


async def test_claude_adapter_allow_returns_empty_dict(workspace: Path, repo_clone: Path) -> None:
    guard = make_workspace_guard(workspace, repo_clone, mode="intake")
    adapter = _make_claude_pretooluse_hook(guard)
    result = await adapter(
        _hook_input("Read", {"file_path": "/etc/passwd"}),
        None,
        HookContext(),
    )
    # Allow: empty dict — SDK treats missing `decision` as continue.
    assert result == {}


async def test_claude_adapter_denies_write_outside_workspace(
    tmp_path: Path, workspace: Path, repo_clone: Path
) -> None:
    guard = make_workspace_guard(workspace, repo_clone, mode="intake")
    adapter = _make_claude_pretooluse_hook(guard)
    elsewhere = tmp_path / "not_in_workspace"
    elsewhere.mkdir()
    result = await adapter(
        _hook_input("Write", {"file_path": str(elsewhere / "evil.py")}),
        None,
        HookContext(),
    )
    hso = result["hookSpecificOutput"]
    assert hso["permissionDecision"] == "deny"
    assert "outside the review workspace" in hso["permissionDecisionReason"]


async def test_claude_adapter_denies_destructive_bash(workspace: Path, repo_clone: Path) -> None:
    guard = make_workspace_guard(workspace, repo_clone, mode="probe")
    adapter = _make_claude_pretooluse_hook(guard)
    result = await adapter(
        _hook_input("Bash", {"command": "rm -rf /"}),
        None,
        HookContext(),
    )
    hso = result["hookSpecificOutput"]
    assert hso["permissionDecision"] == "deny"
    assert "recursive" in hso["permissionDecisionReason"].lower()


async def test_claude_adapter_handles_missing_tool_name_gracefully(
    workspace: Path, repo_clone: Path
) -> None:
    guard = make_workspace_guard(workspace, repo_clone, mode="probe")
    adapter = _make_claude_pretooluse_hook(guard)
    # Unknown tool defaults to deny in the guard
    result = await adapter({"session_id": "s"}, None, HookContext())
    hso = result["hookSpecificOutput"]
    assert hso["permissionDecision"] == "deny"


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
