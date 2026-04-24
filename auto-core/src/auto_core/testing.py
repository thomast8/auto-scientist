"""Shared test-harness helpers for auto-scientist and auto-reviewer.

Guards a dev box where the real ``claude_code_sdk`` is installed from spawning
live CLI subagents during ``pytest``. See the 2026-04-23 zombie-CLI incident.
"""

from __future__ import annotations

import os
import sys
from types import ModuleType
from typing import Any
from unittest.mock import MagicMock

import pytest

_LIVE_CLAUDE_ENV = "LIVE_CLAUDE"


def install_claude_sdk_mock() -> None:
    """Swap a mock ``claude_code_sdk`` into sys.modules when the real one is absent.

    Call from conftest *before* importing any agent code so tests remain
    importable on CI / machines that do not have ``claude-code-sdk`` installed.
    If the real SDK is already imported, this is a no-op — the
    ``block_live_claude_sdk`` fixture is what prevents live spawns on dev boxes.
    """
    if "claude_code_sdk" in sys.modules:
        return

    mock_sdk: Any = ModuleType("claude_code_sdk")
    mock_sdk.__path__ = []

    mock_sdk.AssistantMessage = type("AssistantMessage", (), {})
    mock_sdk.ClaudeCodeOptions = MagicMock
    mock_sdk.ClaudeSDKClient = MagicMock
    mock_sdk.Message = type("Message", (), {})

    def _deny_init(self: Any, message: str = "", interrupt: bool = False) -> None:
        self.message = message
        self.interrupt = interrupt

    mock_sdk.PermissionResultAllow = type(
        "PermissionResultAllow", (), {"__init__": lambda self: None}
    )
    mock_sdk.PermissionResultDeny = type("PermissionResultDeny", (), {"__init__": _deny_init})
    mock_sdk.ResultMessage = type("ResultMessage", (), {})
    mock_sdk.TextBlock = type("TextBlock", (), {})
    mock_sdk.ToolUseBlock = type("ToolUseBlock", (), {})
    mock_sdk.ToolResultBlock = type("ToolResultBlock", (), {})
    mock_sdk.ToolPermissionContext = type("ToolPermissionContext", (), {})
    mock_sdk.HookContext = type("HookContext", (), {"__init__": lambda self: None})

    def _hook_matcher_init(self: Any, matcher: str | None = None, hooks: Any = None) -> None:
        self.matcher = matcher
        self.hooks = hooks if hooks is not None else []

    mock_sdk.HookMatcher = type("HookMatcher", (), {"__init__": _hook_matcher_init})
    mock_sdk.query = MagicMock()
    mock_sdk.McpSdkServerConfig = type("McpSdkServerConfig", (dict,), {})
    mock_sdk.create_sdk_mcp_server = lambda name, version="1.0.0", tools=None: {
        "type": "sdk",
        "name": name,
        "instance": MagicMock(),
    }
    mock_sdk.tool = lambda name, desc, schema: lambda fn: fn
    mock_sdk.SdkMcpTool = type("SdkMcpTool", (), {})

    sys.modules["claude_code_sdk"] = mock_sdk

    # `HookJSONOutput` lives under claude_code_sdk.types; stub the
    # submodule so `from claude_code_sdk.types import HookJSONOutput`
    # resolves under the mock.
    mock_types: Any = ModuleType("claude_code_sdk.types")
    mock_types.HookJSONOutput = dict
    mock_types.HookContext = mock_sdk.HookContext
    mock_types.HookMatcher = mock_sdk.HookMatcher
    mock_types.PermissionResultAllow = mock_sdk.PermissionResultAllow
    mock_types.PermissionResultDeny = mock_sdk.PermissionResultDeny
    mock_types.ToolPermissionContext = mock_sdk.ToolPermissionContext
    sys.modules["claude_code_sdk.types"] = mock_types

    mock_internal = ModuleType("claude_code_sdk._internal")
    sys.modules["claude_code_sdk._internal"] = mock_internal

    mock_client: Any = ModuleType("claude_code_sdk._internal.client")
    mock_client.parse_message = MagicMock()
    sys.modules["claude_code_sdk._internal.client"] = mock_client

    mock_parser: Any = ModuleType("claude_code_sdk._internal.message_parser")
    mock_parser.parse_message = MagicMock()
    sys.modules["claude_code_sdk._internal.message_parser"] = mock_parser

    mock_errors: Any = ModuleType("claude_code_sdk._errors")
    mock_errors.MessageParseError = type("MessageParseError", (Exception,), {})
    sys.modules["claude_code_sdk._errors"] = mock_errors


def is_live_claude_allowed() -> bool:
    """Return True when the suite is explicitly opted in to real CLI invocation."""
    return os.environ.get(_LIVE_CLAUDE_ENV) == "1"


def install_live_claude_block(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch ``auto_core.sdk_backend.claude_query`` with a raiser and reset cost state.

    Intended as the body of an autouse fixture in each test conftest.
    When ``LIVE_CLAUDE=1`` is set the patch is skipped so smoke tests marked
    ``@pytest.mark.live`` can reach the real CLI; the cost accumulator is
    still reset so each test starts from a clean budget.
    """
    from auto_core.cost_ceiling import reset_budget

    reset_budget()

    if is_live_claude_allowed():
        return

    async def _blocked(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError(
            "Live Claude CLI spawn blocked in tests. "
            "Patch auto_core.sdk_backend.claude_query in your test, or set "
            "LIVE_CLAUDE=1 and mark the test with @pytest.mark.live."
        )
        yield  # pragma: no cover -- makes this an async generator

    monkeypatch.setattr("auto_core.sdk_backend.claude_query", _blocked)
