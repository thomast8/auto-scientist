"""Test configuration: mock unavailable SDK modules before collection."""

import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _block_summarizer_api(monkeypatch):
    """Prevent real OpenAI API calls from the summarizer in all tests."""

    async def _fake_query(*args, **kwargs):
        return "mocked summary"

    monkeypatch.setattr("auto_core.summarizer._query_summary", _fake_query)


# claude_code_sdk may not be installed in CI/test environments.
# Mock it at the sys.modules level so agent modules can be imported in tests.
if "claude_code_sdk" not in sys.modules:
    mock_sdk = ModuleType("claude_code_sdk")
    mock_sdk.__path__ = []  # make it a package so submodule imports work

    # Create mock classes that agents import
    mock_sdk.AssistantMessage = type("AssistantMessage", (), {})
    mock_sdk.ClaudeCodeOptions = MagicMock
    mock_sdk.ClaudeSDKClient = MagicMock
    mock_sdk.Message = type("Message", (), {})
    mock_sdk.PermissionResultAllow = type(
        "PermissionResultAllow", (), {"__init__": lambda self: None}
    )
    mock_sdk.PermissionResultDeny = type(
        "PermissionResultDeny",
        (),
        {"__init__": lambda self, message="": setattr(self, "message", message) or None},
    )
    mock_sdk.ResultMessage = type("ResultMessage", (), {})
    mock_sdk.TextBlock = type("TextBlock", (), {})
    mock_sdk.ToolUseBlock = type("ToolUseBlock", (), {})
    mock_sdk.ToolResultBlock = type("ToolResultBlock", (), {})
    mock_sdk.ToolPermissionContext = type("ToolPermissionContext", (), {})
    mock_sdk.query = MagicMock()
    mock_sdk.McpSdkServerConfig = type("McpSdkServerConfig", (dict,), {})
    mock_sdk.create_sdk_mcp_server = lambda name, version="1.0.0", tools=None: {
        "type": "sdk",
        "name": name,
        "instance": MagicMock(),
    }
    mock_sdk.tool = lambda name, desc, schema: lambda fn: fn  # passthrough decorator
    mock_sdk.SdkMcpTool = type("SdkMcpTool", (), {})

    sys.modules["claude_code_sdk"] = mock_sdk

    # Mock internal subpackages used by sdk_utils.py
    mock_internal = ModuleType("claude_code_sdk._internal")
    sys.modules["claude_code_sdk._internal"] = mock_internal

    mock_client = ModuleType("claude_code_sdk._internal.client")
    mock_client.parse_message = MagicMock()
    sys.modules["claude_code_sdk._internal.client"] = mock_client

    mock_parser = ModuleType("claude_code_sdk._internal.message_parser")
    mock_parser.parse_message = MagicMock()
    sys.modules["claude_code_sdk._internal.message_parser"] = mock_parser

    mock_errors = ModuleType("claude_code_sdk._errors")
    mock_errors.MessageParseError = type("MessageParseError", (Exception,), {})
    sys.modules["claude_code_sdk._errors"] = mock_errors
