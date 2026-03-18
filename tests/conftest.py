"""Test configuration: mock unavailable SDK modules before collection."""

import sys
from types import ModuleType
from unittest.mock import MagicMock

# claude_code_sdk may not be installed in CI/test environments.
# Mock it at the sys.modules level so agent modules can be imported in tests.
if "claude_code_sdk" not in sys.modules:
    mock_sdk = ModuleType("claude_code_sdk")

    # Create mock classes that agents import
    mock_sdk.AssistantMessage = type("AssistantMessage", (), {})
    mock_sdk.ClaudeCodeOptions = MagicMock
    mock_sdk.ClaudeSDKClient = MagicMock
    mock_sdk.PermissionResultAllow = type("PermissionResultAllow", (), {"__init__": lambda self: None})
    mock_sdk.PermissionResultDeny = type(
        "PermissionResultDeny", (), {"__init__": lambda self, message="": setattr(self, "message", message) or None}
    )
    mock_sdk.ResultMessage = type("ResultMessage", (), {})
    mock_sdk.TextBlock = type("TextBlock", (), {})
    mock_sdk.ToolPermissionContext = type("ToolPermissionContext", (), {})
    mock_sdk.query = MagicMock()

    sys.modules["claude_code_sdk"] = mock_sdk
