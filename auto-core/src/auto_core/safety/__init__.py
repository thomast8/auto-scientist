"""Cross-app safety primitives used by the orchestrator and SDK backends.

`tool_guard` produces PreToolUse hooks that confine agent tool calls to a
workspace directory and reject a fixed allowlist of destructive shell
verbs. The guard is one of several defence layers (the others live in the
app that owns the workspace): it is intentionally strict and simple.
"""

from auto_core.safety.tool_guard import (
    Decision,
    GuardMode,
    PreToolUseHook,
    make_workspace_guard,
)

__all__ = [
    "Decision",
    "GuardMode",
    "PreToolUseHook",
    "make_workspace_guard",
]
