"""Backward-compat shim. PipelineApp et al. live in auto_core.app now.

Auto-Reviewer and Auto-Scientist share the same Textual UI; the class moved
to `auto_core.app` so the reviewer can import it without triggering the
scientist registry install as a side effect.
"""

from auto_core.app import (  # noqa: F401 - re-export for back-compat
    AgentDetailScreen,
    PipelineApp,
    PipelineCommandProvider,
    QuitConfirmScreen,
    ShowApp,
)
