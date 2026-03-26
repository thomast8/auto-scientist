"""Auto-scientist UI package. Re-exports all public names for backward compatibility."""

from auto_scientist.ui.app import PipelineApp
from auto_scientist.ui.bridge import PipelineLive
from auto_scientist.ui.commands import PipelineCommandProvider
from auto_scientist.ui.detail_screen import AgentDetailScreen, QuitConfirmScreen
from auto_scientist.ui.styles import (
    AGENT_DESCRIPTIONS,
    AGENT_STYLES,
    PHASE_STYLES,
    _format_elapsed,
    _load_prefs,
    _save_prefs,
    _score_style,
    console,
)
from auto_scientist.ui.widgets import AgentPanel, IterationContainer, MetricsBar

__all__ = [
    "AGENT_DESCRIPTIONS",
    "AGENT_STYLES",
    "AgentDetailScreen",
    "AgentPanel",
    "IterationContainer",
    "MetricsBar",
    "PHASE_STYLES",
    "PipelineApp",
    "PipelineCommandProvider",
    "PipelineLive",
    "QuitConfirmScreen",
    "_format_elapsed",
    "_load_prefs",
    "_save_prefs",
    "_score_style",
    "console",
]
