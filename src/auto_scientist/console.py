"""Backward-compatibility shim. All implementations live in auto_scientist.ui."""

from auto_scientist.ui import (  # noqa: F401
    AGENT_DESCRIPTIONS,
    AGENT_STYLES,
    PHASE_STYLES,
    AgentDetailScreen,
    AgentPanel,
    IterationContainer,
    MetricsBar,
    PipelineApp,
    PipelineCommandProvider,
    PipelineLive,
    QuitConfirmScreen,
    _format_elapsed,
    _load_prefs,
    _save_prefs,
    _score_style,
    console,
)
