"""Role registry: one place an app declares its agent styles, descriptions,
artifacts, summary prompts, and ModelConfig field names.

The auto_core machinery (widgets, summarizer, persistence, resume, model_config)
keeps its legacy module-level dicts (AGENT_STYLES, AGENT_DESCRIPTIONS,
PHASE_STYLES, SUMMARY_PROMPTS, _AGENT_ARTIFACTS, _AGENT_BUFFER_PREFIXES,
_AGENT_FIELDS) so existing call sites keep working unchanged. `RoleRegistry`
is the single input that populates them at app bootstrap.

Usage:

    from auto_core.roles import RoleRegistry, install
    install(RoleRegistry(
        agent_styles={"Analyst": "green", ...},
        ...
    ))

This is a CLI-process global. Running two apps in the same process would
clobber each other's registries - that is by design (each CLI entrypoint owns
its own process).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RoleRegistry:
    """Everything the core machinery needs to know about the app's agents."""

    # Keyed by agent display name (the string that appears in panels / logs).
    agent_styles: Mapping[str, str]
    agent_descriptions: Mapping[str, str]
    summary_prompts: Mapping[str, str]

    # Keyed by orchestrator phase name (uppercase: "ANALYZE", "PLAN", etc.).
    phase_styles: Mapping[str, str]

    # Keyed by canonical role identifier used in iteration manifests
    # ("analyst", "scientist", "coder", ...). artifact_specs gives the list of
    # JSON artifact filenames the role writes; buffer_prefixes gives the
    # message-buffer prefixes it logs.
    artifact_specs: Mapping[str, list[str]] = field(default_factory=dict)
    buffer_prefixes: Mapping[str, list[str]] = field(default_factory=dict)
    notebook_sources: Mapping[str, list[str]] = field(default_factory=dict)

    # ModelConfig field names valid for `.resolve()`. Replaces the hardcoded
    # `_AGENT_FIELDS` ClassVar in auto_core.model_config.ModelConfig.
    agent_fields: frozenset[str] = field(default_factory=frozenset)

    # Agent dispatch table. Keys are canonical agent identifiers the
    # orchestrator looks up: ingestor, analyst, scientist, scientist_revision,
    # coder, report, debate, single_critic_debate, completeness_assessment,
    # scientist_stop_revision, single_stop_debate. Each value is the async
    # function that implements that role for this app.
    agent_fns: Mapping[str, Callable[..., Any]] = field(default_factory=dict)

    # Adversary debate catalog. Each entry has at least {"name", "system_text"}
    # and optionally {"instructions"}.
    debate_personas: list[dict[str, str]] = field(default_factory=list)

    # Personas active only from iteration 1+ (skipped on iter 0).
    iteration_0_personas: frozenset[str] = field(default_factory=frozenset)

    # Personas that receive the prediction tree + MCP prediction tool.
    prediction_personas: frozenset[str] = field(default_factory=frozenset)

    # Default critic instructions block appended to non-specialized personas.
    default_critic_instructions: str = ""

    # Stop-gate debate persona catalog.
    stop_personas: list[dict[str, str]] = field(default_factory=list)

    # Model-index rotation fn for debate: (persona_index, iteration, num_models) -> index.
    get_model_index_for_debate: Callable[[int, int, int], int] | None = None

    # Startup-banner customization. The TUI banner (title + per-agent model
    # rows) is rendered by `auto_core.orchestrator._build_startup_banner`,
    # which reads from `auto_core.widgets` at call time. Defaults carry
    # scientist-flavored labels so auto-scientist works without setting
    # anything explicit here.
    app_label: str = "Auto-Scientist"
    banner_agents_before_critics: list[tuple[str, str]] = field(
        default_factory=lambda: [
            ("Ingestor", "ingestor"),
            ("Analyst", "analyst"),
            ("Scientist", "scientist"),
        ]
    )
    banner_agents_after_critics: list[tuple[str, str]] = field(
        default_factory=lambda: [
            ("Coder", "coder"),
            ("Report", "report"),
        ]
    )
    banner_critic_label: str = "Critic"

    # Per-panel rename map. Canonical panel name -> app-specific display name
    # (e.g. "Analyst" -> "Surveyor"). Also works for "Prefix/Detail" shapes:
    # "Critic" mapped to "Adversary" rewrites "Critic/Security" to
    # "Adversary/Security". Default is identity. The canonical name is kept
    # for persistence / state files.
    panel_display_names: Mapping[str, str] = field(default_factory=dict)


def install(registry: RoleRegistry) -> None:
    """Populate the core machinery's module-level lookup tables.

    Called once per CLI process, before the orchestrator starts. Clears
    existing entries so repeated calls behave deterministically.
    """
    from auto_core import model_config, summarizer, widgets

    widgets.AGENT_STYLES.clear()
    widgets.AGENT_STYLES.update(dict(registry.agent_styles))
    widgets.AGENT_DESCRIPTIONS.clear()
    widgets.AGENT_DESCRIPTIONS.update(dict(registry.agent_descriptions))
    widgets.PHASE_STYLES.clear()
    widgets.PHASE_STYLES.update(dict(registry.phase_styles))

    widgets.APP_LABEL = registry.app_label
    widgets.BANNER_AGENTS_BEFORE_CRITICS = list(registry.banner_agents_before_critics)
    widgets.BANNER_AGENTS_AFTER_CRITICS = list(registry.banner_agents_after_critics)
    widgets.BANNER_CRITIC_LABEL = registry.banner_critic_label
    widgets.PANEL_DISPLAY_NAMES.clear()
    widgets.PANEL_DISPLAY_NAMES.update(dict(registry.panel_display_names))

    summarizer.SUMMARY_PROMPTS.clear()
    summarizer.SUMMARY_PROMPTS.update(dict(registry.summary_prompts))

    # resume.py lives in auto_scientist today; import lazily so tests that
    # import auto_core.roles without resume don't fail.
    try:
        from auto_core import resume as _resume

        _resume._AGENT_ARTIFACTS.clear()
        _resume._AGENT_ARTIFACTS.update({k: list(v) for k, v in registry.artifact_specs.items()})
        _resume._AGENT_BUFFER_PREFIXES.clear()
        _resume._AGENT_BUFFER_PREFIXES.update(
            {k: list(v) for k, v in registry.buffer_prefixes.items()}
        )
        _resume._AGENT_NOTEBOOK_SOURCES.clear()
        _resume._AGENT_NOTEBOOK_SOURCES.update(
            {k: list(v) for k, v in registry.notebook_sources.items()}
        )
    except ImportError:
        pass

    model_config.install_agent_fields(registry.agent_fields)

    # Populate the agent dispatch table the orchestrator reads.
    from auto_core import agent_dispatch

    agent_dispatch.AGENT_FNS.clear()
    # Auto-migrate legacy scientist-canonical keys (ingestor / analyst /
    # scientist / etc.) to their generic-role equivalents so the orchestrator
    # only has to know the new vocabulary.
    for key, fn in registry.agent_fns.items():
        canonical = agent_dispatch.LEGACY_KEY_MAP.get(key, key)
        agent_dispatch.AGENT_FNS[canonical] = fn
    agent_dispatch.DEBATE_PERSONAS[:] = list(registry.debate_personas)
    agent_dispatch.ITERATION_0_PERSONAS = frozenset(registry.iteration_0_personas)
    agent_dispatch.PREDICTION_PERSONAS = frozenset(registry.prediction_personas)
    agent_dispatch.DEFAULT_CRITIC_INSTRUCTIONS = registry.default_critic_instructions
    agent_dispatch.STOP_PERSONAS[:] = list(registry.stop_personas)
    if registry.get_model_index_for_debate is not None:
        agent_dispatch.GET_MODEL_INDEX_FOR_DEBATE = registry.get_model_index_for_debate
