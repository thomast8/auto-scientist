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

from collections.abc import Mapping
from dataclasses import dataclass, field


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
