"""Module-level dispatch table the orchestrator reads to find agent callables.

The orchestrator does not know whether it's running inside auto-scientist or
auto-reviewer. Each app populates this dispatch table at import time by
calling `auto_core.roles.install(RoleRegistry(..., agent_fns=..., debate=...))`.

Keys under `AGENT_FNS`:

    "ingestor"                 -> run_ingestor(...)
    "analyst"                  -> run_analyst(...)
    "scientist"                -> run_scientist(...)
    "scientist_revision"       -> run_scientist_revision(...)
    "coder"                    -> run_coder(...)
    "report"                   -> run_report(...)
    "debate"                   -> run_debate(...)
    "single_critic_debate"     -> run_single_critic_debate(...)
    "completeness_assessment"  -> run_completeness_assessment(...)
    "scientist_stop_revision"  -> run_scientist_stop_revision(...)
    "single_stop_debate"       -> run_single_stop_debate(...)

Apps can rebind any of these; the orchestrator dispatches through the table
regardless of which concrete implementation is registered. Auto-Reviewer
binds its own review-flavored agents to the same canonical keys.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

# Agent entry functions the orchestrator calls. Populated by
# `auto_core.roles.install(...)`. Empty by default so a registry-less
# `import auto_core.orchestrator` doesn't silently pull the wrong app's
# agents.
AGENT_FNS: dict[str, Callable[..., Any]] = {}

# Debate persona catalog (for the main Adversary / Critic debate). Populated
# by the app's registry. Each entry is a dict with at least {"name",
# "system_text"} and optionally {"instructions"} (see auto_scientist.prompts.
# critic.PERSONAS or auto_reviewer.prompts.adversary.PERSONAS).
DEBATE_PERSONAS: list[dict[str, str]] = []

# Personas that run on iteration 0 of the investigation loop. Others are
# skipped on iter 0 because they depend on prior probe outcomes.
ITERATION_0_PERSONAS: frozenset[str] = frozenset()

# Personas that receive the prediction tree and the
# `mcp__predictions__read_predictions` tool (they lean on prior outcomes).
PREDICTION_PERSONAS: frozenset[str] = frozenset()

# Default critic instructions block (XML snippet appended to every
# non-specialized persona's system prompt).
DEFAULT_CRITIC_INSTRUCTIONS: str = ""

# Persona catalog for the stop-gate debate (sub-orchestration inside the
# stop-gate role).
STOP_PERSONAS: list[dict[str, str]] = []


def _default_get_model_index_for_debate(persona_index: int, iteration: int, num_models: int) -> int:
    """Default model-index rotation: (persona_index + iteration) % num_models."""
    return (persona_index + iteration) % num_models


# Model-index rotation fn for debate. Apps install their own via RoleRegistry.
GET_MODEL_INDEX_FOR_DEBATE: Callable[[int, int, int], int] = _default_get_model_index_for_debate


def get_agent_fn(key: str) -> Callable[..., Any]:
    """Return a dispatcher for the agent function registered under `key`.

    The dispatcher re-resolves the target through its original module on
    every call so `unittest.mock.patch("the.original.module.run_x")`
    still intercepts - the orchestrator's own tests (and test fixtures
    for downstream apps) rely on that patch semantics.

    Raises RuntimeError if the app did not register a binding - that is a
    fatal wiring bug; the orchestrator would otherwise fail silently.
    """
    import importlib
    import sys

    fn = AGENT_FNS.get(key)
    if fn is None:
        raise RuntimeError(
            f"No agent function registered for {key!r}. "
            "The app's _roles.install_*() helper must bind every canonical "
            "agent key before the orchestrator runs."
        )

    module_name = getattr(fn, "__module__", None)
    attr_name = getattr(fn, "__name__", None)
    if not module_name or not attr_name:
        return fn

    def dispatch(*args: Any, **kwargs: Any) -> Any:
        module = sys.modules.get(module_name)
        if module is None:
            module = importlib.import_module(module_name)
        current = getattr(module, attr_name, fn)
        return current(*args, **kwargs)

    return dispatch
