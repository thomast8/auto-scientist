"""Module-level dispatch table the orchestrator reads to find agent callables.

The orchestrator does not know whether it's running inside auto-scientist or
auto-reviewer. Each app populates this dispatch table at import time by
calling `auto_core.roles.install(RoleRegistry(..., agent_fns=..., debate=...))`.

Keys under `AGENT_FNS` are generic role names, deliberately neither
scientist- nor reviewer-flavored. Auto-Scientist binds its LLM-driven
ingestor/analyst/scientist/... to these keys; Auto-Reviewer binds its own
intake/surveyor/hunter/... (and in the canonicalizer case, a plain Python
function - not every role has to be LLM-backed).

    "canonicalizer"     -> prepare the raw input (ingestor / intake)
    "observer"          -> read results, emit JSON (analyst / surveyor)
    "planner"           -> decide what to try next (scientist / hunter)
    "reviser"           -> revise the plan post-debate (scientist_revision /
                           hunter_revision)
    "adversary"         -> debate the plan (critic / adversary catalog)
    "single_adversary"  -> one persona's critique (single_critic_debate)
    "implementer"       -> write + run code (coder / prober)
    "reporter"          -> compile the final report (report / findings)
    "assessor"          -> stop-gate: completeness assessment
    "stop_reviser"      -> stop-gate: revise the stop decision
    "stop_adversary"    -> stop-gate: one persona's stop critique

The ModelConfig fields themselves keep scientist-canonical names (analyst,
scientist, coder, ingestor, report, assessor, summarizer) so user-facing
presets / TOML configs don't break. `MODEL_CONFIG_SLOT_FOR_ROLE` translates
generic role keys to the model-config slot to resolve against.
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


# Mapping from generic role key to the ModelConfig field the orchestrator
# should resolve. The ModelConfig keeps scientist-canonical slot names (users
# configure per-agent models via `analyst:`, `scientist:`, etc. in their
# TOML / preset). Review-only roles point to the same slot as their science
# analogue.
MODEL_CONFIG_SLOT_FOR_ROLE: dict[str, str] = {
    "canonicalizer": "ingestor",
    "observer": "analyst",
    "planner": "scientist",
    "reviser": "scientist",
    "implementer": "coder",
    "reporter": "report",
    "assessor": "assessor",
    "stop_reviser": "scientist",
    # adversary / single_adversary / stop_adversary resolve via
    # model_config.critics, not a single slot.
}


# Legacy -> generic key migration. Apps that still populate AGENT_FNS with
# the old scientist-canonical keys get auto-migrated at install time. Makes
# the rename one-way-compatible for any test fixtures lagging behind.
LEGACY_KEY_MAP: dict[str, str] = {
    "ingestor": "canonicalizer",
    "analyst": "observer",
    "scientist": "planner",
    "scientist_revision": "reviser",
    "coder": "implementer",
    "report": "reporter",
    "debate": "adversary",
    "single_critic_debate": "single_adversary",
    "completeness_assessment": "assessor",
    "scientist_stop_revision": "stop_reviser",
    "single_stop_debate": "stop_adversary",
}


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
