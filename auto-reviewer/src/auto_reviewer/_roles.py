"""Build and install the auto-reviewer role registry into auto_core.

Called once from `auto_reviewer/__init__.py` at import time. Populates the
core runtime's lookup tables with reviewer-flavored content (Surveyor instead
of Analyst, Hunter instead of Scientist, etc.).

Auto-Reviewer reuses the canonical ModelConfig field names (analyst,
scientist, coder, ingestor, report, assessor, summarizer) so the shared
preset catalog keeps working. The reviewer's Surveyor resolves its model
config from the "analyst" slot, Hunter from "scientist", Prober from
"coder", Intake from "ingestor", Findings from "report", and the stop-gate
Assessor from "assessor".
"""

from collections.abc import Callable
from typing import Any

from auto_core.roles import RoleRegistry, install

AGENT_STYLES: dict[str, str] = {
    "Surveyor": "green",
    "Hunter": "cyan",
    "Prober": "magenta1",
    "Intake": "bright_red",
    "Findings": "blue",
    "Adversary": "yellow",
    "Debate": "yellow",
    "Assessor": "blue",
    "Results": "dim",
    # Also honour the canonical names in case any legacy panel is created
    # by a shared helper before the reviewer's adapter kicks in.
    "Analyst": "green",
    "Scientist": "cyan",
    "Coder": "magenta1",
    "Ingestor": "bright_red",
    "Report": "blue",
    "Critic": "yellow",
}

_INTAKE_DESCRIPTION = "Fetching the PR diff, base refs, and building the review workspace..."
_SURVEYOR_DESCRIPTION = (
    "Reading the diff and probe results; surfacing suspicions and resolutions..."
)
_HUNTER_DESCRIPTION = "Picking which suspicions to chase and writing reproduction recipes..."
_ADVERSARY_DESCRIPTION = (
    "Challenging the Hunter's plan from security, concurrency, API-break, and fuzz angles..."
)
_PROBER_DESCRIPTION = "Writing and running reproduction probes inside review_workspace/..."
_FINDINGS_DESCRIPTION = "Compiling the prioritized findings report with reproducers attached..."

AGENT_DESCRIPTIONS: dict[str, str] = {
    # Reviewer-flavored (display) keys.
    "Intake": _INTAKE_DESCRIPTION,
    "Surveyor": _SURVEYOR_DESCRIPTION,
    "Hunter": _HUNTER_DESCRIPTION,
    "Adversary": _ADVERSARY_DESCRIPTION,
    "Revision": "Revising the BugPlan based on adversary critique...",
    "Prober": _PROBER_DESCRIPTION,
    "Findings": _FINDINGS_DESCRIPTION,
    "Assessor": "Evaluating review coverage against the stated review goal...",
    "Stop Revision": "Deciding whether to uphold or withdraw the stop proposal...",
    # Canonical fallbacks: AgentPanel construction passes the canonical name
    # (Ingestor/Analyst/Scientist/Critic/Coder/Report), but reviewer panels
    # are rendered under display names. Mirror the reviewer-framed text under
    # the canonical keys so the placeholder shows for every launched agent,
    # matching the canonical-fallback pattern used by SUMMARY_PROMPTS below.
    "Ingestor": _INTAKE_DESCRIPTION,
    "Analyst": _SURVEYOR_DESCRIPTION,
    "Scientist": _HUNTER_DESCRIPTION,
    "Critic": _ADVERSARY_DESCRIPTION,
    "Coder": _PROBER_DESCRIPTION,
    "Report": _FINDINGS_DESCRIPTION,
}

PHASE_STYLES: dict[str, str] = {
    "INGESTION": "bright_red",
    "ANALYZE": "green",
    "PLAN": "cyan",
    "DEBATE": "yellow",
    "REVISE": "cyan",
    "IMPLEMENT": "magenta1",
    "REPORT": "blue",
    "ASSESS": "blue",
    "STOP_DEBATE": "yellow",
    "STOP_REVISE": "cyan",
}

SUMMARY_PROMPTS: dict[str, str] = {
    "Intake": (
        "Summarize this intake output in first person. "
        "Focus on: the PR's scope, which files changed, and what review workspace was built."
    ),
    "Surveyor": (
        "Summarize this survey output in first person. "
        "Focus on: which suspicions were surfaced, which touched symbols stand out, "
        "and how prior probes resolved."
    ),
    "Hunter": (
        "Summarize this bug-plan output in first person. "
        "Focus on: which suspected bug is being chased this iteration and what "
        "the reproduction recipe looks like."
    ),
    "Hunter Revision": (
        "Summarize this plan-revision output in first person. "
        "Focus on: what changed from the original BugPlan and which adversary "
        "challenges drove the revision."
    ),
    "Adversary": (
        "Summarize this adversary debate output in first person. "
        "Messages are prefixed [Adversary] or [Hunter]. Focus on: what is being "
        "challenged and what repro failure modes are being surfaced."
    ),
    "Prober": (
        "Summarize this probe output in first person. "
        "Focus on: what repro script or test was written and what approach was taken. "
        "Describe the script structure, not line-by-line details."
    ),
    "Results": (
        "Summarize these probe results in first person. "
        "Focus on: whether the bug reproduced, the signal the probe caught, "
        "and how that resolves the suspected bug."
    ),
    "Findings": (
        "Summarize this findings output in first person. "
        "Focus on: which bugs were confirmed, which suspicions were refuted, "
        "and what open abductions remain."
    ),
    "Completeness Assessment": (
        "Summarize this completeness assessment output in first person. "
        "Focus on: which review sub-questions remain shallow or unexplored."
    ),
    "Stop Debate": (
        "Summarize this stop debate output in first person. "
        "Messages are prefixed [Critic] or [Hunter]. Focus on: whether continuing "
        "or stopping the review is being advocated and why."
    ),
    "Stop Revision": (
        "Summarize this stop revision output in first person. "
        "Focus on: whether the stop decision was upheld or withdrawn and what "
        "gaps or next steps were identified."
    ),
    # Canonical fallbacks.
    "Ingestor": "Summarize intake output in first person.",
    "Analyst": "Summarize surveyor output in first person.",
    "Scientist": "Summarize hunter output in first person.",
    "Coder": "Summarize prober output in first person.",
    "Report": "Summarize findings output in first person.",
    "Debate": "Summarize adversary debate output in first person.",
}

# Artifact + buffer mapping: Auto-Reviewer uses the same canonical on-disk
# filenames as Auto-Scientist so the shared persistence + resume code needs
# no special-casing. (Review-specific filenames can come later if the
# shared tooling grows to care.)
ARTIFACT_SPECS: dict[str, list[str]] = {
    "analyst": ["analysis.json"],  # Surveyor
    "scientist": ["plan.json"],  # Hunter
    "assessment": ["completeness_assessment.json"],
    "stop_debate": ["stop_debate.json"],
    "stop_revision": ["stop_revision_plan.json"],
    "debate": ["debate.json"],
    "revision": ["revision_plan.json"],
    "coder": [],  # Prober
}

BUFFER_PREFIXES: dict[str, list[str]] = {
    "analyst": ["analyst_", "surveyor_"],
    "scientist": ["scientist_", "hunter_"],
    "assessment": ["completeness_assessment_"],
    "stop_debate": ["stop_debate_"],
    "stop_revision": ["stop_revision_"],
    "debate": ["debate_", "adversary_"],
    "revision": ["scientist_revision_", "hunter_revision_"],
    "coder": ["coder_", "prober_"],
}

NOTEBOOK_SOURCES: dict[str, list[str]] = {
    "analyst": [],
    "scientist": ["scientist", "hunter"],
    "assessment": [],
    "stop_debate": [],
    "stop_revision": ["stop_revision", "stop_gate"],
    "debate": [],
    "revision": ["revision"],
    "coder": [],
}

AGENT_FIELDS: frozenset[str] = frozenset(
    {"analyst", "scientist", "coder", "ingestor", "report", "summarizer", "assessor"}
)


def _build_agent_fns() -> dict[str, Callable[..., Any]]:
    """Return the reviewer's agent entry functions, keyed by generic role.

    `canonicalizer` resolves to a deterministic Python function
    (`run_intake`) - PR canonicalization is shell commands, not LLM work.
    The rest are LLM-driven agents mirroring auto-scientist's structural
    machinery with reviewer-flavored prompts + schemas.
    """
    from auto_reviewer.agents.adversary import run_debate, run_single_critic_debate
    from auto_reviewer.agents.findings import run_findings
    from auto_reviewer.agents.hunter import run_hunter, run_hunter_revision
    from auto_reviewer.agents.intake import run_intake
    from auto_reviewer.agents.prober import run_prober
    from auto_reviewer.agents.stop_gate import (
        run_completeness_assessment,
        run_scientist_stop_revision,
        run_single_stop_debate,
    )
    from auto_reviewer.agents.surveyor import run_surveyor

    return {
        "canonicalizer": run_intake,
        "observer": run_surveyor,
        "planner": run_hunter,
        "reviser": run_hunter_revision,
        "implementer": run_prober,
        "reporter": run_findings,
        "adversary": run_debate,
        "single_adversary": run_single_critic_debate,
        "assessor": run_completeness_assessment,
        "stop_reviser": run_scientist_stop_revision,
        "stop_adversary": run_single_stop_debate,
    }


def build_registry() -> RoleRegistry:
    """Return the auto-reviewer RoleRegistry."""
    from auto_reviewer.prompts.adversary import (
        DEFAULT_CRITIC_INSTRUCTIONS,
        ITERATION_0_PERSONAS,
        PERSONAS,
        PREDICTION_PERSONAS,
        get_model_index_for_debate,
    )
    from auto_reviewer.prompts.stop_gate import STOP_PERSONAS

    return RoleRegistry(
        agent_styles=AGENT_STYLES,
        agent_descriptions=AGENT_DESCRIPTIONS,
        phase_styles=PHASE_STYLES,
        summary_prompts=SUMMARY_PROMPTS,
        artifact_specs=ARTIFACT_SPECS,
        buffer_prefixes=BUFFER_PREFIXES,
        notebook_sources=NOTEBOOK_SOURCES,
        agent_fields=AGENT_FIELDS,
        agent_fns=_build_agent_fns(),
        debate_personas=PERSONAS,
        iteration_0_personas=ITERATION_0_PERSONAS,
        prediction_personas=PREDICTION_PERSONAS,
        default_critic_instructions=DEFAULT_CRITIC_INSTRUCTIONS,
        stop_personas=STOP_PERSONAS,
        get_model_index_for_debate=get_model_index_for_debate,
        app_label="Auto-Reviewer",
        banner_agents_before_critics=[
            ("Intake", "ingestor"),
            ("Surveyor", "analyst"),
            ("Hunter", "scientist"),
        ],
        banner_agents_after_critics=[
            ("Prober", "coder"),
            ("Findings", "report"),
        ],
        banner_critic_label="Adversary",
        panel_display_names={
            "Ingestor": "Intake",
            "Analyst": "Surveyor",
            "Scientist": "Hunter",
            "Coder": "Prober",
            "Report": "Findings",
            "Critic": "Adversary",
            "Revision": "Hunter Revision",
        },
        # The Findings agent's report shape is review-oriented, not
        # experiment-oriented: no "executive summary", "methodology",
        # "journey", "version comparison" sections - those belong to
        # auto-scientist. The reviewer produces a punch-list.
        report_expected_headings=[
            "summary",
            "confirmed bugs",
            "refuted suspicions",
            "ungrounded findings",
            "open abductions",
            "known limitations",
        ],
        report_require_version_comparison_table=False,
    )


def install_reviewer_registry() -> None:
    """Install the auto-reviewer role registry into auto_core."""
    install(build_registry())
