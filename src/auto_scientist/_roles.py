"""Build and install the auto-scientist role registry into auto_core.

Called once from `auto_scientist/__init__.py` at import time. Populates the
core's style / description / phase / summary / artifact / notebook-source
lookup tables with the scientist-flavored content that used to be hardcoded
in `widgets.py`, `summarizer.py`, and `resume.py`.
"""

from collections.abc import Callable
from typing import Any

from auto_core.roles import RoleRegistry, install

AGENT_STYLES: dict[str, str] = {
    "Analyst": "green",
    "Scientist": "cyan",
    "Coder": "magenta1",
    "Ingestor": "bright_red",
    "Report": "blue",
    "Critic": "yellow",
    "Debate": "yellow",
    "Assessor": "blue",
    "Results": "dim",
}

AGENT_DESCRIPTIONS: dict[str, str] = {
    "Ingestor": "Preparing and canonicalizing raw data for experiment scripts...",
    "Analyst": "Analyzing experiment results and producing quantitative assessments...",
    "Scientist": "Formulating hypotheses and planning the next experiment...",
    "Critic": "Challenging the plan through critical debate...",
    "Revision": "Revising the experiment plan based on critique feedback...",
    "Coder": "Implementing the experiment plan as a Python script and running it...",
    "Report": "Generating a comprehensive summary report of all findings...",
    "Assessor": "Evaluating investigation completeness against the stated goal...",
    "Stop Revision": "Deciding whether to uphold or withdraw the stop proposal...",
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
    "Ingestor": (
        "Summarize this data ingestion output in first person. "
        "Focus on: what files were processed and what transformations were applied."
    ),
    "Analyst": (
        "Summarize this analysis output in first person. "
        "Focus on: what findings were observed, what metrics stand out, "
        "any improvements or regressions."
    ),
    "Scientist": (
        "Summarize this planning output in first person. "
        "Focus on: what hypothesis and strategy are being formulated, "
        "what is the expected impact."
    ),
    "Scientist Revision": (
        "Summarize this plan revision output in first person. "
        "Focus on: what changed from the original plan, "
        "what revisions were adopted from the debate."
    ),
    "Debate": (
        "Summarize this debate output in first person. "
        "Messages are prefixed [Critic] or [Scientist]. "
        "Do not prefix with 'Critic:' since the panel already identifies who this is. "
        "Focus on: what is being challenged and what positions are forming."
    ),
    "Coder": (
        "Summarize this coding output in first person. "
        "Focus on: what code was written and what approach was taken. "
        "Describe the script structure, not line-by-line details."
    ),
    "Results": (
        "Summarize these experiment results in first person. "
        "Focus on: what the experiment produced, key numeric outcomes, "
        "whether the hypothesis was supported, comparison to previous iteration."
    ),
    "Report": (
        "Summarize this report output in first person. "
        "Focus on: what key findings and results are being documented."
    ),
    "Completeness Assessment": (
        "Summarize this completeness assessment output in first person. "
        "Focus on: what coverage gaps were identified, "
        "which sub-questions remain shallow or unexplored."
    ),
    "Stop Debate": (
        "Summarize this stop debate output in first person. "
        "Messages are prefixed [Critic] or [Scientist]. "
        "Do not prefix with 'Critic:' since the panel already identifies who this is. "
        "Focus on: whether continuing or stopping is being advocated and the key reasons."
    ),
    "Stop Revision": (
        "Summarize this stop revision output in first person. "
        "Focus on: whether the stop decision was upheld or withdrawn, "
        "and what gaps or next steps were identified."
    ),
}

ARTIFACT_SPECS: dict[str, list[str]] = {
    "analyst": ["analysis.json"],
    "scientist": ["plan.json"],
    "assessment": ["completeness_assessment.json"],
    "stop_debate": ["stop_debate.json"],
    "stop_revision": ["stop_revision_plan.json"],
    "debate": ["debate.json"],
    "revision": ["revision_plan.json"],
    "coder": [],
}

BUFFER_PREFIXES: dict[str, list[str]] = {
    "analyst": ["analyst_"],
    "scientist": ["scientist_"],
    "assessment": ["completeness_assessment_"],
    "stop_debate": ["stop_debate_"],
    "stop_revision": ["stop_revision_"],
    "debate": ["debate_"],
    "revision": ["scientist_revision_"],
    "coder": ["coder_"],
}

NOTEBOOK_SOURCES: dict[str, list[str]] = {
    "analyst": [],
    "scientist": ["scientist"],
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
    """Import + return the science agent entry functions.

    Deferred to function scope so `import auto_scientist` doesn't pay the
    cost of loading every agent module when only a subset of the package
    is actually used (e.g. tests importing just `auto_scientist.schemas`).
    """
    from auto_scientist.agents.analyst import run_analyst
    from auto_scientist.agents.coder import run_coder
    from auto_scientist.agents.critic import run_debate, run_single_critic_debate
    from auto_scientist.agents.ingestor import run_ingestor
    from auto_scientist.agents.report import run_report
    from auto_scientist.agents.scientist import run_scientist, run_scientist_revision
    from auto_scientist.agents.stop_gate import (
        run_completeness_assessment,
        run_scientist_stop_revision,
        run_single_stop_debate,
    )

    return {
        "canonicalizer": run_ingestor,
        "observer": run_analyst,
        "planner": run_scientist,
        "reviser": run_scientist_revision,
        "implementer": run_coder,
        "reporter": run_report,
        "adversary": run_debate,
        "single_adversary": run_single_critic_debate,
        "assessor": run_completeness_assessment,
        "stop_reviser": run_scientist_stop_revision,
        "stop_adversary": run_single_stop_debate,
    }


def build_registry() -> RoleRegistry:
    """Return the auto-scientist RoleRegistry."""
    from auto_scientist.prompts.critic import (
        DEFAULT_CRITIC_INSTRUCTIONS,
        ITERATION_0_PERSONAS,
        PERSONAS,
        PREDICTION_PERSONAS,
        get_model_index_for_debate,
    )
    from auto_scientist.prompts.stop_gate import STOP_PERSONAS

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
        app_label="Auto-Scientist",
        banner_agents_before_critics=[
            ("Ingestor", "ingestor"),
            ("Analyst", "analyst"),
            ("Scientist", "scientist"),
        ],
        banner_agents_after_critics=[
            ("Coder", "coder"),
            ("Report", "report"),
        ],
        banner_critic_label="Critic",
    )


def install_scientist_registry() -> None:
    """Install the auto-scientist role registry into auto_core."""
    install(build_registry())
