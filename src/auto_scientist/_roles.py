"""Build and install the auto-scientist role registry into auto_core.

Called once from `auto_scientist/__init__.py` at import time. Populates the
core's style / description / phase / summary / artifact / notebook-source
lookup tables with the scientist-flavored content that used to be hardcoded
in `widgets.py`, `summarizer.py`, and `resume.py`.
"""

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


def build_registry() -> RoleRegistry:
    """Return the auto-scientist RoleRegistry."""
    return RoleRegistry(
        agent_styles=AGENT_STYLES,
        agent_descriptions=AGENT_DESCRIPTIONS,
        phase_styles=PHASE_STYLES,
        summary_prompts=SUMMARY_PROMPTS,
        artifact_specs=ARTIFACT_SPECS,
        buffer_prefixes=BUFFER_PREFIXES,
        notebook_sources=NOTEBOOK_SOURCES,
        agent_fields=AGENT_FIELDS,
    )


def install_scientist_registry() -> None:
    """Install the auto-scientist role registry into auto_core."""
    install(build_registry())
