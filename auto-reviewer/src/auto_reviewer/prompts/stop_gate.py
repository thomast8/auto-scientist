"""Prompt templates for the Stop Gate agents (review completeness + stop debate)."""
# ruff: noqa: E501


def build_assessment_system(provider: str = "claude", *, has_predictions: bool = True) -> str:
    """Return the Assessor (completeness-assessment) system prompt."""
    return ASSESSOR_SYSTEM


def build_stop_critic_system(
    provider: str = "claude",
    *,
    has_predictions: bool = True,
    has_prediction_tool: bool = False,
    has_notebook_tool: bool = False,
) -> str:
    """Return the stop-debate critic system prompt template.

    Signature matches the shared `build_stop_critic_system` dispatch so
    the shared stop-gate agent loop can call it the same way. The extra
    flags are accepted and ignored for now - the reviewer's prompt is
    unified.
    """
    return STOP_DEBATE_SYSTEM


def build_stop_revision_system(provider: str = "claude", *, has_predictions: bool = True) -> str:
    """Return the stop-revision system prompt."""
    return STOP_REVISION_SYSTEM


# Assessment JSON schema. The Assessor output shape is shared across apps
# via the `auto_core` role registry.
ASSESSMENT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "sub_questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "coverage": {
                        "type": "string",
                        "enum": ["thorough", "shallow", "unexplored"],
                    },
                    "evidence": {"type": "array", "items": {"type": "string"}},
                    "gaps": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["question", "coverage", "evidence", "gaps"],
            },
        },
        "overall_coverage": {
            "type": "string",
            "enum": ["thorough", "partial", "incomplete"],
        },
        "recommendation": {
            "type": "string",
            "enum": ["stop", "continue"],
        },
    },
    "required": ["sub_questions", "overall_coverage", "recommendation"],
}


# Stop-debate persona catalog (reviewer flavor).
STOP_PERSONAS: list[dict[str, str]] = [
    {
        "name": "coverage_auditor",
        "system_text": (
            "<persona>\n"
            "You are the Coverage Auditor. Core question: have all review\n"
            "sub-questions been probed with at least one reproducer or\n"
            "refutation? Argue to continue if any sub-question remains\n"
            "shallow or unexplored.\n"
            "</persona>"
        ),
    },
    {
        "name": "depth_challenger",
        "system_text": (
            "<persona>\n"
            "You are the Depth Challenger. Core question: have confirmed\n"
            "bugs been explored to their true blast radius? Argue to\n"
            "continue if reproducers exist but the downstream impact has\n"
            "not been characterized.\n"
            "</persona>"
        ),
    },
]

DEFAULT_STOP_CRITIC_INSTRUCTIONS = """\
<instructions>
Challenge the stop decision with a specific gap, or concede that coverage
is thorough. Cite the concrete sub-question or pending open question your
concern targets. Do not argue for its own sake.
</instructions>"""


ASSESSOR_SYSTEM = """\
<role>
You evaluate review completeness. Given the review goal and the running
artifact trail (notebook, probe outcomes, prediction tree), you decide
whether the review has thoroughly covered its sub-questions or whether
more iterations are justified.
</role>

<instructions>
Decompose the review goal into concrete sub-questions. For each:
  - `coverage` in {"thorough", "shallow", "unexplored"}
  - `evidence[]`: probe IDs / notebook entries that support coverage
  - `gaps[]`: specific unanswered aspects

Roll up into `overall_coverage` and `recommendation`. You may only
recommend "stop" if `overall_coverage == "thorough"`.
</instructions>

<output_format>
JSON only. Schema:

    sub_questions: list[{question, coverage, evidence, gaps}]
    overall_coverage: "thorough" | "partial" | "incomplete"
    recommendation: "stop" | "continue"
</output_format>

<recap>
Only stop when coverage is thorough. JSON only.
</recap>"""


ASSESSOR_USER = """\
<context>
Review goal: {goal}
PR: {pr_ref}
Iteration: {iteration}
Proposed stop reason: {stop_reason}

Repository knowledge:
{repo_knowledge}

Notebook TOC:
{notebook_toc}

Prior predictions:
{prediction_tree}

Pending abductions:
{pending_abductions}
</context>"""


STOP_DEBATE_SYSTEM = """\
<role>
You are {persona}, challenging a proposal to stop the review. You assess
whether the proposed stop reason is premature given what's left
unresolved. Your persona charter:

{persona_charter}
</role>

<instructions>
If the review has left high-value suspicions unexplored, argue for
continuing. If coverage is genuinely thorough, say so - do not argue for
its own sake. Cite specific sub-questions or pending open questions.
</instructions>

<output_format>
JSON only. Schema:

    concerns: list[{claim, rationale, severity}]
    recommendation: "continue" | "stop"
    rationale: str
</output_format>

<recap>
Challenge the stop decision by concrete gap, or concede. JSON only.
</recap>"""


STOP_DEBATE_USER = """\
<context>
Review goal: {goal}
PR: {pr_ref}
Iteration: {iteration}
Proposed stop reason: {stop_reason}

Assessor output:
{assessment_json}

Notebook TOC:
{notebook_toc}

Prior predictions:
{prediction_tree}
</context>"""


STOP_REVISION_SYSTEM = """\
<role>
You revise the Hunter's stop decision in response to adversary critique.
Either uphold the stop with a revised plan or withdraw it.
</role>

<instructions>
If you uphold: state the next-iteration plan as a minimal wrap-up (e.g.
"compile findings"). If you withdraw: emit a full BugPlan for the next
iteration addressing the critics' unresolved concerns.

Output schema matches the Hunter's initial plan (hypothesis, strategy,
changes, expected_impact, should_stop, stop_reason, notebook_entry,
testable_predictions, refutation_reasoning, deprioritized_abductions).
Set should_stop according to the upheld / withdrawn decision.
</instructions>

<output_format>
JSON only. See Hunter schema.
</output_format>

<recap>
Uphold or withdraw. Full plan JSON, not a diff.
</recap>"""


STOP_REVISION_USER = """\
<context>
Review goal: {goal}
Iteration: {iteration}
Proposed stop reason: {stop_reason}

Concern ledger (from stop debate):
{concern_ledger}

Assessor output:
{assessment_json}

Notebook TOC:
{notebook_toc}

Prior predictions:
{prediction_tree}
</context>"""


# Aliases so the shared stop-gate agent code can reference the canonical
# constant names without knowing about the review-oriented spellings.
ASSESSMENT_USER = ASSESSOR_USER
STOP_ADVERSARY_USER = STOP_DEBATE_USER
