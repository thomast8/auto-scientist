"""Scientist agent: prompt-in, JSON-out strategic planner with web search.

Does not read Python code or data files. Receives analysis + notebook via prompt.
Has web search access to ground hypotheses in real-world knowledge.
Output: structured JSON plan with hypothesis, strategy, changes, notebook entry.
"""

import json
import logging
from pathlib import Path
from typing import Any

from claude_code_sdk import ClaudeCodeOptions

from auto_scientist.config import SuccessCriterion
from auto_scientist.prompts.scientist import (
    SCIENTIST_REVISION_SYSTEM,
    SCIENTIST_REVISION_USER,
    SCIENTIST_SYSTEM,
    SCIENTIST_USER,
)
from auto_scientist.schemas import ScientistPlanOutput
from auto_scientist.sdk_utils import (
    OutputValidationError,
    collect_text_from_query,
    validate_json_output,
)

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 3

SCIENTIST_TOOLS = ["WebSearch"]

# JSON schema for structured output (injected into the prompt for LLM guidance)
SCIENTIST_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "hypothesis": {"type": "string"},
        "strategy": {"type": "string", "enum": ["incremental", "structural", "exploratory"]},
        "changes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "what": {"type": "string"},
                    "why": {"type": "string"},
                    "how": {"type": "string"},
                    "priority": {"type": "integer", "enum": [1, 2, 3]},
                },
                "required": ["what", "why", "how", "priority"],
            },
        },
        "expected_impact": {"type": "string"},
        "should_stop": {"type": "boolean"},
        "stop_reason": {"type": ["string", "null"]},
        "notebook_entry": {"type": "string"},
        "success_criteria": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "metric_key": {"type": "string"},
                    "condition": {"type": "string"},
                },
                "required": ["name", "description", "metric_key", "condition"],
            },
        },
        "top_level_criteria": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "metric_key": {"type": "string"},
                    "condition": {"type": "string"},
                },
                "required": ["name", "description", "metric_key", "condition"],
            },
        },
        "criteria_revision": {
            "type": "object",
            "properties": {
                "changes": {"type": "string"},
                "revised_criteria": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                            "metric_key": {"type": "string"},
                            "condition": {"type": "string"},
                        },
                        "required": ["name", "description", "metric_key", "condition"],
                    },
                },
            },
            "required": ["changes", "revised_criteria"],
        },
    },
    "required": [
        "hypothesis",
        "strategy",
        "changes",
        "expected_impact",
        "should_stop",
        "stop_reason",
        "notebook_entry",
        "success_criteria",
    ],
}


def _format_criteria_for_prompt(criteria: list[SuccessCriterion] | None) -> str:
    """Format existing success criteria for injection into the Scientist prompt."""
    if not criteria:
        return "(no top-level success criteria defined yet)"
    lines = []
    for i, c in enumerate(criteria, 1):
        target = ""
        if c.target_min is not None and c.target_max is not None:
            target = f"[{c.target_min}, {c.target_max}]"
        elif c.target_min is not None:
            target = f">= {c.target_min}"
        elif c.target_max is not None:
            target = f"<= {c.target_max}"
        required = "REQUIRED" if c.required else "optional"
        lines.append(
            f"{i}. [{required}] {c.name} (metric: {c.metric_key}, {target}): {c.description}"
        )
    return "\n".join(lines)


async def run_scientist(
    analysis: dict[str, Any],
    notebook_path: Path,
    version: str,
    domain_knowledge: str = "",
    success_criteria: list[SuccessCriterion] | None = None,
    model: str | None = None,
    message_buffer: list[str] | None = None,
) -> dict[str, Any]:
    """Formulate hypothesis and plan based on analysis.

    The Scientist does not read code or data files. It receives the analysis
    JSON and notebook content via prompt injection and returns a structured plan.
    Has web search access.

    Args:
        analysis: Structured analysis JSON from the Analyst.
        notebook_path: Path to the lab notebook (read for context).
        version: Version string for the new experiment (e.g., 'v01').
        domain_knowledge: Domain-specific context.
        success_criteria: Existing top-level criteria (None if not yet defined).
        model: Model override.
        message_buffer: Optional buffer for streaming messages.

    Returns:
        Structured plan dict with keys: hypothesis, strategy, changes,
        expected_impact, should_stop, stop_reason, notebook_entry.
        Optionally: top_level_criteria, criteria_revision.
    """
    notebook_path = Path(notebook_path)
    notebook_content = notebook_path.read_text() if notebook_path.exists() else ""

    user_prompt = SCIENTIST_USER.format(
        domain_knowledge=domain_knowledge or "(no domain knowledge provided)",
        analysis_json=(
            json.dumps(analysis, indent=2) if analysis else "(no analysis yet - first iteration)"
        ),
        notebook_content=notebook_content or "(empty notebook - first iteration)",
        success_criteria=_format_criteria_for_prompt(success_criteria),
        version=version,
    )

    system_prompt = SCIENTIST_SYSTEM

    json_instruction = (
        "\n\n## Output Format\n"
        "You MUST respond with ONLY valid JSON matching the schema below.\n"
        "No markdown fencing. No explanation. No other text.\n\n"
        f"Schema:\n{json.dumps(SCIENTIST_PLAN_SCHEMA, indent=2)}"
    )

    options = ClaudeCodeOptions(
        system_prompt=system_prompt + json_instruction,
        allowed_tools=SCIENTIST_TOOLS,
        max_turns=10,
        model=model,
        extra_args={"setting-sources": ""},
    )

    correction_hint = ""
    for attempt in range(MAX_ATTEMPTS):
        effective_prompt = user_prompt + correction_hint

        try:
            raw = await collect_text_from_query(
                effective_prompt, options, message_buffer, agent_name="Scientist",
            )
        except Exception as e:
            if attempt == MAX_ATTEMPTS - 1:
                raise
            logger.warning(f"Scientist attempt {attempt + 1}: SDK error ({e}), retrying")
            continue

        try:
            return validate_json_output(raw, ScientistPlanOutput, "Scientist")
        except OutputValidationError as e:
            if attempt == MAX_ATTEMPTS - 1:
                raise
            correction_hint = f"\n\n{e.correction_prompt()}"
            logger.warning(f"Scientist attempt {attempt + 1} failed, retrying: {e}")

    raise RuntimeError("Scientist: exhausted retries")  # unreachable


async def run_scientist_revision(
    original_plan: dict[str, Any],
    concern_ledger: list[dict[str, Any]],
    analysis: dict[str, Any],
    notebook_path: Path,
    version: str,
    domain_knowledge: str = "",
    model: str | None = None,
    message_buffer: list[str] | None = None,
) -> dict[str, Any]:
    """Revise the plan after a critic debate.

    Args:
        original_plan: The initial plan that was debated.
        concern_ledger: Structured list of ConcernLedgerEntry dicts.
        analysis: Structured analysis JSON from the Analyst.
        notebook_path: Path to the lab notebook.
        version: Version string.
        domain_knowledge: Domain-specific context.
        model: Model override.
        message_buffer: Optional buffer for streaming messages.

    Returns:
        Revised plan dict (same schema as the initial plan).
    """
    notebook_path = Path(notebook_path)
    notebook_content = notebook_path.read_text() if notebook_path.exists() else ""

    ledger_text = json.dumps(concern_ledger, indent=2) if concern_ledger else "(no concerns raised)"

    user_prompt = SCIENTIST_REVISION_USER.format(
        domain_knowledge=domain_knowledge or "(no domain knowledge provided)",
        analysis_json=(
            json.dumps(analysis, indent=2) if analysis else "(no analysis)"
        ),
        notebook_content=notebook_content or "(empty notebook)",
        original_plan=json.dumps(original_plan, indent=2),
        concern_ledger=ledger_text,
        version=version,
    )

    json_instruction = (
        "\n\n## Output Format\n"
        "You MUST respond with ONLY valid JSON matching the schema below.\n"
        "No markdown fencing. No explanation. No other text.\n\n"
        f"Schema:\n{json.dumps(SCIENTIST_PLAN_SCHEMA, indent=2)}"
    )

    options = ClaudeCodeOptions(
        system_prompt=SCIENTIST_REVISION_SYSTEM + json_instruction,
        allowed_tools=SCIENTIST_TOOLS,
        max_turns=10,
        model=model,
        extra_args={"setting-sources": ""},
    )

    correction_hint = ""
    for attempt in range(MAX_ATTEMPTS):
        effective_prompt = user_prompt + correction_hint

        try:
            raw = await collect_text_from_query(
                effective_prompt, options, message_buffer, agent_name="Scientist revision",
            )
        except Exception as e:
            if attempt == MAX_ATTEMPTS - 1:
                raise
            logger.warning(f"Scientist revision attempt {attempt + 1}: SDK error ({e}), retrying")
            continue

        try:
            return validate_json_output(raw, ScientistPlanOutput, "Scientist revision")
        except OutputValidationError as e:
            if attempt == MAX_ATTEMPTS - 1:
                raise
            correction_hint = f"\n\n{e.correction_prompt()}"
            logger.warning(f"Scientist revision attempt {attempt + 1} failed, retrying: {e}")

    raise RuntimeError("Scientist revision: exhausted retries")  # unreachable
