"""Scientist agent: pure prompt-in, JSON-out strategic planner.

No tools. Does not read Python code. Receives analysis + notebook via prompt.
Output: structured JSON plan with hypothesis, strategy, changes, notebook entry.
"""

import json
from pathlib import Path
from typing import Any

from claude_code_sdk import (
    AssistantMessage,
    ClaudeCodeOptions,
    ResultMessage,
    TextBlock,
    query,
)

from auto_scientist.config import SuccessCriterion
from auto_scientist.prompts.scientist import (
    SCIENTIST_REVISION_SYSTEM,
    SCIENTIST_REVISION_USER,
    SCIENTIST_SYSTEM,
    SCIENTIST_USER,
)

# JSON schema for structured output
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

    The Scientist does not read code. It receives the analysis JSON and
    notebook content via prompt injection and returns a structured plan.

    Args:
        analysis: Structured analysis JSON from the Analyst.
        notebook_path: Path to the lab notebook (read for context).
        version: Version string for the new experiment (e.g., 'v01').
        domain_knowledge: Domain-specific context.
        success_criteria: Existing top-level criteria (None if not yet defined).
        model: Model override.

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

    json_instruction = (
        "\n\n## Output Format\n"
        "You MUST respond with ONLY valid JSON matching the schema below.\n"
        "No markdown fencing. No explanation. No other text.\n\n"
        f"Schema:\n{json.dumps(SCIENTIST_PLAN_SCHEMA, indent=2)}"
    )

    options = ClaudeCodeOptions(
        system_prompt=SCIENTIST_SYSTEM + json_instruction,
        allowed_tools=[],
        max_turns=1,
        model=model,
    )

    result_text = ""
    assistant_texts: list[str] = []

    async for message in query(prompt=user_prompt, options=options):
        if isinstance(message, ResultMessage):
            if message.result:
                result_text = message.result
        elif isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    assistant_texts.append(block.text)
                    if message_buffer is not None:
                        message_buffer.append(block.text)

    # Parse the result - prefer ResultMessage.result, fallback to assistant text
    raw = result_text
    if not raw:
        raw = "\n".join(assistant_texts)

    if not raw:
        raise RuntimeError("Scientist agent returned no output")

    # Extract JSON from the response (handle possible markdown fencing)
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        raw = "\n".join(lines)

    return json.loads(raw)


def _parse_json_response(raw: str, label: str) -> dict[str, Any]:
    """Parse JSON from a response, handling markdown fencing."""
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        raw = "\n".join(lines)
    return json.loads(raw)


async def run_scientist_revision(
    original_plan: dict[str, Any],
    debate_transcript: list[dict[str, str]],
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
        debate_transcript: List of {"role": "critic"|"scientist", "content": str}.
        analysis: Structured analysis JSON from the Analyst.
        notebook_path: Path to the lab notebook.
        version: Version string.
        domain_knowledge: Domain-specific context.

    Returns:
        Revised plan dict (same schema as the initial plan).
    """
    notebook_path = Path(notebook_path)
    notebook_content = notebook_path.read_text() if notebook_path.exists() else ""

    # Format debate transcript
    transcript_parts = []
    for entry in debate_transcript:
        role = entry["role"].capitalize()
        transcript_parts.append(f"### {role}\n{entry['content']}")
    transcript_text = "\n\n".join(transcript_parts)

    user_prompt = SCIENTIST_REVISION_USER.format(
        domain_knowledge=domain_knowledge or "(no domain knowledge provided)",
        analysis_json=(
            json.dumps(analysis, indent=2) if analysis else "(no analysis)"
        ),
        notebook_content=notebook_content or "(empty notebook)",
        original_plan=json.dumps(original_plan, indent=2),
        debate_transcript=transcript_text or "(no debate - critique was skipped)",
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
        allowed_tools=[],
        max_turns=1,
        model=model,
    )

    result_text = ""
    assistant_texts: list[str] = []

    async for message in query(prompt=user_prompt, options=options):
        if isinstance(message, ResultMessage):
            if message.result:
                result_text = message.result
        elif isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    assistant_texts.append(block.text)
                    if message_buffer is not None:
                        message_buffer.append(block.text)

    raw = result_text
    if not raw:
        raw = "\n".join(assistant_texts)

    if not raw:
        raise RuntimeError("Scientist revision returned no output")

    return _parse_json_response(raw, "Scientist revision")
