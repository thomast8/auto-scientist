"""Analyst agent: structured observation of experiment results + plots.

Uses query() (fresh session each iteration, bounded context).
Tools: Read (results file + plot PNGs), Glob (find output files).
Input: results text + lab notebook + success criteria.
Output: structured JSON with success_score, criteria_results, metrics, observations.
max_turns: 5
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
from auto_scientist.prompts.analyst import ANALYST_SYSTEM, ANALYST_USER

# JSON schema for structured output
ANALYST_SCHEMA = {
    "type": "object",
    "properties": {
        "success_score": {"type": "integer", "minimum": 0, "maximum": 100},
        "criteria_results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "measured_value": {"type": ["string", "number", "null"]},
                    "target": {"type": "string"},
                    "status": {"type": "string", "enum": ["pass", "fail", "unable_to_measure"]},
                },
                "required": ["name", "measured_value", "target", "status"],
            },
        },
        "key_metrics": {"type": "object", "additionalProperties": {"type": "number"}},
        "improvements": {"type": "array", "items": {"type": "string"}},
        "regressions": {"type": "array", "items": {"type": "string"}},
        "observations": {"type": "array", "items": {"type": "string"}},
        "iteration_criteria_results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "status": {"type": "string", "enum": ["pass", "fail"]},
                    "measured_value": {"type": "string"},
                },
                "required": ["name", "status", "measured_value"],
            },
        },
    },
    "required": [
        "success_score",
        "criteria_results",
        "key_metrics",
        "improvements",
        "regressions",
        "observations",
        "iteration_criteria_results",
    ],
}


def _format_success_criteria(criteria: list[SuccessCriterion]) -> str:
    """Format success criteria into a readable string for the prompt."""
    if not criteria:
        return "(no success criteria defined)"
    lines = []
    for i, c in enumerate(criteria, 1):
        target = ""
        if c.target_min is not None and c.target_max is not None:
            target = f"target: [{c.target_min}, {c.target_max}]"
        elif c.target_min is not None:
            target = f"target: >= {c.target_min}"
        elif c.target_max is not None:
            target = f"target: <= {c.target_max}"
        required = "REQUIRED" if c.required else "optional"
        lines.append(
            f"{i}. [{required}] {c.name} (metric: {c.metric_key}, {target}): {c.description}"
        )
    return "\n".join(lines)


async def run_analyst(
    results_path: Path,
    plot_paths: list[Path],
    notebook_path: Path,
    domain_knowledge: str = "",
    success_criteria: list[SuccessCriterion] | None = None,
) -> dict[str, Any]:
    """Analyze experiment results and produce structured observation.

    Args:
        results_path: Path to the results text file.
        plot_paths: Paths to output plot PNGs (read as images).
        notebook_path: Path to the lab notebook.
        domain_knowledge: Domain-specific context injected into the prompt.
        success_criteria: List of success criteria to evaluate against.

    Returns:
        Structured dict with keys:
            success_score: int (0-100)
            criteria_results: list[dict] (name, measured_value, target, status)
            key_metrics: dict[str, float]
            improvements: list[str]
            regressions: list[str]
            observations: list[str]
    """
    results_content = results_path.read_text() if results_path.exists() else "(no results file)"
    notebook_content = notebook_path.read_text() if notebook_path.exists() else "(no notebook)"

    plot_list = "\n".join(f"- {p}" for p in plot_paths) if plot_paths else "(no plots available)"

    user_prompt = ANALYST_USER.format(
        domain_knowledge=domain_knowledge or "(no domain knowledge provided)",
        success_criteria=_format_success_criteria(success_criteria or []),
        results_content=results_content,
        notebook_content=notebook_content,
        plot_list=plot_list,
    )

    json_instruction = (
        "\n\n## Output Format\n"
        "You MUST respond with ONLY valid JSON matching the schema below.\n"
        "No markdown fencing. No explanation. No other text.\n\n"
        f"Schema:\n{json.dumps(ANALYST_SCHEMA, indent=2)}"
    )

    options = ClaudeCodeOptions(
        system_prompt=ANALYST_SYSTEM + json_instruction,
        allowed_tools=["Read", "Glob"],
        max_turns=5,
        permission_mode="default",
        cwd=results_path.parent,
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

    # Parse the result - prefer ResultMessage.result, fallback to assistant text
    raw = result_text
    if not raw:
        raw = "\n".join(assistant_texts)

    if not raw:
        raise RuntimeError("Analyst agent returned no output")

    # Extract JSON from the response (handle possible markdown fencing)
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        raw = "\n".join(lines)

    return json.loads(raw)
