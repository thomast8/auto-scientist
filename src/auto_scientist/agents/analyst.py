"""Analyst agent: structured observation of experiment results + plots.

Uses query() (fresh session each iteration, bounded context).
Tools: Read (results file + plot PNGs), Glob (find output files).
Input: results text + lab notebook + success criteria.
Output: structured JSON with criteria_results, metrics, observations.
max_turns: 30
"""

import json
import logging
from pathlib import Path
from typing import Any

from claude_code_sdk import ClaudeCodeOptions

from auto_scientist.config import SuccessCriterion
from auto_scientist.prompts.analyst import ANALYST_SYSTEM, ANALYST_USER
from auto_scientist.schemas import AnalystOutput
from auto_scientist.sdk_utils import (
    OutputValidationError,
    collect_text_from_query,
    validate_json_output,
)

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 3

# JSON schema for structured output (injected into the prompt for LLM guidance)
ANALYST_SCHEMA = {
    "type": "object",
    "properties": {
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
        "domain_knowledge": {"type": "string"},
        "data_summary": {"type": "object"},
    },
    "required": [
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
    results_path: Path | None,
    plot_paths: list[Path],
    notebook_path: Path,
    domain_knowledge: str = "",
    success_criteria: list[SuccessCriterion] | None = None,
    data_dir: Path | None = None,
    model: str | None = None,
    message_buffer: list[str] | None = None,
) -> dict[str, Any]:
    """Analyze experiment results and produce structured observation.

    Args:
        results_path: Path to the results text file (None on iteration 0).
        plot_paths: Paths to output plot PNGs (read as images).
        notebook_path: Path to the lab notebook.
        domain_knowledge: Domain-specific context injected into the prompt.
        success_criteria: List of success criteria to evaluate against.
        data_dir: Path to canonical data directory (set on iteration 0).
        model: Model override.
        message_buffer: Optional buffer for streaming messages.

    Returns:
        Structured dict with keys:
            criteria_results: list[dict] (name, measured_value, target, status)
            key_metrics: dict[str, float]
            improvements: list[str]
            regressions: list[str]
            observations: list[str]
            domain_knowledge: str (optional, iteration 0 only)
            data_summary: dict (optional, iteration 0 only)
    """
    notebook_content = notebook_path.read_text() if notebook_path.exists() else "(no notebook)"

    # Build the data section depending on iteration 0 vs normal
    if data_dir is not None and (results_path is None or not results_path.exists()):
        abs_data_dir = Path(data_dir).resolve()
        # Pre-compute file listing so the analyst doesn't need to Glob
        file_listing = "\n".join(
            f"- {f.name}" for f in sorted(abs_data_dir.iterdir()) if f.is_file()
        )
        data_section = (
            f"<data_directory>{abs_data_dir}</data_directory>\n"
            f"<data_files>\n{file_listing}\n</data_files>\n"
            "Use the Read tool to examine each data file listed above. "
            "Describe the structure of each file factually."
        )
        cwd = abs_data_dir
    else:
        results_content = (
            results_path.read_text()
            if results_path and results_path.exists()
            else "(no results file)"
        )
        plot_list = (
            "\n".join(f"- {p}" for p in plot_paths)
            if plot_paths
            else "(no plots available)"
        )
        data_section = (
            f"<results>{results_content}</results>\n"
            "<plots>\n"
            "Use the Read tool to examine each of these plot files. For each\n"
            "plot, describe what you see: trends, patterns, deviations,\n"
            "outliers. Extract any numeric values visible in the plots.\n"
            f"{plot_list}\n"
            "</plots>"
        )
        cwd = results_path.parent if results_path else notebook_path.parent

    user_prompt = ANALYST_USER.format(
        domain_knowledge=domain_knowledge or "(no domain knowledge provided)",
        success_criteria=_format_success_criteria(success_criteria or []),
        data_section=data_section,
        notebook_content=notebook_content,
    )

    json_instruction = (
        "\n\n## Output Format\n"
        "You MUST respond with ONLY valid JSON matching the schema below.\n"
        "No markdown fencing. No explanation. No other text.\n\n"
        f"Schema:\n{json.dumps(ANALYST_SCHEMA, indent=2)}"
    )

    # The analyst always uses tools (Read for plot PNGs, Glob + Read for data
    # files on iteration 0). Use acceptEdits to avoid interactive permission
    # prompts when running as a sub-agent via the SDK.
    options = ClaudeCodeOptions(
        system_prompt=ANALYST_SYSTEM + json_instruction,
        allowed_tools=["Read", "Glob"],
        max_turns=30,
        permission_mode="acceptEdits",
        cwd=cwd,
        model=model,
        extra_args={"setting-sources": ""},
    )

    correction_hint = ""
    for attempt in range(MAX_ATTEMPTS):
        effective_prompt = user_prompt + correction_hint

        try:
            raw = await collect_text_from_query(
                effective_prompt, options, message_buffer, agent_name="Analyst",
            )
        except Exception as e:
            if attempt == MAX_ATTEMPTS - 1:
                raise
            logger.warning(f"Analyst attempt {attempt + 1}: SDK error ({e}), retrying")
            continue

        try:
            return validate_json_output(raw, AnalystOutput, "Analyst")
        except OutputValidationError as e:
            if attempt == MAX_ATTEMPTS - 1:
                raise
            correction_hint = f"\n\n{e.correction_prompt()}"
            logger.warning(f"Analyst attempt {attempt + 1} failed, retrying: {e}")

    raise RuntimeError("Analyst: exhausted retries")  # unreachable
