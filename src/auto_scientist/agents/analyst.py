"""Analyst agent: structured observation of experiment results + plots.

Uses query() (fresh session each iteration, bounded context).
Tools: Read (results file + plot PNGs), Glob (find output files).
Input: results text + lab notebook.
Output: structured JSON with metrics, observations.
max_turns: 30
"""

import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

from auto_scientist.prompts.analyst import ANALYST_SYSTEM, ANALYST_USER
from auto_scientist.retry import QueryResult, agent_retry_loop
from auto_scientist.schemas import AnalystOutput
from auto_scientist.sdk_backend import SDKOptions, get_backend
from auto_scientist.sdk_utils import (
    collect_text_from_query,
    validate_json_output,
    with_turn_budget,
)

logger = logging.getLogger(__name__)

# JSON schema for structured output (injected into the prompt for LLM guidance)
ANALYST_SCHEMA = {
    "type": "object",
    "properties": {
        "key_metrics": {"type": "object", "additionalProperties": {"type": "number"}},
        "improvements": {"type": "array", "items": {"type": "string"}},
        "regressions": {"type": "array", "items": {"type": "string"}},
        "observations": {"type": "array", "items": {"type": "string"}},
        "prediction_outcomes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "pred_id": {"type": "string"},
                    "prediction": {"type": "string"},
                    "outcome": {
                        "type": "string",
                        "enum": ["confirmed", "refuted", "inconclusive"],
                    },
                    "evidence": {"type": "string"},
                },
                "required": ["pred_id", "prediction", "outcome", "evidence"],
            },
        },
        "domain_knowledge": {"type": "string"},
        "data_summary": {"type": "object"},
    },
    "required": [
        "key_metrics",
        "improvements",
        "regressions",
        "observations",
    ],
}


async def run_analyst(
    results_path: Path | None,
    plot_paths: list[Path],
    notebook_path: Path,
    domain_knowledge: str = "",
    data_dir: Path | None = None,
    model: str | None = None,
    message_buffer: list[str] | None = None,
    provider: str = "anthropic",
) -> dict[str, Any]:
    """Analyze experiment results and produce structured observation.

    Args:
        results_path: Path to the results text file (None on iteration 0).
        plot_paths: Paths to output plot PNGs (read as images).
        notebook_path: Path to the lab notebook.
        domain_knowledge: Domain-specific context injected into the prompt.
        data_dir: Path to canonical data directory (set on iteration 0).
        model: Model override.
        message_buffer: Optional buffer for streaming messages.

    Returns:
        Structured dict with keys:
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
            "\n".join(f"- {p}" for p in plot_paths) if plot_paths else "(no plots available)"
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
    max_turns = 30
    allowed_tools = ["Read", "Glob"]
    backend = get_backend(provider)
    options = SDKOptions(
        system_prompt=with_turn_budget(ANALYST_SYSTEM + json_instruction, max_turns, allowed_tools),
        allowed_tools=allowed_tools,
        max_turns=max_turns,
        permission_mode="acceptEdits",
        cwd=cwd,
        model=model,
        extra_args={"setting-sources": ""},
    )

    async def _query(prompt: str, resume_session_id: str | None) -> QueryResult:
        opts = replace(options, resume=resume_session_id) if resume_session_id else options
        raw, usage, session_id = await collect_text_from_query(
            prompt, opts, backend, message_buffer, agent_name="Analyst"
        )
        return QueryResult(raw_output=raw, session_id=session_id, usage=usage)

    def _validate(result: QueryResult) -> dict[str, Any]:
        return validate_json_output(result.raw_output, AnalystOutput, "Analyst")

    return await agent_retry_loop(
        query_fn=_query,
        validate_fn=_validate,
        prompt=user_prompt,
        agent_name="Analyst",
    )
