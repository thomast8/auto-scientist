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

from auto_scientist.prompts.analyst import ANALYST_USER, build_analyst_system
from auto_scientist.retry import QueryResult, agent_retry_loop
from auto_scientist.schemas import AnalystOutput
from auto_scientist.sdk_backend import SDKOptions, get_backend
from auto_scientist.sdk_utils import (
    collect_text_from_query,
    prepare_turn_budget,
    validate_json_output,
)

logger = logging.getLogger(__name__)

# JSON schema for structured output (injected into the prompt for LLM guidance)
ANALYST_SCHEMA = {
    "type": "object",
    "properties": {
        "key_metrics": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "value": {"type": "number"},
                },
                "required": ["name", "value"],
            },
        },
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
        "data_diagnostics": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "variables": {"type": "array", "items": {"type": "string"}},
                    "pattern": {"type": "string"},
                    "evidence": {"type": "string"},
                },
                "required": ["variables", "pattern", "evidence"],
            },
        },
        "domain_knowledge": {"type": "string"},
        "data_summary": {"type": "string"},
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
    timeout_context: dict | None = None,
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
        timeout_context: If provided, indicates the previous script timed out.
            Keys: timeout_minutes (int), hypothesis (str).

    Returns:
        Structured dict with keys:
            key_metrics: list[{name: str, value: float}]
            improvements: list[str]
            regressions: list[str]
            observations: list[str]
            domain_knowledge: str (optional, iteration 0 only)
            data_summary: str (optional, iteration 0 only)
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

    # Prepend timeout context when the previous script timed out
    if timeout_context:
        has_partial = results_path is not None and results_path.exists()
        timeout_block = (
            "<timeout_info>\n"
            "IMPORTANT: The previous experiment script TIMED OUT after "
            f"{timeout_context['timeout_minutes']} minutes.\n"
            f"Hypothesis being tested: {timeout_context.get('hypothesis', '(unknown)')}\n"
            f"Partial results available: {'yes' if has_partial else 'no'}\n"
            "</timeout_info>\n\n"
        )
        data_section = timeout_block + data_section

    user_prompt = ANALYST_USER.format(
        domain_knowledge=domain_knowledge or "(no domain knowledge provided)",
        data_section=data_section,
        notebook_content=notebook_content,
    )

    json_instruction = (
        "\n\n## Output Format\n"
        "Tool calls are allowed before the final JSON response.\n"
        'The "JSON only" rule applies only to the final assistant message.\n'
        "Respond with valid JSON matching the schema below.\n"
        "No markdown fencing. No explanation. No other text.\n\n"
        f"Schema:\n{json.dumps(ANALYST_SCHEMA, indent=2)}"
    )

    # The analyst always uses tools (Read for plot PNGs, Glob + Read for data
    # files on iteration 0). Use acceptEdits to avoid interactive permission
    # prompts when running as a sub-agent via the SDK.
    max_turns = 30
    allowed_tools = ["Read", "Glob"]
    prompt_provider = "gpt" if provider == "openai" else "claude"
    analyst_system = build_analyst_system(prompt_provider)
    budget = prepare_turn_budget(
        analyst_system + json_instruction, max_turns, allowed_tools, provider=provider
    )
    backend = get_backend(provider)
    options = SDKOptions(
        system_prompt=budget.system_prompt,
        allowed_tools=budget.allowed_tools,
        max_turns=budget.max_turns,
        permission_mode="acceptEdits",
        cwd=cwd,
        model=model,
        extra_args={},
        response_schema=AnalystOutput,
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
