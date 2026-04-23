"""Surveyor agent: structured observation of the PR diff and probe outcomes.

Uses query() (fresh session each iteration, bounded context).
Tools: Read (diff + touched files + probe results), Glob (find output files).
Input: the review workspace (diff.patch, touched_files/, probe_results/) and
the investigation log.
Output: structured JSON with suspicions, touched symbols, observations.
max_turns: 30
"""

import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

from auto_core.agents.notebook_tool import (
    NOTEBOOK_SPEC,
    build_notebook_mcp_server,
    format_notebook_toc,
)
from auto_core.notebook import parse_notebook_entries
from auto_core.retry import QueryResult, agent_retry_loop
from auto_core.sdk_backend import SDKOptions, get_backend
from auto_core.sdk_utils import (
    collect_text_from_query,
    prepare_turn_budget,
    validate_json_output,
)
from auto_core.state import RunState

from auto_reviewer.prompts.surveyor import SURVEYOR_USER, build_surveyor_system
from auto_reviewer.schemas import SurveyorOutput

logger = logging.getLogger(__name__)


def _format_prediction_tree(state: RunState | None) -> str:
    """One-line-per-prediction summary of prior suspected bugs for the prompt."""
    if state is None or not state.prediction_history:
        return "(none yet)"
    lines = []
    for rec in state.prediction_history:
        label = rec.summary.strip() or rec.prediction.strip()[:80]
        lines.append(f"- {rec.pred_id}: {label} -> {rec.outcome}")
    return "\n".join(lines)


# JSON schema for structured output (injected into the prompt for LLM guidance)
SURVEYOR_SCHEMA = {
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


async def run_surveyor(
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
    """Survey the PR diff + probe results and produce structured observations.

    Signature matches the shared `auto_core.agent_dispatch` contract for the
    "observer" role, so several args (`results_path`, `plot_paths`,
    `data_dir`, `domain_knowledge`, `timeout_context`) are accepted and
    ignored - the reviewer's context comes from the review workspace
    (diff.patch, touched_files/, probe_results/) and the persisted RunState.

    Args:
        notebook_path: Path to the investigation log; its parent is the review
            workspace root where state.json and intake artifacts live.
        model: Model override.
        message_buffer: Optional buffer for streaming messages.
        provider: LLM provider name ("anthropic", "openai", etc.).

    Returns:
        Dict shaped like `SurveyorOutput` (suspicions, touched_symbols,
        observations, prediction_outcomes, repo_knowledge, diff_summary).
    """
    notebook_entries = parse_notebook_entries(notebook_path)

    # The review workspace holds diff.patch, touched_files/, probe_results/,
    # and state.json. Intake writes the first three on ingestion; the
    # orchestrator persists state.json after each phase transition.
    workspace_path = notebook_path.parent.resolve()
    state_snapshot: RunState | None = None
    state_path = workspace_path / "state.json"
    if state_path.exists():
        try:
            state_snapshot = RunState.load(state_path)
        except Exception as e:
            logger.warning(f"Surveyor: failed to load {state_path}: {e}")

    goal = state_snapshot.goal if state_snapshot else ""
    pr_ref = state_snapshot.domain if state_snapshot else ""
    iteration = state_snapshot.iteration if state_snapshot else 0
    prediction_tree = _format_prediction_tree(state_snapshot)

    user_prompt = SURVEYOR_USER.format(
        goal=goal or "(no goal set)",
        pr_ref=pr_ref or "(unknown)",
        iteration=iteration,
        workspace_path=workspace_path,
        diff_path=workspace_path / "diff.patch",
        touched_files_dir=workspace_path / "touched_files",
        probe_results_dir=workspace_path / "probe_results",
        notebook_toc=format_notebook_toc(notebook_entries),
        prediction_tree=prediction_tree,
    )

    cwd = workspace_path

    json_instruction = (
        "\n\n## Output Format\n"
        "Tool calls are allowed before the final JSON response.\n"
        'The "JSON only" rule applies only to the final assistant message.\n'
        "Respond with valid JSON matching the schema below.\n"
        "No markdown fencing. No explanation. No other text.\n\n"
        f"Schema:\n{json.dumps(SURVEYOR_SCHEMA, indent=2)}"
    )

    # The analyst always uses tools (Read for plot PNGs, Glob + Read for data
    # files on iteration 0). Use acceptEdits to avoid interactive permission
    # prompts when running as a sub-agent via the SDK.
    max_turns = 30
    allowed_tools = ["Read", "Glob"]
    # Write the scratch file next to the notebook (the run directory),
    # not `cwd` - on iteration 0 cwd is the read-only data directory.
    mcp_servers: dict[str, Any] = {
        "notebook": build_notebook_mcp_server(notebook_path, output_dir=notebook_path.parent),
    }
    allowed_tools.append(NOTEBOOK_SPEC.mcp_tool_name)
    prompt_provider = "gpt" if provider == "openai" else "claude"
    analyst_system = build_surveyor_system(prompt_provider)
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
        mcp_servers=mcp_servers,
        response_schema=SurveyorOutput,
    )

    async def _query(prompt: str, resume_session_id: str | None) -> QueryResult:
        opts = replace(options, resume=resume_session_id) if resume_session_id else options
        raw, usage, session_id = await collect_text_from_query(
            prompt, opts, backend, message_buffer, agent_name="Surveyor"
        )
        return QueryResult(raw_output=raw, session_id=session_id, usage=usage)

    def _validate(result: QueryResult) -> dict[str, Any]:
        return cast(
            dict[str, Any],
            validate_json_output(result.raw_output, SurveyorOutput, "Surveyor"),
        )

    return cast(
        dict[str, Any],
        await agent_retry_loop(
            query_fn=_query,
            validate_fn=_validate,
            prompt=user_prompt,
            agent_name="Surveyor",
        ),
    )
