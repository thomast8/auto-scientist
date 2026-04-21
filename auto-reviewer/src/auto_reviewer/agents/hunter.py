"""Scientist agent: prompt-in, JSON-out strategic planner with web search.

Does not read Python code or data files. Receives analysis + notebook via prompt.
Has web search access to ground hypotheses in real-world knowledge.
Output: structured JSON plan with hypothesis, strategy, changes, notebook entry.
"""

import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

from auto_core.agents.notebook_tool import (
    NOTEBOOK_SPEC,
    build_notebook_mcp_server,
    format_notebook_toc,
)
from auto_core.agents.prediction_tool import (
    PREDICTION_SPEC,
    build_prediction_mcp_server,
    format_compact_tree,
    format_full_detail,
)
from auto_core.model_config import ReasoningConfig, reasoning_to_cc_extra_args
from auto_core.notebook import parse_notebook_entries
from auto_core.retry import QueryResult, agent_retry_loop
from auto_core.sdk_backend import SDKOptions, get_backend
from auto_core.sdk_utils import (
    collect_text_from_query,
    prepare_turn_budget,
    validate_json_output,
)
from auto_core.state import PredictionRecord

from auto_reviewer.prompts.hunter import (
    HUNTER_REVISION_USER,
    HUNTER_USER,
    build_hunter_system,
    build_revision_system,
)
from auto_reviewer.schemas import HunterPlanOutput

logger = logging.getLogger(__name__)

HUNTER_BASE_TOOLS = ["WebSearch"]


def _build_scientist_tools_and_mcp(
    prediction_history: list[PredictionRecord] | None,
    provider: str,
    notebook_path: Path | None = None,
    output_dir: Path | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """Build the tools list and MCP servers dict for a Scientist invocation.

    Both Claude and Codex backends get the same stdio MCP server config.
    Claude passes it via mcp_servers; the CodexBackend writes it to
    an isolated ``$CODEX_HOME/config.toml`` automatically.

    The notebook tool is wired unconditionally when a notebook path is
    supplied - on iteration 0 the notebook is empty or ingestion-only, but
    the tool still returns a useful "no entries" response.
    """
    tools = list(HUNTER_BASE_TOOLS)
    mcp_servers: dict[str, Any] = {}
    if prediction_history:
        mcp_servers["predictions"] = build_prediction_mcp_server(
            prediction_history, output_dir=output_dir
        )
        tools.append(PREDICTION_SPEC.mcp_tool_name)
    if notebook_path is not None:
        mcp_servers["notebook"] = build_notebook_mcp_server(notebook_path, output_dir=output_dir)
        tools.append(NOTEBOOK_SPEC.mcp_tool_name)
    return tools, mcp_servers


# JSON schema for structured output (injected into the prompt for LLM guidance)
HUNTER_PLAN_SCHEMA = {
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
        "testable_predictions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "prediction": {"type": "string"},
                    "diagnostic": {"type": "string"},
                    "if_confirmed": {"type": "string"},
                    "if_refuted": {"type": "string"},
                    "follows_from": {
                        "type": ["string", "null"],
                        "description": (
                            "pred_id of a prior prediction (e.g. '0.3', '1.2'). "
                            "Must be an exact bracketed ID from the prediction "
                            "history, or null for new trajectories."
                        ),
                    },
                },
                "required": ["prediction", "diagnostic", "if_confirmed", "if_refuted"],
            },
        },
        "refutation_reasoning": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "refuted_pred_id": {"type": "string"},
                    "assumptions_violated": {"type": "string"},
                    "alternative_explanation": {"type": "string"},
                    "testable_consequence": {"type": "string"},
                },
                "required": [
                    "refuted_pred_id",
                    "assumptions_violated",
                    "alternative_explanation",
                    "testable_consequence",
                ],
            },
        },
        "deprioritized_abductions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "refuted_pred_id": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["refuted_pred_id", "reason"],
            },
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
    ],
}


# Re-export for backward compatibility (scripts, tests, orchestrator)
_format_compact_tree = format_compact_tree


# Alias preserved so existing imports (test_scientist.py, scripts/compare_personas.py,
# scripts/validate_prediction_tool.py) keep working. New code should import
# format_full_detail from prediction_tool directly.
_format_predictions_for_prompt = format_full_detail


async def run_hunter(
    analysis: dict[str, Any],
    notebook_path: Path,
    version: str,
    domain_knowledge: str = "",
    prediction_history: list[PredictionRecord] | None = None,
    model: str | None = None,
    message_buffer: list[str] | None = None,
    goal: str = "",
    provider: str = "anthropic",
    reasoning: ReasoningConfig | None = None,
    output_dir: Path | None = None,
    pending_abductions: str = "",
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
        prediction_history: Accumulated testable predictions and outcomes.
        model: Model override.
        message_buffer: Optional buffer for streaming messages.
        goal: The user's investigation goal.

    Returns:
        Structured plan dict with keys: hypothesis, strategy, changes,
        expected_impact, should_stop, stop_reason, notebook_entry.
        Optionally: testable_predictions.
    """
    notebook_path = Path(notebook_path)
    notebook_entries = parse_notebook_entries(notebook_path)

    abductions_section = ""
    if pending_abductions:
        abductions_section = (
            "<pending_abductions>\n"
            "In the previous iteration, you generated alternative explanations "
            "for refuted predictions. For each, either include a testable "
            "prediction that addresses it (with follows_from referencing the "
            "refuted_pred_id), or list it in deprioritized_abductions with a "
            "reason.\n\n"
            f"{pending_abductions}\n"
            "</pending_abductions>\n"
        )

    user_prompt = HUNTER_USER.format(
        goal=goal or "(no goal specified)",
        domain_knowledge=domain_knowledge or "(no domain knowledge provided)",
        analysis_json=(
            json.dumps(analysis, indent=2) if analysis else "(no analysis yet - first iteration)"
        ),
        notebook_content=format_notebook_toc(notebook_entries),
        prediction_history=format_compact_tree(prediction_history),
        pending_abductions_section=abductions_section,
        version=version,
    )

    prompt_provider = "gpt" if provider == "openai" else "claude"
    has_predictions = bool(prediction_history)
    system_prompt = build_hunter_system(prompt_provider, has_predictions=has_predictions)

    json_instruction = (
        "\n\n## Output Format\n"
        "Tool calls are allowed before the final JSON response.\n"
        'The "JSON only" rule applies only to the final assistant message.\n'
        "Respond with valid JSON matching the schema below.\n"
        "No markdown fencing. No explanation. No other text.\n\n"
        f"Schema:\n{json.dumps(HUNTER_PLAN_SCHEMA, indent=2)}"
    )

    extra_args: dict[str, str | None] = {}
    if reasoning and reasoning.level != "off":
        extra_args.update(reasoning_to_cc_extra_args(reasoning))

    tools, mcp_servers = _build_scientist_tools_and_mcp(
        prediction_history,
        provider,
        notebook_path=notebook_path,
        output_dir=output_dir,
    )

    max_turns = 18
    budget = prepare_turn_budget(
        system_prompt + json_instruction, max_turns, tools, provider=provider
    )
    backend = get_backend(provider)
    options = SDKOptions(
        system_prompt=budget.system_prompt,
        allowed_tools=budget.allowed_tools,
        max_turns=budget.max_turns,
        model=model,
        extra_args=extra_args,
        mcp_servers=mcp_servers,
        response_schema=HunterPlanOutput,
    )

    async def _query(prompt: str, resume_session_id: str | None) -> QueryResult:
        opts = replace(options, resume=resume_session_id) if resume_session_id else options
        raw, usage, session_id = await collect_text_from_query(
            prompt, opts, backend, message_buffer, agent_name="Scientist"
        )
        return QueryResult(raw_output=raw, session_id=session_id, usage=usage)

    def _validate(result: QueryResult) -> dict[str, Any]:
        return validate_json_output(result.raw_output, HunterPlanOutput, "Scientist")

    return await agent_retry_loop(
        query_fn=_query,
        validate_fn=_validate,
        prompt=user_prompt,
        agent_name="Scientist",
    )


async def run_hunter_revision(
    original_plan: dict[str, Any],
    concern_ledger: list[dict[str, Any]],
    analysis: dict[str, Any],
    notebook_path: Path,
    version: str,
    domain_knowledge: str = "",
    prediction_history: list[PredictionRecord] | None = None,
    model: str | None = None,
    message_buffer: list[str] | None = None,
    goal: str = "",
    provider: str = "anthropic",
    reasoning: ReasoningConfig | None = None,
    output_dir: Path | None = None,
    pending_abductions: str = "",
) -> dict[str, Any]:
    """Revise the plan after a critic debate.

    Args:
        original_plan: The initial plan that was debated.
        concern_ledger: Structured list of ConcernLedgerEntry dicts.
        analysis: Structured analysis JSON from the Analyst.
        notebook_path: Path to the lab notebook.
        version: Version string.
        domain_knowledge: Domain-specific context.
        prediction_history: Accumulated testable predictions and outcomes.
        model: Model override.
        message_buffer: Optional buffer for streaming messages.
        goal: The user's investigation goal.
        reasoning: Reasoning/effort config for the model.

    Returns:
        Revised plan dict (same schema as the initial plan).
    """
    notebook_path = Path(notebook_path)
    notebook_entries = parse_notebook_entries(notebook_path)

    ledger_text = json.dumps(concern_ledger, indent=2) if concern_ledger else "(no concerns raised)"

    abductions_section = ""
    if pending_abductions:
        abductions_section = (
            "<pending_abductions>\n"
            "Address these pending alternative explanations: either include "
            "a testable prediction (with follows_from) or list in "
            "deprioritized_abductions with a reason.\n\n"
            f"{pending_abductions}\n"
            "</pending_abductions>\n"
        )

    user_prompt = HUNTER_REVISION_USER.format(
        goal=goal or "(no goal specified)",
        domain_knowledge=domain_knowledge or "(no domain knowledge provided)",
        analysis_json=(json.dumps(analysis, indent=2) if analysis else "(no analysis)"),
        notebook_content=format_notebook_toc(notebook_entries),
        original_plan=json.dumps(original_plan, indent=2),
        concern_ledger=ledger_text,
        prediction_history=format_compact_tree(prediction_history),
        pending_abductions_section=abductions_section,
        version=version,
    )

    json_instruction = (
        "\n\n## Output Format\n"
        "Tool calls are allowed before the final JSON response.\n"
        'The "JSON only" rule applies only to the final assistant message.\n'
        "Respond with valid JSON matching the schema below.\n"
        "No markdown fencing. No explanation. No other text.\n\n"
        f"Schema:\n{json.dumps(HUNTER_PLAN_SCHEMA, indent=2)}"
    )

    extra_args: dict[str, str | None] = {}
    if reasoning and reasoning.level != "off":
        extra_args.update(reasoning_to_cc_extra_args(reasoning))

    tools, mcp_servers = _build_scientist_tools_and_mcp(
        prediction_history,
        provider,
        notebook_path=notebook_path,
        output_dir=output_dir,
    )

    has_predictions = bool(prediction_history)
    revision_system = build_revision_system(has_predictions=has_predictions)

    max_turns = 18
    budget = prepare_turn_budget(
        revision_system + json_instruction,
        max_turns,
        tools,
        provider=provider,
    )
    backend = get_backend(provider)
    options = SDKOptions(
        system_prompt=budget.system_prompt,
        allowed_tools=budget.allowed_tools,
        max_turns=budget.max_turns,
        model=model,
        extra_args=extra_args,
        mcp_servers=mcp_servers,
        response_schema=HunterPlanOutput,
    )

    async def _query(prompt: str, resume_session_id: str | None) -> QueryResult:
        opts = replace(options, resume=resume_session_id) if resume_session_id else options
        raw, usage, session_id = await collect_text_from_query(
            prompt, opts, backend, message_buffer, agent_name="Scientist revision"
        )
        return QueryResult(raw_output=raw, session_id=session_id, usage=usage)

    def _validate(result: QueryResult) -> dict[str, Any]:
        return validate_json_output(result.raw_output, HunterPlanOutput, "Scientist revision")

    return await agent_retry_loop(
        query_fn=_query,
        validate_fn=_validate,
        prompt=user_prompt,
        agent_name="Scientist revision",
    )
