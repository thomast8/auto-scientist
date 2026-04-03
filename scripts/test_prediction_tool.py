#!/usr/bin/env python3
"""Validation script: test that the Scientist agent uses the read_predictions MCP tool.

Loads real data from experiments/saved/alloy_design/, builds the compact tree
summary + MCP server, runs the Scientist agent, and logs tool usage.

Usage:
    uv run python scripts/test_prediction_tool.py
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from auto_scientist.agents.prediction_tool import build_prediction_mcp_server
from auto_scientist.agents.scientist import _format_compact_tree
from auto_scientist.prompts.scientist import SCIENTIST_SYSTEM, SCIENTIST_USER
from auto_scientist.schemas import ScientistPlanOutput
from auto_scientist.state import ExperimentState

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger("validation")

SAVED_RUN = Path("experiments/saved/alloy_design")


def load_real_data() -> tuple[ExperimentState, dict, str]:
    """Load state, analysis, and notebook from a real saved run."""
    state = ExperimentState.load(SAVED_RUN / "state.json")

    # Find the latest analysis.json
    versions = sorted(SAVED_RUN.glob("v*/analysis.json"))
    analysis = json.loads(versions[-1].read_text()) if versions else {}

    notebook_path = SAVED_RUN / "lab_notebook.xml"
    notebook_content = notebook_path.read_text() if notebook_path.exists() else ""

    return state, analysis, notebook_content


def build_prompt(state: ExperimentState, analysis: dict, notebook_content: str) -> str:
    """Build the Scientist user prompt with compact tree."""
    compact_tree = _format_compact_tree(state.prediction_history)
    return SCIENTIST_USER.format(
        goal=state.goal or "(no goal specified)",
        domain_knowledge=state.domain_knowledge or "(no domain knowledge provided)",
        analysis_json=json.dumps(analysis, indent=2) if analysis else "(no analysis)",
        notebook_content=notebook_content or "(empty notebook)",
        prediction_history=compact_tree,
        version=f"v{state.iteration:02d}",
    )


async def run_validation():
    """Run the Scientist with the MCP tool and log usage."""
    from claude_code_sdk import (
        AssistantMessage,
        ClaudeCodeOptions,
        ResultMessage,
        query,
    )

    if not SAVED_RUN.exists():
        logger.error(f"Saved run not found: {SAVED_RUN}")
        logger.error("Run an alloy_design experiment first to generate test data.")
        return

    state, analysis, notebook_content = load_real_data()
    logger.info(
        f"Loaded state: {len(state.prediction_history)} predictions, iteration {state.iteration}"
    )

    # Build compact tree and MCP server
    compact_tree = _format_compact_tree(state.prediction_history)
    logger.info(f"Compact tree: {len(compact_tree)} chars, {compact_tree.count(chr(10)) + 1} lines")

    mcp_server = build_prediction_mcp_server(state.prediction_history)

    # Build prompt
    user_prompt = build_prompt(state, analysis, notebook_content)
    logger.info(f"User prompt: {len(user_prompt):,} chars")

    # Build system prompt
    plan_schema = ScientistPlanOutput.model_json_schema()
    json_instruction = (
        "\n\n## Output Format\n"
        "You MUST respond with ONLY valid JSON matching the schema below.\n"
        "No markdown fencing. No explanation. No other text.\n\n"
        f"Schema:\n{json.dumps(plan_schema, indent=2)}"
    )
    system_prompt = SCIENTIST_SYSTEM + json_instruction
    system_prompt += (
        "\n\n<available_tools>\n"
        "Your available tools (call directly, do NOT use ToolSearch):\n"
        "  - WebSearch(query: str) - Search the web.\n"
        "  - mcp__predictions__read_predictions(pred_ids?, filter?, iteration?) "
        "- Query prediction history for full detail.\n"
        "</available_tools>"
        "\n<turn_budget>You have a budget of 15 turns for this task.</turn_budget>"
    )

    logger.info(f"System prompt: {len(system_prompt):,} chars")
    logger.info(f"Total prompt: {len(system_prompt) + len(user_prompt):,} chars")

    # Track tool calls
    tool_calls: list[dict] = []

    options = ClaudeCodeOptions(
        system_prompt=system_prompt,
        allowed_tools=["WebSearch", "mcp__predictions__read_predictions"],
        mcp_servers={"predictions": mcp_server},
        max_turns=15,
        permission_mode="acceptEdits",
        extra_args={},
    )

    logger.info("Running Scientist agent...")
    final_result = None
    async for message in query(prompt=user_prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if hasattr(block, "name") and block.name.startswith("mcp__"):
                    call_info = {
                        "tool": block.name,
                        "input": block.input if hasattr(block, "input") else {},
                    }
                    tool_calls.append(call_info)
                    logger.info(f"Tool call: {block.name}({json.dumps(call_info['input'])})")
        elif isinstance(message, ResultMessage):
            final_result = message.result

    # Report results
    logger.info("=" * 60)
    logger.info("VALIDATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total MCP tool calls: {len(tool_calls)}")
    for i, call in enumerate(tool_calls):
        logger.info(f"  Call {i + 1}: {call['tool']}({json.dumps(call['input'])})")

    if tool_calls:
        logger.info("PASS: Scientist used the read_predictions tool")
    else:
        logger.warning("FAIL: Scientist did NOT use the read_predictions tool")

    if final_result:
        try:
            plan = json.loads(final_result)
            logger.info(f"Plan hypothesis: {plan.get('hypothesis', 'N/A')[:100]}")
            predictions = plan.get("testable_predictions", [])
            logger.info(f"Plan predictions: {len(predictions)}")
        except json.JSONDecodeError:
            logger.warning(f"Could not parse plan as JSON: {final_result[:200]}")
    else:
        logger.warning("No final result from Scientist")


if __name__ == "__main__":
    asyncio.run(run_validation())
