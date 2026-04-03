#!/usr/bin/env python3
"""Validate the prediction tree + MCP tool with real run data.

Replays a completed run iteration-by-iteration, showing:
1. How the compact tree evolves at each iteration
2. What the old full-detail format looked like (for comparison)
3. Prompt size savings at each step
4. Optionally runs the Scientist agent (Claude or Codex) to test tool usage

Usage:
    # Show tree evolution only (no API calls):
    uv run python scripts/validate_prediction_tool.py --run alloy_design

    # Run Scientist with Claude backend at a specific iteration:
    uv run python scripts/validate_prediction_tool.py --run alloy_design --invoke --iteration 3

    # Run with Codex backend (no MCP tool, tests fallback):
    uv run python scripts/validate_prediction_tool.py \
      --run alloy_design --invoke --iteration 3 --provider openai
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from auto_scientist.agents.scientist import (
    _build_scientist_tools_and_mcp,
    _format_compact_tree,
    _format_predictions_for_prompt,
)
from auto_scientist.prompts.scientist import SCIENTIST_SYSTEM, SCIENTIST_USER
from auto_scientist.schemas import ScientistPlanOutput
from auto_scientist.state import ExperimentState, PredictionRecord

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("validate")

SAVED_DIR = Path("experiments/saved")


def load_run(name: str) -> tuple[ExperimentState, dict[int, dict], str]:
    """Load state, per-iteration analyses, and notebook from a saved run."""
    run_dir = SAVED_DIR / name
    if not run_dir.exists():
        logger.error(f"Run not found: {run_dir}")
        logger.error(f"Available: {[d.name for d in SAVED_DIR.iterdir() if d.is_dir()]}")
        sys.exit(1)

    state = ExperimentState.load(run_dir / "state.json")

    analyses: dict[int, dict] = {}
    for analysis_path in sorted(run_dir.glob("v*/analysis.json")):
        version_str = analysis_path.parent.name  # e.g. "v03"
        iteration = int(version_str[1:])
        analyses[iteration] = json.loads(analysis_path.read_text())

    notebook_path = run_dir / "lab_notebook.xml"
    notebook = notebook_path.read_text() if notebook_path.exists() else ""

    return state, analyses, notebook


def predictions_at_iteration(
    all_predictions: list[PredictionRecord], iteration: int
) -> list[PredictionRecord]:
    """Return predictions that would exist at the START of a given iteration.

    At iteration N, the Scientist sees:
    - All predictions prescribed in iterations 0..N-1
    - Outcomes resolved through iteration N-1
    """
    return [p for p in all_predictions if p.iteration_prescribed < iteration]


def show_tree_evolution(state: ExperimentState) -> None:
    """Print the compact tree at each iteration, with size comparison."""
    all_preds = state.prediction_history
    max_iter = state.iteration

    print("\n" + "=" * 80)
    print("PREDICTION TREE EVOLUTION")
    print("=" * 80)

    for iteration in range(max_iter + 1):
        preds = predictions_at_iteration(all_preds, iteration)

        compact = _format_compact_tree(preds) if preds else "(no predictions yet)"
        full = _format_predictions_for_prompt(preds) if preds else "(no predictions yet)"

        compact_size = len(compact)
        full_size = len(full)
        savings = ((full_size - compact_size) / full_size * 100) if full_size > 0 else 0

        print(f"\n{'─' * 80}")
        print(f"ITERATION {iteration} (Scientist sees {len(preds)} predictions)")
        print(f"  Compact: {compact_size:,} | Full: {full_size:,} | -{savings:.0f}%")
        print(f"{'─' * 80}")
        print()
        print(compact)
        print()

    # Summary table
    print("\n" + "=" * 80)
    print("SIZE COMPARISON TABLE")
    print("=" * 80)
    print(f"{'Iter':>4}  {'Preds':>5}  {'Compact':>10}  {'Full':>10}  {'Savings':>8}")
    print("-" * 45)
    for iteration in range(max_iter + 1):
        preds = predictions_at_iteration(all_preds, iteration)
        compact_size = len(_format_compact_tree(preds)) if preds else 0
        full_size = len(_format_predictions_for_prompt(preds)) if preds else 0
        savings = ((full_size - compact_size) / full_size * 100) if full_size > 0 else 0
        n = len(preds)
        print(f"{iteration:>4}  {n:>5}  {compact_size:>8,}  {full_size:>8,}  {savings:>6.0f}%")


def show_mcp_tool_demo(state: ExperimentState) -> None:
    """Demonstrate MCP tool queries against the real prediction history."""
    print("\n" + "=" * 80)
    print("MCP TOOL DEMO (what the Scientist can query)")
    print("=" * 80)

    from auto_scientist.agents.prediction_tool import _handle_read_predictions

    preds = state.prediction_history

    queries: list[tuple[str, dict[str, Any]]] = [
        ("All pending predictions", {"filter": "pending"}),
        ("All refuted (dead ends)", {"filter": "refuted"}),
        ("Active chains (pending + ancestors)", {"filter": "active_chains"}),
        ("Specific prediction [0.2]", {"pred_ids": ["0.2"]}),
        ("Iteration 1 predictions", {"iteration": 1}),
    ]

    for label, args in queries:
        print(f"\n{'─' * 60}")
        print(f"Query: {label}")
        print(f"  Args: {json.dumps(args)}")
        print(f"{'─' * 60}")
        result = asyncio.get_event_loop().run_until_complete(_handle_read_predictions(preds, args))
        text = result["content"][0]["text"]
        # Truncate long results for readability
        if len(text) > 1000:
            print(text[:1000])
            print(f"  ... ({len(text) - 1000} more chars)")
        else:
            print(text)


async def run_scientist_with_tool(
    state: ExperimentState,
    analyses: dict[int, dict],
    notebook: str,
    iteration: int,
    provider: str,
) -> None:
    """Actually invoke the Scientist agent and observe tool usage.

    Uses the project's SDKBackend abstraction so Claude runs through the
    Claude Code CLI and Codex runs through the Codex CLI, each with their
    native MCP support.
    """
    from auto_scientist.sdk_backend import SDKOptions, get_backend
    from auto_scientist.sdk_utils import with_turn_budget

    preds = predictions_at_iteration(state.prediction_history, iteration)
    analysis = analyses.get(iteration - 1, {})

    logger.info(f"Running Scientist at iteration {iteration}, {len(preds)} preds")
    logger.info(f"Provider: {provider}")

    compact_tree = _format_compact_tree(preds)
    logger.info(f"Compact tree: {len(compact_tree)} chars")

    user_prompt = SCIENTIST_USER.format(
        goal=state.goal or "(no goal specified)",
        domain_knowledge=state.domain_knowledge or "(none)",
        analysis_json=(json.dumps(analysis, indent=2) if analysis else "(no analysis)"),
        notebook_content=notebook or "(empty notebook)",
        prediction_history=compact_tree,
        version=f"v{iteration:02d}",
    )

    plan_schema = ScientistPlanOutput.model_json_schema()
    json_instruction = (
        "\n\n## Output Format\n"
        "You MUST respond with ONLY valid JSON matching the schema below.\n"
        "No markdown fencing. No explanation. No other text.\n\n"
        f"Schema:\n{json.dumps(plan_schema, indent=2)}"
    )

    tools, mcp_servers = _build_scientist_tools_and_mcp(preds, provider)
    logger.info(f"Tools: {tools}")
    logger.info(f"MCP servers: {list(mcp_servers.keys()) or 'none'}")

    system_prompt = with_turn_budget(SCIENTIST_SYSTEM + json_instruction, 15, tools)

    logger.info(f"System: {len(system_prompt):,} | User: {len(user_prompt):,}")

    print(f"\n{'=' * 80}")
    print(f"SCIENTIST INVOCATION (iter {iteration}, provider={provider})")
    print(f"{'=' * 80}")
    print(f"Predictions: {len(preds)}")
    print(f"Compact tree: {len(compact_tree)} chars")
    print(f"MCP tool: {'yes' if mcp_servers else 'no'}")
    print()

    backend = get_backend(provider)
    options = SDKOptions(
        system_prompt=system_prompt,
        allowed_tools=tools,
        max_turns=15,
        mcp_servers=mcp_servers,
    )

    tool_calls: list[dict[str, Any]] = []
    web_searches: list[str] = []
    final_result = None

    try:
        async for msg in backend.query(user_prompt, options):
            if msg.type == "assistant":
                for block in msg.content_blocks:
                    name = getattr(block, "name", None)
                    inp = getattr(block, "input", {})
                    text = getattr(block, "text", None)

                    if name and name.startswith("mcp__predictions"):
                        call_info = {"tool": name, "input": inp or {}}
                        tool_calls.append(call_info)
                        print(f"  [MCP CALL] {name}")
                        print(f"    Args: {json.dumps(call_info['input'], indent=2)}")
                    elif name == "WebSearch":
                        q = inp.get("query", "") if isinstance(inp, dict) else ""
                        web_searches.append(q)
                        print(f"  [WEB SEARCH] {q}")
                    elif name == "ToolSearch":
                        print(f"  [TOOL SEARCH] {inp}")
                    elif text:
                        txt = text[:300] + ("..." if len(text) > 300 else "")
                        print(f"  [TEXT] {txt}")
            elif msg.type == "result":
                final_result = msg.result
                if msg.usage:
                    print(f"\n  Usage: {msg.usage}")
    except BaseException as exc:
        logger.warning(f"SDK error: {type(exc).__name__}: {exc}")

    # Results
    print(f"\n{'─' * 60}")
    print("RESULTS")
    print(f"{'─' * 60}")
    print(f"MCP tool calls: {len(tool_calls)}")
    for i, call in enumerate(tool_calls):
        print(f"  {i + 1}. {call['tool']}({json.dumps(call['input'])})")
    print(f"Web searches: {len(web_searches)}")
    for i, q in enumerate(web_searches):
        print(f"  {i + 1}. {q}")

    if tool_calls:
        print("\nVERDICT: PASS - Scientist used the read_predictions tool")
    elif mcp_servers:
        print("\nVERDICT: FAIL - MCP tool was available but not used")
    else:
        print("\nVERDICT: N/A - MCP tool not available (Codex backend)")

    if not final_result:
        print("\nNo final result captured (SDK may have errored before completion)")

    if final_result:
        try:
            plan = json.loads(final_result)
            print(f"\nPlan hypothesis: {plan.get('hypothesis', 'N/A')[:200]}")
            preds_out = plan.get("testable_predictions", [])
            print(f"New predictions: {len(preds_out)}")
            for p in preds_out:
                follows = p.get("follows_from", "")
                follows_str = f" (follows {follows})" if follows else ""
                print(f"  - {p.get('prediction', 'N/A')[:100]}{follows_str}")
            should_stop = plan.get("should_stop", False)
            print(f"Should stop: {should_stop}")
            if should_stop:
                print(f"  Reason: {plan.get('stop_reason', 'N/A')[:200]}")
        except json.JSONDecodeError:
            print(f"\nCould not parse plan: {final_result[:300]}")


def main():
    parser = argparse.ArgumentParser(description="Validate prediction tree + MCP tool")
    parser.add_argument(
        "--run",
        default="alloy_design",
        help="Name of saved run to use (default: alloy_design)",
    )
    parser.add_argument(
        "--invoke",
        action="store_true",
        help="Actually invoke the Scientist agent (requires API access)",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=None,
        help="Iteration to simulate (default: last iteration of the run)",
    )
    parser.add_argument(
        "--provider",
        default="anthropic",
        choices=["anthropic", "openai"],
        help="Backend provider (default: anthropic)",
    )
    parser.add_argument(
        "--no-demo",
        action="store_true",
        help="Skip the MCP tool demo section",
    )
    args = parser.parse_args()

    state, analyses, notebook = load_run(args.run)
    n_preds = len(state.prediction_history)
    logger.info(f"Loaded '{args.run}': {state.iteration} iters, {n_preds} preds")

    # Always show tree evolution
    show_tree_evolution(state)

    # MCP tool demo
    if not args.no_demo:
        show_mcp_tool_demo(state)

    # Optionally invoke the Scientist
    if args.invoke:
        iteration = args.iteration if args.iteration is not None else state.iteration
        if iteration > state.iteration:
            logger.error(f"Iteration {iteration} exceeds run's max ({state.iteration})")
            sys.exit(1)
        asyncio.run(run_scientist_with_tool(state, analyses, notebook, iteration, args.provider))
    elif args.iteration is not None:
        logger.info("Use --invoke to actually run the Scientist agent")


if __name__ == "__main__":
    main()
