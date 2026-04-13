#!/usr/bin/env python3
"""Validate the notebook TOC + MCP tool with real run data.

Replays a completed run, showing:
1. How the compact Table of Contents evolves as entries are added
2. What the old full-XML format looked like (for comparison)
3. Prompt size savings at each step
4. Optionally runs the Scientist agent (Claude or Codex) to test tool usage

Usage:
    # Show TOC evolution only (no API calls):
    uv run python scripts/validate_notebook_tool.py --run alloy_design

    # Run Scientist with Claude backend on a specific saved iteration:
    uv run python scripts/validate_notebook_tool.py --run alloy_design --invoke --iteration 3

    # Run with Codex backend (no MCP tool, tests fallback):
    uv run python scripts/validate_notebook_tool.py \
      --run alloy_design --invoke --iteration 3 --provider openai
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from auto_scientist.agents.notebook_tool import (  # noqa: E402
    NOTEBOOK_SPEC,
    _handle_read_notebook,
    format_notebook_toc,
)
from auto_scientist.agents.prediction_tool import format_compact_tree  # noqa: E402
from auto_scientist.agents.scientist import _build_scientist_tools_and_mcp  # noqa: E402
from auto_scientist.notebook import (  # noqa: E402
    NOTEBOOK_FILENAME,
    parse_notebook_entries,
    read_notebook,
)
from auto_scientist.prompts.scientist import (  # noqa: E402
    SCIENTIST_USER,
    build_scientist_system,
)
from auto_scientist.schemas import ScientistPlanOutput  # noqa: E402
from auto_scientist.state import ExperimentState  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("validate-notebook")

SAVED_DIR = Path("experiments/saved")


def load_run(name: str) -> tuple[ExperimentState, dict[int, dict], Path]:
    """Load state, per-iteration analyses, and notebook path from a saved run."""
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

    notebook_path = run_dir / NOTEBOOK_FILENAME
    if not notebook_path.exists():
        logger.error(f"No lab_notebook.xml in {run_dir}")
        sys.exit(1)

    return state, analyses, notebook_path


def _entries_at_or_before(entries: list[dict[str, str]], iteration: int) -> list[dict[str, str]]:
    """Return entries the Scientist would see at the START of *iteration*.

    Versions are strings like ``v00``, ``v01`` plus ``ingestion``. We always
    include ``ingestion`` and any ``v00..v{iteration-1}``. Multiple entries
    can share a version (scientist + revision + stop_gate).
    """
    kept: list[dict[str, str]] = []
    for entry in entries:
        version = entry.get("version", "")
        if version == "ingestion":
            kept.append(entry)
            continue
        if version.startswith("v"):
            try:
                v = int(version[1:])
            except ValueError:
                continue
            if v < iteration:
                kept.append(entry)
    return kept


def _full_xml_for(entries: list[dict[str, str]]) -> str:
    """Serialize an entry list back to the same XML shape `read_notebook()` returns."""
    if not entries:
        return ""
    root = ET.Element("lab_notebook")
    for entry in entries:
        el = ET.SubElement(
            root,
            "entry",
            {"version": entry.get("version", ""), "source": entry.get("source", "")},
        )
        title_el = ET.SubElement(el, "title")
        title_el.text = entry.get("title", "")
        content = entry.get("content", "")
        if content:
            content_el = ET.SubElement(el, "content")
            content_el.text = "\n" + content + "\n  "
        else:
            ET.SubElement(el, "content")
    return '<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(root, encoding="unicode")


def show_toc_evolution(state: ExperimentState, notebook_path: Path) -> None:
    """Print the compact TOC at each iteration, with size comparison."""
    all_entries = parse_notebook_entries(notebook_path)
    max_iter = state.iteration

    print("\n" + "=" * 80)
    print("NOTEBOOK TOC EVOLUTION")
    print("=" * 80)

    rows: list[tuple[int, int, int, int, float]] = []
    for iteration in range(max_iter + 2):
        entries = _entries_at_or_before(all_entries, iteration)
        toc = format_notebook_toc(entries)
        full = _full_xml_for(entries)

        toc_size = len(toc)
        full_size = len(full)
        savings = ((full_size - toc_size) / full_size * 100) if full_size > 0 else 0.0
        rows.append((iteration, len(entries), toc_size, full_size, savings))

        print(f"\n{'-' * 80}")
        print(f"BEFORE ITERATION {iteration} (Scientist sees {len(entries)} entries)")
        print(f"  TOC: {toc_size:,} | Full XML: {full_size:,} | -{savings:.0f}%")
        print(f"{'-' * 80}")
        print()
        print(toc)
        print()

    print("\n" + "=" * 80)
    print("SIZE COMPARISON TABLE")
    print("=" * 80)
    print(f"{'Iter':>4}  {'Entries':>7}  {'TOC':>10}  {'Full':>10}  {'Savings':>8}")
    print("-" * 47)
    for iteration, n, toc_size, full_size, savings in rows:
        print(f"{iteration:>4}  {n:>7}  {toc_size:>8,}  {full_size:>8,}  {savings:>6.0f}%")


def show_mcp_tool_demo(notebook_path: Path) -> None:
    """Demonstrate MCP tool queries against the real notebook entries."""
    print("\n" + "=" * 80)
    print("MCP TOOL DEMO (what an agent can query)")
    print("=" * 80)

    entries = parse_notebook_entries(notebook_path)
    if not entries:
        print("No entries to demo.")
        return

    available_versions = sorted({e.get("version", "") for e in entries if e.get("version")})
    sample_version = available_versions[0] if available_versions else None

    queries: list[tuple[str, dict[str, Any]]] = [
        ("Summary (counts only)", {"summary": True}),
        ("All scientist entries", {"source": "scientist"}),
        ("Most recent 2 entries", {"last_n": 2}),
    ]
    if sample_version:
        queries.append((f"Specific version [{sample_version}]", {"versions": [sample_version]}))
    queries.append(("Search 'hypothesis'", {"search": "hypothesis"}))

    loop = asyncio.new_event_loop()
    try:
        for label, args in queries:
            print(f"\n{'-' * 60}")
            print(f"Query: {label}")
            print(f"  Args: {json.dumps(args)}")
            print(f"{'-' * 60}")
            result = loop.run_until_complete(_handle_read_notebook(entries, args))
            text = result["content"][0]["text"]
            if len(text) > 1000:
                print(text[:1000])
                print(f"  ... ({len(text) - 1000} more chars)")
            else:
                print(text)
    finally:
        loop.close()


async def run_scientist_with_tool(
    state: ExperimentState,
    analyses: dict[int, dict],
    notebook_path: Path,
    iteration: int,
    provider: str,
) -> None:
    """Invoke the Scientist agent and observe notebook tool usage."""
    from auto_scientist.sdk_backend import SDKOptions, get_backend
    from auto_scientist.sdk_utils import prepare_turn_budget

    preds = [p for p in state.prediction_history if p.iteration_prescribed < iteration]
    analysis = analyses.get(iteration - 1, {})

    entries = _entries_at_or_before(parse_notebook_entries(notebook_path), iteration)
    toc = format_notebook_toc(entries)
    full = read_notebook(notebook_path)
    logger.info(f"Notebook entries (visible at iter {iteration}): {len(entries)}")
    logger.info(f"Compact TOC: {len(toc)} chars | Full XML: {len(full)} chars")

    user_prompt = SCIENTIST_USER.format(
        goal=state.goal or "(no goal specified)",
        domain_knowledge=state.domain_knowledge or "(none)",
        analysis_json=(json.dumps(analysis, indent=2) if analysis else "(no analysis)"),
        notebook_content=toc,
        prediction_history=format_compact_tree(preds),
        pending_abductions_section="",
        version=f"v{iteration:02d}",
    )

    plan_schema = ScientistPlanOutput.model_json_schema()
    json_instruction = (
        "\n\n## Output Format\n"
        "You MUST respond with ONLY valid JSON matching the schema below.\n"
        "No markdown fencing. No explanation. No other text.\n\n"
        f"Schema:\n{json.dumps(plan_schema, indent=2)}"
    )

    tools, mcp_servers = _build_scientist_tools_and_mcp(
        preds,
        provider,
        notebook_path=notebook_path,
    )
    logger.info(f"Tools: {tools}")
    logger.info(f"MCP servers: {list(mcp_servers.keys()) or 'none'}")

    prompt_provider = "gpt" if provider == "openai" else "claude"
    system_prompt = build_scientist_system(prompt_provider, has_predictions=bool(preds))
    budget = prepare_turn_budget(system_prompt + json_instruction, 18, tools, provider=provider)

    print(f"\n{'=' * 80}")
    print(f"SCIENTIST INVOCATION (iter {iteration}, provider={provider})")
    print(f"{'=' * 80}")
    print(f"System prompt: {len(budget.system_prompt):,} chars")
    print(f"User prompt:   {len(user_prompt):,} chars")
    print(f"  Notebook TOC slot: {len(toc):,} chars (vs {len(full):,} full XML)")
    print(f"Notebook MCP tool: {'yes' if 'notebook' in mcp_servers else 'no'}")
    print()

    backend = get_backend(provider)
    options = SDKOptions(
        system_prompt=budget.system_prompt,
        allowed_tools=budget.allowed_tools,
        max_turns=budget.max_turns,
        mcp_servers=mcp_servers,
    )

    notebook_calls: list[dict[str, Any]] = []
    other_calls: list[str] = []
    final_result: Any = None

    try:
        async for msg in backend.query(user_prompt, options):
            if msg.type == "assistant":
                for block in msg.content_blocks:
                    name = getattr(block, "name", None)
                    inp = getattr(block, "input", {})
                    text = getattr(block, "text", None)

                    if name == NOTEBOOK_SPEC.mcp_tool_name:
                        call_info = {"tool": name, "input": inp or {}}
                        notebook_calls.append(call_info)
                        print(f"  [NOTEBOOK CALL] {name}")
                        print(f"    Args: {json.dumps(call_info['input'], indent=2)}")
                    elif name and name.startswith("mcp__"):
                        other_calls.append(f"{name}({json.dumps(inp)})")
                        print(f"  [MCP CALL] {name}")
                    elif name == "WebSearch":
                        q = inp.get("query", "") if isinstance(inp, dict) else ""
                        print(f"  [WEB SEARCH] {q}")
                    elif text:
                        txt = text[:300] + ("..." if len(text) > 300 else "")
                        print(f"  [TEXT] {txt}")
            elif msg.type == "result":
                final_result = msg.result
                if msg.usage:
                    print(f"\n  Usage: {msg.usage}")
    except BaseException as exc:  # noqa: BLE001
        logger.warning(f"SDK error: {type(exc).__name__}: {exc}")

    print(f"\n{'-' * 60}")
    print("RESULTS")
    print(f"{'-' * 60}")
    print(f"Notebook tool calls: {len(notebook_calls)}")
    for i, call in enumerate(notebook_calls):
        print(f"  {i + 1}. {call['tool']}({json.dumps(call['input'])})")
    print(f"Other MCP/web calls: {len(other_calls)}")
    for call_text in other_calls:
        print(f"  - {call_text}")

    if notebook_calls:
        print("\nVERDICT: PASS - Scientist used the read_notebook tool")
    elif "notebook" in mcp_servers:
        print("\nVERDICT: WEAK - notebook MCP tool was available but not called this run")
    else:
        print("\nVERDICT: N/A - notebook MCP tool not available")

    if not final_result:
        print("\nNo final result captured (SDK may have errored before completion)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate notebook TOC + MCP tool")
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

    state, analyses, notebook_path = load_run(args.run)
    n_entries = len(parse_notebook_entries(notebook_path))
    logger.info(f"Loaded '{args.run}': {state.iteration} iters, {n_entries} entries")

    show_toc_evolution(state, notebook_path)

    if not args.no_demo:
        show_mcp_tool_demo(notebook_path)

    if args.invoke:
        iteration = args.iteration if args.iteration is not None else state.iteration
        if iteration > state.iteration:
            logger.error(f"Iteration {iteration} exceeds run's max ({state.iteration})")
            sys.exit(1)
        asyncio.run(
            run_scientist_with_tool(state, analyses, notebook_path, iteration, args.provider)
        )
    elif args.iteration is not None:
        logger.info("Use --invoke to actually run the Scientist agent")


if __name__ == "__main__":
    main()
