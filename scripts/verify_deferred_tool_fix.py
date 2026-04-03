"""Verify deferred tool resolution fix.

Run BEFORE the fix:  "before" checks pass, "after" checks fail
Run AFTER the fix:   all checks pass

Usage:
    uv run python scripts/verify_deferred_tool_fix.py
"""

import sys


def check(label: str, condition: bool, detail: str = "") -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}" + (f" -- {detail}" if detail else ""))
    return condition


def run_checks() -> int:
    passed = 0
    failed = 0

    def tally(ok: bool) -> None:
        nonlocal passed, failed
        if ok:
            passed += 1
        else:
            failed += 1

    # --- BEFORE checks: document the broken state ---
    print("\n=== BEFORE-FIX CHECKS (current broken behavior) ===")

    from auto_scientist import sdk_utils

    tally(
        check(
            "with_turn_budget removed",
            not hasattr(sdk_utils, "with_turn_budget"),
        )
    )

    # --- AFTER checks: validate the fix ---
    print("\n=== AFTER-FIX CHECKS (expected after applying fix) ===")

    has_new = hasattr(sdk_utils, "prepare_turn_budget")
    tally(check("prepare_turn_budget exists", has_new))

    if not has_new:
        print("  [SKIP] Skipping remaining after-fix checks (function not found)")
        total = passed + failed
        print(f"\n{'=' * 50}")
        print(f"Results: {passed}/{total} passed, {failed}/{total} failed")
        print("Some checks failed (expected if running before the fix)")
        return 1

    from auto_scientist.sdk_utils import TurnBudgetConfig

    # 1. Deferred tools + anthropic: should add ToolSearch
    budget = sdk_utils.prepare_turn_budget(
        "system prompt", 10, ["WebSearch", "Read"], provider="anthropic"
    )

    tally(check("Returns TurnBudgetConfig", isinstance(budget, TurnBudgetConfig)))

    tally(
        check(
            "ToolSearch IN allowed_tools (anthropic + WebSearch)",
            "ToolSearch" in budget.allowed_tools,
            f"allowed_tools={budget.allowed_tools}",
        )
    )

    tally(
        check(
            "max_turns bumped to 11 (10+1 for ToolSearch)",
            budget.max_turns == 11,
            f"max_turns={budget.max_turns}",
        )
    )

    tally(check("Prompt contains [DEFERRED] tag", "[DEFERRED]" in budget.system_prompt))

    tally(check("Prompt contains select: instruction", "select:" in budget.system_prompt))

    tally(
        check(
            "Prompt does NOT say 'do NOT use ToolSearch'",
            "do NOT use ToolSearch" not in budget.system_prompt,
        )
    )

    # 2. Same tools + openai: should NOT add ToolSearch
    budget_oai = sdk_utils.prepare_turn_budget(
        "system prompt", 10, ["WebSearch", "Read"], provider="openai"
    )

    tally(
        check(
            "No ToolSearch for openai provider",
            "ToolSearch" not in budget_oai.allowed_tools,
            f"allowed_tools={budget_oai.allowed_tools}",
        )
    )

    tally(check("max_turns unchanged for openai", budget_oai.max_turns == 10))

    # 3. Standard tools only: no ToolSearch needed
    budget_std = sdk_utils.prepare_turn_budget(
        "system prompt", 30, ["Read", "Glob"], provider="anthropic"
    )

    tally(
        check("No ToolSearch for standard-only tools", "ToolSearch" not in budget_std.allowed_tools)
    )

    tally(check("max_turns unchanged for standard-only", budget_std.max_turns == 30))

    # 4. MCP tools: not deferred
    budget_mcp = sdk_utils.prepare_turn_budget(
        "system prompt", 10, ["mcp__predictions__read_predictions"], provider="anthropic"
    )

    tally(check("No ToolSearch for MCP-only tools", "ToolSearch" not in budget_mcp.allowed_tools))

    # 5. AskUserQuestion deferred
    budget_ask = sdk_utils.prepare_turn_budget(
        "system prompt", 30, ["Bash", "Read", "AskUserQuestion"], provider="anthropic"
    )

    tally(check("ToolSearch added for AskUserQuestion", "ToolSearch" in budget_ask.allowed_tools))

    # 6. Mixed: WebSearch + MCP + standard
    budget_mix = sdk_utils.prepare_turn_budget(
        "system prompt",
        15,
        ["WebSearch", "mcp__predictions__read_predictions", "Read"],
        provider="anthropic",
    )

    tally(
        check(
            "Mixed: ToolSearch added (WebSearch is deferred)",
            "ToolSearch" in budget_mix.allowed_tools,
        )
    )

    tally(
        check(
            "Mixed: [DEFERRED] only on WebSearch, not MCP",
            "[DEFERRED]" in budget_mix.system_prompt
            and "[DEFERRED] mcp__" not in budget_mix.system_prompt,
        )
    )

    # --- Summary ---
    total = passed + failed
    print(f"\n{'=' * 50}")
    print(f"Results: {passed}/{total} passed, {failed}/{total} failed")
    if failed:
        print("Some checks failed (expected if running before the fix)")
        return 1
    print("All checks passed!")
    return 0


if __name__ == "__main__":
    sys.exit(run_checks())
