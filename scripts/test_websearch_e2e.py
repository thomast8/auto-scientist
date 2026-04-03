"""End-to-end test: verify Claude Code SDK subprocesses have full tool access
and proper isolation from the host environment.

Tests:
1. WebSearch works via ToolSearch resolution
2. Glob and Write available (were stripped by --bare)
3. Host isolation: no CLAUDE.md, no hooks, no plugins leaking

This costs API credits. Run manually:
    uv run python scripts/test_websearch_e2e.py

Exit code 0 = all checks pass, 1 = failures.
"""

import asyncio
import sys

from auto_scientist.sdk_backend import ClaudeBackend, SDKOptions
from auto_scientist.sdk_utils import prepare_turn_budget


async def run_query(
    label: str,
    system_prompt: str,
    allowed_tools: list[str],
    max_turns: int,
    prompt: str,
) -> tuple[list[str], str]:
    """Run SDK query, return (tool_calls, result_text)."""
    print(f"\n--- {label} ---")
    options = SDKOptions(
        system_prompt=system_prompt,
        allowed_tools=allowed_tools,
        max_turns=max_turns,
        model="claude-sonnet-4-6",
    )
    backend = ClaudeBackend()
    tool_calls: list[str] = []
    result_text = ""

    try:
        async for msg in backend.query(prompt, options):
            if msg.type == "assistant" and msg.content_blocks:
                for block in msg.content_blocks:
                    name = getattr(block, "name", None)
                    if name:
                        tool_calls.append(name)
                        inp = getattr(block, "input", {})
                        print(f"  [Tool] {name}({inp})")
                    text = getattr(block, "text", None)
                    if text and text.strip():
                        print(f"  [Text] {text[:200]}")
            elif msg.type == "result":
                result_text = msg.result or ""
                print(f"  [Result] {result_text[:200]}")
    except Exception as e:
        print(f"  [ERROR] {e}")

    return tool_calls, result_text


async def main() -> int:
    print("=" * 60)
    print("E2E: SDK subprocess tool access + isolation")
    print("=" * 60)

    passed = 0
    failed = 0

    def check(label: str, condition: bool, detail: str = "") -> None:
        nonlocal passed, failed
        status = "PASS" if condition else "FAIL"
        print(f"  [{status}] {label}" + (f" -- {detail}" if detail else ""))
        if condition:
            passed += 1
        else:
            failed += 1

    # =================================================================
    # Test 1: WebSearch via prepare_turn_budget
    # =================================================================
    print("\n=== Test 1: WebSearch via ToolSearch resolution ===")
    budget = prepare_turn_budget(
        "You are a research assistant. Search the web to answer the question.",
        5,
        ["WebSearch"],
        provider="anthropic",
    )
    tools1, result1 = await run_query(
        "WebSearch via ToolSearch",
        budget.system_prompt,
        budget.allowed_tools,
        budget.max_turns,
        "Use WebSearch: what year was Python first released? Reply with ONLY the year.",
    )

    check("ToolSearch was called", "ToolSearch" in tools1, f"tools={tools1}")
    check("WebSearch was called", "WebSearch" in tools1, f"tools={tools1}")
    check("Got a result with 1991", "1991" in result1, f"result={result1[:100]!r}")

    # =================================================================
    # Test 2: Glob available (was stripped by --bare)
    # =================================================================
    print("\n=== Test 2: Glob tool ===")
    tools2, _ = await run_query(
        "Use Glob to find files",
        "You are a file system assistant. Use the Glob tool to find files.",
        ["Glob"],
        3,
        "Use Glob to find all *.py files in the current directory. Report the count.",
    )
    check("Glob was called", "Glob" in tools2, f"tools={tools2}")

    # =================================================================
    # Test 3: Write available (was stripped by --bare)
    # =================================================================
    print("\n=== Test 3: Write tool ===")
    tools3, _ = await run_query(
        "Use Write tool",
        "You are a file assistant. Write a small test file when asked.",
        ["Write", "Read"],
        3,
        'Use the Write tool to create /tmp/auto_scientist_e2e_test.txt with content "e2e test ok".',
    )
    check("Write was called", "Write" in tools3, f"tools={tools3}")

    # =================================================================
    # Test 4: Host isolation
    # =================================================================
    print("\n=== Test 4: Host isolation ===")

    # 4a: Ask the subprocess to dump everything it knows about its environment
    _, isolation_result = await run_query(
        "Isolation audit",
        (
            "You are a diagnostic agent. Answer the following questions precisely. "
            "Do NOT speculate or make things up. If you don't see something, say NO."
        ),
        [],
        1,
        (
            "Answer each question on its own line with YES or NO followed by brief details:\n"
            "1. Do you see any CLAUDE.md file content in your system prompt or context?\n"
            "2. Do you see any user memory files (MEMORY.md) in your context?\n"
            "3. Do you see any hooks configuration?\n"
            "4. Do you see any installed plugins or skills?\n"
            "5. Do you have access to the Agent tool?\n"
            "6. Do you have access to the Skill tool?\n"
        ),
    )

    lines = [line.strip() for line in isolation_result.strip().split("\n") if line.strip()]

    def line_says_no(keyword: str) -> bool:
        """Check if the line mentioning keyword starts with NO."""
        for line in lines:
            if keyword.lower() in line.lower():
                return line.lower().lstrip("0123456789. ").startswith("no")
        return False

    check(
        "No CLAUDE.md leakage",
        line_says_no("claude.md"),
        f"response={isolation_result[:300]!r}",
    )
    check(
        "No MEMORY.md leakage",
        line_says_no("memory"),
        f"response={isolation_result[:300]!r}",
    )
    check(
        "No hooks leakage",
        line_says_no("hooks"),
        f"response={isolation_result[:300]!r}",
    )
    check(
        "No plugins/skills leakage",
        line_says_no("plugin"),
        f"response={isolation_result[:300]!r}",
    )
    check(
        "Agent tool blocked",
        line_says_no("agent tool"),
        f"response={isolation_result[:300]!r}",
    )
    check(
        "Skill tool blocked",
        line_says_no("skill tool"),
        f"response={isolation_result[:300]!r}",
    )

    # Summary
    total = passed + failed
    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{total} passed, {failed}/{total} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
