"""Verify Codex MCP isolation fix.

Run BEFORE the fix:  "before" checks pass, "after" checks fail
Run AFTER the fix:   all checks pass

Checks both code structure and live SDK behavior.

Usage:
    uv run python scripts/verify_codex_mcp_isolation.py
    uv run python scripts/verify_codex_mcp_isolation.py --live   # also run SDK integration test
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def check(label: str, condition: bool, detail: str = "") -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}" + (f" -- {detail}" if detail else ""))
    return condition


def run_structural_checks() -> tuple[int, int]:
    """Check code structure changes without running the SDK."""
    passed = 0
    failed = 0

    def tally(ok: bool) -> None:
        nonlocal passed, failed
        if ok:
            passed += 1
        else:
            failed += 1

    # --- REGRESSION GUARD: the old broken behavior must be gone ---
    print("\n=== REGRESSION GUARD (old .codex/config.toml bug must NOT recur) ===")

    from auto_scientist.sdk_backend import CodexBackend

    tmp = Path(tempfile.mkdtemp(prefix="codex_mcp_test_"))
    try:
        mcp_servers = {
            "predictions": {
                "type": "stdio",
                "command": "python3",
                "args": ["/fake/server.py", "/fake/data.json"],
            }
        }
        CodexBackend._write_codex_mcp_config(mcp_servers, tmp)

        old_path = tmp / ".codex" / "config.toml"
        new_path = tmp / "config.toml"

        tally(
            check(
                "Does NOT write to .codex/ subdir (old broken path)",
                not old_path.exists(),
                f".codex/config.toml exists={old_path.exists()}",
            )
        )

        tally(
            check(
                "Writes directly to codex_home/config.toml",
                new_path.exists(),
                f"config.toml exists={new_path.exists()}",
            )
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # --- FIX CHECKS: validate the isolation approach ---
    print("\n=== FIX CHECKS (CODEX_HOME isolation) ===")

    import inspect

    sig = inspect.signature(CodexBackend._write_codex_mcp_config)
    params = list(sig.parameters.keys())

    # 3. Parameter renamed from cwd to codex_home
    has_codex_home_param = "codex_home" in params
    tally(
        check(
            "_write_codex_mcp_config param is 'codex_home' (not 'cwd')",
            has_codex_home_param,
            f"params={params}",
        )
    )

    # 4. Writes config.toml directly (not in .codex/ subdir)
    tmp2 = Path(tempfile.mkdtemp(prefix="codex_mcp_fix_"))
    try:
        CodexBackend._write_codex_mcp_config(mcp_servers, tmp2)
        writes_direct = (tmp2 / "config.toml").exists()
        writes_subdir = (tmp2 / ".codex" / "config.toml").exists()

        tally(
            check(
                "Config written directly to codex_home/config.toml",
                writes_direct,
                f"config.toml exists={writes_direct}",
            )
        )

        tally(
            check(
                "No .codex/ subdir created",
                not writes_subdir,
                f".codex/config.toml exists={writes_subdir}",
            )
        )

        # 5. Config content is valid
        if writes_direct:
            content = (tmp2 / "config.toml").read_text()
            tally(
                check(
                    "Config contains [mcp_servers.predictions]",
                    "[mcp_servers.predictions]" in content,
                )
            )
            tally(
                check(
                    'Config contains command = "python3"',
                    'command = "python3"' in content,
                )
            )
        else:
            tally(check("Config content valid", False, "file not found"))
            tally(check("Config content valid", False, "file not found"))
    finally:
        shutil.rmtree(tmp2, ignore_errors=True)

    # 6. Check that query() uses CODEX_HOME in env
    source = inspect.getsource(CodexBackend.query)
    tally(
        check(
            "query() sets CODEX_HOME in env",
            "CODEX_HOME" in source,
            "checked source for CODEX_HOME string",
        )
    )

    # 7. Check that query() copies auth.json
    tally(
        check(
            "query() copies auth.json for isolation",
            "auth.json" in source,
            "checked source for auth.json string",
        )
    )

    # 8. Check that query() uses tempfile for isolation
    tally(
        check(
            "query() creates temp dir (tempfile.mkdtemp)",
            "mkdtemp" in source,
            "checked source for mkdtemp string",
        )
    )

    # 9. Check that _resolve_sandbox returns danger-full-access for MCP
    tally(
        check(
            "_resolve_sandbox upgrades to danger-full-access with MCP",
            CodexBackend._resolve_sandbox(["WebSearch"], has_mcp=True) == "danger-full-access",
        )
    )

    tally(
        check(
            "_resolve_sandbox stays read-only without MCP",
            CodexBackend._resolve_sandbox(["WebSearch"], has_mcp=False) == "read-only",
        )
    )

    # 11. Check cleanup uses shutil.rmtree
    tally(
        check(
            "query() cleans up with shutil.rmtree",
            "rmtree" in source,
            "checked source for rmtree string",
        )
    )

    return passed, failed


async def run_live_sdk_test() -> tuple[int, int]:
    """Actually run the Codex SDK and verify tool calls."""
    from codex_app_server_sdk import CodexClient, ThreadConfig, TurnOverrides

    passed = 0
    failed = 0

    def tally(ok: bool) -> None:
        nonlocal passed, failed
        if ok:
            passed += 1
        else:
            failed += 1

    print("\n=== LIVE SDK TEST: CODEX_HOME isolation ===")

    codex_home = Path(tempfile.mkdtemp(prefix="codex_home_live_"))
    predictions_path = codex_home / "predictions.json"

    try:
        # Copy auth
        real_auth = Path.home() / ".codex" / "auth.json"
        if real_auth.exists():
            shutil.copy2(real_auth, codex_home / "auth.json")
        else:
            print("  WARNING: No ~/.codex/auth.json - skipping live test")
            return 0, 0

        # Write test predictions
        preds = [
            {
                "pred_id": "1.1",
                "iteration_prescribed": 1,
                "prediction": "X correlates with Y",
                "diagnostic": "basic correlation test",
                "if_confirmed": "proceed to causal analysis",
                "if_refuted": "pivot to alternative hypothesis",
                "outcome": "confirmed",
                "evidence": "Pearson r=0.95, p<0.001",
                "summary": "[1.1] confirmed: X correlates with Y",
            },
            {
                "pred_id": "2.1",
                "iteration_prescribed": 2,
                "follows_from": "1.1",
                "prediction": "Z mediates the X-Y relationship",
                "diagnostic": "mediation analysis",
                "if_confirmed": "build model including Z",
                "if_refuted": "X-Y is a direct effect",
                "outcome": "pending",
                "evidence": "",
                "summary": "[2.1] pending: Z mediates X-Y",
            },
        ]
        predictions_path.write_text(json.dumps(preds))

        # Write MCP config
        server_script = (
            Path(__file__).parent.parent / "src/auto_scientist/agents/_prediction_mcp_server.py"
        )
        config = f"""[mcp_servers.predictions]
command = "python3"
args = ["{server_script}", "{predictions_path}"]
"""
        (codex_home / "config.toml").write_text(config)

        # 1. Verify no host MCP servers leak through
        env = {**os.environ, "CODEX_HOME": str(codex_home), "OPENAI_API_KEY": ""}

        # 2. Spawn Codex and ask it to use the tool
        prompt = (
            "You have a tool called mcp__predictions__read_predictions. "
            "Call it with overview=true. Report the prediction IDs and their statuses. "
            "If you cannot find this tool, say exactly: TOOL_NOT_FOUND"
        )

        client = CodexClient.connect_stdio(
            env=env,
            inactivity_timeout=60.0,
        )

        await client.start()
        thread_config = ThreadConfig(
            model="gpt-5.4-mini",
            base_instructions="You are a test agent. Call the prediction tool if available.",
            sandbox="danger-full-access",
            approval_policy="never",
        )
        turn_overrides = TurnOverrides()
        turn_overrides.effort = "low"

        parts = []
        step_types = []
        async for step in client.chat(
            prompt, thread_config=thread_config, turn_overrides=turn_overrides
        ):
            if step.text:
                parts.append(step.text)
            step_types.append(step.step_type)
            print(f"  [{step.step_type}] {(step.text or '')[:120]}")

        await client.close()

        result = "\n".join(parts)

        # Check results
        tool_not_found = "TOOL_NOT_FOUND" in result
        has_prediction_data = "1.1" in result or "confirmed" in result.lower()
        used_mcp_tool = "tool" in step_types or "mcpToolCall" in step_types

        tally(
            check(
                "Model did NOT say TOOL_NOT_FOUND",
                not tool_not_found,
                f"response contains TOOL_NOT_FOUND={tool_not_found}",
            )
        )

        tally(
            check(
                "Response contains prediction data (1.1 or confirmed)",
                has_prediction_data,
                f"result excerpt: {result[:200]}",
            )
        )

        tally(
            check(
                "Step types include tool call",
                used_mcp_tool,
                f"step_types={step_types}",
            )
        )

        # 3. Verify NO host servers leaked
        # If context7 or playwright appear in the response, isolation failed
        host_leak = "context7" in result.lower() or "playwright" in result.lower()
        tally(
            check(
                "No host MCP servers (context7/playwright) in response",
                not host_leak,
            )
        )

    except Exception as e:
        print(f"  ERROR: {e}")
        tally(check("Live SDK test completed without error", False, str(e)))
    finally:
        shutil.rmtree(codex_home, ignore_errors=True)

    return passed, failed


def main() -> int:
    live = "--live" in sys.argv

    p1, f1 = run_structural_checks()

    p2, f2 = 0, 0
    if live:
        p2, f2 = asyncio.run(run_live_sdk_test())
    else:
        print("\n  (Skipping live SDK test. Run with --live to include.)")

    total_passed = p1 + p2
    total_failed = f1 + f2
    total = total_passed + total_failed

    print(f"\n{'=' * 50}")
    print(f"Results: {total_passed}/{total} passed, {total_failed}/{total} failed")
    if total_failed:
        print("Some checks failed (expected if running before the fix)")
        return 1
    print("All checks passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
