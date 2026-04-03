"""Verify Codex session resumption works end-to-end.

Spins up a real Codex app-server, makes two calls on the same
CodexBackend instance (fresh then resume), and verifies the second
call continues the same thread.

Usage:
    uv run python scripts/verify_codex_session_resume.py
"""

from __future__ import annotations

import asyncio
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def check(label: str, condition: bool, detail: str = "") -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}" + (f" - {detail}" if detail else ""))
    return condition


async def run_live_resume_test() -> tuple[int, int]:
    """Test session resumption with a real Codex app-server."""
    from auto_scientist.sdk_backend import CodexBackend, SDKOptions

    passed = 0
    failed = 0

    def tally(ok: bool) -> None:
        nonlocal passed, failed
        if ok:
            passed += 1
        else:
            failed += 1

    # Pre-flight: check auth exists
    real_auth = Path.home() / ".codex" / "auth.json"
    if not real_auth.exists():
        print("  WARNING: No ~/.codex/auth.json - cannot run live test")
        return 0, 0

    print("\n=== LIVE SDK TEST: Session Resumption ===")

    backend = CodexBackend()
    work_dir = Path(tempfile.mkdtemp(prefix="codex_resume_test_"))

    try:
        # --- Call 1: Fresh session ---
        print("\n--- Call 1: Fresh session ---")

        fresh_opts = SDKOptions(
            system_prompt=(
                "You are a test agent. When asked to remember a word, "
                "just confirm you remembered it. Keep responses very short."
            ),
            allowed_tools=[],
            max_turns=3,
            model="gpt-5.4-mini",
            cwd=work_dir,
            extra_args={},
        )

        session_id: str | None = None
        first_response = ""
        async for msg in backend.query(
            'Remember this secret word: "capybara". Just confirm you remembered it.',
            fresh_opts,
        ):
            if msg.type == "result":
                session_id = msg.session_id
                first_response = msg.result or ""
            elif msg.type == "assistant":
                for block in msg.content_blocks:
                    if hasattr(block, "text"):
                        print(f"  [call1] {block.text[:120]}")

        tally(
            check(
                "Call 1 returned a session_id",
                session_id is not None,
                f"session_id={session_id}",
            )
        )
        tally(
            check(
                "Call 1 got a response",
                len(first_response) > 0,
                f"response_len={len(first_response)}",
            )
        )

        # Verify client is still alive (stateful backend)
        tally(
            check(
                "Client persisted after call 1",
                backend._client is not None,
            )
        )

        if session_id is None:
            print("  Cannot continue without session_id")
            return passed, failed

        # --- Call 2: Resume same session ---
        print("\n--- Call 2: Resume session ---")

        resume_opts = SDKOptions(
            system_prompt=(
                "You are a test agent. When asked to remember a word, "
                "just confirm you remembered it. Keep responses very short."
            ),
            allowed_tools=[],
            max_turns=3,
            model="gpt-5.4-mini",
            cwd=work_dir,
            resume=session_id,
            extra_args={},
        )

        second_response = ""
        resume_session_id: str | None = None
        async for msg in backend.query(
            "What was the secret word I asked you to remember?",
            resume_opts,
        ):
            if msg.type == "result":
                resume_session_id = msg.session_id
                second_response = msg.result or ""
            elif msg.type == "assistant":
                for block in msg.content_blocks:
                    if hasattr(block, "text"):
                        print(f"  [call2] {block.text[:120]}")

        tally(
            check(
                "Call 2 completed without error",
                len(second_response) > 0,
                f"response_len={len(second_response)}",
            )
        )
        tally(
            check(
                "Call 2 reused the same thread",
                resume_session_id == session_id,
                f"original={session_id}, resumed={resume_session_id}",
            )
        )
        tally(
            check(
                "Call 2 remembers the secret word",
                "capybara" in second_response.lower(),
                f"response: {second_response[:200]}",
            )
        )

    except Exception as e:
        print(f"\n  ERROR: {e}")
        import traceback

        traceback.print_exc()
        tally(check("Live resume test completed without error", False, str(e)))
    finally:
        await backend.close()
        shutil.rmtree(work_dir, ignore_errors=True)

        tally(
            check(
                "Cleanup: client closed",
                backend._client is None,
            )
        )
        tally(
            check(
                "Cleanup: codex_home removed",
                backend._codex_home is None,
            )
        )

    return passed, failed


def main() -> int:
    p, f = asyncio.run(run_live_resume_test())

    total = p + f
    print(f"\n{'=' * 50}")
    print(f"Results: {p}/{total} passed, {f}/{total} failed")
    if f:
        return 1
    if total == 0:
        print("No tests ran (missing auth?)")
        return 1
    print("All checks passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
