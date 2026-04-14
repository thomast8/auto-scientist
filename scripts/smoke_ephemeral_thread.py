"""Smoke tests for the Codex ephemeral-thread fix.

Three modes:

1. ``--mode ephemeral`` (default): sends one turn through CodexBackend with
   ephemeral=True. Expected: response comes back, thread_id returned, thread
   is NOT visible in the Codex desktop app sidebar.

2. ``--mode control``: sends one turn using the same isolated setup as
   ``CodexBackend._ensure_client`` (temp CODEX_HOME, auth.json copy,
   OPENAI_API_KEY stripped) but with a bare ThreadConfig (ephemeral omitted).
   Expected: response comes back, thread_id returned, thread IS visible in
   the Codex desktop app sidebar. The only variable vs. mode ephemeral is
   the ephemeral flag, so any behavior difference is attributable to the fix.

3. ``--mode resume``: sends a first turn through CodexBackend (ephemeral),
   captures thread_id, then sends a follow-up using options.resume=thread_id
   whose answer depends on information from the first turn. Confirms both
   that the second turn returns the SAME thread_id AND that the model
   actually carried context forward (a fresh thread cannot answer correctly).

Usage:
    uv run python scripts/smoke_ephemeral_thread.py --mode ephemeral
    uv run python scripts/smoke_ephemeral_thread.py --mode control
    uv run python scripts/smoke_ephemeral_thread.py --mode resume
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

from codex_app_server_sdk import CodexClient, ThreadConfig, TurnOverrides

from auto_scientist.sdk_backend import SDKOptions, close_all_backends, get_backend

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

MODEL = "gpt-5.4-mini"
SYSTEM = "You are a terse assistant. Reply in one short word when asked."
SECRET_WORD = "thundercrab"  # rare, model will not guess this on a fresh thread
PROMPT_REMEMBER = (
    f"Please remember this secret word for later: {SECRET_WORD}. "
    "Acknowledge with the single word 'ok' and nothing else."
)
PROMPT_RECALL = (
    "What was the secret word I asked you to remember earlier in this "
    "conversation? Reply with that word only, in lowercase, and nothing else."
)
PROMPT_PING = "Say the single word 'pong' and nothing else."


async def _run_backend_turn(
    backend, prompt: str, resume: str | None = None
) -> tuple[str, str | None, int]:
    options = SDKOptions(
        system_prompt=SYSTEM,
        allowed_tools=(),
        max_turns=1,
        model=MODEL,
        resume=resume,
    )
    final_text = ""
    thread_id: str | None = None
    num_turns = 0
    async for msg in backend.query(prompt, options):
        if msg.type == "result":
            final_text = msg.result or ""
            thread_id = msg.session_id
            num_turns = int(msg.usage.get("num_turns", 0))
    return final_text, thread_id, num_turns


async def mode_ephemeral() -> int:
    backend = get_backend("openai")
    try:
        print(f"[smoke:ephemeral] prompt: {PROMPT_PING!r}")
        final_text, thread_id, num_turns = await _run_backend_turn(backend, PROMPT_PING)
    finally:
        await close_all_backends()

    print()
    print("=" * 60)
    print(f"[smoke:ephemeral] thread_id:   {thread_id}")
    print(f"[smoke:ephemeral] num_turns:   {num_turns}")
    print(f"[smoke:ephemeral] final_text:  {final_text!r}")
    print("=" * 60)

    if not thread_id or not final_text:
        print("[smoke:ephemeral] FAIL: missing thread_id or text", file=sys.stderr)
        return 1

    print()
    print("[smoke:ephemeral] PASS: Codex call completed end-to-end.")
    print("[smoke:ephemeral] Verify: thread should NOT appear in Codex app sidebar.")
    return 0


async def mode_control() -> int:
    """Isolated Codex SDK call with a bare ThreadConfig (no ephemeral flag).

    Mirrors CodexBackend._ensure_client's isolation setup (temp CODEX_HOME,
    auth.json copy, env stripping) so the ONLY variable vs. mode ephemeral is
    the ephemeral flag on ThreadConfig. Any difference in sidebar behavior is
    therefore attributable to the fix and not workstation state.
    """
    codex_home = Path(tempfile.mkdtemp(prefix="codex_home_smoke_"))
    try:
        real_auth = Path.home() / ".codex" / "auth.json"
        if real_auth.exists():
            shutil.copy2(real_auth, codex_home / "auth.json")

        env: dict[str, str] = {
            **os.environ,
            "CODEX_HOME": str(codex_home),
        }
        if os.environ.get("OPENAI_API_KEY"):
            env["OPENAI_API_KEY"] = ""

        client = CodexClient.connect_stdio(
            cwd=str(Path.cwd()),
            env=env,
            inactivity_timeout=600.0,
        )
        await client.start()

        try:
            # Bare ThreadConfig — ephemeral deliberately omitted.
            thread_config = ThreadConfig(
                model=MODEL,
                base_instructions=SYSTEM,
                sandbox="read-only",
                approval_policy="never",
            )
            turn_overrides = TurnOverrides()
            turn_overrides.effort = "low"

            print(f"[smoke:control] prompt: {PROMPT_PING!r}")
            parts: list[str] = []
            thread_id: str | None = None
            step_count = 0
            async for step in client.chat(
                PROMPT_PING,
                thread_config=thread_config,
                turn_overrides=turn_overrides,
            ):
                step_count += 1
                if step.text:
                    parts.append(step.text)
                if step.thread_id:
                    thread_id = step.thread_id

            final_text = "".join(parts)
        finally:
            await client.close()
    finally:
        shutil.rmtree(codex_home, ignore_errors=True)

    print()
    print("=" * 60)
    print(f"[smoke:control] thread_id:   {thread_id}")
    print(f"[smoke:control] num_steps:   {step_count}")
    print(f"[smoke:control] final_text:  {final_text!r}")
    print("=" * 60)

    if not thread_id or not final_text:
        print("[smoke:control] FAIL: missing thread_id or text", file=sys.stderr)
        return 1

    print()
    print("[smoke:control] PASS: raw SDK call completed end-to-end.")
    print("[smoke:control] Verify: thread SHOULD appear in Codex app sidebar.")
    return 0


async def mode_resume() -> int:
    """First turn plants a secret; second turn must recall it via resume.

    Proves both (a) the resumed turn continues on the same thread_id and
    (b) the model actually sees turn 1's context. A fresh thread cannot
    produce the secret word — that is the real invariant a Coder error
    correction loop depends on.
    """
    backend = get_backend("openai")
    try:
        print(f"[smoke:resume] turn 1 prompt: {PROMPT_REMEMBER!r}")
        text1, thread1, turns1 = await _run_backend_turn(backend, PROMPT_REMEMBER)
        print(f"[smoke:resume] turn 1 result: text={text1!r} thread={thread1} turns={turns1}")
        if not thread1:
            print("[smoke:resume] FAIL: no thread_id after turn 1", file=sys.stderr)
            return 1

        print(f"[smoke:resume] turn 2 prompt (resume={thread1}): {PROMPT_RECALL!r}")
        text2, thread2, turns2 = await _run_backend_turn(backend, PROMPT_RECALL, resume=thread1)
    finally:
        await close_all_backends()

    print()
    print("=" * 60)
    print(f"[smoke:resume] turn 1 thread: {thread1}")
    print(f"[smoke:resume] turn 2 thread: {thread2}")
    print(f"[smoke:resume] turn 1 text:   {text1!r}")
    print(f"[smoke:resume] turn 2 text:   {text2!r}")
    print(f"[smoke:resume] secret word:   {SECRET_WORD!r}")
    print("=" * 60)

    if not thread2 or not text2:
        print("[smoke:resume] FAIL: missing thread_id or text on turn 2", file=sys.stderr)
        return 1
    if thread1 != thread2:
        print(
            f"[smoke:resume] FAIL: thread_id changed across turns "
            f"({thread1} -> {thread2}); resume did not continue same thread",
            file=sys.stderr,
        )
        return 1
    if SECRET_WORD not in text2.lower():
        print(
            f"[smoke:resume] FAIL: turn 2 did not recall the secret word "
            f"{SECRET_WORD!r}; got {text2!r}. The model did not see turn 1's "
            f"context — resume is broken even though thread_id matched.",
            file=sys.stderr,
        )
        return 1

    print()
    print("[smoke:resume] PASS: correction continued on same ephemeral thread")
    print("[smoke:resume]       and carried conversation context forward.")
    print("[smoke:resume] Verify: no new thread should appear in Codex app sidebar.")
    return 0


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["ephemeral", "control", "resume"],
        default="ephemeral",
    )
    args = parser.parse_args()

    if args.mode == "ephemeral":
        return await mode_ephemeral()
    if args.mode == "control":
        return await mode_control()
    if args.mode == "resume":
        return await mode_resume()
    return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
