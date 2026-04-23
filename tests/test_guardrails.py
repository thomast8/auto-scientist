"""Guardrails against live Claude CLI runaways (see 2026-04-23 zombie incident)."""

from __future__ import annotations

import os
import subprocess
import sys
import time

import pytest
from auto_core import cli_cleanup, cost_ceiling, sdk_backend
from auto_core.testing import install_live_claude_block


async def test_block_live_claude_sdk_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """The blocker must raise as soon as production code awaits claude_query."""
    monkeypatch.delenv("LIVE_CLAUDE", raising=False)
    install_live_claude_block(monkeypatch)

    # Iterate the async generator to actually hit the raise — creating the
    # generator alone does not trigger the body.
    with pytest.raises(RuntimeError, match="Live Claude CLI spawn blocked"):
        async for _ in sdk_backend.claude_query(prompt="hi"):  # pragma: no cover
            pass


def test_block_live_claude_sdk_noop_when_opted_in(monkeypatch: pytest.MonkeyPatch) -> None:
    """Setting LIVE_CLAUDE=1 skips the patch so smoke tests can reach the real CLI."""
    monkeypatch.setenv("LIVE_CLAUDE", "1")
    sentinel = sdk_backend.claude_query
    install_live_claude_block(monkeypatch)
    assert sdk_backend.claude_query is sentinel


def test_record_cost_accumulates_then_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTO_SCIENTIST_MAX_RUN_USD", "1.00")
    cost_ceiling.reset_budget()

    cost_ceiling.record_cost(0.40)
    cost_ceiling.record_cost(0.50)
    assert cost_ceiling.total_usd() == pytest.approx(0.90)

    with pytest.raises(cost_ceiling.RunBudgetExceededError, match=r"\$1\.1"):
        cost_ceiling.record_cost(0.20)


def test_record_cost_ignores_none_and_nonpositive() -> None:
    cost_ceiling.reset_budget(limit_usd=1.0)
    cost_ceiling.record_cost(None)
    cost_ceiling.record_cost(0.0)
    cost_ceiling.record_cost(-5.0)
    assert cost_ceiling.total_usd() == 0.0


def test_record_cost_invalid_env_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AUTO_SCIENTIST_MAX_RUN_USD", "not-a-float")
    cost_ceiling.reset_budget()
    with pytest.raises(RuntimeError, match="not a valid float"):
        cost_ceiling.record_cost(0.01)


def test_resolve_message_timeout_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CLAUDE_QUERY_MESSAGE_TIMEOUT_SECONDS", raising=False)
    assert sdk_backend._resolve_message_timeout() == pytest.approx(900.0)


def test_resolve_message_timeout_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAUDE_QUERY_MESSAGE_TIMEOUT_SECONDS", "3.5")
    assert sdk_backend._resolve_message_timeout() == pytest.approx(3.5)


def test_resolve_message_timeout_invalid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAUDE_QUERY_MESSAGE_TIMEOUT_SECONDS", "nope")
    with pytest.raises(RuntimeError, match="not a valid float"):
        sdk_backend._resolve_message_timeout()


def test_descendant_pids_recurses() -> None:
    """Spawn sh → sleep; pgrep-based walk must find the sleep grandchild."""
    parent = subprocess.Popen(
        ["/bin/sh", "-c", "sleep 5 & wait"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        # Give the shell a moment to fork the sleep.
        time.sleep(0.3)
        descendants = cli_cleanup._descendant_pids(parent.pid)
        assert parent.pid not in descendants, "root must be excluded"
        assert len(descendants) >= 1, f"expected at least one descendant, got {descendants}"
    finally:
        parent.terminate()
        parent.wait(timeout=5)


def test_parent_death_watchdog_reaps_on_orphan(monkeypatch: pytest.MonkeyPatch) -> None:
    """When os.getppid() returns 1, the watchdog must reap and os._exit."""
    reap_called = False
    exit_code: list[int] = []

    def fake_kill() -> None:
        nonlocal reap_called
        reap_called = True

    def fake_exit(code: int) -> None:
        exit_code.append(code)
        raise SystemExit(code)  # break the otherwise-infinite loop

    monkeypatch.setattr(cli_cleanup, "kill_child_processes", fake_kill)
    monkeypatch.setattr(cli_cleanup.os, "_exit", fake_exit)
    monkeypatch.setattr(cli_cleanup.os, "getppid", lambda: 1)
    monkeypatch.setattr(cli_cleanup.time, "sleep", lambda _s: None)

    with pytest.raises(SystemExit) as exc:
        cli_cleanup._parent_death_watchdog()

    assert reap_called
    assert exc.value.code == 129
    assert exit_code == [129]


def test_parent_death_watchdog_thread_starts(monkeypatch: pytest.MonkeyPatch) -> None:
    """install_child_cleanup_handlers launches the watchdog thread."""
    import threading as _threading

    cli_cleanup._cleanup_handlers_installed.clear()
    monkeypatch.setattr(cli_cleanup.signal, "signal", lambda *_a, **_k: None)
    monkeypatch.setattr(cli_cleanup.atexit, "register", lambda *_a, **_k: None)
    monkeypatch.setattr(cli_cleanup.os, "getppid", lambda: 9999)  # not orphaned

    try:
        cli_cleanup.install_child_cleanup_handlers()
        names = [t.name for t in _threading.enumerate()]
        assert "parent-death-watchdog" in names
    finally:
        cli_cleanup._cleanup_handlers_installed.clear()


def test_integration_parent_dies_child_exits(tmp_path) -> None:
    """End-to-end: kill the parent Python, confirm the child self-exits via the watchdog."""
    src_root = os.path.dirname(os.path.dirname(os.path.abspath(cli_cleanup.__file__)))
    script = tmp_path / "orphan_child.py"
    script.write_text(
        f"""
import sys, time
sys.path.insert(0, {src_root!r})
from auto_core.cli_cleanup import install_child_cleanup_handlers

install_child_cleanup_handlers()
print("ready", flush=True)
for _ in range(120):
    time.sleep(1)
"""
    )

    launcher = (
        "import subprocess, sys; "
        f"p = subprocess.Popen([sys.executable, {str(script)!r}], stdout=sys.stdout); "
        "p.wait()"
    )
    parent = subprocess.Popen(
        [sys.executable, "-c", launcher],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        # Kill the intermediate Python; the script subprocess reparents to init.
        time.sleep(2.0)
        parent.kill()
        parent.wait(timeout=5)

        # Within ~3 watchdog polls the orphan should be gone.
        deadline = time.monotonic() + (cli_cleanup._WATCHDOG_POLL_SECONDS * 3 + 5)
        while time.monotonic() < deadline:
            out = subprocess.run(
                ["pgrep", "-f", str(script)],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if not out.stdout.strip():
                return
            time.sleep(0.5)
        pytest.fail("orphan child survived past watchdog deadline")
    finally:
        subprocess.run(["pkill", "-f", str(script)], check=False, timeout=5)
