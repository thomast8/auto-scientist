"""Shared CLI cleanup for reaping descendant processes on exit.

Beyond the original SIGHUP/SIGTERM/atexit handlers, we also run a daemon
thread that polls ``os.getppid()`` — when the parent disappears (reparent
to init / launchd) we reap the whole descendant tree ourselves. This
guards against the 2026-04-23 incident where a detached pytest survived
9 days after its terminal died, continuously spawning ``claude`` CLIs.
"""

from __future__ import annotations

import atexit
import logging
import os
import signal
import subprocess
import threading
import time
from contextlib import suppress
from typing import Any

logger = logging.getLogger(__name__)

_cleanup_done = threading.Event()
_cleanup_handlers_installed = threading.Event()
_WATCHDOG_POLL_SECONDS = 5.0


def _descendant_pids(root_pid: int) -> list[int]:
    """Return descendant PIDs via recursive ``pgrep -P``.

    Leaf-first order so parents can be killed only after their children are
    already down. Best-effort; silently returns what it found if pgrep fails.
    """
    collected: list[int] = []
    frontier: list[int] = [root_pid]
    seen: set[int] = {root_pid}

    while frontier:
        parent = frontier.pop()
        try:
            result = subprocess.run(
                ["pgrep", "-P", str(parent)],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
        except Exception:
            logger.debug(f"pgrep -P {parent} failed", exc_info=True)
            continue
        for line in result.stdout.strip().splitlines():
            try:
                child_pid = int(line.strip())
            except ValueError:
                continue
            if child_pid in seen or child_pid == root_pid:
                continue
            seen.add(child_pid)
            frontier.append(child_pid)
            collected.append(child_pid)

    # Leaf-first: later-discovered PIDs are deeper in the tree.
    collected.reverse()
    return collected


def kill_child_processes() -> None:
    """Terminate every descendant of this process with SIGTERM then SIGKILL."""
    if _cleanup_done.is_set():
        return
    _cleanup_done.set()

    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)

    pids = _descendant_pids(os.getpid())
    if not pids:
        return

    for pid in pids:
        with suppress(ProcessLookupError, PermissionError, OSError):
            os.kill(pid, signal.SIGTERM)

    # Give graceful handlers a brief moment, then SIGKILL the holdouts.
    time.sleep(0.5)
    for pid in pids:
        with suppress(ProcessLookupError, PermissionError, OSError):
            os.kill(pid, signal.SIGKILL)


def _fatal_signal_handler(signum: int, _frame: Any) -> None:
    """Handle SIGHUP / SIGTERM by killing descendants and exiting."""
    kill_child_processes()
    os._exit(128 + signum)


def _parent_death_watchdog() -> None:
    """Daemon thread: reap descendants if the parent disappears.

    On macOS there is no ``PR_SET_PDEATHSIG``; polling ``os.getppid()`` is
    portable and costs nothing. When the parent dies we become reparented
    (typically to launchd / init, PID 1), which is the cue to clean up and
    exit so orphaned Claude CLIs don't keep burning tokens for days.
    """
    while True:
        try:
            if os.getppid() == 1:
                logger.warning(
                    "Parent process is gone (reparented to init); reaping descendants and exiting."
                )
                kill_child_processes()
                os._exit(129)  # 128 + SIGHUP
        except Exception:
            logger.debug("parent-death watchdog poll failed", exc_info=True)
        time.sleep(_WATCHDOG_POLL_SECONDS)


def install_child_cleanup_handlers() -> None:
    """Register signal, atexit, and parent-death handlers for child cleanup."""
    if _cleanup_handlers_installed.is_set():
        return
    signal.signal(signal.SIGHUP, _fatal_signal_handler)
    signal.signal(signal.SIGTERM, _fatal_signal_handler)
    atexit.register(kill_child_processes)

    # Parent-death watchdog. Daemon thread dies with the process; in the
    # normal case the process exits before the next 5s poll anyway.
    if os.getppid() != 1:
        threading.Thread(
            target=_parent_death_watchdog,
            name="parent-death-watchdog",
            daemon=True,
        ).start()

    _cleanup_handlers_installed.set()
