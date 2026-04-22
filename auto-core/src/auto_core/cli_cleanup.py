"""Shared CLI cleanup for shutting down direct child processes on exit."""

from __future__ import annotations

import atexit
import logging
import os
import signal
import subprocess
import threading
from contextlib import suppress
from typing import Any

logger = logging.getLogger(__name__)

_cleanup_done = threading.Event()
_cleanup_handlers_installed = threading.Event()


def kill_child_processes() -> None:
    """Terminate direct child processes via ``pgrep -P``."""
    if _cleanup_done.is_set():
        return
    _cleanup_done.set()

    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)

    pid = os.getpid()
    try:
        result = subprocess.run(
            ["pgrep", "-P", str(pid)],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        for line in result.stdout.strip().splitlines():
            with suppress(ProcessLookupError, PermissionError, ValueError, OSError):
                os.kill(int(line.strip()), signal.SIGTERM)
    except Exception:
        logger.debug("Best-effort child cleanup failed", exc_info=True)


def _fatal_signal_handler(signum: int, _frame: Any) -> None:
    """Handle SIGHUP / SIGTERM by killing children and exiting."""
    kill_child_processes()
    os._exit(128 + signum)


def install_child_cleanup_handlers() -> None:
    """Register signal and atexit handlers for child-process cleanup."""
    if _cleanup_handlers_installed.is_set():
        return
    signal.signal(signal.SIGHUP, _fatal_signal_handler)
    signal.signal(signal.SIGTERM, _fatal_signal_handler)
    atexit.register(kill_child_processes)
    _cleanup_handlers_installed.set()
