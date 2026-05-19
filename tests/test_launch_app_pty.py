"""PTY smoke tests for the real terminal launcher."""

from __future__ import annotations

import contextlib
import os
import re
import select
import signal
import struct
import subprocess
import sys
import termios
import time
from pathlib import Path

import pytest
import yaml

ANSI_RE = re.compile(
    r"(?:\x1b\][^\x07]*(?:\x07|\x1b\\))|(?:\x1b[@-Z\\-_])|(?:\x1b\[[0-?]*[ -/]*[@-~])"
)


def _strip_terminal_controls(text: str) -> str:
    """Return screen text without common ANSI/OSC control sequences."""
    return ANSI_RE.sub("", text).replace("\r", "")


def _read_pty(master_fd: int, *, timeout: float) -> str:
    """Read whatever the PTY emits until timeout expires."""
    deadline = time.monotonic() + timeout
    chunks: list[bytes] = []
    while time.monotonic() < deadline:
        ready, _, _ = select.select([master_fd], [], [], 0.05)
        if not ready:
            continue
        try:
            chunk = os.read(master_fd, 65536)
        except OSError:
            break
        if not chunk:
            break
        chunks.append(chunk)
    return b"".join(chunks).decode(errors="ignore")


def _read_until(master_fd: int, needle: str, *, timeout: float) -> str:
    """Read PTY output until stripped terminal text contains needle."""
    deadline = time.monotonic() + timeout
    chunks: list[bytes] = []
    while time.monotonic() < deadline:
        ready, _, _ = select.select([master_fd], [], [], 0.1)
        if ready:
            try:
                chunk = os.read(master_fd, 65536)
            except OSError:
                break
            if not chunk:
                break
            chunks.append(chunk)
            screen = _strip_terminal_controls(b"".join(chunks).decode(errors="ignore"))
            if needle in screen:
                return screen
    screen = _strip_terminal_controls(b"".join(chunks).decode(errors="ignore"))
    raise AssertionError(f"Timed out waiting for {needle!r}; last screen:\n{screen[-2000:]}")


def _wait_for_file(path: Path, *, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.05)
    raise AssertionError(f"Timed out waiting for {path}")


@pytest.mark.skipif(sys.platform == "win32", reason="PTY smoke requires POSIX pty support")
def test_bare_tui_saves_prefilled_config_in_real_pty(tmp_path: Path):
    """Launch the real TUI in a PTY, press keys, and assert saved config output."""
    data_path = tmp_path / "data.csv"
    data_path.write_text("x,y\n1,2\n")
    output_dir = tmp_path / "runs"
    prefill_path = tmp_path / "prefill.yaml"
    prefill_path.write_text(
        yaml.dump(
            {
                "data": str(data_path),
                "goal": "PTY smoke goal",
                "max_iterations": 2,
                "output_dir": str(output_dir),
                "preset": "turbo",
                "provider": "openai",
            },
            sort_keys=False,
        )
    )

    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.update(
        {
            "CODEX_HOME": str(tmp_path / "codex-home"),
            "HOME": str(tmp_path / "home"),
            "PYTHON_DOTENV_DISABLED": "1",
            "PYTHONUNBUFFERED": "1",
            "TERM": "xterm-256color",
        }
    )
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY"):
        env.pop(key, None)
    python_path = os.pathsep.join(
        [
            str(repo_root / "src"),
            str(repo_root / "auto-core" / "src"),
            str(repo_root / "auto-reviewer" / "src"),
            env.get("PYTHONPATH", ""),
        ]
    )
    env["PYTHONPATH"] = python_path

    child_code = (
        "from auto_scientist import cli as cli_mod\n"
        "from auto_scientist.launch_app import LaunchApp\n"
        "def blocked_validate(self, config):\n"
        "    raise SystemExit('Model validation blocked in PTY smoke')\n"
        "def blocked_run(*args, **kwargs):\n"
        "    raise SystemExit('LLM run blocked in PTY smoke')\n"
        "LaunchApp._validate_models = blocked_validate\n"
        "cli_mod._run_from_experiment_config = blocked_run\n"
        "cli_mod.cli()\n"
    )

    master_fd, slave_fd = os.openpty()
    try:
        size = struct.pack("HHHH", 40, 120, 0, 0)
        termios.tcsetwinsize(slave_fd, (40, 120))
    except AttributeError:
        import fcntl

        fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, size)

    proc = subprocess.Popen(
        [sys.executable, "-c", child_code, "-c", str(prefill_path)],
        cwd=tmp_path,
        env=env,
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        close_fds=True,
    )
    os.close(slave_fd)

    saved_path = tmp_path / "experiment.yaml"
    terminal_output = ""
    try:
        screen = _read_until(master_fd, "Data path:", timeout=10)
        terminal_output += screen
        assert "Auto-Scientist" in screen
        screen = _read_until(master_fd, "gpt-5.4-nano", timeout=10)
        terminal_output += screen

        os.write(master_fd, b"\x13")  # Ctrl+S, bound to LaunchApp.action_save.
        _wait_for_file(saved_path, timeout=5)
        terminal_output += _strip_terminal_controls(_read_pty(master_fd, timeout=0.5))

        # Ctrl+S can engage PTY flow control on some systems before Textual
        # receives it, so the first Ctrl+Q may only resume output. Send a
        # second Ctrl+Q so the app-level quit binding receives one reliably.
        os.write(master_fd, b"\x11\x11")  # Ctrl+Q, bound to LaunchApp.action_quit.
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=2)
    finally:
        terminal_output += _strip_terminal_controls(_read_pty(master_fd, timeout=0.2))
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
        os.close(master_fd)

    assert "Model validation blocked in PTY smoke" not in terminal_output
    assert "LLM run blocked in PTY smoke" not in terminal_output
    assert proc.returncode in (0, -signal.SIGTERM)
    saved = yaml.safe_load(saved_path.read_text())
    assert saved["data"] == str(data_path)
    assert saved["goal"] == "PTY smoke goal"
    assert saved["preset"] == "turbo"
    assert saved["provider"] == "openai"
    assert saved["max_iterations"] == 2
    assert saved["output_dir"] == str(output_dir)
    assert saved["models"]["scientist"]["model"] == "gpt-5.4-nano"
    assert saved["models"]["summarizer"]["model"] == "gpt-5.4-nano"
    assert saved["models"]["critics"][0]["model"] == "gpt-5.4-nano"
    assert saved["models"]["critics"][1]["model"] == "gpt-5.4-nano"
