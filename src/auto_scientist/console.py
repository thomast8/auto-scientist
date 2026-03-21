"""Console output helpers for live token streaming."""

import os
import re
import shutil
import sys
import textwrap
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import TextIO

BOLD = "\033[1m"
RESET = "\033[0m"

# Agent color palette
CYAN = "\033[36m"
YELLOW = "\033[33m"
GREEN = "\033[32m"
MAGENTA = "\033[35m"
BLUE = "\033[34m"
RED = "\033[31m"

AGENT_COLORS = {
    "Critic": YELLOW,
    "Scientist": CYAN,
    "Analyst": GREEN,
    "Coder": MAGENTA,
    "Ingestor": RED,
    "Report": BLUE,
    "Debate": YELLOW,
}

# Map orchestrator step prefixes to agent colors
STEP_COLORS = {
    "INGESTION": RED,
    "ANALYZE": GREEN,
    "PLAN": CYAN,
    "DEBATE": YELLOW,
    "REVISE": CYAN,
    "IMPLEMENT": MAGENTA,
    "REPORT": BLUE,
    "ITERATION": BOLD,
}


def _use_color() -> bool:
    return "NO_COLOR" not in os.environ


# ---------------------------------------------------------------------------
# Console log file tee (mirrors print_step / print_summary to a file)
# ---------------------------------------------------------------------------
_log_file: TextIO | None = None
_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def init_console_log(path: Path) -> None:
    """Open the console log file. Called once by the orchestrator at startup."""
    global _log_file
    _log_file = open(path, "a")  # noqa: SIM115 — append mode, closed explicitly


def close_console_log() -> None:
    """Flush and close the console log file."""
    global _log_file
    if _log_file is not None:
        _log_file.close()
        _log_file = None


def _log_to_file(text: str) -> None:
    """Write *text* to the console log, stripping ANSI codes and adding a timestamp."""
    if _log_file is None:
        return
    clean = _ANSI_RE.sub("", text)
    ts = datetime.now().strftime("%H:%M:%S")
    for line in clean.splitlines():
        _log_file.write(f"[{ts}] {line}\n")
    _log_file.flush()


def _wrap(message: str, subsequent_indent: str | None = None) -> str:
    """Wrap *message* to the terminal width with a hanging indent.

    If *subsequent_indent* is not given, continuation lines are indented to
    match the leading whitespace of the message plus two extra spaces.
    """
    width = shutil.get_terminal_size().columns
    if len(message) <= width:
        return message

    if subsequent_indent is None:
        leading = len(message) - len(message.lstrip())
        subsequent_indent = " " * (leading + 2)

    return textwrap.fill(
        message,
        width=width,
        subsequent_indent=subsequent_indent,
        break_long_words=False,
        break_on_hyphens=False,
    )


def print_step(message: str, *, color: str | None = None) -> None:
    """Print a pipeline status message with color based on its prefix.

    Recognizes prefixes like "ANALYZE:", "PLAN:", "ITERATION 3", etc.
    Falls back to plain print when NO_COLOR is set or no prefix matches.

    Pass ``color`` explicitly to override prefix-based detection.
    """
    message = _wrap(message)
    _log_to_file(message)

    if not _use_color():
        print(message)
        return

    if color is None:
        stripped = message.lstrip()
        for prefix, c in STEP_COLORS.items():
            if stripped.startswith(prefix):
                color = c
                break

    if color:
        sys.stdout.write(f"{color}{message}{RESET}\n")
        sys.stdout.flush()
    else:
        print(message)


def _color_for_label(label: str) -> str:
    """Pick a color based on the agent name found in the label."""
    for agent, color in AGENT_COLORS.items():
        if agent in label:
            return color
    return CYAN


def make_stream_printer(label: str) -> Callable[[str], None]:
    """Return a callback that prints tokens to stdout with a one-time label header."""
    printed_label = False
    use_color = _use_color()
    color = _color_for_label(label)

    def on_token(token: str) -> None:
        nonlocal printed_label
        if not printed_label:
            if use_color:
                sys.stdout.write(f"\n{color}{BOLD}{label}{RESET}\n")
            else:
                sys.stdout.write(f"\n{label}\n")
            _log_to_file(label)
            printed_label = True
        sys.stdout.write(token)
        sys.stdout.flush()

    return on_token


def stream_separator() -> None:
    """Print a separator after a streamed response completes."""
    sys.stdout.write("\n\n")
    sys.stdout.flush()


def print_summary(agent_name: str, summary: str, label: str = "") -> None:
    """Print a single-line formatted summary for an agent step.

    Args:
        agent_name: Agent name (used for color lookup).
        summary: Summary text (expected to be short, one sentence).
        label: Optional label like "15s", "done", or "" (for results).
    """
    if not summary:
        return

    use_color = _use_color()
    color = AGENT_COLORS.get(agent_name, CYAN)

    # Clean up: collapse whitespace, strip to one line
    summary = " ".join(summary.split())
    if len(summary) > 200:
        summary = summary[:197] + "..."

    prefix = f"  > [{label}] " if label else "  > "
    line = _wrap(f"{prefix}{summary}", subsequent_indent=" " * len(prefix))
    _log_to_file(line)

    if use_color:
        sys.stdout.write(f"{color}{line}{RESET}\n")
    else:
        sys.stdout.write(f"{line}\n")
    sys.stdout.flush()
