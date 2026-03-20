"""Console output helpers for live token streaming."""

import os
import sys
import textwrap
from collections.abc import Callable

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
    "Ingestor": GREEN,
    "Report": BLUE,
    "Debate": YELLOW,
}

# Map orchestrator step prefixes to agent colors
STEP_COLORS = {
    "INGESTION": GREEN,
    "ANALYZE": GREEN,
    "PLAN": CYAN,
    "DEBATE": YELLOW,
    "REVISE": CYAN,
    "IMPLEMENT": MAGENTA,
    "VALIDATE": MAGENTA,
    "RUN": BLUE,
    "REPORT": BLUE,
    "ITERATION": BOLD,
}


def _use_color() -> bool:
    return "NO_COLOR" not in os.environ


def print_step(message: str) -> None:
    """Print a pipeline status message with color based on its prefix.

    Recognizes prefixes like "ANALYZE:", "PLAN:", "ITERATION 3", etc.
    Falls back to plain print when NO_COLOR is set or no prefix matches.
    """
    if not _use_color():
        print(message)
        return

    stripped = message.lstrip()
    for prefix, color in STEP_COLORS.items():
        if stripped.startswith(prefix):
            sys.stdout.write(f"{color}{message}{RESET}\n")
            sys.stdout.flush()
            return

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
    line = f"{prefix}{summary}"

    if use_color:
        sys.stdout.write(f"{color}{line}{RESET}\n")
    else:
        sys.stdout.write(f"{line}\n")
    sys.stdout.flush()
