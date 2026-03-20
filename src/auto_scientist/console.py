"""Console output helpers for live token streaming."""

import os
import sys
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
    "Discovery": BLUE,
    "Ingestor": GREEN,
    "Report": BLUE,
}


def _use_color() -> bool:
    return "NO_COLOR" not in os.environ


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
