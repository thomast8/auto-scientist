"""Rich-based CLI dashboard for the auto-scientist pipeline.

Provides three main components:
- AgentPanel: Fixed-height scrolling panel for each agent phase
- StatusBar: Persistent status bar showing iteration, phase, elapsed, best score
- PipelineLive: Wraps Rich.Live to manage panels + status bar as one layout
"""

import time
from collections import deque
from pathlib import Path

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

# Module-level console for one-time prints (startup banner, etc.)
console = Console()

PANEL_MAX_LINES = 5

# Agent style palette
AGENT_STYLES = {
    "Analyst": "green",
    "Scientist": "cyan",
    "Coder": "magenta",
    "Ingestor": "red",
    "Report": "blue",
    "Critic": "yellow",
    "Debate": "yellow",
    "Results": "dim",
}


def _score_style(score: int) -> str:
    """Return a Rich style string for a 0-100 score."""
    if score >= 70:
        return "green"
    if score >= 40:
        return "yellow"
    return "red"


def _format_elapsed(seconds: float) -> str:
    """Format seconds into a human-readable duration like '4m 32s'."""
    m, s = divmod(int(seconds), 60)
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


class AgentPanel:
    """A fixed-height scrolling panel for a single agent phase.

    Shows the most recent PANEL_MAX_LINES summary lines. On completion,
    collapses to just the done summary + footer stats. Implements the
    Rich console protocol for rendering.
    """

    def __init__(self, name: str, model: str, style: str = "cyan") -> None:
        self.name = name
        self.model = model
        self.style = style
        self.lines: deque[str] = deque(maxlen=PANEL_MAX_LINES)
        self.start_time = time.monotonic()
        self.input_tokens = 0
        self.output_tokens = 0
        self.done = False
        self.done_summary = ""
        self.error_msg = ""
        self._end_time: float | None = None

    def add_line(self, text: str) -> None:
        """Append a summary line. Older lines scroll off."""
        cleaned = " ".join(text.split())
        self.lines.append(cleaned)

    def complete(self, done_summary: str) -> None:
        """Mark this panel as done. Collapses to the summary line."""
        self.done = True
        self.done_summary = done_summary
        self._end_time = time.monotonic()

    def error(self, msg: str) -> None:
        """Mark this panel as errored."""
        self.done = True
        self.error_msg = msg
        self._end_time = time.monotonic()

    def set_tokens(self, input_tokens: int, output_tokens: int) -> None:
        """Set token usage metadata."""
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens

    @property
    def elapsed(self) -> float:
        """Elapsed seconds since panel creation."""
        end = self._end_time if self._end_time else time.monotonic()
        return end - self.start_time

    def _build_footer(self) -> str:
        """Build the footer subtitle string."""
        parts = [_format_elapsed(self.elapsed)]
        if self.input_tokens or self.output_tokens:
            parts.append(f"{self.input_tokens:,} in / {self.output_tokens:,} out tokens")
        return " | ".join(parts)

    def _build_body(self) -> RenderableType:
        """Build the panel body content."""
        if self.error_msg:
            return Text(f"[error] {self.error_msg}", style="red")

        if self.done and self.done_summary:
            return Text(f"[done] {self.done_summary}", style="bold")

        if not self.lines:
            return Text("  working...", style="dim")

        body_lines = [Text(line) for line in self.lines]
        return Group(*body_lines)

    def __rich_console__(self, console: Console, options):
        """Rich console protocol: render as a Panel."""
        border_style = "red" if self.error_msg else self.style
        subtitle = self._build_footer() if self.done else None

        panel = Panel(
            self._build_body(),
            title=f"{self.name} ({self.model})",
            title_align="left",
            subtitle=subtitle,
            subtitle_align="left",
            border_style=border_style,
            expand=True,
        )
        yield from panel.__rich_console__(console, options)


class StatusBar:
    """Persistent status bar showing iteration, phase, elapsed, best score.

    Elapsed time is computed dynamically at render time so the clock
    ticks smoothly with Live's refresh_per_second.
    """

    def __init__(self, start_time: float | None = None) -> None:
        self.start_time = start_time or time.monotonic()
        self.iteration = 0
        self.phase = ""
        self.best_version = ""
        self.best_score: int | None = None

    def update(
        self,
        iteration: int | None = None,
        phase: str | None = None,
        best_version: str | None = None,
        best_score: int | None = None,
    ) -> None:
        """Update status bar fields. Only non-None values are changed."""
        if iteration is not None:
            self.iteration = iteration
        if phase is not None:
            self.phase = phase
        if best_version is not None:
            self.best_version = best_version
        if best_score is not None:
            self.best_score = best_score

    def __rich_console__(self, console: Console, options):
        """Rich console protocol: render as a single-row Table."""
        elapsed = _format_elapsed(time.monotonic() - self.start_time)

        table = Table(show_header=False, expand=True, box=None, padding=(0, 1))
        table.add_column("iter", style="bold", no_wrap=True)
        table.add_column("phase", style="cyan", no_wrap=True)
        table.add_column("elapsed", style="dim", no_wrap=True)
        table.add_column("best", no_wrap=True, justify="right")

        # Format best score with color
        if self.best_score is not None:
            score_text = Text(
                f"{self.best_version} (score {self.best_score})",
                style=_score_style(self.best_score),
            )
        else:
            score_text = Text("-", style="dim")

        table.add_row(
            f"Iteration {self.iteration}",
            self.phase,
            elapsed,
            score_text,
        )
        yield from table.__rich_console__(console, options)


class PipelineLive:
    """Manages a Rich.Live display with agent panels and a status bar.

    There is exactly one PipelineLive instance for the entire run.
    All phases (including debate) render within this single Live context.
    """

    def __init__(self) -> None:
        self._panels: list[AgentPanel] = []
        self._status_bar = StatusBar()
        self._live: Live | None = None
        self._file_console: Console | None = None
        self._file_handle = None

    def start(self, log_path: Path | None = None) -> None:
        """Start the Live display and optionally open a log file."""
        if log_path:
            self._file_handle = open(log_path, "a")
            self._file_console = Console(
                file=self._file_handle, no_color=True, width=120,
            )

        self._live = Live(
            self._build_renderable(),
            refresh_per_second=4,
            transient=False,
            console=console,
        )
        self._live.start()

    def stop(self) -> None:
        """Stop the Live display and close the log file."""
        if self._live is not None:
            self._live.stop()
            self._live = None
        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None
            self._file_console = None

    def add_panel(self, panel: AgentPanel) -> None:
        """Add an agent panel to the live display."""
        self._panels.append(panel)
        self._refresh()

    def remove_panel(self, panel: AgentPanel) -> None:
        """Remove an agent panel from the live display."""
        if panel in self._panels:
            self._panels.remove(panel)
            self._refresh()

    def collapse_panel(self, panel: AgentPanel, done_summary: str = "") -> None:
        """Mark a panel as complete and refresh."""
        if done_summary:
            panel.complete(done_summary)
        self._refresh()

    def update_status(self, **kwargs) -> None:
        """Update the status bar fields and refresh."""
        self._status_bar.update(**kwargs)
        self._refresh()

    def log(self, message: str) -> None:
        """Write a message to the log file (no terminal output)."""
        if self._file_console is not None:
            self._file_console.print(message)

    def print_static(self, renderable: RenderableType) -> None:
        """Print a renderable above the live display (e.g., Rules, banners).

        Uses live.console.print() which Rich handles correctly within
        a Live context.
        """
        if self._live is not None:
            self._live.console.print(renderable)
        else:
            console.print(renderable)
        # Also log to file
        if self._file_console is not None:
            self._file_console.print(renderable)

    def _build_renderable(self) -> RenderableType:
        """Build the full Live renderable: panels + status bar."""
        parts: list[RenderableType] = list(self._panels)
        parts.append(Rule(style="dim"))
        parts.append(self._status_bar)
        return Group(*parts)

    def _refresh(self) -> None:
        """Rebuild and push the renderable to Live."""
        if self._live is not None:
            self._live.update(self._build_renderable())


# ---------------------------------------------------------------------------
# Backward-compatibility shims (used by orchestrator.py and summarizer.py
# until they are migrated to the new Rich API in the next commit).
# These will be removed once the migration is complete.
# ---------------------------------------------------------------------------

BOLD = "\033[1m"
RESET = "\033[0m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
GREEN = "\033[32m"
MAGENTA = "\033[35m"
BLUE = "\033[34m"
RED = "\033[31m"
DIM = "\033[2m"

AGENT_COLORS = {
    "Critic": YELLOW, "Scientist": CYAN, "Analyst": GREEN,
    "Coder": MAGENTA, "Ingestor": RED, "Report": BLUE,
    "Debate": YELLOW, "Results": DIM,
}

STEP_COLORS = {
    "INGESTION": RED, "ANALYZE": GREEN, "PLAN": CYAN,
    "DEBATE": YELLOW, "REVISE": CYAN, "IMPLEMENT": MAGENTA,
    "REPORT": BLUE, "ITERATION": BOLD,
}

import os
import re
import shutil
import sys
import textwrap
from collections.abc import Callable
from datetime import datetime
from typing import TextIO

_log_file: TextIO | None = None
_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def _use_color() -> bool:
    return "NO_COLOR" not in os.environ


def colorize(text: str, color: str) -> str:
    if not _use_color():
        return text
    return f"{color}{text}{RESET}"


def score_color(score: int) -> str:
    if score >= 70:
        return GREEN
    if score >= 40:
        return YELLOW
    return RED


def init_console_log(path: Path) -> None:
    global _log_file
    _log_file = open(path, "a")


def close_console_log() -> None:
    global _log_file
    if _log_file is not None:
        _log_file.close()
        _log_file = None


def _log_to_file(text: str) -> None:
    if _log_file is None:
        return
    clean = _ANSI_RE.sub("", text)
    ts = datetime.now().strftime("%H:%M:%S")
    for line in clean.splitlines():
        _log_file.write(f"[{ts}] {line}\n")
    _log_file.flush()


def _wrap(message: str, subsequent_indent: str | None = None) -> str:
    width = shutil.get_terminal_size().columns
    if len(message) <= width:
        return message
    if subsequent_indent is None:
        leading = len(message) - len(message.lstrip())
        subsequent_indent = " " * (leading + 2)
    return textwrap.fill(
        message, width=width, subsequent_indent=subsequent_indent,
        break_long_words=False, break_on_hyphens=False,
    )


def print_step(message: str, *, color: str | None = None) -> None:
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
    for agent, color in AGENT_COLORS.items():
        if agent in label:
            return color
    return CYAN


def make_stream_printer(label: str) -> Callable[[str], None]:
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
    sys.stdout.write("\n\n")
    sys.stdout.flush()


def print_header(title: str, fields: dict[str, str] | None = None) -> None:
    width = min(shutil.get_terminal_size().columns, 60)
    separator_len = max(width - len(title) - 1, 10)
    separator = "\u2500" * separator_len
    header_line = f"{title} {separator}"
    _log_to_file(header_line)
    if _use_color():
        sys.stdout.write(f"\n{BOLD}{header_line}{RESET}\n")
    else:
        sys.stdout.write(f"\n{header_line}\n")
    if fields:
        for key, value in fields.items():
            line = _wrap(f"  {f'{key}:':<12s}{value}")
            _log_to_file(line)
            if _use_color():
                sys.stdout.write(f"{DIM}{line}{RESET}\n")
            else:
                sys.stdout.write(f"{line}\n")
    sys.stdout.write("\n")
    sys.stdout.flush()


def print_iteration_header(iteration: int) -> None:
    width = min(shutil.get_terminal_size().columns, 60)
    separator = "\u2501" * width
    title = f"ITERATION {iteration}"
    _log_to_file(separator)
    _log_to_file(title)
    _log_to_file(separator)
    if _use_color():
        sys.stdout.write(f"\n\n{BOLD}{separator}\n{title}\n{separator}{RESET}\n")
    else:
        sys.stdout.write(f"\n\n{separator}\n{title}\n{separator}\n")
    sys.stdout.flush()


def print_summary(agent_name: str, summary: str, label: str = "") -> None:
    if not summary:
        return
    use_color = _use_color()
    color = _color_for_label(agent_name)
    summary = " ".join(summary.split())
    max_len = 400 if label == "done" else 200
    if len(summary) > max_len:
        summary = summary[:max_len - 3] + "..."
    prefix = f"  > [{label}] " if label else "  > "
    line = _wrap(f"{prefix}{summary}", subsequent_indent=" " * len(prefix))
    _log_to_file(line)
    if use_color:
        sys.stdout.write(f"{color}{line}{RESET}\n")
    else:
        sys.stdout.write(f"{line}\n")
    sys.stdout.flush()


class DebateLiveDisplay:
    """Backward-compat shim for the old debate display."""

    def __init__(self, critic_labels: list[str]) -> None:
        self._labels = critic_labels
        self._lines: dict[str, list[str]] = {lb: [] for lb in critic_labels}
        self._live = None

    def start(self) -> None:
        if not _use_color():
            return
        from rich.live import Live as RichLive
        self._live = RichLive(self._render(), refresh_per_second=4, transient=False)
        self._live.start()

    def update(self, label: str, summary: str, time_label: str) -> None:
        summary = " ".join(summary.split())
        is_done = time_label.endswith("done")
        max_len = 400 if is_done else 200
        if len(summary) > max_len:
            summary = summary[:max_len - 3] + "..."
        line = f"  > [{time_label}] {summary}"
        self._lines[label].append(line)
        _log_to_file(f"[{label}] {line}")
        if self._live is not None:
            self._live.update(self._render())

    def stop(self) -> None:
        if self._live is not None:
            self._live.update(self._render())
            self._live.stop()
            self._live = None
        elif not _use_color():
            for label in self._labels:
                print(f"  --- {label} ---")
                for line in self._lines[label]:
                    print(line)

    def _render(self):
        from rich.console import Group as RichGroup
        from rich.rule import Rule as RichRule
        from rich.text import Text as RichText
        sections = []
        for label in self._labels:
            sections.append(RichRule(label, style="yellow"))
            if self._lines[label]:
                for line in self._lines[label]:
                    sections.append(RichText(line, style="yellow"))
            else:
                sections.append(RichText("  waiting...", style="dim yellow"))
        return RichGroup(*sections)
