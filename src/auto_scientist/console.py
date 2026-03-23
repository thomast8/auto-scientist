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

# Maps orchestrator phase names to colors (matches the active agent)
PHASE_STYLES = {
    "INGESTION": "red",
    "ANALYZE": "green",
    "PLAN": "cyan",
    "DEBATE": "yellow",
    "REVISE": "cyan",
    "IMPLEMENT": "magenta",
    "REPORT": "blue",
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
        self.num_turns = 0
        self.done = False
        self.done_summary = ""
        self.error_msg = ""
        self._end_time: float | None = None

    def add_line(self, text: str) -> None:
        """Append a summary line. Older lines scroll off. No-op after done."""
        if self.done:
            return
        cleaned = " ".join(text.split())
        self.lines.append(cleaned)

    def complete(self, done_summary: str) -> None:
        """Mark this panel as done. Collapses to the summary line."""
        if self.done:
            return
        self.done = True
        self.done_summary = done_summary
        self._end_time = time.monotonic()

    def error(self, msg: str) -> None:
        """Mark this panel as errored."""
        if self.done:
            return
        self.done = True
        self.error_msg = msg
        self._end_time = time.monotonic()

    def set_tokens(self, input_tokens: int, output_tokens: int) -> None:
        """Set token usage metadata."""
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens

    def set_stats(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        num_turns: int = 0,
    ) -> None:
        """Set rich usage stats from SDK ResultMessage or direct API calls."""
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.num_turns = num_turns

    @property
    def elapsed(self) -> float:
        """Elapsed seconds since panel creation."""
        end = self._end_time if self._end_time else time.monotonic()
        return end - self.start_time

    def _build_footer(self) -> str:
        """Build the footer subtitle string."""
        parts = [_format_elapsed(self.elapsed)]
        if self.input_tokens or self.output_tokens:
            parts.append(f"{self.input_tokens:,} in / {self.output_tokens:,} out")
        if self.num_turns:
            parts.append(f"{self.num_turns} turns")
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
        subtitle = Text(self._build_footer(), style=border_style)

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
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_turns = 0

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

    def add_agent_stats(self, panel: "AgentPanel") -> None:
        """Accumulate a completed agent's stats into the running totals."""
        self.total_input_tokens += panel.input_tokens
        self.total_output_tokens += panel.output_tokens
        self.total_turns += panel.num_turns

    def __rich_console__(self, console: Console, options):
        """Rich console protocol: render as a single-row Table."""
        elapsed = _format_elapsed(time.monotonic() - self.start_time)
        phase_style = PHASE_STYLES.get(self.phase, "cyan")

        table = Table(show_header=False, expand=True, box=None, padding=(0, 1))
        table.add_column("iter", style="bold", no_wrap=True)
        table.add_column("phase", style=phase_style, no_wrap=True)
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

        # Row 1: iteration, phase, elapsed, best score
        table.add_row(
            f"Iteration {self.iteration}",
            self.phase,
            elapsed,
            score_text,
        )

        # Row 2: running totals (tokens, turns, cost)
        total_tokens = self.total_input_tokens + self.total_output_tokens
        if total_tokens > 0:
            stats_parts = [f"{self.total_input_tokens:,} in / {self.total_output_tokens:,} out"]
            if self.total_turns:
                stats_parts.append(f"{self.total_turns} turns")
            table.add_row(
                "",
                "",
                Text(" | ".join(stats_parts), style="dim"),
                "",
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
        if self._live is not None:
            return  # already started

        if log_path:
            self._file_handle = open(log_path, "a")
            self._file_console = Console(
                file=self._file_handle, no_color=True, width=120,
            )

        try:
            self._live = Live(
                self._build_renderable(),
                refresh_per_second=4,
                transient=False,
                console=console,
            )
            self._live.start()
        except Exception:
            if self._file_handle is not None:
                self._file_handle.close()
                self._file_handle = None
                self._file_console = None
            raise

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
        self.refresh()

    def remove_panel(self, panel: AgentPanel) -> None:
        """Remove an agent panel from the live display."""
        if panel in self._panels:
            self._panels.remove(panel)
            self.refresh()

    def collapse_panel(self, panel: AgentPanel, done_summary: str = "") -> None:
        """Mark a panel as complete, accumulate stats, and refresh."""
        if done_summary:
            panel.complete(done_summary)
        self._status_bar.add_agent_stats(panel)
        self.refresh()

    def update_status(self, **kwargs) -> None:
        """Update the status bar fields and refresh."""
        self._status_bar.update(**kwargs)
        self.refresh()

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

    def has_panel(self, panel: AgentPanel) -> bool:
        """Check if a panel is in the live display."""
        return panel in self._panels

    @property
    def panel_count(self) -> int:
        """Number of active panels."""
        return len(self._panels)

    def refresh(self) -> None:
        """Rebuild and push the renderable to Live."""
        if self._live is not None:
            self._live.update(self._build_renderable())

    def _build_renderable(self) -> RenderableType:
        """Build the full Live renderable: panels + status bar."""
        parts: list[RenderableType] = list(self._panels)
        parts.append(Rule(style="dim"))
        parts.append(self._status_bar)
        return Group(*parts)
