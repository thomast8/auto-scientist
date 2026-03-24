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

    def complete(self, done_summary: str = "") -> None:
        """Mark this panel as done. Collapses to the summary line.

        If done_summary is empty and the panel has lines, the last line
        is used as the done summary (preserves the summarizer's [done] entry).
        """
        if self.done:
            return
        self.done = True
        if done_summary:
            self.done_summary = done_summary
        elif self.lines:
            self.done_summary = self.lines[-1]
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
            parts.append(f"{self.num_turns} {'turn' if self.num_turns == 1 else 'turns'}")
        return " | ".join(parts)

    def _build_body(self) -> RenderableType:
        """Build the panel body content."""
        if self.error_msg:
            return Text(f"[error] {self.error_msg}", style="red")

        done_style = f"bold {self.style}"

        if self.done and self.done_summary:
            summary = self.done_summary
            if summary.startswith("[done] "):
                summary = summary[len("[done] "):]
            return Text(f"[done] {summary}", style=done_style)

        if not self.lines:
            return Text("  working...", style="dim")

        body_lines = [
            Text(line, style=done_style) if line.startswith("[done]") else Text(line)
            for line in self.lines
        ]
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
        """Rich console protocol: render as a compact status line."""
        elapsed = _format_elapsed(time.monotonic() - self.start_time)
        phase_style = PHASE_STYLES.get(self.phase, "cyan")

        line = Text()
        line.append(f" Iteration {self.iteration}", style="bold")
        line.append("  ")
        line.append(self.phase, style=phase_style)
        line.append("  ", style="dim")
        line.append(elapsed, style="dim")

        total_tokens = self.total_input_tokens + self.total_output_tokens
        if total_tokens > 0:
            tokens = f"{self.total_input_tokens:,} in / {self.total_output_tokens:,} out"
            line.append(f" | {tokens}", style="dim")
        if self.total_turns:
            label = "turn" if self.total_turns == 1 else "turns"
            line.append(f" | {self.total_turns} {label}", style="dim")

        if self.best_score is not None:
            # best_version is "vNN" format; v00 = iteration 1, v01 = iteration 2
            try:
                best_iter = int(self.best_version.lstrip("v")) + 1
            except (ValueError, AttributeError):
                best_iter = "?"
            style = _score_style(self.best_score)
            line.append(f"  best: iter {best_iter} ({self.best_score})", style=style)

        yield from line.__rich_console__(console, options)


class PipelineLive:
    """Manages a Rich.Live display with agent panels and a status bar.

    There is exactly one PipelineLive instance for the entire run.
    All phases (including debate) render within this single Live context.
    """

    def __init__(self) -> None:
        self._items: list[RenderableType] = []
        self._status_bar = StatusBar()
        self._live: Live | None = None
        self._file_console: Console | None = None
        self._file_handle = None
        # Iteration border state
        self._iter_title: str | None = None
        self._iter_subtitle: str | None = None
        self._iter_style: str = "bold"

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
        self._items.append(panel)
        self.refresh()

    def add_rule(self, rule: Rule) -> None:
        """Add a Rule inline with panels (renders in chronological order)."""
        self._items.append(rule)
        self.refresh()
        if self._file_console is not None:
            self._file_console.print(rule)

    def start_iteration(self, title: int | str) -> None:
        """Begin an iteration Panel. All subsequent items render inside it."""
        if isinstance(title, int):
            self._iter_title = f"Iteration {title}"
        else:
            self._iter_title = title
        self._iter_subtitle = None
        self._iter_style = "bold"
        self.refresh()
        if self._file_console is not None:
            self._file_console.print(Rule(self._iter_title, style="bold"))

    def end_iteration(self, subtitle: str, style: str) -> None:
        """Finalize the iteration Panel with a subtitle and border color."""
        self._iter_subtitle = subtitle
        self._iter_style = style
        self.refresh()
        if self._file_console is not None:
            if self._iter_title is not None:
                label = f"{self._iter_title}: {subtitle}"
            else:
                label = subtitle
            self._file_console.print(Rule(label, style=style))

    def remove_panel(self, panel: AgentPanel) -> None:
        """Remove an agent panel from the live display."""
        if panel in self._items:
            self._items.remove(panel)
            self.refresh()

    def collapse_panel(self, panel: AgentPanel, done_summary: str = "") -> None:
        """Mark a panel as complete and accumulate stats.

        Outside an iteration: flushes everything up to and including this
        panel statically (rules that precede it + the panel itself).
        Inside an iteration: keeps items in _items so they render inside
        the iteration Panel border until flush_completed() is called.
        """
        panel.complete(done_summary)
        self._status_bar.add_agent_stats(panel)

        if self._iter_title is None and panel in self._items:
            idx = self._items.index(panel)
            to_flush = self._items[:idx + 1]
            target = self._live.console if self._live is not None else console
            for item in to_flush:
                target.print(item)
            del self._items[:idx + 1]

        self.refresh()

    def flush_completed(self) -> None:
        """Print all remaining items statically and clear the live area.

        Inside an iteration: wraps all items in a Panel with the iteration
        title/subtitle/border, prints as one unit, and resets iteration state.
        Outside: prints items individually.
        """
        target = self._live.console if self._live is not None else console
        if self._iter_title is not None:
            body = Group(*self._items) if self._items else Text("")
            iter_panel = Panel(
                body,
                title=self._iter_title,
                title_align="left",
                subtitle=self._iter_subtitle,
                subtitle_align="left",
                border_style=self._iter_style,
                expand=True,
            )
            target.print(iter_panel)
            self._iter_title = None
            self._iter_subtitle = None
            self._iter_style = "bold"
        else:
            for item in self._items:
                target.print(item)
        self._items.clear()
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
        return panel in self._items

    @property
    def panel_count(self) -> int:
        """Number of active panels (excludes rules)."""
        return sum(1 for item in self._items if isinstance(item, AgentPanel))

    def refresh(self) -> None:
        """Rebuild and push the renderable to Live."""
        if self._live is not None:
            self._live.update(self._build_renderable())

    def _build_renderable(self) -> RenderableType:
        """Build the full Live renderable: items + status bar."""
        parts: list[RenderableType] = []
        if self._iter_title is not None:
            body = Group(*self._items) if self._items else Text("")
            iter_panel = Panel(
                body,
                title=self._iter_title,
                title_align="left",
                subtitle=self._iter_subtitle,
                subtitle_align="left",
                border_style=self._iter_style,
                expand=True,
            )
            parts.append(iter_panel)
        else:
            parts.extend(self._items)
        parts.append(Rule(style="dim"))
        parts.append(self._status_bar)
        return Group(*parts)
