"""Textual-based CLI dashboard for the auto-scientist pipeline.

Provides:
- AgentPanel: Auto-sizing widget for each agent phase (scrolling deque / full expand)
- StatusBarWidget: Persistent status bar docked to bottom
- IterationContainer: Bordered container grouping panels per iteration
- PipelineLive: Bridge between orchestrator (worker thread) and Textual app
- PipelineApp: Textual App that composes the widget tree and runs the orchestrator
"""

import asyncio
import time
from collections import deque
from pathlib import Path

from rich.console import Console, Group, RenderableType
from rich.text import Text
from textual.app import App, ComposeResult, RenderResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.widget import Widget
from textual.widgets import Static
from textual.worker import Worker, WorkerState

# Module-level console for one-time prints (startup banner in headless mode, etc.)
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


# ---------------------------------------------------------------------------
# AgentPanel widget
# ---------------------------------------------------------------------------


class AgentPanel(Widget):
    """Auto-sizing panel widget for a single agent phase.

    Shows the most recent PANEL_MAX_LINES summary lines by default.
    When ``expanded`` is True, shows ALL accumulated lines and the widget
    grows to fit (``height: auto``).  On completion, collapses to just the
    done summary + footer stats.
    """

    DEFAULT_CSS = """
    AgentPanel {
        height: auto;
        min-height: 3;
        padding: 0 1;
        margin: 0 0;
    }
    """

    def __init__(self, name: str, model: str, style: str = "cyan") -> None:
        super().__init__()
        self._panel_name = name
        self.model = model
        self.panel_style = style
        self.lines: deque[str] = deque(maxlen=PANEL_MAX_LINES)
        self.all_lines: list[str] = []
        self.start_time = time.monotonic()
        self.input_tokens = 0
        self.output_tokens = 0
        self.num_turns = 0
        self.done = False
        self.done_summary = ""
        self.error_msg = ""
        self._end_time: float | None = None
        self.expanded: bool = False

    def on_mount(self) -> None:
        self._refresh_timer = self.set_interval(1 / 4, self._tick)

    def _tick(self) -> None:
        """Periodic refresh for the elapsed timer. Stops after done."""
        if self.done and hasattr(self, "_refresh_timer"):
            self._refresh_timer.stop()
        self.refresh()

    @property
    def panel_name(self) -> str:
        return self._panel_name

    def add_line(self, text: str) -> None:
        """Append a summary line. Older lines scroll off. No-op after done."""
        if self.done:
            return
        cleaned = " ".join(text.split())
        self.all_lines.append(cleaned)
        self.lines.append(cleaned)

    def complete(self, done_summary: str = "") -> None:
        """Mark this panel as done.

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

        done_style = f"bold {self.panel_style}"

        if self.done and self.done_summary:
            summary = self.done_summary
            if summary.startswith("[done] "):
                summary = summary[len("[done] "):]
            if self.expanded and self.all_lines:
                body_lines = [
                    Text(line, style=done_style) if line.startswith("[done]") else Text(line)
                    for line in self.all_lines
                ]
                done_text = f"[done] {summary}"
                last = self.all_lines[-1]
                if last != done_text and last != summary:
                    body_lines.append(Text(done_text, style=done_style))
                return Group(*body_lines)
            return Text(f"[done] {summary}", style=done_style)

        if not self.lines:
            return Text("  working...", style="dim")

        source = self.all_lines if self.expanded else self.lines
        body_lines = [
            Text(line, style=done_style) if line.startswith("[done]") else Text(line)
            for line in source
        ]
        return Group(*body_lines)

    def render(self) -> RenderResult:
        """Render the panel body. Border title/subtitle set dynamically."""
        border_color = "red" if self.error_msg else self.panel_style
        self.border_title = f"{self._panel_name} ({self.model})"
        self.border_subtitle = self._build_footer()
        self.styles.border = ("solid", border_color)
        return self._build_body()

    def on_click(self) -> None:
        """Toggle expanded state on click."""
        self.expanded = not self.expanded
        self.refresh()


# ---------------------------------------------------------------------------
# StatusBarWidget
# ---------------------------------------------------------------------------


class StatusBarWidget(Widget):
    """Persistent status bar docked to the bottom of the screen."""

    DEFAULT_CSS = """
    StatusBarWidget {
        dock: bottom;
        height: 1;
        background: $surface;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self.start_time = time.monotonic()
        self.iteration = 0
        self.phase = ""
        self.best_version = ""
        self.best_score: int | None = None
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_turns = 0
        self.finished: bool = False
        self._end_time: float | None = None

    def on_mount(self) -> None:
        self.set_interval(1 / 4, self.refresh)

    def set_status(
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

    def finish(self) -> None:
        """Freeze the elapsed timer at the current value."""
        self.finished = True
        self._end_time = time.monotonic()

    def add_agent_stats(self, panel: AgentPanel) -> None:
        """Accumulate a completed agent's stats into the running totals."""
        self.total_input_tokens += panel.input_tokens
        self.total_output_tokens += panel.output_tokens
        self.total_turns += panel.num_turns

    def render(self) -> RenderResult:
        end = self._end_time if self._end_time else time.monotonic()
        elapsed = _format_elapsed(end - self.start_time)
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
            try:
                best_iter = int(self.best_version.lstrip("v")) + 1
            except (ValueError, AttributeError):
                best_iter = "?"
            style = _score_style(self.best_score)
            line.append(f"  best: iter {best_iter} ({self.best_score})", style=style)

        try:
            has_expanded = any(p.expanded for p in self.app.query(AgentPanel))
        except Exception:
            has_expanded = False
        toggle_action = "compact" if has_expanded else "expand"
        line.append(f"  Ctrl+O: {toggle_action}", style="dim italic")
        if self.finished:
            line.append("  q: exit", style="dim italic")

        return line


# ---------------------------------------------------------------------------
# IterationContainer
# ---------------------------------------------------------------------------


class IterationContainer(Vertical):
    """Bordered container grouping panels for one iteration."""

    DEFAULT_CSS = """
    IterationContainer {
        height: auto;
        border: solid $accent;
    }
    """

    def __init__(self, iter_title: str) -> None:
        super().__init__()
        self.border_title = iter_title

    def set_result(self, text: str, style: str) -> None:
        """Set the iteration result as border subtitle."""
        self.border_subtitle = text
        # Only set border color for actual color names, not Rich style modifiers
        valid = {"red", "green", "yellow", "blue", "cyan", "magenta", "white"}
        if style in valid:
            self.styles.border = ("solid", style)


# ---------------------------------------------------------------------------
# PipelineLive bridge
# ---------------------------------------------------------------------------


class PipelineLive:
    """Bridge between the orchestrator (worker thread) and the Textual app.

    In app mode (_app is set): delegates widget mounting/refresh to PipelineApp
    via call_from_thread for thread safety.
    In headless mode (_app is None): tracks state only, no widget rendering.
    """

    def __init__(self) -> None:
        self._panels: list[AgentPanel] = []
        self._app: PipelineApp | None = None
        self._status_bar: StatusBarWidget | None = None
        self._current_iteration: IterationContainer | None = None
        self._file_console: Console | None = None
        self._file_handle = None

    def start(self, log_path: Path | None = None) -> None:
        """Open the optional log file."""
        if log_path:
            self._file_handle = open(log_path, "a")  # noqa: SIM115
            self._file_console = Console(
                file=self._file_handle, no_color=True, width=120,
            )

    def stop(self) -> None:
        """Close the log file and reset panel state."""
        for panel in self._panels:
            panel.expanded = False
        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None
            self._file_console = None

    def add_panel(self, panel: AgentPanel) -> None:
        """Track a panel and mount it in the app if running."""
        self._panels.append(panel)
        if self._app is not None:
            self._app.call_from_thread(self._app._mount_panel, panel)

    def collapse_panel(self, panel: AgentPanel, done_summary: str = "") -> None:
        """Mark a panel as complete and accumulate stats."""
        panel.complete(done_summary)
        if self._status_bar is not None:
            self._status_bar.add_agent_stats(panel)
        if self._app is not None:
            self._app.call_from_thread(panel.refresh)
        if self._file_console is not None:
            self._file_console.print(
                f"[{panel.panel_name}] {panel.done_summary} ({panel._build_footer()})"
            )

    def start_iteration(self, title: int | str) -> None:
        """Begin an iteration container."""
        iter_title = f"Iteration {title}" if isinstance(title, int) else title
        container = IterationContainer(iter_title=iter_title)
        self._current_iteration = container
        if self._app is not None:
            self._app.call_from_thread(self._app._mount_iteration, container)
        if self._file_console is not None:
            self._file_console.print(f"\n{'=' * 60}")
            self._file_console.print(iter_title)
            self._file_console.print(f"{'=' * 60}")

    def end_iteration(self, subtitle: str, style: str) -> None:
        """Finalize the iteration container with a result."""
        if self._current_iteration is not None:
            if self._app is not None:
                self._app.call_from_thread(
                    self._current_iteration.set_result, subtitle, style,
                )
            else:
                self._current_iteration.set_result(subtitle, style)
        if self._file_console is not None:
            label = subtitle
            if self._current_iteration is not None:
                label = f"{self._current_iteration.border_title}: {subtitle}"
            self._file_console.print(f"--- {label} ---")

    def flush_completed(self) -> None:
        """Clear iteration state. Panels stay mounted in the DOM."""
        self._current_iteration = None

    def remove_panel(self, panel: AgentPanel) -> None:
        """Remove a panel from tracking."""
        if panel in self._panels:
            self._panels.remove(panel)
        if self._app is not None:
            self._app.call_from_thread(panel.remove)

    def update_status(self, **kwargs) -> None:
        """Update the status bar fields."""
        if self._status_bar is not None:
            self._status_bar.set_status(**kwargs)

    def log(self, message: str) -> None:
        """Write a message to the log file only (no terminal output)."""
        if self._file_console is not None:
            self._file_console.print(message)

    def print_static(self, renderable: RenderableType) -> None:
        """Print a renderable. In app mode, mount as Static widget."""
        if self._app is not None:
            self._app.call_from_thread(self._app._mount_static, renderable)
        else:
            console.print(renderable)
        if self._file_console is not None:
            self._file_console.print(renderable)

    def add_rule(self, rule: RenderableType) -> None:
        """Add a rule/separator. In app mode, mount as Static widget."""
        if self._app is not None:
            self._app.call_from_thread(self._app._mount_static, rule)
        if self._file_console is not None:
            self._file_console.print(rule)

    def has_panel(self, panel: AgentPanel) -> bool:
        """Check if a panel is tracked."""
        return panel in self._panels

    @property
    def panel_count(self) -> int:
        """Number of tracked panels that are not yet done."""
        return sum(1 for p in self._panels if not p.done)

    def wait_for_dismiss(self) -> None:
        """No-op. PipelineApp handles dismiss via q key binding."""

    def refresh(self) -> None:
        """Refresh all active panels in the app."""
        if self._app is not None:
            for panel in self._panels:
                if not panel.done:
                    self._app.call_from_thread(panel.refresh)


# ---------------------------------------------------------------------------
# PipelineApp
# ---------------------------------------------------------------------------


class PipelineApp(App):
    """Textual application that runs the orchestrator and displays the dashboard."""

    BINDINGS = [
        Binding("ctrl+o", "toggle_expand", show=False),
        Binding("q", "quit_app", show=False),
    ]

    DEFAULT_CSS = """
    #main-scroll {
        height: 1fr;
    }
    """

    def __init__(self, orchestrator) -> None:
        super().__init__()
        self._orchestrator = orchestrator
        self._finished: bool = False
        self._live: PipelineLive = PipelineLive()

    def compose(self) -> ComposeResult:
        yield VerticalScroll(id="main-scroll")
        yield StatusBarWidget()

    def on_mount(self) -> None:
        # Wire the bridge to this app
        self._live._app = self
        self._live._status_bar = self.query_one(StatusBarWidget)

        # Override the orchestrator's default headless PipelineLive
        self._orchestrator._live = self._live

        # Log file is opened by orchestrator.run() after it creates the
        # output directory. Don't open it here since the dir may not exist.

        # Run orchestrator in a thread worker
        self.run_worker(self._run_pipeline, thread=True, exit_on_error=False)

    def _run_pipeline(self) -> None:
        """Sync wrapper that runs the async orchestrator in its own event loop."""
        asyncio.run(self._orchestrator.run())

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if event.state in (WorkerState.SUCCESS, WorkerState.ERROR):
            self._finished = True
            bar = self.query_one(StatusBarWidget)
            bar.finish()

    # -- Key binding actions --

    def action_toggle_expand(self) -> None:
        """Toggle expanded state on all AgentPanels."""
        panels = list(self.query(AgentPanel))
        if not panels:
            return
        # Toggle to the opposite of the first panel's state
        new_state = not panels[0].expanded
        for panel in panels:
            panel.expanded = new_state
            panel.refresh()

    def action_quit_app(self) -> None:
        """Exit the app, but only after the pipeline finishes."""
        if self._finished:
            self.exit()

    # -- Mount helpers (called via call_from_thread from PipelineLive) --

    def _mount_panel(self, panel: AgentPanel) -> None:
        """Mount a panel into the current iteration container or main scroll."""
        target = self._live._current_iteration or self.query_one("#main-scroll")
        target.mount(panel)
        panel.scroll_visible()

    def _mount_iteration(self, container: IterationContainer) -> None:
        """Mount an iteration container into the main scroll."""
        self.query_one("#main-scroll").mount(container)

    def _mount_static(self, renderable: RenderableType) -> None:
        """Mount a Rich renderable as a Static widget."""
        self.query_one("#main-scroll").mount(Static(renderable))
