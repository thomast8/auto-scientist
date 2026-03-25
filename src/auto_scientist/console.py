"""Textual-based CLI dashboard for the auto-scientist pipeline.

Provides:
- AgentPanel: Collapsible + RichLog widget for each agent phase
- MetricsBar: Persistent metrics display (sparkline, tokens, phase)
- IterationContainer: Bordered container grouping panels per iteration
- AgentDetailScreen: Full-screen view of one agent's output
- QuitConfirmScreen: Modal confirmation dialog for quit
- PipelineCommandProvider: Command palette with navigation and control
- PipelineLive: Bridge between orchestrator (worker thread) and Textual app
- PipelineApp: Textual App with screens, command palette, and message handlers
"""

import asyncio
import json
import subprocess
import threading
import time
from functools import partial
from pathlib import Path

_PREFS_PATH = Path.home() / ".config" / "auto-scientist" / "preferences.json"


def _load_prefs() -> dict:
    """Load user preferences from disk."""
    try:
        return json.loads(_PREFS_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _save_prefs(prefs: dict) -> None:
    """Save user preferences to disk."""
    _PREFS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _PREFS_PATH.write_text(json.dumps(prefs, indent=2))

from rich.console import Console, RenderableType
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.command import Hit, Hits, Provider
from textual.containers import Center, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.screen import ModalScreen
from textual.widget import Widget
from textual.widgets import (
    Button,
    Collapsible,
    Footer,
    Header,
    Label,
    LoadingIndicator,
    RichLog,
    Static,
)
from textual.worker import Worker, WorkerState

# Module-level console for one-time prints (startup banner in headless mode, etc.)
console = Console()

# Agent style palette
AGENT_STYLES = {
    "Analyst": "green",
    "Scientist": "cyan",
    "Coder": "magenta1",
    "Ingestor": "red",
    "Report": "blue",
    "Critic": "yellow",
    "Debate": "yellow",
    "Results": "dim",
}

# Short descriptions shown immediately when an agent panel opens (before first summary)
AGENT_DESCRIPTIONS: dict[str, str] = {
    "Ingestor": "Preparing and canonicalizing raw data for experiment scripts...",
    "Analyst": "Analyzing experiment results and producing quantitative assessments...",
    "Scientist": "Formulating hypotheses and planning the next experiment...",
    "Critic": "Challenging the plan through critical debate...",
    "Revision": "Revising the experiment plan based on critique feedback...",
    "Coder": "Implementing the experiment plan as a Python script and running it...",
    "Report": "Generating a comprehensive summary report of all findings...",
}

# Maps orchestrator phase names to colors (matches the active agent)
PHASE_STYLES = {
    "INGESTION": "red",
    "ANALYZE": "green",
    "PLAN": "cyan",
    "DEBATE": "yellow",
    "REVISE": "cyan",
    "IMPLEMENT": "magenta1",
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
    """Collapsible panel widget for a single agent phase.

    Contains a Collapsible wrapping a RichLog. The Collapsible title shows
    agent name, model, and stats. On completion, collapses to show just
    the done summary.
    """

    DEFAULT_CSS = """
    AgentPanel {
        width: 100%;
        height: auto;
        min-height: 3;
        padding: 0 0;
        margin: 0 0;
    }
    AgentPanel:hover {
        background: $surface;
    }
    AgentPanel RichLog {
        height: auto;
        max-height: 20;
    }
    AgentPanel LoadingIndicator {
        height: 1;
    }
    AgentPanel .agent-description {
        color: $text-muted;
        text-style: italic;
        padding: 0 1;
    }
    AgentPanel CollapsibleTitle {
        width: 100%;
    }
    """

    def __init__(self, name: str, model: str, style: str = "cyan", description: str = "") -> None:
        super().__init__()
        self._panel_name = name
        self.model = model
        self.panel_style = style
        self.all_lines: list[str] = []
        self.start_time = time.monotonic()
        self.input_tokens = 0
        self.output_tokens = 0
        self.num_turns = 0
        self.done = False
        self.done_summary = ""
        self.error_msg = ""
        self._end_time: float | None = None
        # Resolve description: explicit > exact lookup > prefix lookup (e.g. "Critic/X" -> "Critic")
        if not description:
            description = AGENT_DESCRIPTIONS.get(
                name, AGENT_DESCRIPTIONS.get(name.split("/")[0], "")
            )
        self._description = description

    def compose(self) -> ComposeResult:
        with Collapsible(title=self._make_title(), collapsed=False):
            yield LoadingIndicator()
            if self._description:
                yield Static(self._description, classes="agent-description")
            yield RichLog(auto_scroll=True, markup=True, wrap=True)

    def on_mount(self) -> None:
        self._refresh_timer = self.set_interval(1, self._tick)

    def _tick(self) -> None:
        """Update the Collapsible title with elapsed time. Stops after done."""
        if self.done and hasattr(self, "_refresh_timer"):
            self._refresh_timer.stop()
            return
        self._update_title()

    def on_resize(self, event) -> None:
        """Re-render RichLog content at new width."""
        if not self.all_lines:
            return
        try:
            rich_log = self.query_one(RichLog)
        except NoMatches:
            return
        rich_log.clear()
        for line in self.all_lines:
            rich_log.write(Text(line), expand=True)

    def _make_title(self) -> str:
        """Build the Collapsible title string."""
        footer = self._build_footer()
        return f"[{self.panel_style}]{self._panel_name} ({self.model}) | {footer}[/]"

    def _update_title(self) -> None:
        """Update the Collapsible title in the DOM."""
        try:
            collapsible = self.query_one(Collapsible)
        except NoMatches:
            return
        collapsible.title = self._make_title()

    @property
    def panel_name(self) -> str:
        return self._panel_name

    @property
    def lines(self) -> list[str]:
        """Backward-compatible property. Returns all_lines."""
        return self.all_lines

    def add_line(self, text: str) -> None:
        """Append a summary line. Thread-safe: routes DOM update to UI thread."""
        if self.done:
            return
        cleaned = " ".join(text.split())
        self.all_lines.append(cleaned)
        try:
            app = self.app
        except Exception:
            return
        if app._thread_id == threading.get_ident():
            self._write_to_richlog(cleaned)
        else:
            app.call_from_thread(self._write_to_richlog, cleaned)

    def _write_to_richlog(self, text: str) -> None:
        """Write a line to the RichLog widget (must be called from UI thread)."""
        if len(self.all_lines) == 1:
            for indicator in self.query(LoadingIndicator):
                indicator.remove()
            for desc in self.query(".agent-description"):
                desc.remove()
        try:
            rich_log = self.query_one(RichLog)
        except NoMatches:
            return
        rich_log.write(Text(text), expand=True)

    def complete(self, done_summary: str = "") -> None:
        """Mark this panel as done.

        If done_summary is empty and the panel has lines, the last line
        is used as the done summary. Sets metadata immediately (thread-safe),
        defers DOM updates to the UI thread.
        """
        if self.done:
            return
        self.done = True
        if done_summary:
            self.done_summary = done_summary
        elif self.all_lines:
            self.done_summary = self.all_lines[-1]
        self._end_time = time.monotonic()

    def _apply_complete_dom(self) -> None:
        """Apply completion state to DOM (must be called from UI thread)."""
        for indicator in self.query(LoadingIndicator):
            indicator.remove()
        for desc in self.query(".agent-description"):
            desc.remove()
        try:
            collapsible = self.query_one(Collapsible)
        except NoMatches:
            return
        summary = self.done_summary
        if summary.startswith("[done] "):
            summary = summary[len("[done] "):]
        collapsible.title = (
            f"[{self.panel_style}]{self._panel_name}: {summary} | {self._build_footer()}[/]"
        )
        collapsible.collapsed = True

    def error(self, msg: str) -> None:
        """Mark this panel as errored. Thread-safe: routes DOM update to UI thread."""
        if self.done:
            return
        self.done = True
        self.error_msg = msg
        self._end_time = time.monotonic()
        try:
            app = self.app
        except Exception:
            return
        if app._thread_id == threading.get_ident():
            self._apply_error_dom(msg)
        else:
            app.call_from_thread(self._apply_error_dom, msg)

    def _apply_error_dom(self, msg: str) -> None:
        """Apply error state to DOM (must be called from UI thread)."""
        for indicator in self.query(LoadingIndicator):
            indicator.remove()
        for desc in self.query(".agent-description"):
            desc.remove()
        try:
            collapsible = self.query_one(Collapsible)
            rich_log = self.query_one(RichLog)
        except NoMatches:
            return
        collapsible.title = (
            f"[{self.panel_style}]{self._panel_name}:[/] [red][error] {msg}[/red] | {self._build_footer()}"
        )
        rich_log.write(Text(f"[error] {msg}", style="red"))

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
            parts.append(
                f"{self.input_tokens:,} in / {self.output_tokens:,} out"
            )
        if self.num_turns:
            parts.append(
                f"{self.num_turns} "
                f"{'turn' if self.num_turns == 1 else 'turns'}"
            )
        return " | ".join(parts)


# ---------------------------------------------------------------------------
# MetricsBar widget
# ---------------------------------------------------------------------------


class MetricsBar(Widget):
    """Persistent metrics bar showing iteration, phase, scores, tokens."""

    DEFAULT_CSS = """
    MetricsBar {
        dock: top;
        height: 1;
        background: $surface;
        padding: 0 1;
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
        self.scores: list[float] = []
        self.finished: bool = False
        self._end_time: float | None = None

    def on_mount(self) -> None:
        self.set_interval(1, self.refresh)

    def set_status(
        self,
        iteration: int | None = None,
        phase: str | None = None,
        best_version: str | None = None,
        best_score: int | None = None,
    ) -> None:
        """Update metrics bar fields. Only non-None values are changed."""
        if iteration is not None:
            self.iteration = iteration
        if phase is not None:
            self.phase = phase
        if best_version is not None:
            self.best_version = best_version
        if best_score is not None:
            self.best_score = best_score
        self.refresh()

    def finish(self) -> None:
        """Freeze the elapsed timer at the current value."""
        self.finished = True
        self._end_time = time.monotonic()

    def add_agent_stats(self, panel: "AgentPanel") -> None:
        """Accumulate a completed agent's stats into the running totals."""
        self.total_input_tokens += panel.input_tokens
        self.total_output_tokens += panel.output_tokens
        self.total_turns += panel.num_turns

    def render(self) -> Text:
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
            tokens = (
                f"{self.total_input_tokens:,} in"
                f" / {self.total_output_tokens:,} out"
            )
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
            line.append(
                f"  best: iter {best_iter} ({self.best_score})",
                style=style,
            )

        if self.scores:
            blocks = " \u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"
            max_s = max(self.scores) or 1
            spark = ""
            for s in self.scores[-20:]:
                idx = min(int(s / max_s * 8), 8)
                spark += blocks[idx]
            line.append(f"  [{spark}]", style="dim")

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
        transition: border 300ms in_out_cubic;
    }
    """

    def __init__(self, iter_title: str) -> None:
        super().__init__()
        self.border_title = iter_title

    def set_result(self, text: str, style: str) -> None:
        """Set the iteration result as border subtitle."""
        self.border_subtitle = text
        valid = {
            "red", "green", "yellow", "blue", "cyan", "magenta", "white",
        }
        if style in valid:
            self.styles.border = ("solid", style)


# ---------------------------------------------------------------------------
# AgentDetailScreen
# ---------------------------------------------------------------------------


class AgentDetailScreen(ModalScreen):
    """Full-screen view of one agent's complete output."""

    DEFAULT_CSS = """
    AgentDetailScreen {
        align: center middle;
    }
    AgentDetailScreen > Vertical {
        width: 90%;
        height: 90%;
        border: solid $accent;
        background: $surface;
    }
    AgentDetailScreen > Vertical > RichLog {
        height: 1fr;
    }
    AgentDetailScreen > Vertical > Static {
        height: auto;
        padding: 0 1;
        background: $primary-background;
    }
    """

    BINDINGS = [
        Binding("escape", "dismiss", "Back", show=True),
    ]

    def __init__(
        self,
        panel_name: str,
        model: str,
        stats: str,
        lines: list[str],
    ) -> None:
        super().__init__()
        self._panel_name = panel_name
        self._model = model
        self._stats = stats
        self._lines = lines

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(
                f"[bold]{self._panel_name}[/bold] ({self._model})"
                f" | {self._stats}"
            )
            yield RichLog(auto_scroll=False, markup=True, wrap=True)

    def on_mount(self) -> None:
        rich_log = self.query_one(RichLog)
        for line in self._lines:
            rich_log.write(Text(line))

    def action_dismiss(self) -> None:
        self.app.pop_screen()


# ---------------------------------------------------------------------------
# QuitConfirmScreen
# ---------------------------------------------------------------------------


class QuitConfirmScreen(ModalScreen[bool]):
    """Modal confirmation dialog for quitting while pipeline runs."""

    DEFAULT_CSS = """
    QuitConfirmScreen {
        align: center middle;
    }
    QuitConfirmScreen > Vertical {
        width: 50;
        height: auto;
        border: solid $error;
        background: $surface;
        padding: 1 2;
    }
    QuitConfirmScreen > Vertical > Label {
        width: 100%;
        text-align: center;
        margin-bottom: 1;
    }
    QuitConfirmScreen > Vertical > Center {
        height: auto;
    }
    """

    BINDINGS = [
        Binding("y", "yes", "Yes", show=True),
        Binding("n", "no", "No", show=True),
        Binding("escape", "no", show=False),
    ]

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("Pipeline is still running. Quit anyway?")
            with Center():
                yield Button("Yes", variant="error", id="yes-btn")
                yield Button("No", variant="primary", id="no-btn")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "yes-btn")

    def action_yes(self) -> None:
        self.dismiss(True)

    def action_no(self) -> None:
        self.dismiss(False)


# ---------------------------------------------------------------------------
# PipelineCommandProvider
# ---------------------------------------------------------------------------


class PipelineCommandProvider(Provider):
    """Command palette provider for pipeline navigation and control."""

    async def search(self, query: str) -> Hits:
        matcher = self.matcher(query)
        app = self.app
        if not isinstance(app, PipelineApp):
            return

        # Static commands
        commands = [
            ("Expand all panels", app.action_toggle_expand),
            ("Collapse all panels", app.action_toggle_expand),
            ("Go to top", partial(app._scroll_to, "top")),
            ("Go to bottom", partial(app._scroll_to, "bottom")),
            ("Quit", app.action_quit),
        ]

        # Theme switching
        for theme_name in sorted(app.available_themes):
            commands.append((
                f"Switch theme: {theme_name}",
                partial(app._set_theme, theme_name),
            ))

        # Pipeline control
        if hasattr(app._orchestrator, "pause_requested"):
            commands.append((
                "Pause after current iteration",
                partial(app._set_orchestrator_flag, "pause_requested"),
            ))
        if hasattr(app._orchestrator, "skip_to_report"):
            commands.append((
                "Skip to report",
                partial(app._set_orchestrator_flag, "skip_to_report"),
            ))

        # Dynamic: go to iteration N
        for container in app.query(IterationContainer):
            title = container.border_title or "?"
            commands.append((
                f"Go to {title}",
                partial(app._scroll_to_widget, container),
            ))

        # Dynamic: view agent details
        for panel in app.query(AgentPanel):
            commands.append((
                f"View {panel.panel_name} details",
                partial(app._open_agent_detail, panel),
            ))

        # Open experiment directory (macOS)
        if app._orchestrator and hasattr(app._orchestrator, "output_dir"):
            commands.append((
                "Open experiment directory",
                partial(
                    app._open_directory,
                    app._orchestrator.output_dir,
                ),
            ))

        for label, callback in commands:
            score = matcher.match(label)
            if score > 0:
                yield Hit(
                    score, matcher.highlight(label), callback,
                )


# ---------------------------------------------------------------------------
# PipelineLive bridge
# ---------------------------------------------------------------------------


class PipelineLive:
    """Bridge between the orchestrator (worker thread) and the Textual app.

    In app mode (_app is set): mounts widgets via call_from_thread.
    In headless mode (_app is None): tracks state only, no rendering.
    """

    def __init__(self) -> None:
        self._panels: list[AgentPanel] = []
        self._app: PipelineApp | None = None
        self._current_iteration: IterationContainer | None = None
        self._file_console: Console | None = None
        self._file_handle = None

    def start(self, log_path: Path | None = None) -> None:
        """Open the optional log file."""
        if log_path:
            self._file_handle = log_path.open("a")
            self._file_console = Console(
                file=self._file_handle, no_color=True, width=120,
            )

    def stop(self) -> None:
        """Close the log file."""
        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None
            self._file_console = None

    def add_panel(self, panel: AgentPanel) -> None:
        """Track a panel and mount it in the app if running."""
        self._panels.append(panel)
        if self._app is not None:
            self._app.call_from_thread(self._app._mount_panel, panel)

    def collapse_panel(
        self, panel: AgentPanel, done_summary: str = "",
    ) -> None:
        """Mark a panel as complete and accumulate stats."""
        panel.complete(done_summary)
        if self._app is not None:
            self._app.call_from_thread(
                self._app._do_panel_collapse, panel,
            )
        if self._file_console is not None:
            self._file_console.print(
                f"[{panel.panel_name}] "
                f"{panel.done_summary} ({panel._build_footer()})"
            )

    def start_iteration(self, title: int | str) -> None:
        """Begin an iteration container."""
        iter_title = (
            f"Iteration {title}" if isinstance(title, int) else title
        )
        container = IterationContainer(iter_title=iter_title)
        self._current_iteration = container
        if self._app is not None:
            self._app.call_from_thread(
                self._app._mount_iteration, container,
            )
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
                label = (
                    f"{self._current_iteration.border_title}: {subtitle}"
                )
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
        """Update the metrics bar fields."""
        if self._app is not None:
            self._app.call_from_thread(
                self._app._on_status_update, **kwargs,
            )

    def log(self, message: str) -> None:
        """Write a message to the log file only (no terminal output)."""
        if self._file_console is not None:
            self._file_console.print(message)

    def print_static(self, renderable: RenderableType) -> None:
        """Print a renderable. In app mode, mount as Static widget."""
        if self._app is not None:
            self._app.call_from_thread(
                self._app._mount_static, renderable,
            )
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
        """No-op. PipelineApp handles dismiss via key binding."""

    def refresh(self) -> None:
        """No-op. Widget refresh is handled automatically."""


# ---------------------------------------------------------------------------
# PipelineApp
# ---------------------------------------------------------------------------


class PipelineApp(App):
    """Textual app that runs the orchestrator and displays the dashboard."""

    COMMANDS = App.COMMANDS | {PipelineCommandProvider}

    BINDINGS = [
        Binding("ctrl+o", "toggle_expand", "Expand/Collapse", show=True),
        Binding("ctrl+q", "quit", "Quit", show=True),
        Binding("ctrl+t", "cycle_theme", "Theme", show=True),
        Binding(
            "enter", "open_focused_detail", "Detail", show=False,
        ),
    ]

    DEFAULT_CSS = """
    #main-scroll {
        height: 1fr;
    }
    #main-scroll > Static {
        width: 100%;
    }
    """

    def __init__(self, orchestrator) -> None:
        super().__init__()
        self._orchestrator = orchestrator
        self._finished: bool = False
        self._live: PipelineLive = PipelineLive()
        self._worker_loop: asyncio.AbstractEventLoop | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield MetricsBar()
        yield VerticalScroll(id="main-scroll")
        yield Footer()

    def on_mount(self) -> None:
        saved_theme = _load_prefs().get("theme")
        if saved_theme and saved_theme in self.available_themes:
            self.theme = saved_theme
        self.title = "Auto-Scientist"
        self._live._app = self
        self._orchestrator._live = self._live
        self.run_worker(
            self._run_pipeline, thread=True, exit_on_error=False,
        )

    def _run_pipeline(self) -> None:
        """Run the async orchestrator in its own event loop (worker thread)."""
        loop = asyncio.new_event_loop()
        self._worker_loop = loop
        try:
            loop.run_until_complete(self._orchestrator.run())
        except asyncio.CancelledError:
            pass
        finally:
            loop.close()

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if event.state in (WorkerState.SUCCESS, WorkerState.ERROR):
            self._finished = True
            try:
                bar = self.query_one(MetricsBar)
            except NoMatches:
                bar = None
            if bar is not None:
                bar.finish()

            if event.state == WorkerState.ERROR:
                self.notify(
                    "Pipeline failed! Check logs.",
                    severity="error",
                    timeout=0,
                )
            else:
                self.notify(
                    "Pipeline complete!",
                    severity="information",
                    timeout=0,
                )

    # -- Callbacks from PipelineLive (called via call_from_thread) --

    def _do_panel_collapse(self, panel: AgentPanel) -> None:
        """Apply panel DOM collapse and handle post-collapse UI (runs on UI thread)."""
        panel._apply_complete_dom()
        self._on_panel_collapsed(panel)

    def _on_panel_collapsed(self, panel: AgentPanel) -> None:
        """Handle panel completion: accumulate stats, fire toast."""
        try:
            bar = self.query_one(MetricsBar)
        except NoMatches:
            bar = None
        if bar is not None:
            bar.add_agent_stats(panel)
        elapsed = _format_elapsed(panel.elapsed)
        self.notify(f"{panel.panel_name} complete ({elapsed})")

    def _on_status_update(self, **kwargs) -> None:
        """Handle status update: update MetricsBar."""
        try:
            bar = self.query_one(MetricsBar)
        except NoMatches:
            return
        bar.set_status(**kwargs)

    # -- Scroll helpers --

    def _is_near_bottom(self) -> bool:
        """Check if the scroll view is at or near the bottom."""
        scroll = self.query_one("#main-scroll", VerticalScroll)
        return scroll.scroll_offset.y >= scroll.max_scroll_y - 2

    def _scroll_to_end(self) -> None:
        """Scroll to the bottom of the main scroll area."""
        self.call_after_refresh(
            self.query_one("#main-scroll").scroll_end, animate=False,
        )

    # -- Key binding actions --

    def action_toggle_expand(self) -> None:
        """Toggle expanded state on all AgentPanel Collapsibles."""
        collapsibles = list(self.query("AgentPanel Collapsible"))
        if not collapsibles:
            return
        new_state = not collapsibles[0].collapsed
        for c in collapsibles:
            c.collapsed = new_state
        self._scroll_to_end()

    def action_quit(self) -> None:
        """Quit with confirmation if pipeline is still running."""
        if not self._finished:
            self.push_screen(
                QuitConfirmScreen(), callback=self._handle_quit_confirm,
            )
        else:
            self.exit()

    def _handle_quit_confirm(self, confirmed: bool | None) -> None:
        """Handle quit confirmation result."""
        if confirmed:
            self._force_quit()

    def _force_quit(self) -> None:
        """Cancel the pipeline and exit."""
        if self._worker_loop is not None and self._worker_loop.is_running():
            self._worker_loop.call_soon_threadsafe(self._cancel_all_tasks)
        self.exit()

    def _cancel_all_tasks(self) -> None:
        """Cancel all tasks on the worker event loop."""
        if self._worker_loop:
            for task in asyncio.all_tasks(self._worker_loop):
                task.cancel()

    def _persist_theme(self, theme_name: str) -> None:
        """Save the selected theme to user preferences."""
        prefs = _load_prefs()
        prefs["theme"] = theme_name
        _save_prefs(prefs)

    def action_cycle_theme(self) -> None:
        """Cycle through available themes."""
        themes = sorted(self.available_themes)
        if not themes:
            return
        current = self.theme or themes[0]
        try:
            idx = themes.index(current)
            self.theme = themes[(idx + 1) % len(themes)]
        except ValueError:
            self.theme = themes[0]
        self._persist_theme(self.theme)
        self.notify(f"Theme: {self.theme}")

    def action_open_focused_detail(self) -> None:
        """Open detail view for the currently focused AgentPanel."""
        focused = self.focused
        if focused is None:
            return
        panel = None
        widget = focused
        while widget is not None:
            if isinstance(widget, AgentPanel):
                panel = widget
                break
            widget = widget.parent
        if panel is not None:
            self._open_agent_detail(panel)

    # -- Command palette helpers --

    def _open_agent_detail(self, panel: AgentPanel) -> None:
        """Push the AgentDetailScreen for a panel."""
        self.push_screen(AgentDetailScreen(
            panel_name=panel.panel_name,
            model=panel.model,
            stats=panel._build_footer(),
            lines=list(panel.all_lines),
        ))

    def _scroll_to(self, direction: str) -> None:
        """Scroll the main area to top or bottom."""
        scroll = self.query_one("#main-scroll", VerticalScroll)
        if direction == "top":
            scroll.scroll_home(animate=False)
        else:
            scroll.scroll_end(animate=False)

    def _scroll_to_widget(self, widget: Widget) -> None:
        """Scroll to make a widget visible."""
        widget.scroll_visible(animate=False)

    def _set_theme(self, theme_name: str) -> None:
        """Switch to a named theme."""
        self.theme = theme_name
        self._persist_theme(theme_name)
        self.notify(f"Theme: {theme_name}")

    def _set_orchestrator_flag(self, flag_name: str) -> None:
        """Set a boolean flag on the orchestrator."""
        setattr(self._orchestrator, flag_name, True)
        self.notify(f"{flag_name.replace('_', ' ').title()} requested")

    def _open_directory(self, path: Path) -> None:
        """Open a directory in the system file manager."""
        try:
            subprocess.Popen(["open", str(path)])
        except FileNotFoundError:
            self.notify(f"Directory: {path}", timeout=10)

    # -- Mount helpers (called via call_from_thread from PipelineLive) --

    def _mount_panel(self, panel: AgentPanel) -> None:
        """Mount a panel into the current iteration container or scroll."""
        near_bottom = self._is_near_bottom()
        target = (
            self._live._current_iteration
            or self.query_one("#main-scroll")
        )
        target.mount(panel)
        if near_bottom:
            self._scroll_to_end()

    def _mount_iteration(self, container: IterationContainer) -> None:
        """Mount an iteration container into the main scroll."""
        near_bottom = self._is_near_bottom()
        self.query_one("#main-scroll").mount(container)
        if near_bottom:
            self._scroll_to_end()

    def _mount_static(self, renderable: RenderableType) -> None:
        """Mount a Rich renderable as a Static widget."""
        near_bottom = self._is_near_bottom()
        self.query_one("#main-scroll").mount(Static(renderable))
        if near_bottom:
            self._scroll_to_end()
