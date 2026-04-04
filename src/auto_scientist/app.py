"""Textual application classes, screens, and command palette.

Provides:
- AgentDetailScreen: Full-screen view of one agent's output
- QuitConfirmScreen: Modal confirmation dialog for quit
- PipelineCommandProvider: Command palette with navigation and control
- PipelineApp: Textual App with screens, command palette, and message handlers
- ShowApp: Read-only viewer for a completed run's TUI panels
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import subprocess
from collections.abc import Callable
from functools import partial
from pathlib import Path

from rich.console import RenderableType
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.command import Hit, Hits, Provider
from textual.containers import Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.screen import ModalScreen
from textual.widget import Widget
from textual.widgets import (
    Collapsible,
    Footer,
    Header,
    Label,
    RichLog,
    Static,
)
from textual.worker import Worker, WorkerState

from auto_scientist.pipeline_live import PipelineLive
from auto_scientist.preferences import load_theme, save_theme
from auto_scientist.widgets import (
    AgentPanel,
    IterationContainer,
    MetricsBar,
    _format_elapsed,
)

logger = logging.getLogger(__name__)


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
            yield Static(f"[bold]{self._panel_name}[/bold] ({self._model}) | {self._stats}")
            yield RichLog(auto_scroll=False, markup=True, wrap=True)

    def on_mount(self) -> None:
        rich_log = self.query_one(RichLog)
        for line in self._lines:
            rich_log.write(Text(line))

    async def action_dismiss(self, result: object = None) -> None:
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
        border: round $error;
        background: $surface;
        padding: 1 2;
    }
    QuitConfirmScreen > Vertical > Label {
        width: 100%;
        text-align: center;
        margin-bottom: 1;
    }
    QuitConfirmScreen > Vertical > Static {
        width: 100%;
        text-align: center;
        color: $text-muted;
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
            yield Static("y/n")

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
        commands: list[tuple[str, Callable[[], object]]] = [
            ("Expand all panels", app.action_toggle_expand),
            ("Collapse all panels", app.action_toggle_expand),
            ("Go to top", partial(app._scroll_to, "top")),
            ("Go to bottom", partial(app._scroll_to, "bottom")),
            ("Quit", app.action_quit),
        ]

        # Theme switching
        for theme_name in sorted(app.available_themes):
            commands.append(
                (
                    f"Switch theme: {theme_name}",
                    partial(app._set_theme, theme_name),
                )
            )

        # Pipeline control
        if hasattr(app._orchestrator, "pause_requested"):
            commands.append(
                (
                    "Pause after current iteration",
                    partial(app._set_orchestrator_flag, "pause_requested"),
                )
            )
        if hasattr(app._orchestrator, "skip_to_report"):
            commands.append(
                (
                    "Skip to report",
                    partial(app._set_orchestrator_flag, "skip_to_report"),
                )
            )

        # Dynamic: go to iteration N
        for container in app.query(IterationContainer):
            title = getattr(container, "_iter_title", container.border_title) or "?"
            commands.append(
                (
                    f"Go to {title}",
                    partial(app._scroll_to_widget, container),
                )
            )

        # Dynamic: view agent details
        for panel in app.query(AgentPanel):
            commands.append(
                (
                    f"View {panel.panel_name} details",
                    partial(app._open_agent_detail, panel),
                )
            )

        # Open experiment directory (macOS)
        if app._orchestrator and hasattr(app._orchestrator, "output_dir"):
            commands.append(
                (
                    "Open experiment directory",
                    partial(
                        app._open_directory,
                        app._orchestrator.output_dir,
                    ),
                )
            )

        for label, callback in commands:
            score = matcher.match(label)
            if score > 0:
                yield Hit(
                    score,
                    matcher.highlight(label),
                    callback,
                )


# ---------------------------------------------------------------------------
# PipelineApp
# ---------------------------------------------------------------------------


class PipelineApp(App):
    """Textual app that runs the orchestrator and displays the dashboard."""

    ALLOW_SELECT = True

    COMMANDS = App.COMMANDS | {PipelineCommandProvider}

    BINDINGS = [
        Binding("ctrl+o", "toggle_expand", "Expand/Collapse", show=True),
        Binding("ctrl+q", "quit", "Quit", show=True),
        Binding(
            "enter",
            "open_focused_detail",
            "Detail",
            show=False,
        ),
    ]

    DEFAULT_CSS = """
    #outer-container {
        height: 1fr;
        border: round grey;
        padding: 0 1;
    }
    #banner-area {
        height: auto;
    }
    #run-area {
        height: auto;
        border: round grey;
        transition: border 300ms in_out_cubic;
    }
    #run-area > Static {
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
        with VerticalScroll(id="outer-container") as outer:
            outer.border_title = "Auto-Scientist"
            yield Vertical(id="banner-area")
            with Vertical(id="run-area") as run:
                run.border_title = "Run"
                pass
        yield Footer()

    def on_mount(self) -> None:
        saved_theme = load_theme()
        if saved_theme in self.available_themes:
            self.theme = saved_theme
        self.title = "Auto-Scientist"
        self._live._app = self
        self._orchestrator._live = self._live
        self.run_worker(
            self._run_pipeline,
            thread=True,
            exit_on_error=False,
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

            run_area = self.query_one("#run-area")
            if event.state == WorkerState.ERROR:
                run_area.styles.border = ("round", "red")
                error = event.worker.error
                error_msg = f"{type(error).__name__}: {error}" if error else "Unknown error"
                self.notify(
                    "Pipeline failed! See error below.",
                    severity="error",
                    timeout=0,
                )
                self._mount_static(
                    Text(f"\n{error_msg}", style="red"),
                )
            else:
                run_area.styles.border = ("round", "green")
                self.notify(
                    "Pipeline complete!",
                    severity="information",
                    timeout=0,
                )

    # -- Callbacks from PipelineLive (called via call_from_thread) --

    def _do_panel_collapse(self, panel: AgentPanel) -> None:
        """Apply panel DOM collapse and handle post-collapse UI (runs on UI thread)."""
        near_bottom = self._is_near_bottom()
        panel._apply_complete_dom()
        self._on_panel_collapsed(panel)
        if near_bottom:
            self._scroll_to_end()

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
        scroll = self.query_one("#outer-container", VerticalScroll)
        return scroll.scroll_offset.y >= scroll.max_scroll_y - 2

    def _scroll_to_end(self) -> None:
        """Scroll to the bottom of the main scroll area."""
        self.call_after_refresh(
            self.query_one("#outer-container").scroll_end,
            animate=False,
        )

    # -- Key binding actions --

    def action_toggle_expand(self) -> None:
        """Toggle expanded state on all AgentPanel Collapsibles and IterationContainers.

        When collapsing, skip panels that are still running so their
        live output remains visible.
        """
        panels = list(self.query(AgentPanel))
        containers = list(self.query(IterationContainer))
        if not panels and not containers:
            return

        # Determine direction from first available panel
        collapsing = True
        if panels:
            try:
                first_collapsible = panels[0].query_one(Collapsible)
                collapsing = not first_collapsible.collapsed
            except NoMatches:
                pass

        was_near_bottom = self._is_near_bottom()

        # Toggle agent panels
        for panel in panels:
            if collapsing and not panel.done:
                continue
            try:
                c = panel.query_one(Collapsible)
            except NoMatches:
                continue
            # Suppress Textual's scroll_visible during batch toggle
            object.__setattr__(c, "scroll_visible", lambda *a, **kw: None)
            c.collapsed = collapsing
            object.__delattr__(c, "scroll_visible")

        # Toggle finished iteration containers
        for container in containers:
            if container._in_progress or not container._panels:
                continue
            if (collapsing and not container._is_collapsed) or (
                not collapsing and container._is_collapsed
            ):
                container.toggle_iteration()

        if was_near_bottom:
            self._scroll_to_end()

    async def action_quit(self) -> None:
        """Quit with confirmation if pipeline is still running."""
        if not self._finished:
            self.push_screen(
                QuitConfirmScreen(),
                callback=self._handle_quit_confirm,
            )
        else:
            self.exit()

    def _handle_quit_confirm(self, confirmed: bool | None) -> None:
        """Handle quit confirmation result."""
        if confirmed:
            self._force_quit()

    def _force_quit(self) -> None:
        """Cancel the pipeline and exit.

        Schedules cancellation of all asyncio tasks on the worker event loop,
        waits briefly for SDK transport cleanup (subprocess termination) to
        complete, then exits the Textual app.
        """
        if self._worker_loop is not None and self._worker_loop.is_running():
            self._worker_loop.call_soon_threadsafe(self._cancel_all_tasks)
            # Give the worker thread time to propagate cancellation through
            # the SDK transports, which terminate their child processes in
            # ``finally`` blocks.  Without this pause, Textual's exit can
            # tear down the process before cleanup completes, orphaning SDK
            # subprocesses (claude CLI, codex app-server).
            import time

            time.sleep(1.0)
        self.exit()

    def _cancel_all_tasks(self) -> None:
        """Cancel all tasks on the worker event loop."""
        if self._worker_loop:
            for task in asyncio.all_tasks(self._worker_loop):
                task.cancel()

    def watch_theme(self, theme_name: str) -> None:
        """Persist every theme change, regardless of how it was triggered.

        Catches changes from the custom command palette AND the built-in
        Textual ThemeProvider (which otherwise bypasses persistence).
        """
        save_theme(theme_name)

    def action_open_focused_detail(self) -> None:
        """Open detail view for the currently focused AgentPanel."""
        focused = self.focused
        if focused is None:
            return
        panel = None
        widget: Widget | None = focused
        while widget is not None:
            if isinstance(widget, AgentPanel):
                panel = widget
                break
            widget = widget.parent  # type: ignore[assignment]
        if panel is not None:
            self._open_agent_detail(panel)

    # -- Command palette helpers --

    def _open_agent_detail(self, panel: AgentPanel) -> None:
        """Push the AgentDetailScreen for a panel."""
        self.push_screen(
            AgentDetailScreen(
                panel_name=panel.panel_name,
                model=panel.model,
                stats=panel._build_footer(),
                lines=list(panel.all_lines),
            )
        )

    def _scroll_to(self, direction: str) -> None:
        """Scroll the main area to top or bottom."""
        scroll = self.query_one("#outer-container", VerticalScroll)
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
        self.notify(f"Theme: {theme_name}")

    def _set_orchestrator_flag(self, flag_name: str) -> None:
        """Set a boolean flag on the orchestrator."""
        setattr(self._orchestrator, flag_name, True)
        self.notify(f"{flag_name.replace('_', ' ').title()} requested")

    def _open_directory(self, path: Path) -> None:
        """Open a directory in the system file manager."""
        try:
            subprocess.Popen(["open", str(path)])
        except OSError:
            self.notify(f"Directory: {path}", timeout=10)

    # -- Mount helpers (called via call_from_thread from PipelineLive) --

    def _mount_panel(self, panel: AgentPanel) -> None:
        """Mount a panel into the current iteration container or scroll."""
        near_bottom = self._is_near_bottom()
        target = self._live._current_iteration or self.query_one("#run-area")
        target.mount(panel)
        if isinstance(target, IterationContainer):
            target.add_panel(panel)
        if near_bottom:
            self._scroll_to_end()

    def _mount_iteration(self, container: IterationContainer) -> None:
        """Mount an iteration container into the run area."""
        near_bottom = self._is_near_bottom()
        self.query_one("#run-area").mount(container)
        if near_bottom:
            self._scroll_to_end()

    def _mount_banner(self, renderable: RenderableType) -> None:
        """Mount the startup banner into the banner area."""
        self.query_one("#banner-area").mount(Static(renderable))

    def _mount_static(self, renderable: RenderableType) -> None:
        """Mount a Rich renderable as a Static widget."""
        near_bottom = self._is_near_bottom()
        self.query_one("#run-area").mount(Static(renderable))
        if near_bottom:
            self._scroll_to_end()


class ShowApp(App):
    """Read-only viewer for a completed run's TUI panels."""

    ALLOW_SELECT = True

    BINDINGS = [
        Binding("ctrl+o", "toggle_expand", "Expand/Collapse", show=True),
        Binding("ctrl+q", "quit", "Quit", show=True),
        Binding("enter", "open_focused_detail", "Detail", show=False),
    ]

    DEFAULT_CSS = PipelineApp.DEFAULT_CSS

    def __init__(self, manifest_records: list, run_title: str = "Run") -> None:
        super().__init__()
        self._manifest_records = manifest_records
        self._run_title = run_title

    def compose(self) -> ComposeResult:
        yield Header()
        with VerticalScroll(id="outer-container") as outer:
            outer.border_title = "Auto-Scientist"
            with Vertical(id="run-area") as run:
                run.border_title = self._run_title
                pass
        yield Footer()

    def on_mount(self) -> None:
        saved_theme = load_theme()
        if saved_theme in self.available_themes:
            self.theme = saved_theme
        self.title = f"Auto-Scientist — {self._run_title}"

        run_area = self.query_one("#run-area")
        for record in self._manifest_records:
            container = IterationContainer(iter_title=record.title)
            run_area.mount(container)
            for p in record.panels:
                panel = AgentPanel(name=p.name, model=p.model, style=p.style)
                container.mount(panel)
                container.add_panel(panel)
                panel.input_tokens = p.input_tokens
                panel.output_tokens = p.output_tokens
                panel.thinking_tokens = p.thinking_tokens
                panel.num_turns = p.num_turns
                for line in p.lines:
                    panel.all_lines.append(line)
                    panel._write_to_richlog(line)
                panel.complete(p.done_summary)
                panel._apply_complete_dom()
                panel._end_time = panel.start_time + p.elapsed_seconds
            container.set_result(record.result_text, record.result_style, record.summary)

        run_area.styles.border = ("round", "green")

    def action_toggle_expand(self) -> None:
        """Toggle expanded state on all panels and iteration containers."""
        panels = list(self.query(AgentPanel))
        containers = list(self.query(IterationContainer))
        if not panels and not containers:
            return
        collapsing = True
        if panels:
            with contextlib.suppress(NoMatches):
                collapsing = not panels[0].query_one(Collapsible).collapsed
        for panel in panels:
            try:
                c = panel.query_one(Collapsible)
            except NoMatches:
                continue
            object.__setattr__(c, "scroll_visible", lambda *a, **kw: None)
            c.collapsed = collapsing
            object.__delattr__(c, "scroll_visible")
        for container in containers:
            if not container._panels:
                continue
            if (collapsing and not container._is_collapsed) or (
                not collapsing and container._is_collapsed
            ):
                container.toggle_iteration()

    async def action_quit(self) -> None:
        self.exit()

    def action_open_focused_detail(self) -> None:
        focused = self.focused
        if focused is None:
            return
        widget: Widget | None = focused
        while widget is not None:
            if isinstance(widget, AgentPanel):
                self.push_screen(
                    AgentDetailScreen(
                        panel_name=widget.panel_name,
                        model=widget.model,
                        stats=widget._build_footer(),
                        lines=list(widget.all_lines),
                    )
                )
                return
            widget = widget.parent  # type: ignore[assignment]

    def watch_theme(self, theme_name: str) -> None:
        save_theme(theme_name)
