"""PipelineApp: Textual App with multi-screen support, command palette, and tab management."""

import asyncio
import subprocess
from pathlib import Path

from rich.console import RenderableType
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.css.query import NoMatches
from textual.widget import Widget
from textual.widgets import (
    Collapsible,
    Footer,
    Header,
    Static,
)
from textual.worker import Worker, WorkerState

from auto_scientist.experiment_store import (
    ExperimentStore,
    FilesystemStore,
    next_output_dir,
)
from auto_scientist.ui.bridge import PipelineLive
from auto_scientist.ui.commands import PipelineCommandProvider
from auto_scientist.ui.config_form import ConfigForm
from auto_scientist.ui.detail_screen import AgentDetailScreen, QuitConfirmScreen
from auto_scientist.ui.styles import _format_elapsed, _load_prefs, _save_prefs
from auto_scientist.ui.widgets import AgentPanel, IterationContainer, MetricsBar


class PipelineApp(App):
    """Textual app with HomeScreen or direct ExperimentScreen mode.

    When orchestrator is None: shows HomeScreen with presets, config form, past runs.
    When orchestrator is provided: shows single ExperimentScreen (backward compatible).
    """

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

    def __init__(
        self,
        orchestrator=None,
        store: ExperimentStore | None = None,
    ) -> None:
        super().__init__()
        saved_theme = _load_prefs().get("theme")
        if saved_theme and saved_theme in self.available_themes:
            self.theme = saved_theme
        self._orchestrator = orchestrator
        self._finished: bool = False
        self._live: PipelineLive = PipelineLive()
        self._worker_loop: asyncio.AbstractEventLoop | None = None
        self._store = store or FilesystemStore(Path("experiments"))
        self._experiment_screens: dict[str, object] = {}

    def compose(self) -> ComposeResult:
        yield Header()
        if self._orchestrator is not None:
            # Legacy single-experiment mode
            yield MetricsBar()
            yield VerticalScroll(id="main-scroll")
        yield Footer()

    def on_mount(self) -> None:
        self.title = "Auto-Scientist"
        if self._orchestrator is not None:
            # Legacy mode: run single experiment directly
            self._live._app = self
            self._orchestrator._live = self._live
            self.run_worker(
                self._run_pipeline, thread=True, exit_on_error=False,
            )
        else:
            # Multi-experiment mode: show HomeScreen
            from auto_scientist.ui.home_screen import HomeScreen

            self.push_screen(HomeScreen(store=self._store))

    def on_config_form_launch_requested(self, event: ConfigForm.LaunchRequested) -> None:
        """Handle launch from HomeScreen config form."""
        # Create orchestrator with the config from the form
        # The actual Orchestrator creation needs state + output_dir
        from auto_scientist.orchestrator import Orchestrator
        from auto_scientist.state import ExperimentState
        from auto_scientist.ui.experiment_screen import ExperimentScreen

        output_dir = next_output_dir(Path("experiments"))
        output_dir.mkdir(parents=True, exist_ok=True)

        state = ExperimentState(
            domain="auto",
            goal=event.goal,
            data_path=event.data_path or None,
            ingestion_source=event.ingestion_source,
        )

        orchestrator = Orchestrator(
            state=state,
            data_path=event.data_path,
            output_dir=output_dir,
            model_config=event.model_config,
            max_iterations=event.max_iterations,
            debate_rounds=event.debate_rounds,
        )

        label = f"{event.goal[:20]}"
        screen = ExperimentScreen(
            orchestrator=orchestrator,
            experiment_label=label,
        )
        self.push_screen(screen)

    # -- Legacy single-experiment mode methods --

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
        near_bottom = self._is_near_bottom()
        panel._apply_complete_dom()
        self._on_panel_collapsed(panel)
        if near_bottom:
            self._scroll_to_end()

    def _on_panel_collapsed(self, panel: AgentPanel) -> None:
        try:
            bar = self.query_one(MetricsBar)
        except NoMatches:
            bar = None
        if bar is not None:
            bar.add_agent_stats(panel)
        elapsed = _format_elapsed(panel.elapsed)
        self.notify(f"{panel.panel_name} complete ({elapsed})")

    def _on_status_update(self, **kwargs) -> None:
        try:
            bar = self.query_one(MetricsBar)
        except NoMatches:
            return
        bar.set_status(**kwargs)

    # -- Scroll helpers --

    def _is_near_bottom(self) -> bool:
        try:
            scroll = self.query_one("#main-scroll", VerticalScroll)
            return scroll.scroll_offset.y >= scroll.max_scroll_y - 2
        except NoMatches:
            return False

    def _scroll_to_end(self) -> None:
        try:
            scroll = self.query_one("#main-scroll")
            self.call_after_refresh(scroll.scroll_end, animate=False)
        except NoMatches:
            pass

    # -- Key binding actions --

    def action_toggle_expand(self) -> None:
        panels = list(self.query(AgentPanel))
        if not panels:
            return
        first_collapsible = panels[0].query_one(Collapsible)
        collapsing = not first_collapsible.collapsed
        for panel in panels:
            if collapsing and not panel.done:
                continue
            try:
                c = panel.query_one(Collapsible)
            except NoMatches:
                continue
            c.collapsed = collapsing
        self._scroll_to_end()

    def action_quit(self) -> None:
        if self._orchestrator is not None and not self._finished:
            self.push_screen(
                QuitConfirmScreen(), callback=self._handle_quit_confirm,
            )
        else:
            self.exit()

    def _handle_quit_confirm(self, confirmed: bool | None) -> None:
        if confirmed:
            self._force_quit()

    def _force_quit(self) -> None:
        if self._worker_loop is not None and self._worker_loop.is_running():
            self._worker_loop.call_soon_threadsafe(self._cancel_all_tasks)
        self.exit()

    def _cancel_all_tasks(self) -> None:
        if self._worker_loop:
            for task in asyncio.all_tasks(self._worker_loop):
                task.cancel()

    def _persist_theme(self, theme_name: str) -> None:
        prefs = _load_prefs()
        prefs["theme"] = theme_name
        _save_prefs(prefs)

    def action_cycle_theme(self) -> None:
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
        self.push_screen(AgentDetailScreen(
            panel_name=panel.panel_name,
            model=panel.model,
            stats=panel._build_footer(),
            lines=list(panel.all_lines),
        ))

    def _scroll_to(self, direction: str) -> None:
        try:
            scroll = self.query_one("#main-scroll", VerticalScroll)
        except NoMatches:
            return
        if direction == "top":
            scroll.scroll_home(animate=False)
        else:
            scroll.scroll_end(animate=False)

    def _scroll_to_widget(self, widget: Widget) -> None:
        widget.scroll_visible(animate=False)

    def _set_theme(self, theme_name: str) -> None:
        self.theme = theme_name
        self._persist_theme(theme_name)
        self.notify(f"Theme: {theme_name}")

    def _set_orchestrator_flag(self, flag_name: str) -> None:
        if self._orchestrator:
            setattr(self._orchestrator, flag_name, True)
            self.notify(f"{flag_name.replace('_', ' ').title()} requested")

    def _open_directory(self, path: Path) -> None:
        try:
            subprocess.Popen(["open", str(path)])
        except FileNotFoundError:
            self.notify(f"Directory: {path}", timeout=10)

    # -- Mount helpers (called via call_from_thread from PipelineLive) --

    def _mount_panel(self, panel: AgentPanel) -> None:
        near_bottom = self._is_near_bottom()
        target = (
            self._live._current_iteration
            or self.query_one("#main-scroll")
        )
        target.mount(panel)
        if near_bottom:
            self._scroll_to_end()

    def _mount_iteration(self, container: IterationContainer) -> None:
        near_bottom = self._is_near_bottom()
        self.query_one("#main-scroll").mount(container)
        if near_bottom:
            self._scroll_to_end()

    def _mount_static(self, renderable: RenderableType) -> None:
        near_bottom = self._is_near_bottom()
        self.query_one("#main-scroll").mount(Static(renderable))
        if near_bottom:
            self._scroll_to_end()
