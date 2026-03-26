"""ExperimentScreen: standalone Screen for running one experiment dashboard."""

from __future__ import annotations

import asyncio

from rich.console import RenderableType
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.css.query import NoMatches
from textual.screen import Screen
from textual.widgets import (
    Collapsible,
    Footer,
    Header,
    Static,
)
from textual.worker import Worker, WorkerState

from auto_scientist.ui.bridge import PipelineLive
from auto_scientist.ui.detail_screen import AgentDetailScreen
from auto_scientist.ui.styles import _format_elapsed
from auto_scientist.ui.widgets import AgentPanel, IterationContainer, MetricsBar


class ExperimentScreen(Screen):
    """Dashboard screen for a single experiment.

    In live mode (orchestrator provided): runs the orchestrator in a worker
    thread and displays real-time agent output.
    In read-only mode: displays historical data from a completed run.
    """

    BINDINGS = [
        Binding("ctrl+o", "toggle_expand", "Expand/Collapse", show=True),
        Binding(
            "enter", "open_focused_detail", "Detail", show=False,
        ),
    ]

    DEFAULT_CSS = """
    ExperimentScreen #exp-scroll {
        height: 1fr;
    }
    ExperimentScreen #exp-scroll > Static {
        width: 100%;
    }
    """

    def __init__(
        self,
        orchestrator=None,
        read_only: bool = False,
        experiment_label: str = "Experiment",
    ) -> None:
        super().__init__()
        self._orchestrator = orchestrator
        self._read_only = read_only
        self._finished: bool = read_only
        self._live: PipelineLive = PipelineLive()
        self._worker_loop: asyncio.AbstractEventLoop | None = None
        self._experiment_label = experiment_label

    def compose(self) -> ComposeResult:
        yield Header()
        yield MetricsBar()
        yield VerticalScroll(id="exp-scroll")
        yield Footer()

    def on_mount(self) -> None:
        if self._read_only or self._orchestrator is None:
            return
        self._live._app = self
        self._orchestrator._live = self._live
        self.app.run_worker(
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
                self.app.notify(
                    "Pipeline failed! Check logs.",
                    severity="error",
                    timeout=0,
                )
            else:
                self.app.notify(
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
        self.app.notify(f"{panel.panel_name} complete ({elapsed})")

    def _on_status_update(self, **kwargs) -> None:
        try:
            bar = self.query_one(MetricsBar)
        except NoMatches:
            return
        bar.set_status(**kwargs)

    # -- Scroll helpers --

    def _is_near_bottom(self) -> bool:
        scroll = self.query_one("#exp-scroll", VerticalScroll)
        return scroll.scroll_offset.y >= scroll.max_scroll_y - 2

    def _scroll_to_end(self) -> None:
        self.app.call_after_refresh(
            self.query_one("#exp-scroll").scroll_end, animate=False,
        )

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

    def action_open_focused_detail(self) -> None:
        focused = self.app.focused
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
            self.app.push_screen(AgentDetailScreen(
                panel_name=panel.panel_name,
                model=panel.model,
                stats=panel._build_footer(),
                lines=list(panel.all_lines),
            ))

    # -- Mount helpers (called via call_from_thread from PipelineLive) --

    def _mount_panel(self, panel: AgentPanel) -> None:
        near_bottom = self._is_near_bottom()
        target = (
            self._live._current_iteration
            or self.query_one("#exp-scroll")
        )
        target.mount(panel)
        if near_bottom:
            self._scroll_to_end()

    def _mount_iteration(self, container: IterationContainer) -> None:
        near_bottom = self._is_near_bottom()
        self.query_one("#exp-scroll").mount(container)
        if near_bottom:
            self._scroll_to_end()

    def _mount_static(self, renderable: RenderableType) -> None:
        near_bottom = self._is_near_bottom()
        self.query_one("#exp-scroll").mount(Static(renderable))
        if near_bottom:
            self._scroll_to_end()

    # -- Lifecycle helpers --

    def cancel_pipeline(self) -> None:
        """Cancel the running pipeline (called on quit)."""
        if self._worker_loop is not None and self._worker_loop.is_running():
            self._worker_loop.call_soon_threadsafe(self._cancel_all_tasks)

    def _cancel_all_tasks(self) -> None:
        if self._worker_loop:
            for task in asyncio.all_tasks(self._worker_loop):
                task.cancel()
