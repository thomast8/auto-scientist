"""Tests for Textual console components: AgentPanel, StatusBarWidget, PipelineLive, PipelineApp."""

import threading

import pytest
from textual.app import App, ComposeResult

from auto_scientist.console import (
    PANEL_MAX_LINES,
    AgentPanel,
    IterationContainer,
    PipelineApp,
    PipelineLive,
    StatusBarWidget,
    _format_elapsed,
    _score_style,
)

# ---------------------------------------------------------------------------
# Minimal test apps for widget-level testing
# ---------------------------------------------------------------------------


class PanelTestApp(App):
    """Minimal app that mounts a single AgentPanel for testing."""

    def __init__(self, panel: AgentPanel) -> None:
        super().__init__()
        self._panel = panel

    def compose(self) -> ComposeResult:
        yield self._panel


class StatusBarTestApp(App):
    """Minimal app that mounts a StatusBarWidget for testing."""

    def __init__(self, bar: StatusBarWidget) -> None:
        super().__init__()
        self._bar = bar

    def compose(self) -> ComposeResult:
        yield self._bar


class IterationTestApp(App):
    """Minimal app that mounts an IterationContainer for testing."""

    def __init__(self, container: IterationContainer) -> None:
        super().__init__()
        self._container = container

    def compose(self) -> ComposeResult:
        yield self._container


# ---------------------------------------------------------------------------
# AgentPanel tests
# ---------------------------------------------------------------------------


class TestAgentPanel:
    @pytest.mark.asyncio
    async def test_construction(self):
        panel = AgentPanel(name="Analyst", model="claude-sonnet-4-6", style="green")
        async with PanelTestApp(panel).run_test():
            assert panel.panel_name == "Analyst"
            assert panel.model == "claude-sonnet-4-6"
            assert not panel.done

    @pytest.mark.asyncio
    async def test_add_line(self):
        panel = AgentPanel(name="Analyst", model="claude-sonnet-4-6", style="green")
        async with PanelTestApp(panel).run_test():
            panel.add_line("[15s] Analyzing data")
            assert len(panel.lines) == 1
            assert panel.all_lines[0] == "[15s] Analyzing data"

    @pytest.mark.asyncio
    async def test_deque_scrolling(self):
        panel = AgentPanel(name="Analyst", model="claude-sonnet-4-6", style="green")
        async with PanelTestApp(panel).run_test():
            for i in range(7):
                panel.add_line(f"line {i}")
            assert len(panel.lines) == PANEL_MAX_LINES
            assert len(panel.all_lines) == 7
            assert panel.lines[0] == "line 2"
            assert panel.lines[-1] == "line 6"

    @pytest.mark.asyncio
    async def test_complete(self):
        panel = AgentPanel(name="Analyst", model="claude-sonnet-4-6", style="green")
        async with PanelTestApp(panel).run_test():
            panel.add_line("[15s] Working...")
            panel.complete("Analysis complete, found 3 metrics")
            assert panel.done
            assert panel.done_summary == "Analysis complete, found 3 metrics"

    @pytest.mark.asyncio
    async def test_complete_uses_last_line_as_fallback(self):
        panel = AgentPanel(name="Analyst", model="claude-sonnet-4-6", style="green")
        async with PanelTestApp(panel).run_test():
            panel.add_line("[done] strategy=incremental")
            panel.complete()
            assert panel.done_summary == "[done] strategy=incremental"

    @pytest.mark.asyncio
    async def test_error(self):
        panel = AgentPanel(name="Analyst", model="claude-sonnet-4-6", style="green")
        async with PanelTestApp(panel).run_test():
            panel.error("Connection timeout")
            assert panel.done
            assert panel.error_msg == "Connection timeout"

    @pytest.mark.asyncio
    async def test_set_tokens(self):
        panel = AgentPanel(name="Critic", model="gpt-4o", style="yellow")
        async with PanelTestApp(panel).run_test():
            panel.set_tokens(2340, 890)
            assert panel.input_tokens == 2340
            assert panel.output_tokens == 890

    @pytest.mark.asyncio
    async def test_set_stats(self):
        panel = AgentPanel(name="Critic", model="gpt-4o", style="yellow")
        async with PanelTestApp(panel).run_test():
            panel.set_stats(input_tokens=100, output_tokens=50, num_turns=3)
            assert panel.input_tokens == 100
            assert panel.output_tokens == 50
            assert panel.num_turns == 3

    @pytest.mark.asyncio
    async def test_add_line_noop_after_done(self):
        panel = AgentPanel(name="Analyst", model="claude-sonnet-4-6", style="green")
        async with PanelTestApp(panel).run_test():
            panel.add_line("working")
            panel.complete("done")
            panel.add_line("this should be ignored")
            assert len(panel.all_lines) == 1

    @pytest.mark.asyncio
    async def test_elapsed_freezes_on_complete(self):
        panel = AgentPanel(name="Analyst", model="claude-sonnet-4-6", style="green")
        async with PanelTestApp(panel).run_test():
            panel.complete("done")
            assert panel._end_time is not None

    @pytest.mark.asyncio
    async def test_expanded_default_false(self):
        panel = AgentPanel(name="Analyst", model="claude-sonnet-4-6", style="green")
        async with PanelTestApp(panel).run_test():
            assert panel.expanded is False

    @pytest.mark.asyncio
    async def test_expanded_toggle(self):
        panel = AgentPanel(name="Analyst", model="claude-sonnet-4-6", style="green")
        async with PanelTestApp(panel).run_test():
            panel.expanded = True
            assert panel.expanded is True
            panel.expanded = False
            assert panel.expanded is False


# ---------------------------------------------------------------------------
# StatusBarWidget tests
# ---------------------------------------------------------------------------


class TestStatusBarWidget:
    @pytest.mark.asyncio
    async def test_construction(self):
        bar = StatusBarWidget()
        async with StatusBarTestApp(bar).run_test():
            assert bar.iteration == 0
            assert bar.phase == ""
            assert bar.finished is False

    @pytest.mark.asyncio
    async def test_update(self):
        bar = StatusBarWidget()
        async with StatusBarTestApp(bar).run_test():
            bar.set_status(iteration=3, phase="DEBATE", best_version="v02", best_score=85)
            assert bar.iteration == 3
            assert bar.phase == "DEBATE"
            assert bar.best_version == "v02"
            assert bar.best_score == 85

    @pytest.mark.asyncio
    async def test_add_agent_stats(self):
        bar = StatusBarWidget()
        panel = AgentPanel(name="Analyst", model="claude-sonnet-4-6", style="green")
        async with StatusBarTestApp(bar).run_test():
            panel.set_stats(input_tokens=100, output_tokens=50, num_turns=3)
            bar.add_agent_stats(panel)
            assert bar.total_input_tokens == 100
            assert bar.total_output_tokens == 50
            assert bar.total_turns == 3

    @pytest.mark.asyncio
    async def test_finish_freezes_timer(self):
        bar = StatusBarWidget()
        async with StatusBarTestApp(bar).run_test():
            bar.finish()
            assert bar.finished is True
            assert bar._end_time is not None


# ---------------------------------------------------------------------------
# IterationContainer tests
# ---------------------------------------------------------------------------


class TestIterationContainer:
    @pytest.mark.asyncio
    async def test_construction(self):
        container = IterationContainer(iter_title="Iteration 0")
        async with IterationTestApp(container).run_test():
            assert container.border_title == "Iteration 0"

    @pytest.mark.asyncio
    async def test_set_result(self):
        container = IterationContainer(iter_title="Iteration 1")
        async with IterationTestApp(container).run_test():
            container.set_result("completed (85)", "green")
            assert container.border_subtitle == "completed (85)"


# ---------------------------------------------------------------------------
# PipelineLive headless tests
# ---------------------------------------------------------------------------


class TestPipelineLiveHeadless:
    def test_lifecycle(self):
        live = PipelineLive()
        live.start()
        live.stop()

    def test_add_panel_tracks_internally(self):
        live = PipelineLive()
        live.start()
        panel = AgentPanel(name="Analyst", model="claude-sonnet-4-6", style="green")
        live.add_panel(panel)
        assert live.has_panel(panel)
        assert live.panel_count == 1
        live.stop()

    def test_collapse_panel(self):
        live = PipelineLive()
        live.start()
        panel = AgentPanel(name="Analyst", model="claude-sonnet-4-6", style="green")
        live.add_panel(panel)
        live.collapse_panel(panel, "done")
        assert panel.done
        assert panel.done_summary == "done"
        live.stop()

    def test_file_logging(self, tmp_path):
        log_path = tmp_path / "console.log"
        live = PipelineLive()
        live.start(log_path=log_path)
        live.log("Test log message")
        live.stop()
        content = log_path.read_text()
        assert "Test log message" in content

    def test_panel_count(self):
        live = PipelineLive()
        live.start()
        p1 = AgentPanel(name="Analyst", model="claude-sonnet-4-6", style="green")
        p2 = AgentPanel(name="Scientist", model="claude-sonnet-4-6", style="cyan")
        live.add_panel(p1)
        live.add_panel(p2)
        assert live.panel_count == 2
        live.stop()

    def test_remove_panel(self):
        live = PipelineLive()
        live.start()
        panel = AgentPanel(name="Analyst", model="claude-sonnet-4-6", style="green")
        live.add_panel(panel)
        live.remove_panel(panel)
        assert not live.has_panel(panel)
        assert live.panel_count == 0
        live.stop()

    def test_update_status(self):
        live = PipelineLive()
        live.start()
        live.update_status(iteration=1, phase="PLAN")
        # In headless mode, status updates are tracked internally
        live.stop()

    def test_wait_for_dismiss_noop(self):
        live = PipelineLive()
        live.start()
        live.wait_for_dismiss()  # should not block
        live.stop()

    def test_flush_completed(self):
        live = PipelineLive()
        live.start()
        panel = AgentPanel(name="Analyst", model="claude-sonnet-4-6", style="green")
        live.add_panel(panel)
        live.collapse_panel(panel, "done")
        live.flush_completed()
        # Should not raise
        live.stop()

    def test_start_end_iteration(self):
        live = PipelineLive()
        live.start()
        live.start_iteration(0)
        live.end_iteration("completed (85)", "green")
        live.flush_completed()
        live.stop()

    def test_panels_list_tracks_all(self):
        """The _panels list preserves all panels ever added for validation."""
        live = PipelineLive()
        live.start()
        p1 = AgentPanel(name="Analyst", model="claude-sonnet-4-6", style="green")
        p2 = AgentPanel(name="Scientist", model="claude-sonnet-4-6", style="cyan")
        live.add_panel(p1)
        live.add_panel(p2)
        assert p1 in live._panels
        assert p2 in live._panels
        live.stop()


# ---------------------------------------------------------------------------
# PipelineApp tests
# ---------------------------------------------------------------------------


class TestPipelineApp:
    @pytest.mark.asyncio
    async def test_lifecycle_with_mock_orchestrator(self):
        """App starts, runs mock orchestrator, and can be dismissed."""

        class FakeOrch:
            _live: PipelineLive | None = None

            async def run(self):
                pass

        orch = FakeOrch()
        app = PipelineApp(orch)
        async with app.run_test():
            # App should have composed the widget tree
            assert app.query_one("#main-scroll") is not None
            assert app.query_one(StatusBarWidget) is not None

    @pytest.mark.asyncio
    async def test_add_panel_mounts_widget(self):
        class FakeOrch:
            _live: PipelineLive | None = None

            async def run(self):
                self._live.add_panel(
                    AgentPanel(name="Analyst", model="claude-sonnet-4-6", style="green")
                )

        orch = FakeOrch()
        app = PipelineApp(orch)
        async with app.run_test() as pilot:
            await pilot.pause()
            panels = app.query(AgentPanel)
            assert len(panels) >= 1

    @pytest.mark.asyncio
    async def test_ctrl_o_toggles_all_panels(self):
        class FakeOrch:
            _live: PipelineLive | None = None

            async def run(self):
                self._live.add_panel(
                    AgentPanel(name="Analyst", model="claude-sonnet-4-6", style="green")
                )
                self._live.add_panel(
                    AgentPanel(name="Scientist", model="claude-sonnet-4-6", style="cyan")
                )

        orch = FakeOrch()
        app = PipelineApp(orch)
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("ctrl+o")
            await pilot.pause()
            panels = list(app.query(AgentPanel))
            assert all(p.expanded for p in panels)
            await pilot.press("ctrl+o")
            await pilot.pause()
            assert all(not p.expanded for p in panels)

    @pytest.mark.asyncio
    async def test_per_panel_click_toggles_one(self):
        class FakeOrch:
            _live: PipelineLive | None = None

            async def run(self):
                self._live.add_panel(
                    AgentPanel(name="Analyst", model="claude-sonnet-4-6", style="green")
                )
                self._live.add_panel(
                    AgentPanel(name="Scientist", model="claude-sonnet-4-6", style="cyan")
                )

        orch = FakeOrch()
        app = PipelineApp(orch)
        async with app.run_test() as pilot:
            await pilot.pause()
            panels = list(app.query(AgentPanel))
            assert len(panels) == 2
            await pilot.click(AgentPanel, offset=(5, 0))
            await pilot.pause()
            # At least one should be expanded
            expanded_count = sum(1 for p in panels if p.expanded)
            assert expanded_count >= 1

    @pytest.mark.asyncio
    async def test_start_iteration_creates_container(self):
        class FakeOrch:
            _live: PipelineLive | None = None

            async def run(self):
                self._live.start_iteration(0)

        orch = FakeOrch()
        app = PipelineApp(orch)
        async with app.run_test() as pilot:
            await pilot.pause()
            containers = app.query(IterationContainer)
            assert len(containers) >= 1

    @pytest.mark.asyncio
    async def test_update_status(self):
        class FakeOrch:
            _live: PipelineLive | None = None

            async def run(self):
                self._live.update_status(iteration=1, phase="PLAN")

        orch = FakeOrch()
        app = PipelineApp(orch)
        async with app.run_test() as pilot:
            await pilot.pause()
            bar = app.query_one(StatusBarWidget)
            assert bar.iteration == 1
            assert bar.phase == "PLAN"

    @pytest.mark.asyncio
    async def test_q_noop_when_not_finished(self):
        """Pressing q before pipeline finishes should not exit the app."""
        gate = threading.Event()

        class FakeOrch:
            _live: PipelineLive | None = None

            async def run(self):
                # Worker runs in a thread, so use threading.Event
                gate.wait()

        orch = FakeOrch()
        app = PipelineApp(orch)
        async with app.run_test() as pilot:
            await pilot.press("q")
            await pilot.pause()
            # App should still be running (not exited)
            assert app.query_one(StatusBarWidget) is not None
            gate.set()

    @pytest.mark.asyncio
    async def test_q_exits_when_finished(self):
        """Pressing q after pipeline finishes should exit the app."""

        class FakeOrch:
            _live: PipelineLive | None = None

            async def run(self):
                pass

        orch = FakeOrch()
        app = PipelineApp(orch)
        async with app.run_test() as pilot:
            await pilot.pause()
            # Worker should have completed, setting _finished
            assert app._finished is True
            await pilot.press("q")
            await pilot.pause()

    @pytest.mark.asyncio
    async def test_worker_completion_sets_finished(self):
        """Worker completing sets _finished and freezes status bar."""

        class FakeOrch:
            _live: PipelineLive | None = None

            async def run(self):
                pass

        orch = FakeOrch()
        app = PipelineApp(orch)
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app._finished is True
            bar = app.query_one(StatusBarWidget)
            assert bar.finished is True
            assert bar._end_time is not None

    @pytest.mark.asyncio
    async def test_collapse_panel_accumulates_stats_in_app(self):
        """In app mode, collapse_panel accumulates stats on the status bar."""

        class FakeOrch:
            _live: PipelineLive | None = None

            async def run(self):
                panel = AgentPanel(name="Analyst", model="claude-sonnet-4-6", style="green")
                panel.set_stats(input_tokens=500, output_tokens=200, num_turns=3)
                self._live.add_panel(panel)
                self._live.collapse_panel(panel, "done")

        orch = FakeOrch()
        app = PipelineApp(orch)
        async with app.run_test() as pilot:
            await pilot.pause()
            bar = app.query_one(StatusBarWidget)
            assert bar.total_input_tokens == 500
            assert bar.total_output_tokens == 200
            assert bar.total_turns == 3

    @pytest.mark.asyncio
    async def test_end_iteration_and_flush_separates_containers(self):
        """After flush, a new iteration gets its own container."""

        class FakeOrch:
            _live: PipelineLive | None = None

            async def run(self):
                self._live.start_iteration(0)
                self._live.add_panel(
                    AgentPanel(name="Analyst", model="claude-sonnet-4-6", style="green")
                )
                self._live.end_iteration("done", "green")
                self._live.flush_completed()
                self._live.start_iteration(1)
                self._live.add_panel(
                    AgentPanel(name="Scientist", model="claude-sonnet-4-6", style="cyan")
                )

        orch = FakeOrch()
        app = PipelineApp(orch)
        async with app.run_test() as pilot:
            await pilot.pause()
            containers = list(app.query(IterationContainer))
            assert len(containers) == 2


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestHelperFunctions:
    def test_format_elapsed_zero(self):
        assert _format_elapsed(0) == "0s"

    def test_format_elapsed_seconds(self):
        assert _format_elapsed(45) == "45s"

    def test_format_elapsed_one_minute(self):
        assert _format_elapsed(60) == "1m 0s"

    def test_format_elapsed_minutes_and_seconds(self):
        assert _format_elapsed(125) == "2m 5s"

    def test_score_style_green(self):
        assert _score_style(70) == "green"
        assert _score_style(100) == "green"

    def test_score_style_yellow(self):
        assert _score_style(40) == "yellow"
        assert _score_style(69) == "yellow"

    def test_score_style_red(self):
        assert _score_style(0) == "red"
        assert _score_style(39) == "red"


# ---------------------------------------------------------------------------
# AgentPanel idempotency tests
# ---------------------------------------------------------------------------


class TestAgentPanelIdempotency:
    @pytest.mark.asyncio
    async def test_double_complete(self):
        panel = AgentPanel(name="Analyst", model="claude-sonnet-4-6", style="green")
        async with PanelTestApp(panel).run_test():
            panel.complete("first")
            panel.complete("second")
            assert panel.done_summary == "first"

    @pytest.mark.asyncio
    async def test_double_error(self):
        panel = AgentPanel(name="Analyst", model="claude-sonnet-4-6", style="green")
        async with PanelTestApp(panel).run_test():
            panel.error("err1")
            panel.error("err2")
            assert panel.error_msg == "err1"

    @pytest.mark.asyncio
    async def test_error_after_complete_is_noop(self):
        panel = AgentPanel(name="Analyst", model="claude-sonnet-4-6", style="green")
        async with PanelTestApp(panel).run_test():
            panel.complete("done")
            panel.error("err")
            assert panel.error_msg == ""
            assert panel.done_summary == "done"
