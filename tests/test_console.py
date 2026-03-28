"""Tests for Textual console components: AgentPanel, MetricsBar, PipelineLive, PipelineApp."""

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Collapsible, RichLog

from auto_scientist.console import (
    AGENT_STYLES,
    PHASE_STYLES,
    AgentDetailScreen,
    AgentPanel,
    IterationContainer,
    IterationToggle,
    MetricsBar,
    PipelineApp,
    PipelineLive,
    QuitConfirmScreen,
    _format_elapsed,
)

# ---------------------------------------------------------------------------
# Minimal test apps for widget-level testing
# ---------------------------------------------------------------------------


class PanelTestApp(App):
    def __init__(self, panel: AgentPanel) -> None:
        super().__init__()
        self._panel = panel

    def compose(self) -> ComposeResult:
        yield self._panel


class MetricsBarTestApp(App):
    def __init__(self, bar: MetricsBar) -> None:
        super().__init__()
        self._bar = bar

    def compose(self) -> ComposeResult:
        yield self._bar


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


# ---------------------------------------------------------------------------
# Constants tests
# ---------------------------------------------------------------------------


class TestConstants:
    def test_agent_styles_has_all_agents(self):
        expected = {
            "Analyst",
            "Scientist",
            "Coder",
            "Ingestor",
            "Report",
            "Critic",
            "Debate",
            "Results",
        }
        assert set(AGENT_STYLES.keys()) == expected

    def test_phase_styles_has_all_phases(self):
        expected = {
            "INGESTION",
            "ANALYZE",
            "PLAN",
            "DEBATE",
            "REVISE",
            "IMPLEMENT",
            "REPORT",
        }
        assert set(PHASE_STYLES.keys()) == expected


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
    async def test_has_collapsible_and_richlog(self):
        panel = AgentPanel(name="Analyst", model="claude-sonnet-4-6", style="green")
        async with PanelTestApp(panel).run_test():
            assert panel.query_one(Collapsible) is not None
            assert panel.query_one(RichLog) is not None

    @pytest.mark.asyncio
    async def test_add_line(self):
        panel = AgentPanel(name="Analyst", model="claude-sonnet-4-6", style="green")
        async with PanelTestApp(panel).run_test():
            panel.add_line("[15s] Analyzing data")
            assert len(panel.all_lines) == 1
            assert panel.all_lines[0] == "[15s] Analyzing data"

    @pytest.mark.asyncio
    async def test_lines_property_returns_all_lines(self):
        panel = AgentPanel(name="Analyst", model="claude-sonnet-4-6", style="green")
        async with PanelTestApp(panel).run_test():
            panel.add_line("line 1")
            panel.add_line("line 2")
            assert panel.lines is panel.all_lines
            assert len(panel.lines) == 2

    @pytest.mark.asyncio
    async def test_multiple_lines_accumulate(self):
        panel = AgentPanel(name="Analyst", model="claude-sonnet-4-6", style="green")
        async with PanelTestApp(panel).run_test():
            for i in range(7):
                panel.add_line(f"line {i}")
            assert len(panel.all_lines) == 7

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
    async def test_complete_collapses_collapsible(self):
        panel = AgentPanel(name="Analyst", model="claude-sonnet-4-6", style="green")
        async with PanelTestApp(panel).run_test():
            panel.complete("done")
            panel._apply_complete_dom()
            collapsible = panel.query_one(Collapsible)
            assert collapsible.collapsed is True

    @pytest.mark.asyncio
    async def test_complete_single_entry_not_expandable(self):
        panel = AgentPanel(name="Scientist", model="claude-sonnet-4-6", style="cyan")
        async with PanelTestApp(panel).run_test():
            panel.add_line("Planned experiment A")
            panel.complete("Planned experiment A")
            panel._apply_complete_dom()
            collapsible = panel.query_one(Collapsible)
            assert collapsible.collapsed is True
            assert collapsible.disabled is True
            title_widget = collapsible.query_one("CollapsibleTitle")
            assert title_widget.collapsed_symbol == "●"

    @pytest.mark.asyncio
    async def test_complete_multi_entry_stays_expandable(self):
        panel = AgentPanel(name="Analyst", model="claude-sonnet-4-6", style="green")
        async with PanelTestApp(panel).run_test():
            panel.add_line("Step 1")
            panel.add_line("Step 2")
            panel.add_line("Step 3")
            panel.complete("Analysis complete")
            panel._apply_complete_dom()
            collapsible = panel.query_one(Collapsible)
            assert collapsible.collapsed is True
            assert collapsible.disabled is False

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
    async def test_build_footer_with_tokens_and_turns(self):
        panel = AgentPanel(name="Analyst", model="claude-sonnet-4-6", style="green")
        async with PanelTestApp(panel).run_test():
            panel.set_stats(input_tokens=100, output_tokens=50, num_turns=3)
            footer = panel._build_footer()
            assert "100 in / 50 out" in footer
            assert "3 turns" in footer

    @pytest.mark.asyncio
    async def test_build_footer_singular_turn(self):
        panel = AgentPanel(name="Analyst", model="claude-sonnet-4-6", style="green")
        async with PanelTestApp(panel).run_test():
            panel.set_stats(input_tokens=10, output_tokens=5, num_turns=1)
            footer = panel._build_footer()
            assert "1 turn" in footer
            assert "turns" not in footer


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


# ---------------------------------------------------------------------------
# IterationContainer tests
# ---------------------------------------------------------------------------


class IterationTestApp(App):
    """Test app hosting an IterationContainer with panels inside."""

    def __init__(
        self, container: IterationContainer, panels: list[AgentPanel] | None = None
    ) -> None:
        super().__init__()
        self._container = container
        self._test_panels = panels or []

    def compose(self) -> ComposeResult:
        yield self._container

    def on_mount(self) -> None:
        for panel in self._test_panels:
            self._container.mount(panel)
            self._container.add_panel(panel)


class TestIterationContainer:
    def test_construction(self):
        container = IterationContainer(iter_title="Iteration 0")
        assert container._iter_title == "Iteration 0"
        assert container.border_title == "Iteration 0"
        assert container._in_progress is True
        assert container._panels == []
        assert container._is_collapsed is False

    def test_set_result(self):
        container = IterationContainer(iter_title="Iteration 1")
        container.set_result("completed (85)", "green")
        # Status text is not shown (green border already signals completion)
        assert container.border_subtitle == ""
        assert container._in_progress is False
        assert container.border_title == "Iteration 1"

    def test_add_panel(self):
        container = IterationContainer(iter_title="Iteration 0")
        panel = AgentPanel(name="Analyst", model="m", style="green")
        container.add_panel(panel)
        assert len(container._panels) == 1
        assert container._panels[0] is panel

    @pytest.mark.asyncio
    async def test_collapse_hides_panels(self):
        container = IterationContainer(iter_title="Iteration 1")
        p1 = AgentPanel(name="Analyst", model="m", style="green")
        p2 = AgentPanel(name="Coder", model="m", style="magenta1")
        p1.complete("Analysis done")
        p1.set_stats(input_tokens=100, output_tokens=50, num_turns=2)
        p2.complete("Code written")
        p2.set_stats(input_tokens=200, output_tokens=100, num_turns=3)
        async with IterationTestApp(container, [p1, p2]).run_test():
            container.set_result("completed", "green")
            assert container._is_collapsed is True
            # Panels should be hidden
            assert str(p1.styles.display) == "none"
            assert str(p2.styles.display) == "none"
            # Toggle widget should be mounted
            toggles = list(container.query(IterationToggle))
            assert len(toggles) == 1
            assert "2 agents" in str(toggles[0].render())

    @pytest.mark.asyncio
    async def test_collapse_aggregates_metrics(self):
        container = IterationContainer(iter_title="Iteration 1")
        p1 = AgentPanel(name="Analyst", model="m", style="green")
        p2 = AgentPanel(name="Coder", model="m", style="magenta1")
        p1.complete("done")
        p1.set_stats(input_tokens=1000, output_tokens=500, num_turns=2)
        p2.complete("done")
        p2.set_stats(input_tokens=2000, output_tokens=1000, num_turns=3)
        async with IterationTestApp(container, [p1, p2]).run_test():
            container.set_result("completed", "green")
            subtitle = str(container.border_subtitle)
            assert "3,000 in" in subtitle
            assert "1,500 out" in subtitle
            assert "5 turns" in subtitle

    @pytest.mark.asyncio
    async def test_collapse_with_summary_text(self):
        container = IterationContainer(iter_title="Iteration 1")
        p1 = AgentPanel(name="Analyst", model="m", style="green")
        p1.complete("done")
        async with IterationTestApp(container, [p1]).run_test():
            container.set_result("completed", "green", "We explored the dataset.")
            recaps = list(container.query(".iteration-recap"))
            assert len(recaps) == 1
            assert "explored" in str(recaps[0].render())

    @pytest.mark.asyncio
    async def test_toggle_iteration(self):
        container = IterationContainer(iter_title="Iteration 1")
        p1 = AgentPanel(name="Analyst", model="m", style="green")
        p1.complete("done")
        async with IterationTestApp(container, [p1]).run_test():
            container.set_result("completed", "green")
            assert container._is_collapsed is True
            assert str(p1.styles.display) == "none"
            # Expand
            container.toggle_iteration()
            assert container._is_collapsed is False
            assert str(p1.styles.display) == "block"
            # Collapse again
            container.toggle_iteration()
            assert container._is_collapsed is True
            assert str(p1.styles.display) == "none"

    @pytest.mark.asyncio
    async def test_collapse_idempotent(self):
        container = IterationContainer(iter_title="Iteration 1")
        p1 = AgentPanel(name="Analyst", model="m", style="green")
        p1.complete("done")
        async with IterationTestApp(container, [p1]).run_test():
            container.set_result("completed", "green", "Summary")
            # Calling collapse again should be a no-op
            container.collapse_iteration("Another summary")
            # Should still only have one toggle and one recap
            toggles = list(container.query(IterationToggle))
            assert len(toggles) == 1
            recaps = list(container.query(".iteration-recap"))
            assert len(recaps) == 1


# ---------------------------------------------------------------------------
# MetricsBar tests
# ---------------------------------------------------------------------------


class TestMetricsBar:
    @pytest.mark.asyncio
    async def test_construction(self):
        bar = MetricsBar()
        async with MetricsBarTestApp(bar).run_test():
            assert bar.iteration == 0
            assert bar.phase == ""
            assert bar.finished is False

    @pytest.mark.asyncio
    async def test_set_status(self):
        bar = MetricsBar()
        async with MetricsBarTestApp(bar).run_test():
            bar.set_status(iteration=3, phase="DEBATE")
            assert bar.iteration == 3
            assert bar.phase == "DEBATE"

    @pytest.mark.asyncio
    async def test_set_status_partial_update(self):
        bar = MetricsBar()
        async with MetricsBarTestApp(bar).run_test():
            bar.set_status(iteration=5)
            bar.set_status(phase="DEBATE")
            assert bar.iteration == 5
            assert bar.phase == "DEBATE"

    @pytest.mark.asyncio
    async def test_add_agent_stats(self):
        bar = MetricsBar()
        panel = AgentPanel(
            name="Analyst",
            model="claude-sonnet-4-6",
            style="green",
        )
        async with MetricsBarTestApp(bar).run_test():
            panel.set_stats(input_tokens=100, output_tokens=50, num_turns=3)
            bar.add_agent_stats(panel)
            assert bar.total_input_tokens == 100
            assert bar.total_output_tokens == 50
            assert bar.total_turns == 3

    @pytest.mark.asyncio
    async def test_finish_freezes_timer(self):
        bar = MetricsBar()
        async with MetricsBarTestApp(bar).run_test():
            bar.finish()
            assert bar.finished is True
            assert bar._end_time is not None

    @pytest.mark.asyncio
    async def test_render_with_scores(self):
        bar = MetricsBar()
        async with MetricsBarTestApp(bar).run_test():
            bar.scores = [20, 40, 60, 80]
            rendered = bar.render()
            assert "[" in rendered.plain
            assert "]" in rendered.plain

    @pytest.mark.asyncio
    async def test_render_with_all_zero_scores(self):
        bar = MetricsBar()
        async with MetricsBarTestApp(bar).run_test():
            bar.scores = [0, 0, 0]
            rendered = bar.render()
            assert "[" in rendered.plain


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
        panel = AgentPanel(
            name="Analyst",
            model="claude-sonnet-4-6",
            style="green",
        )
        live.add_panel(panel)
        assert live.has_panel(panel)
        assert live.panel_count == 1
        live.stop()

    def test_collapse_panel(self):
        live = PipelineLive()
        live.start()
        panel = AgentPanel(
            name="Analyst",
            model="claude-sonnet-4-6",
            style="green",
        )
        live.add_panel(panel)
        live.collapse_panel(panel, "done")
        assert panel.done
        assert panel.done_summary == "done"
        live.stop()

    def test_collapse_panel_file_logging(self, tmp_path):
        log_path = tmp_path / "console.log"
        live = PipelineLive()
        live.start(log_path=log_path)
        panel = AgentPanel(
            name="Analyst",
            model="claude-sonnet-4-6",
            style="green",
        )
        live.add_panel(panel)
        live.collapse_panel(panel, "analysis done")
        live.stop()
        content = log_path.read_text()
        assert "Analyst" in content
        assert "analysis done" in content

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
        p1 = AgentPanel(
            name="Analyst",
            model="claude-sonnet-4-6",
            style="green",
        )
        p2 = AgentPanel(
            name="Scientist",
            model="claude-sonnet-4-6",
            style="cyan",
        )
        live.add_panel(p1)
        live.add_panel(p2)
        assert live.panel_count == 2
        live.stop()

    def test_remove_panel(self):
        live = PipelineLive()
        live.start()
        panel = AgentPanel(
            name="Analyst",
            model="claude-sonnet-4-6",
            style="green",
        )
        live.add_panel(panel)
        live.remove_panel(panel)
        assert not live.has_panel(panel)
        assert live.panel_count == 0
        live.stop()

    def test_update_status(self):
        live = PipelineLive()
        live.start()
        live.update_status(iteration=1, phase="PLAN")
        live.stop()

    def test_wait_for_dismiss_noop(self):
        live = PipelineLive()
        live.start()
        live.wait_for_dismiss()
        live.stop()

    def test_flush_completed(self):
        live = PipelineLive()
        live.start()
        panel = AgentPanel(
            name="Analyst",
            model="claude-sonnet-4-6",
            style="green",
        )
        live.add_panel(panel)
        live.collapse_panel(panel, "done")
        live.flush_completed()
        live.stop()

    def test_start_end_iteration(self):
        live = PipelineLive()
        live.start()
        live.start_iteration(0)
        live.end_iteration("completed (85)", "green")
        live.flush_completed()
        live.stop()

    def test_panels_list_tracks_all(self):
        live = PipelineLive()
        live.start()
        p1 = AgentPanel(
            name="Analyst",
            model="claude-sonnet-4-6",
            style="green",
        )
        p2 = AgentPanel(
            name="Scientist",
            model="claude-sonnet-4-6",
            style="cyan",
        )
        live.add_panel(p1)
        live.add_panel(p2)
        assert p1 in live._panels
        assert p2 in live._panels
        live.stop()

    def test_refresh_is_noop(self):
        live = PipelineLive()
        live.start()
        live.refresh()
        live.stop()

    def test_print_static_headless(self):
        live = PipelineLive()
        live.start()
        live.print_static("hello")
        live.stop()

    def test_add_rule_headless(self):
        live = PipelineLive()
        live.start()
        live.add_rule("---")
        live.stop()

    def test_print_static_writes_to_log(self, tmp_path):
        log_path = tmp_path / "console.log"
        live = PipelineLive()
        live.start(log_path=log_path)
        live.print_static("banner text")
        live.stop()
        content = log_path.read_text()
        assert "banner text" in content

    def test_iteration_flush_and_restart(self):
        live = PipelineLive()
        live.start()
        live.start_iteration(0)
        live.end_iteration("done", "green")
        live.flush_completed()
        assert live._current_iteration is None
        live.start_iteration(1)
        assert live._current_iteration is not None
        assert live._current_iteration._iter_title == "Iteration 1"
        live.stop()

    def test_end_iteration_passes_summary(self):
        live = PipelineLive()
        live.start()
        live.start_iteration(0)
        container = live._current_iteration
        # In headless mode, set_result is called directly with summary_text
        live.end_iteration("done", "green", "Iteration recap text")
        # Status text is not shown (green border already signals completion)
        assert container.border_subtitle == ""
        assert container._in_progress is False
        live.stop()


# ---------------------------------------------------------------------------
# PipelineApp tests
# ---------------------------------------------------------------------------


class TestPipelineApp:
    @pytest.mark.asyncio
    async def test_lifecycle_with_mock_orchestrator(self):
        class FakeOrch:
            _live: PipelineLive | None = None

            async def run(self):
                pass

        orch = FakeOrch()
        app = PipelineApp(orch)
        async with app.run_test():
            assert app.query_one("#main-scroll") is not None
            assert app.query_one(MetricsBar) is not None

    @pytest.mark.asyncio
    async def test_add_panel_mounts_widget(self):
        class FakeOrch:
            _live: PipelineLive | None = None

            async def run(self):
                self._live.add_panel(
                    AgentPanel(
                        name="Analyst",
                        model="claude-sonnet-4-6",
                        style="green",
                    )
                )

        orch = FakeOrch()
        app = PipelineApp(orch)
        async with app.run_test() as pilot:
            await pilot.pause()
            panels = app.query(AgentPanel)
            assert len(panels) >= 1

    def test_start_iteration_creates_container_headless(self):
        live = PipelineLive()
        live.start()
        live.start_iteration(0)
        assert live._current_iteration is not None
        assert live._current_iteration._iter_title == "Iteration 0"
        live.stop()

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
            bar = app.query_one(MetricsBar)
            assert bar.iteration == 1
            assert bar.phase == "PLAN"

    @pytest.mark.asyncio
    async def test_worker_completion_sets_finished(self):
        class FakeOrch:
            _live: PipelineLive | None = None

            async def run(self):
                pass

        orch = FakeOrch()
        app = PipelineApp(orch)
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app._finished is True
            bar = app.query_one(MetricsBar)
            assert bar.finished is True
            assert bar._end_time is not None

    @pytest.mark.asyncio
    async def test_collapse_panel_accumulates_stats(self):
        class FakeOrch:
            _live: PipelineLive | None = None

            async def run(self):
                panel = AgentPanel(
                    name="Analyst",
                    model="claude-sonnet-4-6",
                    style="green",
                )
                panel.set_stats(
                    input_tokens=500,
                    output_tokens=200,
                    num_turns=3,
                )
                self._live.add_panel(panel)
                self._live.collapse_panel(panel, "done")

        orch = FakeOrch()
        app = PipelineApp(orch)
        async with app.run_test() as pilot:
            await pilot.pause()
            bar = app.query_one(MetricsBar)
            assert bar.total_input_tokens == 500
            assert bar.total_output_tokens == 200
            assert bar.total_turns == 3


# ---------------------------------------------------------------------------
# Orchestrator pause/skip flag tests
# ---------------------------------------------------------------------------


class TestOrchestratorFlags:
    def test_flags_default_false(self):
        from unittest.mock import MagicMock

        from auto_scientist.orchestrator import Orchestrator

        state = MagicMock()
        state.phase = "ingestion"
        orch = Orchestrator(
            state=state,
            data_path=None,
            output_dir=MagicMock(),
        )
        assert orch.pause_requested is False
        assert orch.skip_to_report is False


# ---------------------------------------------------------------------------
# Screen tests
# ---------------------------------------------------------------------------


class TestAgentDetailScreen:
    @pytest.mark.asyncio
    async def test_detail_screen_shows_lines(self):
        class FakeOrch:
            _live: PipelineLive | None = None

            async def run(self):
                pass

        orch = FakeOrch()
        app = PipelineApp(orch)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.push_screen(
                AgentDetailScreen(
                    panel_name="Analyst",
                    model="claude-sonnet-4-6",
                    stats="5s | 100 in / 50 out",
                    lines=["line 1", "line 2", "line 3"],
                )
            )
            await pilot.pause()
            rich_log = app.screen.query_one(RichLog)
            assert rich_log is not None

    @pytest.mark.asyncio
    async def test_detail_screen_escape_dismisses(self):
        class FakeOrch:
            _live: PipelineLive | None = None

            async def run(self):
                pass

        orch = FakeOrch()
        app = PipelineApp(orch)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.push_screen(
                AgentDetailScreen(
                    panel_name="Analyst",
                    model="claude-sonnet-4-6",
                    stats="5s",
                    lines=["line 1"],
                )
            )
            await pilot.pause()
            assert isinstance(app.screen, AgentDetailScreen)
            await pilot.press("escape")
            await pilot.pause()
            assert not isinstance(app.screen, AgentDetailScreen)


class TestQuitConfirmScreen:
    @pytest.mark.asyncio
    async def test_quit_confirm_y_dismisses_with_true(self):
        class FakeOrch:
            _live: PipelineLive | None = None

            async def run(self):
                pass

        results = []
        orch = FakeOrch()
        app = PipelineApp(orch)
        async with app.run_test() as pilot:
            await pilot.pause()
            app._finished = False  # Pretend still running
            app.push_screen(
                QuitConfirmScreen(),
                callback=results.append,
            )
            await pilot.pause()
            await pilot.press("y")
            await pilot.pause()
            assert results == [True]

    @pytest.mark.asyncio
    async def test_quit_confirm_n_dismisses_with_false(self):
        class FakeOrch:
            _live: PipelineLive | None = None

            async def run(self):
                pass

        results = []
        orch = FakeOrch()
        app = PipelineApp(orch)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.push_screen(
                QuitConfirmScreen(),
                callback=results.append,
            )
            await pilot.pause()
            await pilot.press("n")
            await pilot.pause()
            assert results == [False]

    @pytest.mark.asyncio
    async def test_quit_when_finished_exits_immediately(self):
        class FakeOrch:
            _live: PipelineLive | None = None

            async def run(self):
                pass

        orch = FakeOrch()
        app = PipelineApp(orch)
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app._finished is True
            # Should exit without showing modal
            await pilot.press("ctrl+q")
            await pilot.pause()

    @pytest.mark.asyncio
    async def test_quit_when_running_shows_modal(self):
        import threading

        gate = threading.Event()

        class FakeOrch:
            _live: PipelineLive | None = None

            async def run(self):
                gate.wait()

        orch = FakeOrch()
        app = PipelineApp(orch)
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app._finished is False
            await pilot.press("ctrl+q")
            await pilot.pause()
            assert isinstance(app.screen, QuitConfirmScreen)
            await pilot.press("n")
            await pilot.pause()
            gate.set()


# ---------------------------------------------------------------------------
# Theme cycling test
# ---------------------------------------------------------------------------


class TestThemeCycling:
    @pytest.mark.asyncio
    async def test_ctrl_t_changes_theme(self):
        class FakeOrch:
            _live: PipelineLive | None = None

            async def run(self):
                pass

        orch = FakeOrch()
        app = PipelineApp(orch)
        async with app.run_test() as pilot:
            await pilot.pause()
            initial_theme = app.theme
            await pilot.press("ctrl+t")
            await pilot.pause()
            assert app.theme != initial_theme
