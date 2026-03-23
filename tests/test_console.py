"""Tests for Rich console components: AgentPanel, StatusBar, PipelineLive."""

import time
from io import StringIO
from unittest.mock import MagicMock, patch

from rich.console import Console

from auto_scientist.console import AgentPanel, PipelineLive, StatusBar


class TestAgentPanel:
    def test_construction(self):
        panel = AgentPanel(name="Analyst", model="claude-sonnet-4-6", style="green")
        assert panel.name == "Analyst"
        assert panel.model == "claude-sonnet-4-6"
        assert not panel.done

    def test_add_line(self):
        panel = AgentPanel(name="Analyst", model="claude-sonnet-4-6", style="green")
        panel.add_line("[15s] Analyzing data")
        assert len(panel.lines) == 1
        assert panel.lines[0] == "[15s] Analyzing data"

    def test_deque_scrolling(self):
        panel = AgentPanel(name="Analyst", model="claude-sonnet-4-6", style="green")
        for i in range(7):
            panel.add_line(f"line {i}")
        assert len(panel.lines) == 5
        assert panel.lines[0] == "line 2"
        assert panel.lines[-1] == "line 6"

    def test_complete(self):
        panel = AgentPanel(name="Analyst", model="claude-sonnet-4-6", style="green")
        panel.add_line("[15s] Working...")
        panel.complete("Analysis complete, found 3 metrics")
        assert panel.done
        assert panel.done_summary == "Analysis complete, found 3 metrics"

    def test_error(self):
        panel = AgentPanel(name="Analyst", model="claude-sonnet-4-6", style="green")
        panel.error("Connection timeout")
        assert panel.done
        assert panel.error_msg == "Connection timeout"

    def test_set_tokens(self):
        panel = AgentPanel(name="Critic", model="gpt-4o", style="yellow")
        panel.set_tokens(2340, 890)
        assert panel.input_tokens == 2340
        assert panel.output_tokens == 890

    def test_renders_panel_with_title(self):
        panel = AgentPanel(name="Analyst", model="claude-sonnet-4-6", style="green")
        panel.add_line("[15s] Analyzing data")
        buf = StringIO()
        console = Console(file=buf, width=80, no_color=True)
        console.print(panel)
        output = buf.getvalue()
        assert "Analyst" in output
        assert "claude-sonnet-4-6" in output
        assert "Analyzing data" in output

    def test_renders_footer_with_tokens(self):
        panel = AgentPanel(name="Critic", model="gpt-4o", style="yellow")
        panel.set_tokens(100, 50)
        panel.complete("Done")
        buf = StringIO()
        console = Console(file=buf, width=80, no_color=True)
        console.print(panel)
        output = buf.getvalue()
        assert "100 in" in output
        assert "50 out" in output

    def test_renders_footer_without_tokens(self):
        panel = AgentPanel(name="Analyst", model="claude-sonnet-4-6", style="green")
        panel.complete("Done")
        buf = StringIO()
        console = Console(file=buf, width=80, no_color=True)
        console.print(panel)
        output = buf.getvalue()
        # Should not have token counts
        assert " in /" not in output

    def test_collapsed_shows_done_summary(self):
        panel = AgentPanel(name="Analyst", model="claude-sonnet-4-6", style="green")
        panel.add_line("[15s] Working...")
        panel.add_line("[30s] Still working...")
        panel.complete("Found 3 key metrics with R2=0.85")
        buf = StringIO()
        console = Console(file=buf, width=80, no_color=True)
        console.print(panel)
        output = buf.getvalue()
        assert "Found 3 key metrics" in output
        # Should not show progress lines after collapse
        assert "Working" not in output

    def test_error_panel_shows_error_msg(self):
        panel = AgentPanel(name="Analyst", model="claude-sonnet-4-6", style="green")
        panel.error("API rate limit exceeded")
        buf = StringIO()
        console = Console(file=buf, width=80, no_color=True)
        console.print(panel)
        output = buf.getvalue()
        assert "API rate limit exceeded" in output

    def test_empty_panel_shows_spinner_text(self):
        panel = AgentPanel(name="Analyst", model="claude-sonnet-4-6", style="green")
        buf = StringIO()
        console = Console(file=buf, width=80, no_color=True)
        console.print(panel)
        output = buf.getvalue()
        assert "working" in output.lower() or "..." in output


class TestStatusBar:
    def test_construction(self):
        bar = StatusBar(start_time=time.monotonic())
        assert bar.iteration == 0
        assert bar.phase == ""

    def test_update(self):
        bar = StatusBar(start_time=time.monotonic())
        bar.update(iteration=3, phase="DEBATE", best_version="v02", best_score=85)
        assert bar.iteration == 3
        assert bar.phase == "DEBATE"
        assert bar.best_version == "v02"
        assert bar.best_score == 85

    def test_renders_table(self):
        bar = StatusBar(start_time=time.monotonic())
        bar.update(iteration=2, phase="ANALYZE", best_version="v01", best_score=72)
        buf = StringIO()
        console = Console(file=buf, width=80, no_color=True)
        console.print(bar)
        output = buf.getvalue()
        assert "Iteration 2" in output
        assert "ANALYZE" in output
        assert "v01" in output
        assert "72" in output

    def test_score_color_green(self):
        bar = StatusBar(start_time=time.monotonic())
        bar.update(iteration=1, phase="PLAN", best_version="v00", best_score=85)
        buf = StringIO()
        console = Console(file=buf, width=80, force_terminal=True)
        console.print(bar)
        output = buf.getvalue()
        # Score 85 should render (green styling applied)
        assert "85" in output

    def test_no_score_shows_dash(self):
        bar = StatusBar(start_time=time.monotonic())
        bar.update(iteration=0, phase="INGESTION")
        buf = StringIO()
        console = Console(file=buf, width=80, no_color=True)
        console.print(bar)
        output = buf.getvalue()
        assert "-" in output


class TestPipelineLive:
    def test_lifecycle(self):
        live = PipelineLive()
        live.start()
        live.stop()

    def test_add_and_remove_panel(self):
        live = PipelineLive()
        live.start()
        panel = AgentPanel(name="Analyst", model="claude-sonnet-4-6", style="green")
        live.add_panel(panel)
        assert panel in live._panels
        live.remove_panel(panel)
        assert panel not in live._panels
        live.stop()

    def test_collapse_panel(self):
        live = PipelineLive()
        live.start()
        panel = AgentPanel(name="Analyst", model="claude-sonnet-4-6", style="green")
        live.add_panel(panel)
        live.collapse_panel(panel, "Analysis done")
        assert panel.done
        assert panel.done_summary == "Analysis done"
        live.stop()

    def test_update_status(self):
        live = PipelineLive()
        live.start()
        live.update_status(iteration=1, phase="PLAN")
        assert live._status_bar.iteration == 1
        assert live._status_bar.phase == "PLAN"
        live.stop()

    def test_file_logging(self, tmp_path):
        log_path = tmp_path / "console.log"
        live = PipelineLive()
        live.start(log_path=log_path)
        live.log("Test log message")
        live.stop()
        content = log_path.read_text()
        assert "Test log message" in content

    def test_file_logging_no_ansi(self, tmp_path):
        log_path = tmp_path / "console.log"
        live = PipelineLive()
        live.start(log_path=log_path)
        live.log("[bold red]Styled text[/]")
        live.stop()
        content = log_path.read_text()
        assert "\033[" not in content
        assert "Styled text" in content

    def test_file_logging_append_mode(self, tmp_path):
        log_path = tmp_path / "console.log"
        live = PipelineLive()
        live.start(log_path=log_path)
        live.log("first run")
        live.stop()

        live2 = PipelineLive()
        live2.start(log_path=log_path)
        live2.log("second run")
        live2.stop()

        content = log_path.read_text()
        assert "first run" in content
        assert "second run" in content
