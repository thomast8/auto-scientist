"""Tests for ExperimentScreen."""

import pytest
from textual.app import App, ComposeResult

from auto_scientist.ui.experiment_screen import ExperimentScreen
from auto_scientist.ui.widgets import MetricsBar


class ExperimentScreenApp(App):
    """Test app wrapping an ExperimentScreen."""

    def __init__(self, screen: ExperimentScreen) -> None:
        super().__init__()
        self._screen = screen

    def on_mount(self) -> None:
        self.push_screen(self._screen)


class TestExperimentScreenComposition:
    async def test_read_only_composes_with_metrics_bar(self):
        screen = ExperimentScreen(read_only=True)
        app = ExperimentScreenApp(screen)
        async with app.run_test() as pilot:
            await pilot.pause()
            bars = screen.query(MetricsBar)
            assert len(bars) == 1

    async def test_read_only_is_finished(self):
        screen = ExperimentScreen(read_only=True)
        app = ExperimentScreenApp(screen)
        async with app.run_test():
            assert screen._finished is True

    async def test_read_only_does_not_start_worker(self):
        screen = ExperimentScreen(read_only=True)
        app = ExperimentScreenApp(screen)
        async with app.run_test():
            assert screen._worker_loop is None

    async def test_no_orchestrator_does_not_start_worker(self):
        screen = ExperimentScreen(orchestrator=None)
        app = ExperimentScreenApp(screen)
        async with app.run_test():
            assert screen._worker_loop is None

    async def test_has_experiment_label(self):
        screen = ExperimentScreen(
            read_only=True, experiment_label="SpO2 fast",
        )
        assert screen._experiment_label == "SpO2 fast"
