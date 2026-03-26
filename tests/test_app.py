"""Tests for PipelineApp multi-screen support."""

import pytest

from auto_scientist.ui.app import PipelineApp
from auto_scientist.ui.home_screen import HomeScreen


class TestPipelineAppNoOrchestrator:
    async def test_shows_home_screen(self):
        app = PipelineApp(orchestrator=None)
        async with app.run_test() as pilot:
            await pilot.pause()
            # The top screen should be a HomeScreen
            assert isinstance(app.screen, HomeScreen)

    async def test_has_store(self):
        app = PipelineApp(orchestrator=None)
        async with app.run_test() as pilot:
            await pilot.pause()
            assert app._store is not None
