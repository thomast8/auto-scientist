"""Tests for HomeScreen."""

import json
from pathlib import Path

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, DataTable

from auto_scientist.experiment_store import FilesystemStore
from auto_scientist.state import ExperimentState
from auto_scientist.ui.config_form import ConfigForm
from auto_scientist.ui.home_screen import HomeScreen


class HomeScreenApp(App):
    def __init__(self, store=None):
        super().__init__()
        self._store = store
        self._home: HomeScreen | None = None

    def on_mount(self) -> None:
        self._home = HomeScreen(store=self._store)
        self.push_screen(self._home)


def _create_experiment(base: Path, name: str, goal: str = "Test goal", phase: str = "stopped"):
    exp_dir = base / name
    exp_dir.mkdir(parents=True)
    state = ExperimentState(domain="auto", goal=goal, phase=phase, iteration=5)
    state.save(exp_dir / "state.json")
    return exp_dir


class TestHomeScreenComposition:
    async def test_has_preset_buttons(self):
        app = HomeScreenApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            home = app._home
            buttons = home.query("Button")
            ids = [b.id or "" for b in buttons]
            assert "preset-default" in ids
            assert "preset-fast" in ids
            assert "preset-max" in ids

    async def test_has_config_form(self):
        app = HomeScreenApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            forms = app._home.query(ConfigForm)
            assert len(forms) == 1

    async def test_has_past_runs_table(self):
        app = HomeScreenApp()
        async with app.run_test() as pilot:
            await pilot.pause()
            tables = app._home.query(DataTable)
            assert len(tables) == 1


class TestHomeScreenPastRuns:
    async def test_populates_from_store(self, tmp_path: Path):
        _create_experiment(tmp_path, "exp1", goal="My experiment")
        store = FilesystemStore(tmp_path)
        app = HomeScreenApp(store=store)
        async with app.run_test() as pilot:
            await pilot.pause()
            table = app._home.query_one(DataTable)
            assert table.row_count == 1

    async def test_empty_store_shows_empty_table(self, tmp_path: Path):
        store = FilesystemStore(tmp_path)
        app = HomeScreenApp(store=store)
        async with app.run_test() as pilot:
            await pilot.pause()
            table = app._home.query_one(DataTable)
            assert table.row_count == 0

    async def test_no_store_shows_empty_table(self):
        app = HomeScreenApp(store=None)
        async with app.run_test() as pilot:
            await pilot.pause()
            table = app._home.query_one(DataTable)
            assert table.row_count == 0
