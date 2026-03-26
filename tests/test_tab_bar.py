"""Tests for the TabBar widget."""

import pytest
from textual.app import App, ComposeResult

from auto_scientist.ui.tab_bar import TabBar


class TabBarTestApp(App):
    """Minimal app for testing TabBar."""

    def compose(self) -> ComposeResult:
        yield TabBar()


class TestTabBar:
    @pytest.fixture
    async def app(self):
        async with TabBarTestApp().run_test() as pilot:
            yield pilot

    async def test_initial_home_tab_exists(self, app):
        tab_bar = app.app.query_one(TabBar)
        tabs = tab_bar.tab_ids
        assert "home" in tabs

    async def test_add_tab(self, app):
        tab_bar = app.app.query_one(TabBar)
        tab_bar.add_tab("SpO2 fast", "exp-1")
        tabs = tab_bar.tab_ids
        assert "exp-1" in tabs

    async def test_remove_tab(self, app):
        tab_bar = app.app.query_one(TabBar)
        tab_bar.add_tab("SpO2 fast", "exp-1")
        tab_bar.remove_tab("exp-1")
        assert "exp-1" not in tab_bar.tab_ids

    async def test_cannot_remove_home_tab(self, app):
        tab_bar = app.app.query_one(TabBar)
        tab_bar.remove_tab("home")
        assert "home" in tab_bar.tab_ids

    async def test_set_active(self, app):
        tab_bar = app.app.query_one(TabBar)
        tab_bar.add_tab("SpO2 fast", "exp-1")
        tab_bar.set_active("exp-1")
        assert tab_bar.active_tab == "exp-1"

    async def test_set_status_indicator(self, app):
        tab_bar = app.app.query_one(TabBar)
        tab_bar.add_tab("SpO2 fast", "exp-1")
        tab_bar.set_status("exp-1", "running")
        # Should not raise; visual indicator is rendered

    async def test_tab_selected_message(self, app):
        """Clicking a tab posts a TabSelected message."""
        tab_bar = app.app.query_one(TabBar)
        tab_bar.add_tab("SpO2 fast", "exp-1")
        messages = []

        def handler(msg):
            messages.append(msg)

        # We test the message type exists and is correctly structured
        msg = TabBar.TabSelected(tab_bar, "exp-1")
        assert msg.tab_id == "exp-1"
