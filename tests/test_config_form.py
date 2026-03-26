"""Tests for ConfigForm widget."""

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Select

from auto_scientist.ui.config_form import ConfigForm


class ConfigFormApp(App):
    def compose(self) -> ComposeResult:
        yield ConfigForm()


class TestConfigForm:
    async def test_composes_without_error(self):
        async with ConfigFormApp().run_test() as pilot:
            await pilot.pause()
            form = pilot.app.query_one(ConfigForm)
            assert form is not None

    async def test_default_preset_applied_on_mount(self):
        async with ConfigFormApp().run_test() as pilot:
            await pilot.pause()
            form = pilot.app.query_one(ConfigForm)
            assert form._selected_preset == "default"
            assert form._modified is False

    async def test_get_model_config_returns_valid(self):
        async with ConfigFormApp().run_test() as pilot:
            await pilot.pause()
            form = pilot.app.query_one(ConfigForm)
            mc = form.get_model_config()
            assert mc is not None
            assert mc.preset_name == "default"

    async def test_preset_select_changes_preset(self):
        async with ConfigFormApp().run_test() as pilot:
            await pilot.pause()
            form = pilot.app.query_one(ConfigForm)
            preset_select = form.query_one("#preset-select", Select)
            preset_select.value = "fast"
            await pilot.pause()
            assert form._selected_preset == "fast"

    async def test_modified_flag_set_on_input_change(self):
        async with ConfigFormApp().run_test() as pilot:
            await pilot.pause()
            form = pilot.app.query_one(ConfigForm)
            assert form._modified is False
            # Simulate changing the max iterations
            from textual.widgets import Input
            max_iter = form.query_one("#max-iter-input", Input)
            max_iter.value = "20"
            await pilot.pause()
            assert form._modified is True

    async def test_modified_config_has_modified_preset_name(self):
        async with ConfigFormApp().run_test() as pilot:
            await pilot.pause()
            form = pilot.app.query_one(ConfigForm)
            from textual.widgets import Input
            max_iter = form.query_one("#max-iter-input", Input)
            max_iter.value = "20"
            await pilot.pause()
            mc = form.get_model_config()
            assert "(modified)" in mc.preset_name
