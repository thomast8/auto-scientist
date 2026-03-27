"""Tests for the TUI launch form."""

import yaml

from auto_scientist.experiment_config import ExperimentConfig
from auto_scientist.launch_app import LaunchApp


class TestLaunchAppConstruction:
    async def test_app_has_expected_widgets(self):
        app = LaunchApp()
        async with app.run_test() as _pilot:
            # Verify key widgets exist
            assert app.query_one("#data-input") is not None
            assert app.query_one("#goal-input") is not None
            assert app.query_one("#preset-select") is not None
            assert app.query_one("#max-iterations-input") is not None
            assert app.query_one("#debate-rounds-input") is not None
            assert app.query_one("#output-dir-input") is not None

    async def test_prefill_from_config(self):
        cfg = ExperimentConfig(
            data="/path/to/data.csv",
            goal="Test goal text",
            max_iterations=10,
            preset="fast",
        )
        app = LaunchApp(prefill=cfg)
        async with app.run_test() as _pilot:
            from textual.widgets import Input, Select, TextArea

            assert app.query_one("#data-input", Input).value == "/path/to/data.csv"
            assert app.query_one("#goal-input", TextArea).text == "Test goal text"
            assert app.query_one("#max-iterations-input", Input).value == "10"
            assert app.query_one("#preset-select", Select).value == "fast"


class TestLaunchAppRun:
    async def test_ctrl_r_stores_config(self):
        app = LaunchApp()
        async with app.run_test(size=(120, 50)) as pilot:
            from textual.widgets import Input, TextArea

            app.query_one("#data-input", Input).value = "/tmp/data.csv"
            app.query_one("#goal-input", TextArea).text = "My test goal"

            await pilot.press("ctrl+r")

            assert app.result_config is not None
            assert app.result_config.data == "/tmp/data.csv"
            assert app.result_config.goal == "My test goal"
            assert app.result_config.preset == "default"

    async def test_ctrl_r_validates_required_fields(self):
        app = LaunchApp()
        async with app.run_test(size=(120, 50)) as pilot:
            # Don't fill in data/goal
            await pilot.press("ctrl+r")

            # App should still be running (not exited)
            assert app.result_config is None


class TestLaunchAppSave:
    async def test_ctrl_s_writes_yaml(self, tmp_path):
        app = LaunchApp(save_path=tmp_path / "saved.yaml")
        async with app.run_test(size=(120, 50)) as pilot:
            from textual.widgets import Input, TextArea

            app.query_one("#data-input", Input).value = "/tmp/data.csv"
            app.query_one("#goal-input", TextArea).text = "Save test"

            await pilot.press("ctrl+s")

        out = tmp_path / "saved.yaml"
        assert out.exists()
        raw = yaml.safe_load(out.read_text())
        assert raw["data"] == "/tmp/data.csv"
        assert raw["goal"] == "Save test"
