"""Tests for the TUI launch form."""

import json
from unittest.mock import patch

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
            assert app.query_one("#top-provider-select") is not None
            assert app.query_one("#max-iterations-input") is not None
            assert app.query_one("#output-dir-input") is not None

    async def test_mode_column_exists_for_all_agents(self):
        app = LaunchApp()
        async with app.run_test() as _pilot:
            from textual.widgets import Select

            # SDK agents
            for agent in ["ingestor", "analyst", "scientist", "coder", "report"]:
                mode_sel = app.query_one(f"#model-{agent}-mode", Select)
                assert mode_sel.value == "sdk"

            # First critic slot
            mode_sel = app.query_one("#model-critic-0-mode", Select)
            assert mode_sel is not None

            # Trailing agents
            mode_sel = app.query_one("#model-summarizer-mode", Select)
            assert mode_sel is not None

    async def test_top_provider_defaults_to_anthropic(self):
        app = LaunchApp()
        async with app.run_test() as _pilot:
            from textual.widgets import Select

            assert app.query_one("#top-provider-select", Select).value == "anthropic"

    async def test_agent_provider_not_locked(self):
        """All agent provider dropdowns should be enabled (not locked to anthropic)."""
        app = LaunchApp()
        async with app.run_test() as _pilot:
            from textual.widgets import Select

            for agent in ["ingestor", "analyst", "scientist", "coder", "report"]:
                sel = app.query_one(f"#model-{agent}-provider", Select)
                assert not sel.disabled

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

    async def test_prefill_with_provider(self):
        cfg = ExperimentConfig(
            data="/path/to/data.csv",
            goal="Test goal",
            provider="openai",
        )
        app = LaunchApp(prefill=cfg)
        async with app.run_test() as _pilot:
            from textual.widgets import Select

            assert app.query_one("#top-provider-select", Select).value == "openai"


class TestLaunchAppRun:
    @patch.object(LaunchApp, "_validate_models", return_value=[])
    async def test_ctrl_r_stores_config(self, _mock_validate):
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

    @patch.object(LaunchApp, "_validate_models", return_value=[])
    async def test_ctrl_r_stores_provider(self, _mock_validate):
        app = LaunchApp()
        async with app.run_test(size=(120, 50)) as pilot:
            from textual.widgets import Input, Select, TextArea

            # Wait for initial mount/refresh to complete before changing values
            await pilot.pause()
            await pilot.pause()

            app.query_one("#data-input", Input).value = "/tmp/data.csv"
            app.query_one("#goal-input", TextArea).text = "Test goal"
            app.query_one("#top-provider-select", Select).value = "openai"
            await pilot.pause()
            await pilot.pause()

            await pilot.press("ctrl+r")

            assert app.result_config is not None
            assert app.result_config.provider == "openai"

    @patch.object(LaunchApp, "_validate_models", return_value=[])
    async def test_ctrl_r_stores_mode(self, _mock_validate):
        app = LaunchApp()
        async with app.run_test(size=(120, 50)) as pilot:
            from textual.widgets import Input, Select, TextArea

            app.query_one("#data-input", Input).value = "/tmp/data.csv"
            app.query_one("#goal-input", TextArea).text = "Test goal"
            # Change coder to API mode
            app.query_one("#model-coder-mode", Select).value = "api"
            await pilot.pause()

            await pilot.press("ctrl+r")

            assert app.result_config is not None
            assert app.result_config.models is not None
            assert app.result_config.models.coder is not None
            assert app.result_config.models.coder.mode == "api"

    async def test_ctrl_r_validates_required_fields(self):
        app = LaunchApp()
        async with app.run_test(size=(120, 50)) as pilot:
            # Don't fill in data/goal
            await pilot.press("ctrl+r")

            # App should still be running (not exited)
            assert app.result_config is None


class TestLaunchAppTheme:
    async def test_saved_theme_is_restored(self, tmp_path):
        prefs_path = tmp_path / "preferences.json"
        prefs_path.write_text(json.dumps({"theme": "atom-one-light"}))

        with (
            patch("auto_scientist.preferences.PREFS_PATH", prefs_path),
            patch("auto_scientist.preferences.system_is_dark", return_value=False),
        ):
            app = LaunchApp()
            async with app.run_test() as pilot:
                await pilot.pause()
                assert app.theme == "atom-one-light"

    async def test_theme_changes_are_persisted(self, tmp_path):
        prefs_path = tmp_path / "preferences.json"

        with patch("auto_scientist.preferences.PREFS_PATH", prefs_path):
            app = LaunchApp()
            async with app.run_test() as pilot:
                await pilot.pause()
                app.theme = "atom-one-light"
                await pilot.pause()

        assert json.loads(prefs_path.read_text())["theme"] == "atom-one-light"


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
