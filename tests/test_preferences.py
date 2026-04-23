"""Tests for user preference persistence (atomic write, load, round-trip)."""

import json

import pytest
from auto_core.preferences import load_preferences, save_preferences


@pytest.fixture()
def prefs_path(tmp_path, monkeypatch):
    """Redirect PREFS_PATH to a temp directory."""
    path = tmp_path / "config" / "auto-scientist" / "preferences.json"
    monkeypatch.setattr("auto_core.preferences.PREFS_PATH", path)
    return path


class TestLoadPreferences:
    def test_returns_empty_when_missing(self, prefs_path):
        assert load_preferences() == {}

    def test_returns_saved_dict(self, prefs_path):
        prefs_path.parent.mkdir(parents=True)
        prefs_path.write_text('{"theme": "dark"}')
        assert load_preferences() == {"theme": "dark"}

    def test_returns_empty_on_corrupt_json(self, prefs_path):
        prefs_path.parent.mkdir(parents=True)
        prefs_path.write_text("{broken")
        assert load_preferences() == {}

    def test_returns_empty_on_non_dict(self, prefs_path):
        prefs_path.parent.mkdir(parents=True)
        prefs_path.write_text("[1, 2, 3]")
        assert load_preferences() == {}


class TestSavePreferences:
    def test_round_trip(self, prefs_path):
        save_preferences({"theme": "nord", "count": 42})
        assert load_preferences() == {"theme": "nord", "count": 42}

    def test_creates_parent_directories(self, prefs_path):
        assert not prefs_path.parent.exists()
        save_preferences({"key": "value"})
        assert prefs_path.exists()
        assert load_preferences() == {"key": "value"}

    def test_overwrites_existing(self, prefs_path):
        save_preferences({"version": 1})
        save_preferences({"version": 2})
        assert load_preferences() == {"version": 2}

    def test_no_temp_files_left_on_success(self, prefs_path):
        save_preferences({"key": "value"})
        tmp_files = list(prefs_path.parent.glob("prefs-*.tmp"))
        assert tmp_files == []

    def test_cleans_temp_on_serialization_failure(self, prefs_path, monkeypatch):
        # Make json.dump fail by passing an unserializable value.
        # TypeError is not OSError, so it propagates after temp cleanup.
        class Boom:
            pass

        with pytest.raises(TypeError, match="not JSON serializable"):
            save_preferences({"bad": Boom()})  # type: ignore[dict-item]
        tmp_files = list(prefs_path.parent.glob("prefs-*.tmp"))
        assert tmp_files == []
        # Original file should not exist (no prior save)
        assert not prefs_path.exists()

    def test_atomic_replace_preserves_valid_on_failure(self, prefs_path):
        save_preferences({"original": True})
        assert load_preferences() == {"original": True}

        class Boom:
            pass

        with pytest.raises(TypeError):
            save_preferences({"bad": Boom()})  # type: ignore[dict-item]
        # Original content preserved since atomic replace never happened
        content = json.loads(prefs_path.read_text())
        assert content == {"original": True}
