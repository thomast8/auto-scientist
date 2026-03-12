"""Tests for experiment state persistence and crash recovery."""

import json
from pathlib import Path

import pytest

from auto_scientist.state import ExperimentState, VersionEntry


@pytest.fixture
def sample_state():
    return ExperimentState(
        domain="test",
        goal="Test goal",
        phase="iteration",
        iteration=3,
    )


@pytest.fixture
def tmp_state_path(tmp_path):
    return tmp_path / "state.json"


class TestExperimentState:
    def test_save_and_load_roundtrip(self, sample_state, tmp_state_path):
        sample_state.save(tmp_state_path)
        loaded = ExperimentState.load(tmp_state_path)
        assert loaded.domain == "test"
        assert loaded.goal == "Test goal"
        assert loaded.phase == "iteration"
        assert loaded.iteration == 3

    def test_save_creates_parent_dirs(self, sample_state, tmp_path):
        nested_path = tmp_path / "a" / "b" / "state.json"
        sample_state.save(nested_path)
        assert nested_path.exists()

    def test_load_preserves_versions(self, tmp_state_path):
        state = ExperimentState(domain="test", goal="g")
        entry = VersionEntry(
            version="v1.01",
            iteration=1,
            script_path="/tmp/script.py",
            results_path="/tmp/results.txt",
            score=7,
            hypothesis="Test hypothesis",
            status="completed",
        )
        state.record_version(entry)
        state.save(tmp_state_path)

        loaded = ExperimentState.load(tmp_state_path)
        assert len(loaded.versions) == 1
        assert loaded.versions[0].version == "v1.01"
        assert loaded.versions[0].score == 7
        assert loaded.best_version == "v1.01"
        assert loaded.best_score == 7

    def test_record_version_updates_best(self):
        state = ExperimentState(domain="test", goal="g")
        state.record_version(VersionEntry(version="v1", iteration=1, script_path="a", score=5))
        assert state.best_version == "v1"
        state.record_version(VersionEntry(version="v2", iteration=2, script_path="b", score=8))
        assert state.best_version == "v2"
        assert state.best_score == 8
        # Lower score doesn't replace best
        state.record_version(VersionEntry(version="v3", iteration=3, script_path="c", score=3))
        assert state.best_version == "v2"

    def test_consecutive_failures(self):
        state = ExperimentState(domain="test", goal="g")
        assert not state.should_stop_on_failures(3)
        state.record_failure()
        state.record_failure()
        assert not state.should_stop_on_failures(3)
        state.record_failure()
        assert state.should_stop_on_failures(3)

    def test_success_resets_failures(self):
        state = ExperimentState(domain="test", goal="g")
        state.record_failure()
        state.record_failure()
        state.record_success()
        assert state.consecutive_failures == 0
        assert not state.should_stop_on_failures(3)

    def test_json_is_valid(self, sample_state, tmp_state_path):
        sample_state.save(tmp_state_path)
        data = json.loads(tmp_state_path.read_text())
        assert isinstance(data, dict)
        assert data["domain"] == "test"
