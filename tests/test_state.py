"""Tests for experiment state persistence and crash recovery."""

import json
from pathlib import Path

import pytest

from auto_scientist.config import SuccessCriterion
from auto_scientist.state import CriteriaRevision, ExperimentState, VersionEntry


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

    def test_raw_data_path_field(self):
        state = ExperimentState(domain="test", goal="g", raw_data_path="/raw/data.csv")
        assert state.raw_data_path == "/raw/data.csv"

    def test_raw_data_path_defaults_to_none(self):
        state = ExperimentState(domain="test", goal="g")
        assert state.raw_data_path is None

    def test_default_phase_is_ingestion(self):
        state = ExperimentState(domain="test", goal="g")
        assert state.phase == "ingestion"

    def test_raw_data_path_roundtrip(self, tmp_state_path):
        state = ExperimentState(domain="test", goal="g", raw_data_path="/raw/data.csv")
        state.save(tmp_state_path)
        loaded = ExperimentState.load(tmp_state_path)
        assert loaded.raw_data_path == "/raw/data.csv"

    def test_success_criteria_defaults_to_none(self):
        state = ExperimentState(domain="test", goal="g")
        assert state.success_criteria is None

    def test_domain_knowledge_defaults_to_empty(self):
        state = ExperimentState(domain="test", goal="g")
        assert state.domain_knowledge == ""

    def test_criteria_history_defaults_to_empty(self):
        state = ExperimentState(domain="test", goal="g")
        assert state.criteria_history == []

    def test_success_criteria_roundtrip(self, tmp_state_path):
        criteria = [
            SuccessCriterion(
                name="acc", description="accuracy", metric_key="accuracy",
                target_min=0.9,
            ),
        ]
        state = ExperimentState(domain="test", goal="g", success_criteria=criteria)
        state.save(tmp_state_path)
        loaded = ExperimentState.load(tmp_state_path)
        assert len(loaded.success_criteria) == 1
        assert loaded.success_criteria[0].name == "acc"
        assert loaded.success_criteria[0].target_min == 0.9

    def test_domain_knowledge_roundtrip(self, tmp_state_path):
        state = ExperimentState(domain="test", goal="g", domain_knowledge="test knowledge")
        state.save(tmp_state_path)
        loaded = ExperimentState.load(tmp_state_path)
        assert loaded.domain_knowledge == "test knowledge"


class TestCriteriaRevision:
    def test_construct(self):
        criteria = [
            SuccessCriterion(name="a", description="b", metric_key="c"),
        ]
        rev = CriteriaRevision(
            iteration=1,
            action="defined",
            changes="Initial criteria definition",
            criteria_snapshot=criteria,
        )
        assert rev.iteration == 1
        assert rev.action == "defined"
        assert len(rev.criteria_snapshot) == 1

    def test_serialize_deserialize_roundtrip(self, tmp_path):
        criteria = [
            SuccessCriterion(name="a", description="b", metric_key="c", target_min=0.5),
        ]
        rev = CriteriaRevision(
            iteration=2,
            action="revised",
            changes="Lowered target",
            criteria_snapshot=criteria,
        )
        path = tmp_path / "rev.json"
        path.write_text(rev.model_dump_json(indent=2))
        loaded = CriteriaRevision.model_validate_json(path.read_text())
        assert loaded.iteration == 2
        assert loaded.action == "revised"
        assert loaded.criteria_snapshot[0].target_min == 0.5

    def test_criteria_history_roundtrip(self, tmp_path):
        criteria = [SuccessCriterion(name="a", description="b", metric_key="c")]
        rev = CriteriaRevision(
            iteration=1, action="defined", changes="Initial",
            criteria_snapshot=criteria,
        )
        state = ExperimentState(
            domain="test", goal="g", criteria_history=[rev],
        )
        state_path = tmp_path / "state.json"
        state.save(state_path)
        loaded = ExperimentState.load(state_path)
        assert len(loaded.criteria_history) == 1
        assert loaded.criteria_history[0].action == "defined"


class TestDiscoveryPhaseMigration:
    def test_discovery_phase_migrates_to_iteration(self, tmp_path):
        """Loading state with phase='discovery' should migrate to phase='iteration', iteration=0."""
        state_path = tmp_path / "state.json"
        data = {
            "domain": "test",
            "goal": "test goal",
            "phase": "discovery",
            "iteration": 0,
            "versions": [],
        }
        state_path.write_text(json.dumps(data))
        loaded = ExperimentState.load(state_path)
        assert loaded.phase == "iteration"
        assert loaded.iteration == 0
