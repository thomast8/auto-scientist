"""Tests for experiment state persistence and crash recovery."""

import json

import pytest
from auto_core.state import ExperimentState, PredictionRecord, VersionEntry


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
            hypothesis="Test hypothesis",
            status="completed",
        )
        state.record_version(entry)
        state.save(tmp_state_path)

        loaded = ExperimentState.load(tmp_state_path)
        assert len(loaded.versions) == 1
        assert loaded.versions[0].version == "v1.01"
        assert loaded.versions[0].hypothesis == "Test hypothesis"

    def test_record_version_appends(self):
        state = ExperimentState(domain="test", goal="g")
        state.record_version(VersionEntry(version="v1", iteration=1, script_path="a"))
        state.record_version(VersionEntry(version="v2", iteration=2, script_path="b"))
        state.record_version(VersionEntry(version="v3", iteration=3, script_path="c"))
        assert len(state.versions) == 3

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

    def test_domain_knowledge_defaults_to_empty(self):
        state = ExperimentState(domain="test", goal="g")
        assert state.domain_knowledge == ""

    def test_domain_knowledge_roundtrip(self, tmp_state_path):
        state = ExperimentState(domain="test", goal="g", domain_knowledge="test knowledge")
        state.save(tmp_state_path)
        loaded = ExperimentState.load(tmp_state_path)
        assert loaded.domain_knowledge == "test knowledge"


class TestVersionEntry:
    def test_defaults(self):
        entry = VersionEntry(version="v01", iteration=1, script_path="/tmp/s.py")
        assert entry.results_path is None
        assert entry.hypothesis == ""
        assert entry.status == "pending"

    def test_serialization_roundtrip(self):
        entry = VersionEntry(
            version="v03",
            iteration=3,
            script_path="/tmp/s.py",
            results_path="/tmp/r.txt",
            hypothesis="Test hypothesis",
            status="completed",
        )
        json_str = entry.model_dump_json()
        loaded = VersionEntry.model_validate_json(json_str)
        assert loaded.version == "v03"
        assert loaded.status == "completed"
        assert loaded.hypothesis == "Test hypothesis"

    def test_all_statuses(self):
        for status in ["pending", "running", "completed", "failed", "crashed"]:
            entry = VersionEntry(
                version="v01",
                iteration=1,
                script_path="/tmp/s.py",
                status=status,
            )
            assert entry.status == status

    def test_failure_reason_defaults_to_none(self):
        entry = VersionEntry(version="v01", iteration=1, script_path="/tmp/s.py")
        assert entry.failure_reason is None

    def test_failure_reason_values(self):
        for reason in ["timed_out", "crash", "no_script", "no_result"]:
            entry = VersionEntry(
                version="v01",
                iteration=1,
                script_path="/tmp/s.py",
                failure_reason=reason,
            )
            assert entry.failure_reason == reason

    def test_failure_reason_roundtrip(self):
        entry = VersionEntry(
            version="v01",
            iteration=1,
            script_path="/tmp/s.py",
            status="failed",
            failure_reason="timed_out",
        )
        json_str = entry.model_dump_json()
        loaded = VersionEntry.model_validate_json(json_str)
        assert loaded.failure_reason == "timed_out"

    def test_failure_reason_backward_compat(self):
        """Loading JSON without failure_reason should default to None."""
        data = {
            "version": "v01",
            "iteration": 1,
            "script_path": "/tmp/s.py",
            "status": "failed",
        }
        entry = VersionEntry.model_validate(data)
        assert entry.failure_reason is None


class TestExperimentStateExtended:
    def test_record_version_appends_all(self):
        state = ExperimentState(domain="test", goal="g")
        state.record_version(VersionEntry(version="v1", iteration=1, script_path="a", score=5))
        state.record_version(VersionEntry(version="v2", iteration=2, script_path="b", score=0))
        assert len(state.versions) == 2

    def test_should_stop_custom_max(self):
        state = ExperimentState(domain="test", goal="g")
        state.record_failure()
        assert state.should_stop_on_failures(1)

    def test_multiple_versions_tracked(self):
        state = ExperimentState(domain="test", goal="g")
        for i in range(5):
            state.record_version(
                VersionEntry(version=f"v{i:02d}", iteration=i, script_path=f"s{i}.py", score=i),
            )
        assert len(state.versions) == 5

    def test_dead_ends_persistence(self, tmp_path):
        state = ExperimentState(domain="test", goal="g", dead_ends=["v01", "v03"])
        path = tmp_path / "state.json"
        state.save(path)
        loaded = ExperimentState.load(path)
        assert loaded.dead_ends == ["v01", "v03"]


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


# ---------------------------------------------------------------------------
# PredictionRecord
# ---------------------------------------------------------------------------


class TestPredictionRecord:
    def test_defaults(self):
        r = PredictionRecord(
            iteration_prescribed=1,
            prediction="test prediction",
            diagnostic="run diagnostic",
            if_confirmed="do A",
            if_refuted="do B",
        )
        assert r.outcome == "pending"
        assert r.evidence == ""
        assert r.iteration_evaluated is None
        assert r.follows_from is None

    def test_with_all_fields(self):
        r = PredictionRecord(
            iteration_prescribed=1,
            iteration_evaluated=2,
            prediction="spline is identifiable",
            diagnostic="profile smoothing parameter",
            if_confirmed="fine-tune",
            if_refuted="fix parameter",
            follows_from="initial hypothesis",
            outcome="refuted",
            evidence="CV loss flat for s in [0.5, 100]",
        )
        assert r.outcome == "refuted"
        assert r.follows_from == "initial hypothesis"

    def test_summary_defaults_to_empty(self):
        r = PredictionRecord(
            iteration_prescribed=1,
            prediction="test",
            diagnostic="d",
            if_confirmed="c",
            if_refuted="r",
        )
        assert r.summary == ""

    def test_summary_roundtrip(self):
        r = PredictionRecord(
            iteration_prescribed=1,
            prediction="test",
            diagnostic="d",
            if_confirmed="c",
            if_refuted="r",
            summary="Cr-corrosion r_s near zero; Ni dominates at 0.613",
        )
        json_str = r.model_dump_json()
        loaded = PredictionRecord.model_validate_json(json_str)
        assert loaded.summary == "Cr-corrosion r_s near zero; Ni dominates at 0.613"

    def test_summary_backward_compat(self):
        """Loading JSON without summary field should default to empty string."""
        data = {
            "iteration_prescribed": 1,
            "prediction": "test",
            "diagnostic": "d",
            "if_confirmed": "c",
            "if_refuted": "r",
        }
        r = PredictionRecord.model_validate(data)
        assert r.summary == ""


# ---------------------------------------------------------------------------
# ExperimentState - prediction_history
# ---------------------------------------------------------------------------


class TestExperimentStatePredictions:
    def test_defaults_to_empty(self):
        state = ExperimentState(domain="test", goal="g")
        assert state.prediction_history == []

    def test_roundtrip(self, tmp_path):
        state = ExperimentState(domain="test", goal="g")
        state.prediction_history.append(
            PredictionRecord(
                iteration_prescribed=1,
                prediction="noise is additive",
                diagnostic="compute residual-x correlation",
                if_confirmed="OLS is appropriate",
                if_refuted="switch to WLS",
                outcome="confirmed",
                evidence="correlation = 0.03",
                iteration_evaluated=1,
            )
        )
        path = tmp_path / "state.json"
        state.save(path)

        loaded = ExperimentState.load(path)
        assert len(loaded.prediction_history) == 1
        rec = loaded.prediction_history[0]
        assert rec.prediction == "noise is additive"
        assert rec.outcome == "confirmed"
        assert rec.iteration_evaluated == 1

    def test_backward_compat_missing_field(self, tmp_path):
        data = {"domain": "test", "goal": "g", "phase": "iteration", "iteration": 0}
        path = tmp_path / "state.json"
        path.write_text(json.dumps(data))
        loaded = ExperimentState.load(path)
        assert loaded.prediction_history == []

    def test_max_iterations_defaults_to_none(self):
        state = ExperimentState(domain="test", goal="g")
        assert state.max_iterations is None

    def test_max_iterations_roundtrip(self, tmp_path):
        state = ExperimentState(domain="test", goal="g", max_iterations=5)
        path = tmp_path / "state.json"
        state.save(path)
        loaded = ExperimentState.load(path)
        assert loaded.max_iterations == 5

    def test_backward_compat_missing_max_iterations(self, tmp_path):
        data = {"domain": "test", "goal": "g", "phase": "iteration", "iteration": 0}
        path = tmp_path / "state.json"
        path.write_text(json.dumps(data))
        loaded = ExperimentState.load(path)
        assert loaded.max_iterations is None
