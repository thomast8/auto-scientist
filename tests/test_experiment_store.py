"""Tests for ExperimentStore protocol and FilesystemStore implementation."""

import json
from pathlib import Path

import pytest

from auto_scientist.experiment_store import (
    ExperimentDetail,
    ExperimentSummary,
    FilesystemStore,
    IngestionSource,
    next_output_dir,
)
from auto_scientist.model_config import ModelConfig
from auto_scientist.state import ExperimentState


# ---------------------------------------------------------------------------
# Data model tests
# ---------------------------------------------------------------------------


class TestExperimentSummary:
    def test_construction(self):
        s = ExperimentSummary(
            id="exp_001",
            goal="Test goal",
            preset_name="fast",
            iteration=3,
            status="completed",
            output_dir="/tmp/experiments/exp_001",
            started_at="2026-03-26T10:00:00",
        )
        assert s.id == "exp_001"
        assert s.goal == "Test goal"
        assert s.preset_name == "fast"
        assert s.iteration == 3
        assert s.status == "completed"

    def test_default_preset_name_is_none(self):
        s = ExperimentSummary(
            id="x",
            goal="g",
            iteration=0,
            status="paused",
            output_dir="/tmp",
            started_at="2026-03-26",
        )
        assert s.preset_name is None


class TestIngestionSource:
    def test_construction(self):
        src = IngestionSource(
            id="exp_001",
            goal="Investigate SpO2",
            data_paths=["/data/spo2.csv"],
            output_dir="/tmp/experiments/exp_001",
        )
        assert src.id == "exp_001"
        assert src.data_paths == ["/data/spo2.csv"]


class TestExperimentDetail:
    def test_wraps_state_and_config(self):
        state = ExperimentState(domain="auto", goal="test")
        config = ModelConfig.builtin_preset("fast")
        detail = ExperimentDetail(state=state, models=config)
        assert detail.state.goal == "test"
        assert detail.models is not None


# ---------------------------------------------------------------------------
# FilesystemStore tests
# ---------------------------------------------------------------------------


def _create_experiment(
    base: Path,
    name: str,
    goal: str = "test goal",
    phase: str = "stopped",
    iteration: int = 5,
    data_path: str | None = "/data/test.csv",
    preset_name: str | None = "fast",
) -> Path:
    """Helper: create a fake experiment directory with state.json and model_config.json."""
    exp_dir = base / name
    exp_dir.mkdir(parents=True)

    state = ExperimentState(
        domain="auto",
        goal=goal,
        phase=phase,
        iteration=iteration,
        data_path=data_path,
    )
    state.save(exp_dir / "state.json")

    mc = ModelConfig.builtin_preset(preset_name or "default")
    (exp_dir / "model_config.json").write_text(mc.model_dump_json(indent=2))

    return exp_dir


class TestFilesystemStoreListExperiments:
    def test_empty_dir(self, tmp_path: Path):
        store = FilesystemStore(tmp_path)
        assert store.list_experiments() == []

    def test_finds_experiment_with_state_json(self, tmp_path: Path):
        _create_experiment(tmp_path, "exp1")
        store = FilesystemStore(tmp_path)
        results = store.list_experiments()
        assert len(results) == 1
        assert results[0].id == "exp1"

    def test_reads_goal_iteration_phase(self, tmp_path: Path):
        _create_experiment(
            tmp_path, "exp1", goal="My goal", phase="iteration", iteration=3,
        )
        store = FilesystemStore(tmp_path)
        result = store.list_experiments()[0]
        assert result.goal == "My goal"
        assert result.iteration == 3

    def test_derives_status_from_phase(self, tmp_path: Path):
        _create_experiment(tmp_path, "done", phase="stopped")
        _create_experiment(tmp_path, "running", phase="iteration")
        store = FilesystemStore(tmp_path)
        results = {r.id: r.status for r in store.list_experiments()}
        assert results["done"] == "completed"
        assert results["running"] == "paused"

    def test_skips_dirs_without_state_json(self, tmp_path: Path):
        (tmp_path / "not_an_experiment").mkdir()
        _create_experiment(tmp_path, "real_exp")
        store = FilesystemStore(tmp_path)
        assert len(store.list_experiments()) == 1

    def test_multiple_experiments_sorted_by_name(self, tmp_path: Path):
        _create_experiment(tmp_path, "b_exp")
        _create_experiment(tmp_path, "a_exp")
        store = FilesystemStore(tmp_path)
        ids = [r.id for r in store.list_experiments()]
        assert ids == ["a_exp", "b_exp"]


class TestFilesystemStoreIngestionSources:
    def test_returns_only_experiments_with_data_path(self, tmp_path: Path):
        _create_experiment(tmp_path, "has_data", data_path="/data/test.csv")
        _create_experiment(tmp_path, "no_data", data_path=None)
        store = FilesystemStore(tmp_path)
        sources = store.get_ingestion_sources()
        assert len(sources) == 1
        assert sources[0].id == "has_data"

    def test_ingestion_source_has_correct_fields(self, tmp_path: Path):
        _create_experiment(
            tmp_path, "exp1", goal="My goal", data_path="/data/test.csv",
        )
        store = FilesystemStore(tmp_path)
        source = store.get_ingestion_sources()[0]
        assert source.goal == "My goal"
        assert source.output_dir == str(tmp_path / "exp1")


class TestFilesystemStoreGetExperiment:
    def test_returns_full_detail(self, tmp_path: Path):
        _create_experiment(tmp_path, "exp1", goal="Test")
        store = FilesystemStore(tmp_path)
        detail = store.get_experiment("exp1")
        assert detail is not None
        assert detail.state.goal == "Test"
        assert detail.model_config is not None

    def test_returns_none_for_missing(self, tmp_path: Path):
        store = FilesystemStore(tmp_path)
        assert store.get_experiment("nonexistent") is None


class TestFilesystemStoreNoops:
    def test_register_is_noop(self, tmp_path: Path):
        store = FilesystemStore(tmp_path)
        state = ExperimentState(domain="auto", goal="test")
        mc = ModelConfig.builtin_preset("fast")
        result = store.register_experiment(tmp_path / "new", state, mc)
        assert result == "new"

    def test_update_status_is_noop(self, tmp_path: Path):
        store = FilesystemStore(tmp_path)
        store.update_status("exp1", "running", 5)  # Should not raise


# ---------------------------------------------------------------------------
# next_output_dir tests
# ---------------------------------------------------------------------------


class TestNextOutputDir:
    def test_returns_base_when_no_state_json(self, tmp_path: Path):
        base = tmp_path / "experiments"
        assert next_output_dir(base) == base

    def test_returns_numbered_when_state_exists(self, tmp_path: Path):
        base = tmp_path / "experiments"
        base.mkdir()
        (base / "state.json").write_text("{}")
        result = next_output_dir(base)
        assert result == tmp_path / "experiments_001"

    def test_increments_number(self, tmp_path: Path):
        base = tmp_path / "experiments"
        base.mkdir()
        (base / "state.json").write_text("{}")
        (tmp_path / "experiments_001").mkdir()
        (tmp_path / "experiments_001" / "state.json").write_text("{}")
        result = next_output_dir(base)
        assert result == tmp_path / "experiments_002"
