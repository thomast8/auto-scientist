"""Tests for iteration manifest persistence."""

import json

import pytest
from auto_core.iteration_manifest import (
    IterationRecord,
    PanelRecord,
    append_record,
    load_manifest,
    save_manifest,
)


@pytest.fixture
def sample_panel():
    return PanelRecord(
        name="Analyst",
        model="claude-sonnet-4-6",
        style="green",
        done_summary="Data characterization complete",
        input_tokens=1200,
        output_tokens=3400,
        num_turns=3,
        elapsed_seconds=43.5,
    )


@pytest.fixture
def sample_record(sample_panel):
    return IterationRecord(
        iteration=0,
        title="Iteration 0",
        result_text="completed",
        result_style="green",
        summary="Explored dataset with 500 rows",
        panels=[sample_panel],
    )


class TestManifestRoundtrip:
    def test_save_and_load_preserves_all_fields(self, tmp_path, sample_record):
        path = tmp_path / "manifest.json"
        save_manifest([sample_record], path)
        loaded = load_manifest(path)

        assert len(loaded) == 1
        rec = loaded[0]
        assert rec.iteration == 0
        assert rec.title == "Iteration 0"
        assert rec.result_text == "completed"
        assert rec.result_style == "green"
        assert rec.summary == "Explored dataset with 500 rows"
        assert len(rec.panels) == 1

        p = rec.panels[0]
        assert p.name == "Analyst"
        assert p.model == "claude-sonnet-4-6"
        assert p.style == "green"
        assert p.done_summary == "Data characterization complete"
        assert p.input_tokens == 1200
        assert p.output_tokens == 3400
        assert p.num_turns == 3
        assert p.elapsed_seconds == 43.5

    def test_load_missing_file_returns_empty(self, tmp_path):
        path = tmp_path / "nonexistent.json"
        assert load_manifest(path) == []

    def test_append_record(self, tmp_path, sample_record):
        path = tmp_path / "manifest.json"

        # First append creates the file
        append_record(sample_record, path)
        assert len(load_manifest(path)) == 1

        # Second append adds to it
        second = IterationRecord(
            iteration=1,
            title="Iteration 1",
            panels=[],
        )
        append_record(second, path)
        records = load_manifest(path)
        assert len(records) == 2
        assert records[0].iteration == 0
        assert records[1].iteration == 1

    def test_ingestion_record(self, tmp_path, sample_panel):
        path = tmp_path / "manifest.json"
        rec = IterationRecord(
            iteration="ingestion",
            title="Ingestion",
            result_text="done",
            result_style="green",
            panels=[sample_panel],
        )
        save_manifest([rec], path)
        loaded = load_manifest(path)
        assert loaded[0].iteration == "ingestion"

    def test_save_creates_parent_dirs(self, tmp_path, sample_record):
        path = tmp_path / "nested" / "dir" / "manifest.json"
        save_manifest([sample_record], path)
        assert path.exists()

    def test_valid_json_output(self, tmp_path, sample_record):
        path = tmp_path / "manifest.json"
        save_manifest([sample_record], path)
        data = json.loads(path.read_text())
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["iteration"] == 0
