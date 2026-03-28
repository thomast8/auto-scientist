"""Tests for the checkpoint replay / rewind functionality."""

import json

import pytest

from auto_scientist.iteration_manifest import (
    MANIFEST_FILENAME,
    IterationRecord,
    PanelRecord,
    load_manifest,
    save_manifest,
)
from auto_scientist.notebook import NOTEBOOK_FILENAME
from auto_scientist.replay import rewind_run
from auto_scientist.state import ExperimentState, PredictionRecord, VersionEntry


@pytest.fixture
def run_dir(tmp_path):
    """Create a synthetic 3-iteration run directory."""
    d = tmp_path / "test_run"
    d.mkdir()

    old_output = "/original/experiments/test"

    # State with 3 completed iterations
    state = ExperimentState(
        domain="test",
        goal="Test goal",
        phase="stopped",
        iteration=3,
        data_path=f"{old_output}/data",
        raw_data_path="/original/seed/data",
        config_path=f"{old_output}/domain_config.json",
        domain_knowledge="Knowledge from v00 analysis",
        versions=[
            VersionEntry(
                version=f"v{i:02d}",
                iteration=i,
                script_path=f"{old_output}/v{i:02d}/experiment.py",
                results_path=f"{old_output}/v{i:02d}/results.txt",
                hypothesis=f"Hypothesis {i}",
                status="completed",
            )
            for i in range(3)
        ],
        prediction_history=[
            PredictionRecord(
                pred_id=f"{i}.1",
                iteration_prescribed=i,
                iteration_evaluated=i + 1 if i < 2 else None,
                prediction=f"Prediction from iteration {i}",
                diagnostic=f"Check metric {i}",
                if_confirmed="Continue",
                if_refuted="Revise",
                outcome="confirmed" if i < 2 else "pending",
            )
            for i in range(3)
        ],
    )
    state.save(d / "state.json")

    # Data directory
    (d / "data").mkdir()
    (d / "data" / "dataset.csv").write_text("col1,col2\n1,2\n")

    # Domain config
    (d / "domain_config.json").write_text('{"name": "test"}')

    # Model config
    (d / "model_config.json").write_text('{"analyst": {}}')

    # Version directories with analysis.json
    for i in range(3):
        vdir = d / f"v{i:02d}"
        vdir.mkdir()
        (vdir / "experiment.py").write_text(f"# iteration {i}")
        (vdir / "results.txt").write_text(f"results {i}")
        analysis = {}
        if i == 0:
            analysis["domain_knowledge"] = "Knowledge from v00 analysis"
        (vdir / "analysis.json").write_text(json.dumps(analysis))

    # Buffers
    buffers = d / "buffers"
    buffers.mkdir()
    for agent in ("analyst", "scientist", "coder"):
        for i in range(3):
            (buffers / f"{agent}_{i:02d}.txt").write_text(f"buffer {agent} {i}")
    (buffers / "report_03.txt").write_text("report buffer")
    (buffers / "debate_methodologist_01.txt").write_text("debate buffer")
    (buffers / "debate_methodologist_02.txt").write_text("debate buffer 2")

    # Lab notebook
    notebook = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        "<lab_notebook>\n"
        '<entry version="ingestion" source="ingestor">\n'
        "  <title>Ingestion</title>\n"
        "  <content>Ingested data</content>\n"
        "</entry>\n"
        '<entry version="v00" source="scientist">\n'
        "  <title>Exploration</title>\n"
        "  <content>First plan</content>\n"
        "</entry>\n"
        '<entry version="v01" source="scientist">\n'
        "  <title>Hypothesis 1</title>\n"
        "  <content>Second plan</content>\n"
        "</entry>\n"
        '<entry version="v02" source="scientist">\n'
        "  <title>Hypothesis 2</title>\n"
        "  <content>Third plan</content>\n"
        "</entry>\n"
        "</lab_notebook>\n"
    )
    (d / NOTEBOOK_FILENAME).write_text(notebook)

    # Final artifacts
    (d / "report.md").write_text("# Report")
    (d / "console.log").write_text("console output")
    (d / "debug.log").write_text("debug output")

    # Iteration manifest
    records = [
        IterationRecord(
            iteration="ingestion",
            title="Ingestion",
            result_text="done",
            result_style="green",
            panels=[PanelRecord(name="Ingestor", model="claude-sonnet-4-6")],
        ),
        IterationRecord(
            iteration=0,
            title="Iteration 0",
            result_text="completed",
            result_style="green",
            panels=[
                PanelRecord(name="Analyst", model="claude-sonnet-4-6", elapsed_seconds=40),
                PanelRecord(name="Scientist", model="claude-opus-4-6", elapsed_seconds=50),
            ],
        ),
        IterationRecord(
            iteration=1,
            title="Iteration 1",
            result_text="completed",
            result_style="green",
            panels=[],
        ),
        IterationRecord(
            iteration=2,
            title="Iteration 2",
            result_text="completed",
            result_style="green",
            panels=[],
        ),
    ]
    save_manifest(records, d / MANIFEST_FILENAME)

    return d


class TestRewindRun:
    def test_sets_phase_and_iteration(self, run_dir):
        state = rewind_run(run_dir, 1)
        assert state.phase == "iteration"
        assert state.iteration == 1

    def test_trims_versions(self, run_dir):
        state = rewind_run(run_dir, 1)
        assert len(state.versions) == 1
        assert state.versions[0].version == "v00"

    def test_trims_predictions(self, run_dir):
        state = rewind_run(run_dir, 1)
        assert len(state.prediction_history) == 1
        assert state.prediction_history[0].pred_id == "0.1"

    def test_resets_consecutive_failures(self, run_dir):
        # Set failures before rewinding
        s = ExperimentState.load(run_dir / "state.json")
        s.consecutive_failures = 3
        s.save(run_dir / "state.json")

        state = rewind_run(run_dir, 1)
        assert state.consecutive_failures == 0

    def test_clears_dead_ends(self, run_dir):
        s = ExperimentState.load(run_dir / "state.json")
        s.dead_ends = ["dead1", "dead2"]
        s.save(run_dir / "state.json")

        state = rewind_run(run_dir, 1)
        assert state.dead_ends == []

    def test_truncates_notebook(self, run_dir):
        rewind_run(run_dir, 1)
        notebook = (run_dir / NOTEBOOK_FILENAME).read_text()
        assert 'version="ingestion"' in notebook
        assert 'version="v00"' in notebook
        assert 'version="v01"' not in notebook
        assert 'version="v02"' not in notebook

    def test_deletes_version_dirs(self, run_dir):
        rewind_run(run_dir, 1)
        assert (run_dir / "v00").exists()
        assert not (run_dir / "v01").exists()
        assert not (run_dir / "v02").exists()

    def test_deletes_buffers(self, run_dir):
        rewind_run(run_dir, 1)
        buffers = run_dir / "buffers"
        remaining = sorted(f.name for f in buffers.iterdir())
        # Only iteration 0 buffers should remain
        assert "analyst_00.txt" in remaining
        assert "scientist_00.txt" in remaining
        assert "coder_00.txt" in remaining
        # Iteration 1+ buffers should be gone
        assert "analyst_01.txt" not in remaining
        assert "scientist_02.txt" not in remaining
        assert "debate_methodologist_01.txt" not in remaining
        assert "debate_methodologist_02.txt" not in remaining
        # Report buffer should be gone
        assert "report_03.txt" not in remaining

    def test_rewrites_paths(self, run_dir):
        state = rewind_run(run_dir, 1)
        resolved = str(run_dir.resolve())
        assert state.data_path.startswith(resolved)
        assert state.config_path.startswith(resolved)
        assert state.versions[0].script_path.startswith(resolved)
        assert state.versions[0].results_path.startswith(resolved)
        # raw_data_path should be unchanged
        assert state.raw_data_path == "/original/seed/data"

    def test_reconstructs_domain_knowledge_iter1(self, run_dir):
        state = rewind_run(run_dir, 1)
        assert state.domain_knowledge == "Knowledge from v00 analysis"

    def test_reconstructs_domain_knowledge_iter0(self, run_dir):
        state = rewind_run(run_dir, 0)
        assert state.domain_knowledge == ""

    def test_deletes_report_artifacts(self, run_dir):
        rewind_run(run_dir, 1)
        assert not (run_dir / "report.md").exists()
        assert not (run_dir / "console.log").exists()
        assert not (run_dir / "debug.log").exists()

    def test_trims_manifest(self, run_dir):
        rewind_run(run_dir, 1)
        records = load_manifest(run_dir / MANIFEST_FILENAME)
        # Should keep ingestion + iteration 0
        assert len(records) == 2
        assert records[0].iteration == "ingestion"
        assert records[1].iteration == 0

    def test_preserves_data_dir(self, run_dir):
        rewind_run(run_dir, 1)
        assert (run_dir / "data" / "dataset.csv").exists()

    def test_preserves_model_config(self, run_dir):
        rewind_run(run_dir, 1)
        assert (run_dir / "model_config.json").exists()

    def test_state_saved_to_disk(self, run_dir):
        rewind_run(run_dir, 1)
        loaded = ExperimentState.load(run_dir / "state.json")
        assert loaded.phase == "iteration"
        assert loaded.iteration == 1


class TestRewindValidation:
    def test_rejects_iteration_beyond_current(self, run_dir):
        with pytest.raises(ValueError, match="must be <="):
            rewind_run(run_dir, 4)

    def test_rejects_negative_iteration(self, run_dir):
        with pytest.raises(ValueError, match="must be >= 0"):
            rewind_run(run_dir, -1)

    def test_rejects_ingestion_phase(self, tmp_path):
        d = tmp_path / "bad_run"
        d.mkdir()
        state = ExperimentState(
            domain="test",
            goal="g",
            phase="ingestion",
            iteration=0,
        )
        state.save(d / "state.json")
        with pytest.raises(ValueError, match="Cannot rewind"):
            rewind_run(d, 0)


class TestRewindToZero:
    def test_rewind_to_zero_clears_everything(self, run_dir):
        state = rewind_run(run_dir, 0)
        assert state.iteration == 0
        assert state.versions == []
        assert state.prediction_history == []
        assert state.domain_knowledge == ""

        # All version dirs gone
        assert not (run_dir / "v00").exists()
        assert not (run_dir / "v01").exists()
        assert not (run_dir / "v02").exists()

        # Notebook only has ingestion
        notebook = (run_dir / NOTEBOOK_FILENAME).read_text()
        assert 'version="ingestion"' in notebook
        assert 'version="v00"' not in notebook

        # Manifest only has ingestion
        records = load_manifest(run_dir / MANIFEST_FILENAME)
        assert len(records) == 1
        assert records[0].iteration == "ingestion"


class TestExtendRun:
    def test_extend_preserves_all_versions(self, run_dir):
        """Rewinding to current iteration = extend (keep everything, reset phase)."""
        state = rewind_run(run_dir, 3)
        assert state.phase == "iteration"
        assert state.iteration == 3
        assert len(state.versions) == 3
        assert (run_dir / "v00").exists()
        assert (run_dir / "v01").exists()
        assert (run_dir / "v02").exists()

    def test_extend_removes_report_artifacts(self, run_dir):
        rewind_run(run_dir, 3)
        assert not (run_dir / "report.md").exists()

    def test_extend_preserves_predictions(self, run_dir):
        state = rewind_run(run_dir, 3)
        assert len(state.prediction_history) == 3

    def test_extend_preserves_report_phase_analyst_buffer(self, run_dir):
        """The analyst buffer at target_iteration was from _resolve_final_predictions."""
        buffers = run_dir / "buffers"
        (buffers / "analyst_03.txt").write_text("report-phase analyst")
        rewind_run(run_dir, 3)
        assert (buffers / "analyst_03.txt").exists()
        assert (buffers / "analyst_03.txt").read_text() == "report-phase analyst"

    def test_extend_deletes_non_analyst_buffers_at_target(self, run_dir):
        """Non-analyst buffers at target iteration should still be deleted."""
        buffers = run_dir / "buffers"
        (buffers / "scientist_03.txt").write_text("should be deleted")
        rewind_run(run_dir, 3)
        assert not (buffers / "scientist_03.txt").exists()


class TestRewindWithoutManifest:
    def test_missing_manifest_is_fine(self, run_dir):
        """Runs from before the manifest feature should work."""
        (run_dir / MANIFEST_FILENAME).unlink()
        state = rewind_run(run_dir, 1)
        assert state.phase == "iteration"
        assert state.iteration == 1
        # Manifest file should not be created
        assert not (run_dir / MANIFEST_FILENAME).exists()
