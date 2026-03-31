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
from auto_scientist.resume import RewindResult, rewind_run
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
        # Add plan.json and debate.json for agent-level resume tests
        plan = {"hypothesis": f"Hypothesis {i}", "strategy": f"Strategy {i}"}
        (vdir / "plan.json").write_text(json.dumps(plan))
        if i > 0:
            debate = {
                "original_plan": {
                    "hypothesis": f"Original {i}",
                    "strategy": f"Original strategy {i}",
                },
                "debate_results": [],
            }
            (vdir / "debate.json").write_text(json.dumps(debate))

    # Buffers
    buffers = d / "buffers"
    buffers.mkdir()
    for agent in ("analyst", "scientist", "coder"):
        for i in range(3):
            (buffers / f"{agent}_{i:02d}.txt").write_text(f"buffer {agent} {i}")
    (buffers / "report_03.txt").write_text("report buffer")
    (buffers / "debate_methodologist_01.txt").write_text("debate buffer")
    (buffers / "debate_methodologist_02.txt").write_text("debate buffer 2")
    (buffers / "scientist_revision_01.txt").write_text("revision buffer")
    (buffers / "scientist_revision_02.txt").write_text("revision buffer 2")

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
        '<entry version="v01" source="revision">\n'
        "  <title>Revised 1</title>\n"
        "  <content>Revised plan</content>\n"
        "</entry>\n"
        '<entry version="v02" source="scientist">\n'
        "  <title>Hypothesis 2</title>\n"
        "  <content>Third plan</content>\n"
        "</entry>\n"
        '<entry version="v02" source="revision">\n'
        "  <title>Revised 2</title>\n"
        "  <content>Revised plan 2</content>\n"
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
            panels=[
                PanelRecord(
                    name="Analyst",
                    model="claude-sonnet-4-6",
                    elapsed_seconds=30,
                    input_tokens=5000,
                    output_tokens=1000,
                    done_summary="Analysis complete",
                ),
                PanelRecord(
                    name="Scientist",
                    model="claude-opus-4-6",
                    elapsed_seconds=45,
                    input_tokens=8000,
                    output_tokens=2000,
                    done_summary="Plan formulated",
                ),
                PanelRecord(
                    name="Critic/Methodologist",
                    model="claude-opus-4-6",
                    elapsed_seconds=20,
                ),
                PanelRecord(
                    name="Coder",
                    model="claude-sonnet-4-6",
                    elapsed_seconds=60,
                    input_tokens=10000,
                    output_tokens=3000,
                    done_summary="Script written",
                ),
            ],
        ),
        IterationRecord(
            iteration=2,
            title="Iteration 2",
            result_text="completed",
            result_style="green",
            panels=[
                PanelRecord(
                    name="Analyst",
                    model="claude-sonnet-4-6",
                    elapsed_seconds=35,
                    input_tokens=6000,
                    output_tokens=1200,
                    done_summary="Analysis v2 complete",
                ),
                PanelRecord(
                    name="Scientist",
                    model="claude-opus-4-6",
                    elapsed_seconds=50,
                    input_tokens=9000,
                    output_tokens=2500,
                    done_summary="Plan v2 formulated",
                ),
                PanelRecord(
                    name="Coder",
                    model="claude-sonnet-4-6",
                    elapsed_seconds=55,
                ),
            ],
        ),
    ]
    save_manifest(records, d / MANIFEST_FILENAME)

    return d


class TestRewindRun:
    def test_sets_phase_and_iteration(self, run_dir):
        result = rewind_run(run_dir, 1)
        assert result.state.phase == "iteration"
        assert result.state.iteration == 1

    def test_returns_rewind_result(self, run_dir):
        result = rewind_run(run_dir, 1)
        assert isinstance(result, RewindResult)
        assert result.from_agent is None
        assert result.restored_panels is None

    def test_trims_versions(self, run_dir):
        result = rewind_run(run_dir, 1)
        assert len(result.state.versions) == 1
        assert result.state.versions[0].version == "v00"

    def test_trims_predictions(self, run_dir):
        result = rewind_run(run_dir, 1)
        assert len(result.state.prediction_history) == 1
        assert result.state.prediction_history[0].pred_id == "0.1"

    def test_resets_consecutive_failures(self, run_dir):
        # Set failures before rewinding
        s = ExperimentState.load(run_dir / "state.json")
        s.consecutive_failures = 3
        s.save(run_dir / "state.json")

        result = rewind_run(run_dir, 1)
        assert result.state.consecutive_failures == 0

    def test_clears_dead_ends(self, run_dir):
        s = ExperimentState.load(run_dir / "state.json")
        s.dead_ends = ["dead1", "dead2"]
        s.save(run_dir / "state.json")

        result = rewind_run(run_dir, 1)
        assert result.state.dead_ends == []

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
        result = rewind_run(run_dir, 1)
        resolved = str(run_dir.resolve())
        assert result.state.data_path.startswith(resolved)
        assert result.state.config_path.startswith(resolved)
        assert result.state.versions[0].script_path.startswith(resolved)
        assert result.state.versions[0].results_path.startswith(resolved)
        # raw_data_path should be unchanged
        assert result.state.raw_data_path == "/original/seed/data"

    def test_reconstructs_domain_knowledge_iter1(self, run_dir):
        result = rewind_run(run_dir, 1)
        assert result.state.domain_knowledge == "Knowledge from v00 analysis"

    def test_reconstructs_domain_knowledge_iter0(self, run_dir):
        result = rewind_run(run_dir, 0)
        assert result.state.domain_knowledge == ""

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
        result = rewind_run(run_dir, 0)
        assert result.state.iteration == 0
        assert result.state.versions == []
        assert result.state.prediction_history == []
        assert result.state.domain_knowledge == ""

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
        result = rewind_run(run_dir, 3)
        assert result.state.phase == "iteration"
        assert result.state.iteration == 3
        assert len(result.state.versions) == 3
        assert (run_dir / "v00").exists()
        assert (run_dir / "v01").exists()
        assert (run_dir / "v02").exists()

    def test_extend_removes_report_artifacts(self, run_dir):
        rewind_run(run_dir, 3)
        assert not (run_dir / "report.md").exists()

    def test_extend_preserves_predictions(self, run_dir):
        result = rewind_run(run_dir, 3)
        assert len(result.state.prediction_history) == 3

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

    def test_from_agent_in_extend_mode_errors_when_version_dir_missing(self, run_dir):
        """--from-agent at current iteration should error if version dir doesn't exist.

        If the iteration hasn't started yet (no version dir), there are no
        prior agent artifacts to load from, so we raise rather than silently
        changing behavior.
        """
        with pytest.raises(ValueError, match="v03.*does not exist"):
            rewind_run(run_dir, 3, from_agent="scientist")


class TestRewindWithoutManifest:
    def test_missing_manifest_is_fine(self, run_dir):
        """Runs from before the manifest feature should work."""
        (run_dir / MANIFEST_FILENAME).unlink()
        result = rewind_run(run_dir, 1)
        assert result.state.phase == "iteration"
        assert result.state.iteration == 1
        # Manifest file should not be created
        assert not (run_dir / MANIFEST_FILENAME).exists()


# ---------------------------------------------------------------------------
# Agent-level resume tests
# ---------------------------------------------------------------------------


class TestAgentLevelResume:
    """Tests for --from-agent functionality in rewind_run."""

    def test_from_agent_scientist_keeps_analysis(self, run_dir):
        """Resuming from scientist keeps analysis.json, removes plan + debate + coder artifacts."""
        result = rewind_run(run_dir, 2, from_agent="scientist")
        assert result.from_agent == "scientist"

        vdir = run_dir / "v02"
        assert vdir.exists()
        assert (vdir / "analysis.json").exists()
        assert not (vdir / "plan.json").exists()
        assert not (vdir / "debate.json").exists()
        assert not (vdir / "experiment.py").exists()

    def test_from_agent_debate_keeps_analysis_and_restores_plan(self, run_dir):
        """Resuming from debate keeps analysis + restores pre-debate plan from debate.json."""
        result = rewind_run(run_dir, 2, from_agent="debate")
        assert result.from_agent == "debate"

        vdir = run_dir / "v02"
        assert (vdir / "analysis.json").exists()
        # plan.json should be the pre-debate original, not the post-debate revision
        assert (vdir / "plan.json").exists()
        plan = json.loads((vdir / "plan.json").read_text())
        assert plan["hypothesis"] == "Original 2"
        # debate.json itself should be deleted
        assert not (vdir / "debate.json").exists()

    def test_from_agent_coder_keeps_analysis_plan_debate(self, run_dir):
        """Resuming from coder keeps analysis + plan + debate, removes coder artifacts."""
        result = rewind_run(run_dir, 2, from_agent="coder")
        assert result.from_agent == "coder"

        vdir = run_dir / "v02"
        assert (vdir / "analysis.json").exists()
        assert (vdir / "plan.json").exists()
        assert (vdir / "debate.json").exists()
        # Coder artifacts should be gone
        assert not (vdir / "experiment.py").exists()
        assert not (vdir / "results.txt").exists()

    def test_from_agent_analyst_normalizes_to_none(self, run_dir):
        """from_agent='analyst' is equivalent to a full restart."""
        result = rewind_run(run_dir, 2, from_agent="analyst")
        assert result.from_agent is None
        # Version dir should be fully deleted (same as no from_agent)
        assert not (run_dir / "v02").exists()

    def test_from_agent_unknown_raises(self, run_dir):
        with pytest.raises(ValueError, match="Unknown agent"):
            rewind_run(run_dir, 2, from_agent="unknown_agent")

    def test_from_agent_buffers_selective_cleanup(self, run_dir):
        """Buffers for skipped agents are kept, buffers for re-run agents are deleted."""
        rewind_run(run_dir, 2, from_agent="scientist")
        buffers = run_dir / "buffers"
        remaining = sorted(f.name for f in buffers.iterdir())
        # Analyst buffer at iteration 2 should be kept
        assert "analyst_02.txt" in remaining
        # Scientist buffer at iteration 2 should be deleted
        assert "scientist_02.txt" not in remaining
        # Debate and coder buffers at iteration 2 should be deleted
        assert "debate_methodologist_02.txt" not in remaining
        assert "coder_02.txt" not in remaining
        # Revision buffer at iteration 2 should be deleted
        assert "scientist_revision_02.txt" not in remaining
        # Earlier iteration buffers should all be preserved
        assert "analyst_01.txt" in remaining
        assert "scientist_01.txt" in remaining

    def test_from_agent_debate_buffers(self, run_dir):
        """Resuming from debate keeps analyst + scientist buffers, removes debate + coder."""
        rewind_run(run_dir, 2, from_agent="debate")
        buffers = run_dir / "buffers"
        remaining = sorted(f.name for f in buffers.iterdir())
        assert "analyst_02.txt" in remaining
        assert "scientist_02.txt" in remaining
        assert "debate_methodologist_02.txt" not in remaining
        assert "scientist_revision_02.txt" not in remaining
        assert "coder_02.txt" not in remaining

    def test_from_agent_notebook_selective_strip(self, run_dir):
        """Notebook entries for re-run agents are stripped, earlier ones kept."""
        rewind_run(run_dir, 2, from_agent="debate")
        notebook = (run_dir / NOTEBOOK_FILENAME).read_text()
        # v02 scientist entry should be kept (scientist is before debate)
        assert 'version="v02" source="scientist"' in notebook
        # v02 revision entry should be removed (revision is part of debate)
        assert 'version="v02" source="revision"' not in notebook
        # Earlier entries should all be preserved
        assert 'version="v01" source="scientist"' in notebook
        assert 'version="v01" source="revision"' in notebook

    def test_from_agent_scientist_strips_both_notebook_entries(self, run_dir):
        """Resuming from scientist strips both scientist and revision entries at target."""
        rewind_run(run_dir, 2, from_agent="scientist")
        notebook = (run_dir / NOTEBOOK_FILENAME).read_text()
        assert 'version="v02" source="scientist"' not in notebook
        assert 'version="v02" source="revision"' not in notebook
        # Earlier iterations preserved
        assert 'version="v01" source="scientist"' in notebook

    def test_from_agent_prediction_trimming_scientist(self, run_dir):
        """Resuming from scientist removes predictions prescribed at target iteration."""
        result = rewind_run(run_dir, 2, from_agent="scientist")
        # Predictions prescribed at iteration 0 and 1 should be kept
        # Prediction prescribed at iteration 2 should be removed (scientist will re-prescribe)
        prescribed_iters = [p.iteration_prescribed for p in result.state.prediction_history]
        assert 0 in prescribed_iters
        assert 1 in prescribed_iters
        assert 2 not in prescribed_iters

    def test_from_agent_prediction_trimming_coder(self, run_dir):
        """Resuming from coder keeps all predictions (coder doesn't touch predictions)."""
        result = rewind_run(run_dir, 2, from_agent="coder")
        prescribed_iters = [p.iteration_prescribed for p in result.state.prediction_history]
        assert 0 in prescribed_iters
        assert 1 in prescribed_iters
        assert 2 in prescribed_iters

    def test_from_agent_domain_knowledge_includes_target(self, run_dir):
        """When from_agent is set, domain_knowledge includes the target iteration's analysis."""
        result = rewind_run(run_dir, 1, from_agent="scientist")
        # v00 has domain_knowledge, v01 does not; should still get v00's
        assert result.state.domain_knowledge == "Knowledge from v00 analysis"

    def test_from_agent_manifest_trimmed(self, run_dir):
        """Manifest should not include the target iteration (it will be re-recorded)."""
        rewind_run(run_dir, 2, from_agent="scientist")
        records = load_manifest(run_dir / MANIFEST_FILENAME)
        iterations = [r.iteration for r in records]
        assert 2 not in iterations
        # Earlier iterations preserved
        assert "ingestion" in iterations
        assert 0 in iterations
        assert 1 in iterations

    def test_from_agent_restored_panels(self, run_dir):
        """restored_panels should contain panel records for agents before from_agent."""
        result = rewind_run(run_dir, 2, from_agent="scientist")
        assert result.restored_panels is not None
        assert len(result.restored_panels) == 1
        assert result.restored_panels[0]["name"] == "Analyst"
        assert result.restored_panels[0]["elapsed_seconds"] == 35
        assert result.restored_panels[0]["input_tokens"] == 6000
        assert result.restored_panels[0]["done_summary"] == "Analysis v2 complete"

    def test_from_agent_restored_panels_debate(self, run_dir):
        """Resuming from debate should restore Analyst + Scientist panels."""
        result = rewind_run(run_dir, 2, from_agent="debate")
        assert result.restored_panels is not None
        names = [p["name"] for p in result.restored_panels]
        assert "Analyst" in names
        assert "Scientist" in names
        assert "Coder" not in names

    def test_from_agent_restored_panels_coder(self, run_dir):
        """Resuming from coder should restore Analyst + Scientist + Critic panels."""
        result = rewind_run(run_dir, 1, from_agent="coder")
        assert result.restored_panels is not None
        names = [p["name"] for p in result.restored_panels]
        assert "Analyst" in names
        assert "Scientist" in names
        assert "Critic/Methodologist" in names
        assert "Coder" not in names

    def test_from_agent_no_manifest_returns_none_panels(self, run_dir):
        """Without a manifest, restored_panels should be None."""
        (run_dir / MANIFEST_FILENAME).unlink()
        result = rewind_run(run_dir, 2, from_agent="scientist")
        assert result.restored_panels is None

    def test_from_agent_prerequisite_validation(self, run_dir):
        """Missing prerequisite artifacts should raise ValueError."""
        # Delete analysis.json from v02
        (run_dir / "v02" / "analysis.json").unlink()
        with pytest.raises(ValueError, match="required artifact 'analysis.json' not found"):
            rewind_run(run_dir, 2, from_agent="scientist")

    def test_from_agent_prerequisite_validation_debate(self, run_dir):
        """Resuming from debate requires analysis.json and plan.json."""
        (run_dir / "v02" / "plan.json").unlink()
        with pytest.raises(ValueError, match="required artifact 'plan.json' not found"):
            rewind_run(run_dir, 2, from_agent="debate")
