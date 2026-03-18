"""Tests for the orchestrator state machine."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from auto_scientist.config import DomainConfig
from auto_scientist.orchestrator import Orchestrator
from auto_scientist.runner import RunResult
from auto_scientist.state import ExperimentState, VersionEntry


@pytest.fixture
def base_state():
    return ExperimentState(domain="test", goal="test goal", phase="ingestion")


@pytest.fixture
def config():
    return DomainConfig(
        name="test", description="Test domain", data_paths=["data.csv"],
        domain_knowledge="test knowledge",
    )


@pytest.fixture
def orchestrator(base_state, tmp_path, config):
    return Orchestrator(
        state=base_state,
        data_path=tmp_path / "data.csv",
        output_dir=tmp_path / "experiments",
        config=config,
    )


class TestOrchestratorInit:
    def test_defaults(self, base_state, tmp_path):
        o = Orchestrator(state=base_state, data_path=tmp_path, output_dir=tmp_path)
        assert o.max_iterations == 20
        assert o.critic_models == []
        assert o.debate_rounds == 2
        assert o.max_consecutive_failures == 5

    def test_custom_values(self, base_state, tmp_path):
        o = Orchestrator(
            state=base_state, data_path=tmp_path, output_dir=tmp_path,
            max_iterations=5, critic_models=["openai:gpt-4o"],
            debate_rounds=3, max_consecutive_failures=2,
        )
        assert o.max_iterations == 5
        assert o.critic_models == ["openai:gpt-4o"]
        assert o.debate_rounds == 3


class TestEvaluate:
    def test_none_result_records_failure(self, orchestrator):
        entry = VersionEntry(version="v01", iteration=1, script_path="/tmp/s.py")
        orchestrator._evaluate(None, entry)
        assert entry.status == "failed"
        assert orchestrator.state.consecutive_failures == 1

    def test_timed_out_records_failure(self, orchestrator):
        result = RunResult(success=False, timed_out=True)
        entry = VersionEntry(version="v01", iteration=1, script_path="/tmp/s.py")
        orchestrator._evaluate(result, entry)
        assert entry.status == "failed"
        assert orchestrator.state.consecutive_failures == 1

    def test_nonzero_exit_records_failure(self, orchestrator):
        result = RunResult(success=False, return_code=1)
        entry = VersionEntry(version="v01", iteration=1, script_path="/tmp/s.py")
        orchestrator._evaluate(result, entry)
        assert entry.status == "failed"
        assert orchestrator.state.consecutive_failures == 1

    def test_success_records_completion(self, orchestrator, tmp_path):
        result = RunResult(success=True, stdout="output", return_code=0)
        script_path = tmp_path / "experiments" / "v01" / "experiment.py"
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text("print('ok')")
        results_path = script_path.parent / "results.txt"
        results_path.write_text("results")

        entry = VersionEntry(version="v01", iteration=1, script_path=str(script_path))
        orchestrator._evaluate(result, entry)
        assert entry.status == "completed"
        assert orchestrator.state.consecutive_failures == 0
        assert entry.results_path == str(results_path)


class TestNotebookContent:
    def test_returns_content_when_exists(self, orchestrator, tmp_path):
        notebook = tmp_path / "experiments" / "lab_notebook.md"
        notebook.parent.mkdir(parents=True, exist_ok=True)
        notebook.write_text("# Notebook")
        assert orchestrator._notebook_content() == "# Notebook"

    def test_returns_empty_when_missing(self, orchestrator):
        assert orchestrator._notebook_content() == ""


class TestRunIngestion:
    @pytest.mark.asyncio
    async def test_raises_without_data_path(self, tmp_path):
        state = ExperimentState(domain="test", goal="g", phase="ingestion")
        o = Orchestrator(state=state, data_path=None, output_dir=tmp_path)
        with pytest.raises(ValueError, match="Cannot run ingestion"):
            await o._run_ingestion()

    @pytest.mark.asyncio
    @patch("auto_scientist.agents.ingestor.run_ingestor", new_callable=AsyncMock)
    async def test_returns_canonical_data_dir(self, mock_ingestor, tmp_path):
        canonical = tmp_path / "experiments" / "data"
        canonical.mkdir(parents=True, exist_ok=True)
        mock_ingestor.return_value = canonical

        state = ExperimentState(
            domain="test", goal="g", phase="ingestion",
            data_path=str(tmp_path / "raw.csv"),
        )
        o = Orchestrator(
            state=state, data_path=tmp_path / "raw.csv",
            output_dir=tmp_path / "experiments",
        )
        result = await o._run_ingestion()

        assert result == canonical
        mock_ingestor.assert_called_once()


class TestPhaseTransitions:
    @pytest.mark.asyncio
    async def test_max_iterations_triggers_report(self, tmp_path):
        state = ExperimentState(
            domain="test", goal="g", phase="iteration",
            iteration=20,
        )
        config = DomainConfig(name="t", description="d", data_paths=[])
        o = Orchestrator(
            state=state, data_path=tmp_path, output_dir=tmp_path,
            max_iterations=20, config=config,
        )

        with patch.object(o, "_run_report", new_callable=AsyncMock):
            await o.run()

        assert state.phase == "stopped"

    @pytest.mark.asyncio
    async def test_consecutive_failures_triggers_report(self, tmp_path):
        state = ExperimentState(
            domain="test", goal="g", phase="iteration",
            consecutive_failures=5,
        )
        config = DomainConfig(name="t", description="d", data_paths=[])
        o = Orchestrator(
            state=state, data_path=tmp_path, output_dir=tmp_path,
            max_consecutive_failures=5, config=config,
        )

        with patch.object(o, "_run_report", new_callable=AsyncMock):
            await o.run()

        assert state.phase == "stopped"

    @pytest.mark.asyncio
    async def test_resume_from_iteration_skips_ingestion_and_discovery(self, tmp_path):
        state = ExperimentState(
            domain="test", goal="g", phase="iteration",
            iteration=20,
        )
        config = DomainConfig(name="t", description="d", data_paths=[])
        o = Orchestrator(
            state=state, data_path=tmp_path, output_dir=tmp_path,
            max_iterations=20, config=config,
        )

        with patch.object(o, "_run_report", new_callable=AsyncMock) as mock_report:
            await o.run()

        mock_report.assert_called_once()
        assert state.phase == "stopped"


class TestRunIteration:
    @pytest.mark.asyncio
    async def test_scientist_stop_sets_report_phase(self, orchestrator, tmp_path):
        orchestrator.output_dir.mkdir(parents=True, exist_ok=True)
        orchestrator.state.phase = "iteration"
        orchestrator.state.versions = [
            VersionEntry(
                version="v00", iteration=0, script_path="/tmp/s.py",
                results_path=str(tmp_path / "results.txt"),
            ),
        ]
        (tmp_path / "results.txt").write_text("data")

        plan = {"should_stop": True, "stop_reason": "goal reached"}

        with (
            patch.object(orchestrator, "_run_analyst", new_callable=AsyncMock, return_value={}),
            patch.object(orchestrator, "_run_scientist_plan", new_callable=AsyncMock, return_value=plan),
        ):
            await orchestrator._run_iteration()

        assert orchestrator.state.phase == "report"

    @pytest.mark.asyncio
    async def test_no_critics_skips_debate(self, orchestrator, tmp_path):
        orchestrator.output_dir.mkdir(parents=True, exist_ok=True)
        orchestrator.critic_models = []
        orchestrator.state.phase = "iteration"
        orchestrator.state.versions = [
            VersionEntry(
                version="v00", iteration=0, script_path="/tmp/s.py",
                results_path=str(tmp_path / "results.txt"),
            ),
        ]
        (tmp_path / "results.txt").write_text("data")

        plan = {"should_stop": False, "hypothesis": "test"}
        script_path = tmp_path / "experiments" / "v01" / "experiment.py"
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text("print('hi')")

        run_result = RunResult(success=True, stdout="ok", return_code=0)

        with (
            patch.object(orchestrator, "_run_analyst", new_callable=AsyncMock, return_value={}),
            patch.object(orchestrator, "_run_scientist_plan", new_callable=AsyncMock, return_value=plan),
            patch.object(orchestrator, "_run_debate", new_callable=AsyncMock, return_value=None) as mock_debate,
            patch.object(orchestrator, "_run_scientist_revision", new_callable=AsyncMock, return_value=None),
            patch.object(orchestrator, "_run_coder", new_callable=AsyncMock, return_value=script_path),
            patch.object(orchestrator, "_validate_script", new_callable=AsyncMock, return_value=True),
            patch.object(orchestrator, "_run_experiment", new_callable=AsyncMock, return_value=run_result),
        ):
            await orchestrator._run_iteration()

        mock_debate.assert_called_once_with(plan)
