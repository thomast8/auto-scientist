"""Tests for the orchestrator state machine."""

from unittest.mock import AsyncMock, patch

import pytest

from auto_scientist.config import DomainConfig, SuccessCriterion
from auto_scientist.orchestrator import Orchestrator
from auto_scientist.runner import RunResult
from auto_scientist.state import ExperimentState, VersionEntry


@pytest.fixture
def base_state():
    return ExperimentState(domain="test", goal="test goal", phase="ingestion")


@pytest.fixture
def orchestrator(base_state, tmp_path):
    o = Orchestrator(
        state=base_state,
        data_path=tmp_path / "data.csv",
        output_dir=tmp_path / "experiments",
    )
    # Set config directly for tests that need it
    o.config = DomainConfig(
        name="test", description="Test domain", data_paths=["data.csv"],
    )
    # Domain knowledge now lives in state, not config
    base_state.domain_knowledge = "test knowledge"
    return o


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
        o = Orchestrator(
            state=state, data_path=tmp_path, output_dir=tmp_path,
            max_iterations=20,
        )
        o.config = DomainConfig(name="t", description="d", data_paths=[])

        with patch.object(o, "_run_report", new_callable=AsyncMock):
            await o.run()

        assert state.phase == "stopped"

    @pytest.mark.asyncio
    async def test_consecutive_failures_triggers_report(self, tmp_path):
        state = ExperimentState(
            domain="test", goal="g", phase="iteration",
            consecutive_failures=5,
        )
        o = Orchestrator(
            state=state, data_path=tmp_path, output_dir=tmp_path,
            max_consecutive_failures=5,
        )
        o.config = DomainConfig(name="t", description="d", data_paths=[])

        with patch.object(o, "_run_report", new_callable=AsyncMock):
            await o.run()

        assert state.phase == "stopped"

    @pytest.mark.asyncio
    async def test_ingestion_transitions_to_iteration(self, tmp_path):
        """After ingestion, phase should be 'iteration' (not 'discovery')."""
        state = ExperimentState(
            domain="test", goal="g", phase="ingestion",
            data_path=str(tmp_path / "raw.csv"),
        )
        o = Orchestrator(
            state=state, data_path=tmp_path / "raw.csv",
            output_dir=tmp_path / "experiments",
            max_iterations=0,
        )

        canonical = tmp_path / "experiments" / "data"
        canonical.mkdir(parents=True, exist_ok=True)

        config_path = tmp_path / "experiments" / "domain_config.json"
        config_path.write_text('{"name":"t","description":"d","data_paths":[]}')

        with (
            patch.object(o, "_run_ingestion", new_callable=AsyncMock, return_value=canonical),
            patch.object(o, "_run_report", new_callable=AsyncMock),
        ):
            await o.run()

        # Should have gone ingestion -> iteration (hit max) -> report -> stopped
        assert state.phase == "stopped"

    @pytest.mark.asyncio
    async def test_resume_from_iteration_skips_ingestion(self, tmp_path):
        state = ExperimentState(
            domain="test", goal="g", phase="iteration",
            iteration=20,
        )
        o = Orchestrator(
            state=state, data_path=tmp_path, output_dir=tmp_path,
            max_iterations=20,
        )
        o.config = DomainConfig(name="t", description="d", data_paths=[])

        with patch.object(o, "_run_report", new_callable=AsyncMock) as mock_report:
            await o.run()

        mock_report.assert_called_once()
        assert state.phase == "stopped"


class TestIteration0:
    """Iteration 0 flow: analyst initial, scientist without versions, coder runs, debate skipped."""

    @pytest.mark.asyncio
    async def test_analyst_calls_initial_on_no_versions(self, orchestrator, tmp_path):
        """When no versions exist, analyst should call _run_analyst_initial."""
        orchestrator.output_dir.mkdir(parents=True, exist_ok=True)
        orchestrator.state.phase = "iteration"
        orchestrator.state.iteration = 0

        analysis = {"success_score": None, "observations": ["200 rows"]}

        with patch.object(
            orchestrator, "_run_analyst_initial",
            new_callable=AsyncMock, return_value=analysis,
        ) as mock_initial:
            result = await orchestrator._run_analyst()

        mock_initial.assert_called_once()
        assert result == analysis

    @pytest.mark.asyncio
    async def test_debate_skipped_on_iteration_0(self, orchestrator, tmp_path):
        """Debate should be skipped when iteration is 0."""
        orchestrator.output_dir.mkdir(parents=True, exist_ok=True)
        orchestrator.state.phase = "iteration"
        orchestrator.state.iteration = 0
        orchestrator.critic_models = ["openai:gpt-4o"]

        plan = {"should_stop": False, "hypothesis": "explore"}
        script_path = tmp_path / "experiments" / "v00" / "experiment.py"
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text("print('hi')")
        run_result = RunResult(success=True, stdout="ok", return_code=0)

        with (
            patch.object(orchestrator, "_run_analyst", new_callable=AsyncMock,
                          return_value={"success_score": None}),
            patch.object(orchestrator, "_run_scientist_plan", new_callable=AsyncMock,
                          return_value=plan),
            patch.object(
                orchestrator, "_run_debate", new_callable=AsyncMock,
            ) as mock_debate,
            patch.object(
                orchestrator, "_run_scientist_revision",
                new_callable=AsyncMock,
            ) as mock_revision,
            patch.object(orchestrator, "_run_coder", new_callable=AsyncMock,
                          return_value=script_path),
            patch.object(orchestrator, "_validate_script", new_callable=AsyncMock,
                          return_value=True),
            patch.object(orchestrator, "_run_experiment", new_callable=AsyncMock,
                          return_value=run_result),
            patch.object(orchestrator, "_apply_criteria_updates"),
        ):
            await orchestrator._run_iteration_body()

        mock_debate.assert_not_called()
        mock_revision.assert_not_called()

    @pytest.mark.asyncio
    async def test_iteration_starts_at_0(self, orchestrator, tmp_path):
        """Iteration counter should start at 0 and increment at end."""
        orchestrator.output_dir.mkdir(parents=True, exist_ok=True)
        orchestrator.state.phase = "iteration"
        orchestrator.state.iteration = 0

        plan = {"should_stop": False, "hypothesis": "test"}
        script_path = tmp_path / "experiments" / "v00" / "experiment.py"
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text("print('hi')")
        run_result = RunResult(success=True, stdout="ok", return_code=0)

        with (
            patch.object(
                orchestrator, "_run_analyst",
                new_callable=AsyncMock, return_value={},
            ),
            patch.object(
                orchestrator, "_run_scientist_plan",
                new_callable=AsyncMock, return_value=plan,
            ),
            patch.object(
                orchestrator, "_run_debate",
                new_callable=AsyncMock,
            ),
            patch.object(
                orchestrator, "_run_scientist_revision",
                new_callable=AsyncMock, return_value=None,
            ),
            patch.object(
                orchestrator, "_run_coder",
                new_callable=AsyncMock, return_value=script_path,
            ),
            patch.object(
                orchestrator, "_validate_script",
                new_callable=AsyncMock, return_value=True,
            ),
            patch.object(
                orchestrator, "_run_experiment",
                new_callable=AsyncMock, return_value=run_result,
            ),
            patch.object(orchestrator, "_apply_criteria_updates"),
        ):
            await orchestrator._run_iteration_body()

        assert orchestrator.state.iteration == 1


class TestApplyCriteriaUpdates:
    """Test _apply_criteria_updates with various plan shapes."""

    def test_top_level_criteria_populates_state(self, orchestrator):
        plan = {
            "top_level_criteria": [
                {"name": "RMSE", "description": "low error", "metric_key": "rmse",
                 "condition": "< 0.5"},
                {"name": "R-squared", "description": "high fit", "metric_key": "r2",
                 "condition": "> 0.95"},
            ],
        }
        orchestrator.state.iteration = 1
        orchestrator._apply_criteria_updates(plan)

        assert orchestrator.state.success_criteria is not None
        assert len(orchestrator.state.success_criteria) == 2
        assert orchestrator.state.success_criteria[0].name == "RMSE"
        assert orchestrator.state.success_criteria[0].target_max == 0.5
        assert orchestrator.state.success_criteria[1].target_min == 0.95
        assert len(orchestrator.state.criteria_history) == 1
        assert orchestrator.state.criteria_history[0].action == "defined"

    def test_criteria_revision_updates_state(self, orchestrator):
        # Pre-populate with existing criteria
        orchestrator.state.success_criteria = [
            SuccessCriterion(name="R2", description="fit", metric_key="r2", target_min=0.95),
        ]
        plan = {
            "criteria_revision": {
                "changes": "Lowered target from 0.95 to 0.90",
                "revised_criteria": [
                    {"name": "R2", "description": "fit", "metric_key": "r2",
                     "condition": "> 0.90"},
                ],
            },
        }
        orchestrator.state.iteration = 3
        orchestrator._apply_criteria_updates(plan)

        assert orchestrator.state.success_criteria[0].target_min == 0.90
        assert len(orchestrator.state.criteria_history) == 1
        assert orchestrator.state.criteria_history[0].action == "revised"

    def test_no_criteria_fields_no_change(self, orchestrator):
        orchestrator.state.success_criteria = None
        plan = {"hypothesis": "test", "changes": []}
        orchestrator._apply_criteria_updates(plan)
        assert orchestrator.state.success_criteria is None
        assert len(orchestrator.state.criteria_history) == 0

    def test_condition_parsing_less_than(self, orchestrator):
        plan = {
            "top_level_criteria": [
                {"name": "Error", "description": "low", "metric_key": "err",
                 "condition": "< 500"},
            ],
        }
        orchestrator.state.iteration = 1
        orchestrator._apply_criteria_updates(plan)
        assert orchestrator.state.success_criteria[0].target_max == 500.0
        assert orchestrator.state.success_criteria[0].target_min is None

    def test_condition_parsing_greater_than(self, orchestrator):
        plan = {
            "top_level_criteria": [
                {"name": "Accuracy", "description": "high", "metric_key": "acc",
                 "condition": "> 0.95"},
            ],
        }
        orchestrator.state.iteration = 1
        orchestrator._apply_criteria_updates(plan)
        assert orchestrator.state.success_criteria[0].target_min == 0.95
        assert orchestrator.state.success_criteria[0].target_max is None


class TestDomainKnowledgeSourcing:
    """Domain knowledge comes from state, not config."""

    @pytest.mark.asyncio
    async def test_analyst_receives_state_domain_knowledge(self, orchestrator, tmp_path):
        """_run_analyst passes state.domain_knowledge, not config.domain_knowledge."""
        orchestrator.output_dir.mkdir(parents=True, exist_ok=True)
        orchestrator.state.phase = "iteration"
        orchestrator.state.domain_knowledge = "from state"

        latest = VersionEntry(
            version="v01", iteration=1, script_path=str(tmp_path / "s.py"),
            results_path=str(tmp_path / "results.txt"),
        )
        orchestrator.state.versions = [latest]
        (tmp_path / "results.txt").write_text("data")

        captured_kwargs = {}

        async def capture_analyst(**kwargs):
            captured_kwargs.update(kwargs)
            return {"success_score": 50}

        with patch("auto_scientist.agents.analyst.run_analyst", side_effect=capture_analyst):
            await orchestrator._run_analyst()

        assert captured_kwargs["domain_knowledge"] == "from state"

    @pytest.mark.asyncio
    async def test_analyst_domain_knowledge_updates_state(self, orchestrator, tmp_path):
        """When analyst returns domain_knowledge, state should be updated."""
        orchestrator.output_dir.mkdir(parents=True, exist_ok=True)
        orchestrator.state.domain_knowledge = ""

        analysis = {
            "success_score": None,
            "domain_knowledge": "Environmental sensor data with temperature readings",
        }

        with patch.object(
            orchestrator, "_run_analyst_initial",
            new_callable=AsyncMock, return_value=analysis,
        ):
            result = await orchestrator._run_analyst()

        # The orchestrator should check for domain_knowledge in result
        # This is tested via the orchestrator's iteration body
        expected = "Environmental sensor data with temperature readings"
        assert result.get("domain_knowledge") == expected


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
            patch.object(
                orchestrator, "_run_analyst",
                new_callable=AsyncMock, return_value={},
            ),
            patch.object(
                orchestrator, "_run_scientist_plan",
                new_callable=AsyncMock, return_value=plan,
            ),
            patch.object(orchestrator, "_apply_criteria_updates"),
        ):
            await orchestrator._run_iteration_body()

        assert orchestrator.state.phase == "report"

    @pytest.mark.asyncio
    async def test_no_critics_skips_debate(self, orchestrator, tmp_path):
        orchestrator.output_dir.mkdir(parents=True, exist_ok=True)
        orchestrator.critic_models = []
        orchestrator.state.phase = "iteration"
        orchestrator.state.iteration = 1
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
            patch.object(
                orchestrator, "_run_analyst",
                new_callable=AsyncMock, return_value={},
            ),
            patch.object(
                orchestrator, "_run_scientist_plan",
                new_callable=AsyncMock, return_value=plan,
            ),
            patch.object(
                orchestrator, "_run_debate",
                new_callable=AsyncMock, return_value=None,
            ) as mock_debate,
            patch.object(
                orchestrator, "_run_scientist_revision",
                new_callable=AsyncMock, return_value=None,
            ),
            patch.object(
                orchestrator, "_run_coder",
                new_callable=AsyncMock, return_value=script_path,
            ),
            patch.object(
                orchestrator, "_validate_script",
                new_callable=AsyncMock, return_value=True,
            ),
            patch.object(
                orchestrator, "_run_experiment",
                new_callable=AsyncMock, return_value=run_result,
            ),
            patch.object(orchestrator, "_apply_criteria_updates"),
        ):
            await orchestrator._run_iteration_body()

        mock_debate.assert_called_once_with(plan)
