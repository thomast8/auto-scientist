"""Tests for the orchestrator state machine."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

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

    def test_success_no_results_file(self, orchestrator, tmp_path):
        result = RunResult(success=True, stdout="output", return_code=0)
        script_path = tmp_path / "experiments" / "v01" / "experiment.py"
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text("print('ok')")

        entry = VersionEntry(version="v01", iteration=1, script_path=str(script_path))
        orchestrator._evaluate(result, entry)
        assert entry.status == "completed"
        assert entry.results_path is None


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


class TestParseCriterion:
    def test_greater_than(self, orchestrator):
        raw = {"name": "acc", "description": "d", "metric_key": "acc", "condition": "> 0.95"}
        sc = Orchestrator._parse_criterion(raw)
        assert sc.target_min == 0.95
        assert sc.target_max is None

    def test_less_than(self, orchestrator):
        raw = {"name": "err", "description": "d", "metric_key": "err", "condition": "< 500"}
        sc = Orchestrator._parse_criterion(raw)
        assert sc.target_max == 500.0
        assert sc.target_min is None

    def test_greater_equal(self, orchestrator):
        raw = {"name": "r2", "description": "d", "metric_key": "r2", "condition": ">= 0.9"}
        sc = Orchestrator._parse_criterion(raw)
        assert sc.target_min == 0.9

    def test_less_equal(self, orchestrator):
        raw = {"name": "err", "description": "d", "metric_key": "err", "condition": "<= 10"}
        sc = Orchestrator._parse_criterion(raw)
        assert sc.target_max == 10.0

    def test_invalid_condition_no_targets(self, orchestrator):
        raw = {"name": "x", "description": "d", "metric_key": "x", "condition": "between 0 and 1"}
        sc = Orchestrator._parse_criterion(raw)
        assert sc.target_min is None
        assert sc.target_max is None

    def test_empty_condition(self, orchestrator):
        raw = {"name": "x", "description": "d", "metric_key": "x", "condition": ""}
        sc = Orchestrator._parse_criterion(raw)
        assert sc.target_min is None
        assert sc.target_max is None

    def test_missing_condition_key(self, orchestrator):
        raw = {"name": "x", "description": "d", "metric_key": "x"}
        sc = Orchestrator._parse_criterion(raw)
        assert sc.target_min is None
        assert sc.target_max is None

    def test_preserves_name_description(self, orchestrator):
        raw = {"name": "Accuracy", "description": "Model accuracy", "metric_key": "acc", "condition": "> 0.9"}
        sc = Orchestrator._parse_criterion(raw)
        assert sc.name == "Accuracy"
        assert sc.description == "Model accuracy"
        assert sc.metric_key == "acc"


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

    def test_top_level_takes_priority_over_revision(self, orchestrator):
        orchestrator.state.iteration = 1
        plan = {
            "top_level_criteria": [
                {"name": "A", "description": "d", "metric_key": "a", "condition": "> 0.9"},
            ],
            "criteria_revision": {
                "changes": "ignored",
                "revised_criteria": [
                    {"name": "B", "description": "d", "metric_key": "b", "condition": "< 1"},
                ],
            },
        }
        orchestrator._apply_criteria_updates(plan)
        assert len(orchestrator.state.success_criteria) == 1
        assert orchestrator.state.success_criteria[0].name == "A"
        assert orchestrator.state.criteria_history[0].action == "defined"

    def test_empty_top_level_list_no_update(self, orchestrator):
        orchestrator.state.success_criteria = None
        plan = {"top_level_criteria": []}
        orchestrator._apply_criteria_updates(plan)
        assert orchestrator.state.success_criteria is None

    def test_revision_with_empty_revised_list(self, orchestrator):
        orchestrator.state.success_criteria = [
            SuccessCriterion(name="X", description="d", metric_key="x", target_min=0.5),
        ]
        plan = {
            "criteria_revision": {
                "changes": "Cleared all criteria",
                "revised_criteria": [],
            },
        }
        orchestrator.state.iteration = 2
        orchestrator._apply_criteria_updates(plan)
        assert orchestrator.state.success_criteria == []

    def test_criteria_history_records_defined_action(self, orchestrator):
        orchestrator.state.iteration = 1
        plan = {
            "top_level_criteria": [
                {"name": "A", "description": "d", "metric_key": "a", "condition": "> 0.9"},
            ],
        }
        orchestrator._apply_criteria_updates(plan)
        assert len(orchestrator.state.criteria_history) == 1
        assert orchestrator.state.criteria_history[0].action == "defined"
        assert orchestrator.state.criteria_history[0].iteration == 1

    def test_criteria_history_records_revised_action(self, orchestrator):
        orchestrator.state.success_criteria = [
            SuccessCriterion(name="A", description="d", metric_key="a", target_min=0.9),
        ]
        orchestrator.state.iteration = 3
        plan = {
            "criteria_revision": {
                "changes": "Lowered target",
                "revised_criteria": [
                    {"name": "A", "description": "d", "metric_key": "a", "condition": "> 0.8"},
                ],
            },
        }
        orchestrator._apply_criteria_updates(plan)
        assert orchestrator.state.criteria_history[0].action == "revised"
        assert orchestrator.state.criteria_history[0].iteration == 3

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

    @pytest.mark.asyncio
    async def test_domain_knowledge_updated_from_analysis(self, orchestrator, tmp_path):
        orchestrator.output_dir.mkdir(parents=True, exist_ok=True)
        orchestrator.state.phase = "iteration"
        orchestrator.state.iteration = 0
        orchestrator.state.domain_knowledge = ""

        analysis = {
            "success_score": None,
            "domain_knowledge": "Sensor data with temperature readings",
        }
        plan = {"should_stop": False, "hypothesis": "explore"}
        script_path = tmp_path / "experiments" / "v00" / "experiment.py"
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text("print('hi')")
        run_result = RunResult(success=True, stdout="ok", return_code=0)

        with (
            patch.object(orchestrator, "_run_analyst", new_callable=AsyncMock,
                          return_value=analysis),
            patch.object(orchestrator, "_run_scientist_plan", new_callable=AsyncMock,
                          return_value=plan),
            patch.object(orchestrator, "_run_coder", new_callable=AsyncMock,
                          return_value=script_path),
            patch.object(orchestrator, "_validate_script", new_callable=AsyncMock,
                          return_value=True),
            patch.object(orchestrator, "_run_experiment", new_callable=AsyncMock,
                          return_value=run_result),
            patch.object(orchestrator, "_apply_criteria_updates"),
        ):
            await orchestrator._run_iteration_body()

        assert orchestrator.state.domain_knowledge == "Sensor data with temperature readings"

    @pytest.mark.asyncio
    async def test_should_stop_transitions_to_report(self, orchestrator, tmp_path):
        orchestrator.output_dir.mkdir(parents=True, exist_ok=True)
        orchestrator.state.phase = "iteration"
        orchestrator.state.iteration = 1
        orchestrator.state.versions = [
            VersionEntry(
                version="v00", iteration=0, script_path="/tmp/s.py",
                results_path=str(tmp_path / "results.txt"),
            ),
        ]
        (tmp_path / "results.txt").write_text("data")

        plan = {"should_stop": True, "stop_reason": "criteria met"}

        with (
            patch.object(orchestrator, "_run_analyst", new_callable=AsyncMock,
                          return_value={}),
            patch.object(orchestrator, "_run_scientist_plan", new_callable=AsyncMock,
                          return_value=plan),
            patch.object(orchestrator, "_apply_criteria_updates"),
        ):
            await orchestrator._run_iteration_body()

        assert orchestrator.state.phase == "report"

    @pytest.mark.asyncio
    async def test_validation_failure_increments_iteration(self, orchestrator, tmp_path):
        orchestrator.output_dir.mkdir(parents=True, exist_ok=True)
        orchestrator.state.phase = "iteration"
        orchestrator.state.iteration = 0

        plan = {"should_stop": False, "hypothesis": "test"}
        script_path = tmp_path / "experiments" / "v00" / "experiment.py"
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text("def foo(")

        with (
            patch.object(orchestrator, "_run_analyst", new_callable=AsyncMock,
                          return_value={}),
            patch.object(orchestrator, "_run_scientist_plan", new_callable=AsyncMock,
                          return_value=plan),
            patch.object(orchestrator, "_run_coder", new_callable=AsyncMock,
                          return_value=script_path),
            patch.object(orchestrator, "_validate_script", new_callable=AsyncMock,
                          return_value=False),
            patch.object(orchestrator, "_apply_criteria_updates"),
        ):
            await orchestrator._run_iteration_body()

        assert orchestrator.state.consecutive_failures == 1
        assert orchestrator.state.iteration == 1

    @pytest.mark.asyncio
    async def test_iteration_summary_prints_artifacts(self, orchestrator, tmp_path, capsys):
        orchestrator.output_dir.mkdir(parents=True, exist_ok=True)
        orchestrator.state.phase = "iteration"
        orchestrator.state.iteration = 0

        plan = {"should_stop": False, "hypothesis": "test"}
        script_path = tmp_path / "experiments" / "v00" / "experiment.py"
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text("print('hi')")
        # Create some artifact files
        (script_path.parent / "plot.png").write_text("fake")
        (script_path.parent / "results.txt").write_text("output")
        run_result = RunResult(success=True, stdout="ok", return_code=0)

        with (
            patch.object(orchestrator, "_run_analyst", new_callable=AsyncMock,
                          return_value={}),
            patch.object(orchestrator, "_run_scientist_plan", new_callable=AsyncMock,
                          return_value=plan),
            patch.object(orchestrator, "_run_coder", new_callable=AsyncMock,
                          return_value=script_path),
            patch.object(orchestrator, "_validate_script", new_callable=AsyncMock,
                          return_value=True),
            patch.object(orchestrator, "_run_experiment", new_callable=AsyncMock,
                          return_value=run_result),
            patch.object(orchestrator, "_apply_criteria_updates"),
        ):
            await orchestrator._run_iteration_body()

        captured = capsys.readouterr()
        assert "Outputs:" in captured.out


class TestRunAnalystInitial:
    @pytest.mark.asyncio
    async def test_calls_run_analyst_with_data_dir(self, orchestrator, tmp_path):
        orchestrator.output_dir.mkdir(parents=True, exist_ok=True)
        orchestrator.data_path = tmp_path / "data"

        captured_kwargs = {}

        async def capture_analyst(**kwargs):
            captured_kwargs.update(kwargs)
            return {"success_score": None}

        with patch("auto_scientist.agents.analyst.run_analyst", side_effect=capture_analyst):
            result = await orchestrator._run_analyst_initial()

        assert captured_kwargs["results_path"] is None
        assert captured_kwargs["data_dir"] == tmp_path / "data"
        assert result["success_score"] is None

    @pytest.mark.asyncio
    async def test_error_returns_none(self, orchestrator, tmp_path):
        orchestrator.output_dir.mkdir(parents=True, exist_ok=True)
        orchestrator.data_path = tmp_path / "data"

        with patch(
            "auto_scientist.agents.analyst.run_analyst",
            new_callable=AsyncMock,
            side_effect=RuntimeError("Agent error"),
        ):
            result = await orchestrator._run_analyst_initial()

        assert result is None


class TestRunAnalystNormal:
    @pytest.mark.asyncio
    async def test_no_results_path_returns_none(self, orchestrator, tmp_path):
        orchestrator.output_dir.mkdir(parents=True, exist_ok=True)
        orchestrator.state.versions = [
            VersionEntry(version="v00", iteration=0, script_path="/tmp/s.py",
                          results_path=None),
        ]

        result = await orchestrator._run_analyst()
        assert result is None

    @pytest.mark.asyncio
    async def test_updates_best_score(self, orchestrator, tmp_path):
        orchestrator.output_dir.mkdir(parents=True, exist_ok=True)

        results_path = tmp_path / "results.txt"
        results_path.write_text("data")
        latest = VersionEntry(
            version="v01", iteration=1, script_path=str(tmp_path / "s.py"),
            results_path=str(results_path),
        )
        orchestrator.state.versions = [latest]

        async def fake_analyst(**kwargs):
            return {"success_score": 90}

        with patch("auto_scientist.agents.analyst.run_analyst", side_effect=fake_analyst):
            await orchestrator._run_analyst()

        assert latest.score == 90
        assert orchestrator.state.best_score == 90
        assert orchestrator.state.best_version == "v01"

    @pytest.mark.asyncio
    async def test_error_returns_none(self, orchestrator, tmp_path):
        orchestrator.output_dir.mkdir(parents=True, exist_ok=True)

        results_path = tmp_path / "results.txt"
        results_path.write_text("data")
        orchestrator.state.versions = [
            VersionEntry(version="v01", iteration=1, script_path=str(tmp_path / "s.py"),
                          results_path=str(results_path)),
        ]

        with patch(
            "auto_scientist.agents.analyst.run_analyst",
            new_callable=AsyncMock,
            side_effect=RuntimeError("Agent error"),
        ):
            result = await orchestrator._run_analyst()

        assert result is None


class TestRunScientistPlan:
    @pytest.mark.asyncio
    async def test_returns_plan(self, orchestrator, tmp_path):
        orchestrator.output_dir.mkdir(parents=True, exist_ok=True)
        plan = {"hypothesis": "test", "strategy": "incremental", "notebook_entry": "## v00"}

        with patch(
            "auto_scientist.agents.scientist.run_scientist",
            new_callable=AsyncMock,
            return_value=plan,
        ):
            result = await orchestrator._run_scientist_plan({"success_score": 50})

        assert result["hypothesis"] == "test"
        notebook = orchestrator.output_dir / "lab_notebook.md"
        assert notebook.exists()
        assert "## v00" in notebook.read_text()

    @pytest.mark.asyncio
    async def test_error_returns_none(self, orchestrator, tmp_path):
        orchestrator.output_dir.mkdir(parents=True, exist_ok=True)

        with patch(
            "auto_scientist.agents.scientist.run_scientist",
            new_callable=AsyncMock,
            side_effect=RuntimeError("Agent error"),
        ):
            result = await orchestrator._run_scientist_plan({})

        assert result is None


class TestRunDebateOrchestrator:
    @pytest.mark.asyncio
    async def test_no_critics_returns_none(self, orchestrator):
        orchestrator.critic_models = []
        result = await orchestrator._run_debate({"hypothesis": "test"})
        assert result is None

    @pytest.mark.asyncio
    async def test_no_plan_returns_none(self, orchestrator):
        orchestrator.critic_models = ["openai:gpt-4o"]
        result = await orchestrator._run_debate(None)
        assert result is None

    @pytest.mark.asyncio
    async def test_calls_run_debate_with_args(self, orchestrator, tmp_path):
        orchestrator.output_dir.mkdir(parents=True, exist_ok=True)
        orchestrator.critic_models = ["openai:gpt-4o"]
        orchestrator.state.domain_knowledge = "test knowledge"
        plan = {"hypothesis": "test"}

        critique = [{"model": "openai:gpt-4o", "critique": "looks good", "transcript": []}]

        with patch(
            "auto_scientist.agents.critic.run_debate",
            new_callable=AsyncMock,
            return_value=critique,
        ) as mock_debate:
            result = await orchestrator._run_debate(plan)

        assert result == critique
        call_kwargs = mock_debate.call_args.kwargs
        assert call_kwargs["critic_specs"] == ["openai:gpt-4o"]
        assert call_kwargs["domain_knowledge"] == "test knowledge"


class TestRunScientistRevisionOrchestrator:
    @pytest.mark.asyncio
    async def test_no_plan_returns_none(self, orchestrator):
        result = await orchestrator._run_scientist_revision(None, [], {})
        assert result is None

    @pytest.mark.asyncio
    async def test_no_debate_result_returns_none(self, orchestrator):
        result = await orchestrator._run_scientist_revision({"h": "x"}, None, {})
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_revised_plan(self, orchestrator, tmp_path):
        orchestrator.output_dir.mkdir(parents=True, exist_ok=True)
        plan = {"hypothesis": "original"}
        debate_result = [
            {"model": "openai:gpt-4o", "critique": "weak",
             "transcript": [{"role": "critic", "content": "weak"}]},
        ]
        revised = {"hypothesis": "revised", "notebook_entry": "## revised"}

        with patch(
            "auto_scientist.agents.scientist.run_scientist_revision",
            new_callable=AsyncMock,
            return_value=revised,
        ):
            result = await orchestrator._run_scientist_revision(plan, debate_result, {})

        assert result["hypothesis"] == "revised"
        notebook = orchestrator.output_dir / "lab_notebook.md"
        assert "## revised" in notebook.read_text()

    @pytest.mark.asyncio
    async def test_error_returns_none(self, orchestrator, tmp_path):
        orchestrator.output_dir.mkdir(parents=True, exist_ok=True)
        plan = {"hypothesis": "original"}
        debate_result = [
            {"model": "openai:gpt-4o", "critique": "weak",
             "transcript": [{"role": "critic", "content": "weak"}]},
        ]

        with patch(
            "auto_scientist.agents.scientist.run_scientist_revision",
            new_callable=AsyncMock,
            side_effect=RuntimeError("Agent error"),
        ):
            result = await orchestrator._run_scientist_revision(plan, debate_result, {})

        assert result is None


class TestRunCoderOrchestrator:
    @pytest.mark.asyncio
    async def test_no_plan_returns_none(self, orchestrator):
        result = await orchestrator._run_coder(None)
        assert result is None

    @pytest.mark.asyncio
    async def test_calls_run_coder(self, orchestrator, tmp_path):
        orchestrator.output_dir.mkdir(parents=True, exist_ok=True)
        orchestrator.state.data_path = str(tmp_path / "data")
        plan = {"hypothesis": "test"}

        script_path = tmp_path / "experiments" / "v00" / "experiment.py"

        with patch(
            "auto_scientist.agents.coder.run_coder",
            new_callable=AsyncMock,
            return_value=script_path,
        ):
            result = await orchestrator._run_coder(plan)

        assert result == script_path

    @pytest.mark.asyncio
    async def test_error_records_failure(self, orchestrator, tmp_path):
        orchestrator.output_dir.mkdir(parents=True, exist_ok=True)

        with patch(
            "auto_scientist.agents.coder.run_coder",
            new_callable=AsyncMock,
            side_effect=RuntimeError("Coder failed"),
        ):
            result = await orchestrator._run_coder({"hypothesis": "test"})

        assert result is None
        assert orchestrator.state.consecutive_failures == 1

    @pytest.mark.asyncio
    async def test_uses_previous_script_from_versions(self, orchestrator, tmp_path):
        orchestrator.output_dir.mkdir(parents=True, exist_ok=True)
        orchestrator.state.versions = [
            VersionEntry(version="v00", iteration=0,
                          script_path=str(tmp_path / "v00" / "experiment.py")),
        ]

        captured_kwargs = {}

        async def capture_coder(**kwargs):
            captured_kwargs.update(kwargs)
            return tmp_path / "v01" / "experiment.py"

        with patch("auto_scientist.agents.coder.run_coder", side_effect=capture_coder):
            await orchestrator._run_coder({"hypothesis": "test"})

        assert str(captured_kwargs["previous_script"]) == str(tmp_path / "v00" / "experiment.py")


class TestValidateScript:
    @pytest.mark.asyncio
    async def test_valid_script_returns_true(self, orchestrator, tmp_path):
        script = tmp_path / "valid.py"
        script.write_text("x = 1 + 2\n")
        result = await orchestrator._validate_script(script)
        assert result is True

    @pytest.mark.asyncio
    async def test_invalid_script_returns_false(self, orchestrator, tmp_path):
        script = tmp_path / "invalid.py"
        script.write_text("def foo(\n")
        result = await orchestrator._validate_script(script)
        assert result is False


class TestRunExperimentOrchestrator:
    @pytest.mark.asyncio
    async def test_none_script_returns_none(self, orchestrator):
        result = await orchestrator._run_experiment(None)
        assert result is None

    @pytest.mark.asyncio
    async def test_uses_config_command(self, orchestrator, tmp_path):
        orchestrator.config = DomainConfig(
            name="t", description="d", data_paths=[],
            run_command="python {script_path}",
            run_cwd=str(tmp_path),
            run_timeout_minutes=5,
        )
        script = tmp_path / "test.py"
        script.write_text("print('hello')\n")

        result = await orchestrator._run_experiment(script)

        assert result.success
        assert "hello" in result.stdout

    @pytest.mark.asyncio
    async def test_saves_stdout_to_results(self, orchestrator, tmp_path):
        orchestrator.config = DomainConfig(
            name="t", description="d", data_paths=[],
            run_command="python {script_path}",
            run_cwd=str(tmp_path),
        )
        script = tmp_path / "test.py"
        script.write_text("print('output data')\n")

        await orchestrator._run_experiment(script)

        results_path = script.parent / "results.txt"
        assert results_path.exists()
        assert "output data" in results_path.read_text()

    @pytest.mark.asyncio
    async def test_timeout_prints_message(self, orchestrator, tmp_path, capsys):
        orchestrator.config = DomainConfig(
            name="t", description="d", data_paths=[],
            run_command="python {script_path}",
            run_cwd=str(tmp_path),
            run_timeout_minutes=0,
        )
        script = tmp_path / "slow.py"
        script.write_text("import time\ntime.sleep(10)\n")

        result = await orchestrator._run_experiment(script)

        assert result.timed_out
        captured = capsys.readouterr()
        assert "timed out" in captured.out

    @pytest.mark.asyncio
    async def test_failure_prints_stderr(self, orchestrator, tmp_path, capsys):
        orchestrator.config = DomainConfig(
            name="t", description="d", data_paths=[],
            run_command="python {script_path}",
            run_cwd=str(tmp_path),
        )
        script = tmp_path / "fail.py"
        script.write_text("raise ValueError('test error')\n")

        result = await orchestrator._run_experiment(script)

        assert not result.success
        captured = capsys.readouterr()
        assert "failed" in captured.out

    @pytest.mark.asyncio
    async def test_default_command_without_config(self, orchestrator, tmp_path):
        orchestrator.config = None
        script = tmp_path / "test.py"
        script.write_text("print('hello')\n")

        result = await orchestrator._run_experiment(script)
        assert result is not None


class TestRunReportOrchestrator:
    @pytest.mark.asyncio
    async def test_calls_run_report(self, orchestrator, tmp_path):
        orchestrator.output_dir.mkdir(parents=True, exist_ok=True)

        report_path = orchestrator.output_dir / "report.md"

        with patch(
            "auto_scientist.agents.report.run_report",
            new_callable=AsyncMock,
            return_value=report_path,
        ) as mock_report:
            await orchestrator._run_report()

        mock_report.assert_called_once()
        call_kwargs = mock_report.call_args.kwargs
        assert call_kwargs["state"] is orchestrator.state
        assert call_kwargs["output_dir"] == orchestrator.output_dir

    @pytest.mark.asyncio
    async def test_error_does_not_raise(self, orchestrator, tmp_path, capsys):
        orchestrator.output_dir.mkdir(parents=True, exist_ok=True)

        with patch(
            "auto_scientist.agents.report.run_report",
            new_callable=AsyncMock,
            side_effect=RuntimeError("Report failed"),
        ):
            await orchestrator._run_report()

        captured = capsys.readouterr()
        assert "error" in captured.out


class TestRunFullOrchestration:
    @pytest.mark.asyncio
    async def test_run_prints_critics(self, tmp_path, capsys):
        state = ExperimentState(
            domain="test", goal="g", phase="iteration",
            iteration=20,
        )
        o = Orchestrator(
            state=state, data_path=tmp_path, output_dir=tmp_path,
            max_iterations=20,
            critic_models=["openai:gpt-4o"],
        )
        o.config = DomainConfig(name="t", description="d", data_paths=[])

        with patch.object(o, "_run_report", new_callable=AsyncMock):
            await o.run()

        captured = capsys.readouterr()
        assert "openai:gpt-4o" in captured.out

    @pytest.mark.asyncio
    async def test_run_iteration_loop_with_schedule(self, tmp_path):
        state = ExperimentState(
            domain="test", goal="g", phase="iteration",
            iteration=0,
        )
        o = Orchestrator(
            state=state, data_path=tmp_path, output_dir=tmp_path,
            max_iterations=1,
        )
        o.config = DomainConfig(name="t", description="d", data_paths=[])
        state.schedule = "00:00-23:59"

        plan = {"should_stop": False, "hypothesis": "test"}
        script_path = tmp_path / "v00" / "experiment.py"
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text("print('hi')")
        run_result = RunResult(success=True, stdout="ok", return_code=0)

        with (
            patch("auto_scientist.orchestrator.wait_for_window", new_callable=AsyncMock),
            patch.object(o, "_run_analyst", new_callable=AsyncMock, return_value={}),
            patch.object(o, "_run_scientist_plan", new_callable=AsyncMock, return_value=plan),
            patch.object(o, "_run_coder", new_callable=AsyncMock, return_value=script_path),
            patch.object(o, "_validate_script", new_callable=AsyncMock, return_value=True),
            patch.object(o, "_run_experiment", new_callable=AsyncMock, return_value=run_result),
            patch.object(o, "_apply_criteria_updates"),
            patch.object(o, "_run_report", new_callable=AsyncMock),
        ):
            await o.run()

        assert state.phase == "stopped"
        assert state.iteration == 1


class TestRunIngestionFull:
    @pytest.mark.asyncio
    async def test_ingestion_loads_config(self, tmp_path):
        state = ExperimentState(
            domain="test", goal="g", phase="ingestion",
        )
        o = Orchestrator(
            state=state, data_path=tmp_path / "raw.csv",
            output_dir=tmp_path / "experiments",
            max_iterations=0,
        )

        canonical = tmp_path / "experiments" / "data"
        canonical.mkdir(parents=True, exist_ok=True)
        (canonical / "clean.csv").write_text("a,b\n1,2\n")

        config_path = tmp_path / "experiments" / "domain_config.json"
        config_path.write_text('{"name":"mydom","description":"test","data_paths":["clean.csv"]}')

        with (
            patch(
                "auto_scientist.agents.ingestor.run_ingestor",
                new_callable=AsyncMock,
                return_value=canonical,
            ),
            patch.object(o, "_run_report", new_callable=AsyncMock),
        ):
            await o.run()

        assert o.config is not None
        assert o.config.name == "mydom"
        assert state.config_path == str(config_path)
        assert state.phase == "stopped"

    @pytest.mark.asyncio
    async def test_ingestion_prints_data_files(self, tmp_path, capsys):
        state = ExperimentState(
            domain="test", goal="g", phase="ingestion",
        )
        o = Orchestrator(
            state=state, data_path=tmp_path / "raw.csv",
            output_dir=tmp_path / "experiments",
            max_iterations=0,
        )

        canonical = tmp_path / "experiments" / "data"
        canonical.mkdir(parents=True, exist_ok=True)
        (canonical / "data.csv").write_text("a,b\n1,2\n")

        with (
            patch(
                "auto_scientist.agents.ingestor.run_ingestor",
                new_callable=AsyncMock,
                return_value=canonical,
            ),
            patch.object(o, "_run_report", new_callable=AsyncMock),
        ):
            await o.run()

        captured = capsys.readouterr()
        assert "data.csv" in captured.out


class TestSummaryIntegration:
    def test_summary_model_defaults_to_none(self, base_state, tmp_path):
        o = Orchestrator(state=base_state, data_path=tmp_path, output_dir=tmp_path)
        assert o.summary_model is None

    def test_summary_model_set(self, base_state, tmp_path):
        o = Orchestrator(
            state=base_state, data_path=tmp_path, output_dir=tmp_path,
            summary_model="gpt-4o-mini",
        )
        assert o.summary_model == "gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_no_summary_when_model_none(self, tmp_path):
        """When summary_model is None, summarizer is never called."""
        state = ExperimentState(
            domain="test", goal="g", phase="iteration", iteration=0,
        )
        o = Orchestrator(
            state=state, data_path=tmp_path, output_dir=tmp_path,
            max_iterations=1,
        )
        o.config = DomainConfig(name="t", description="d", data_paths=[])
        o.output_dir.mkdir(parents=True, exist_ok=True)

        plan = {"should_stop": False, "hypothesis": "test"}
        script_path = tmp_path / "v00" / "experiment.py"
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text("print('hi')")
        run_result = RunResult(success=True, stdout="ok", return_code=0)

        with (
            patch("auto_scientist.orchestrator.wait_for_window", new_callable=AsyncMock),
            patch.object(o, "_run_analyst", new_callable=AsyncMock, return_value={}),
            patch.object(o, "_run_scientist_plan", new_callable=AsyncMock, return_value=plan),
            patch.object(o, "_run_coder", new_callable=AsyncMock, return_value=script_path),
            patch.object(o, "_validate_script", new_callable=AsyncMock, return_value=True),
            patch.object(o, "_run_experiment", new_callable=AsyncMock, return_value=run_result),
            patch.object(o, "_apply_criteria_updates"),
            patch.object(o, "_run_report", new_callable=AsyncMock),
            patch("auto_scientist.orchestrator.run_with_summaries", new_callable=AsyncMock) as mock_rws,
        ):
            await o.run()

        mock_rws.assert_not_called()

    @pytest.mark.asyncio
    async def test_summaries_in_iteration(self, tmp_path):
        """When summary_model is set, run_with_summaries is called for agent steps."""
        state = ExperimentState(
            domain="test", goal="g", phase="iteration", iteration=0,
        )
        o = Orchestrator(
            state=state, data_path=tmp_path, output_dir=tmp_path,
            max_iterations=1, summary_model="gpt-4o-mini",
        )
        o.config = DomainConfig(name="t", description="d", data_paths=[])
        o.output_dir.mkdir(parents=True, exist_ok=True)

        plan = {"should_stop": False, "hypothesis": "test"}
        script_path = tmp_path / "v00" / "experiment.py"
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text("print('hi')")
        run_result = RunResult(success=True, stdout="ok", return_code=0)

        async def rws_passthrough(coro_fn, *args, **kwargs):
            buf = kwargs.get("message_buffer", [])
            return await coro_fn(buf)

        with (
            patch("auto_scientist.orchestrator.wait_for_window", new_callable=AsyncMock),
            patch("auto_scientist.agents.analyst.run_analyst", new_callable=AsyncMock, return_value={}),
            patch("auto_scientist.agents.scientist.run_scientist", new_callable=AsyncMock, return_value=plan),
            patch("auto_scientist.agents.coder.run_coder", new_callable=AsyncMock, return_value=script_path),
            patch.object(o, "_validate_script", new_callable=AsyncMock, return_value=True),
            patch.object(o, "_run_experiment", new_callable=AsyncMock, return_value=run_result),
            patch.object(o, "_apply_criteria_updates"),
            patch.object(o, "_run_report", new_callable=AsyncMock),
            patch("auto_scientist.orchestrator.run_with_summaries", new_callable=AsyncMock) as mock_rws,
        ):
            # Make run_with_summaries pass through to the actual coroutine
            mock_rws.side_effect = rws_passthrough
            await o.run()

        # Should have been called for analyst, scientist, coder at minimum
        assert mock_rws.call_count >= 3

    @pytest.mark.asyncio
    async def test_results_summary_after_run(self, orchestrator, tmp_path):
        """After successful experiment, results.txt should be summarized."""
        orchestrator.output_dir.mkdir(parents=True, exist_ok=True)
        orchestrator.summary_model = "gpt-4o-mini"
        orchestrator.state.phase = "iteration"
        orchestrator.state.iteration = 0

        plan = {"should_stop": False, "hypothesis": "test"}
        script_path = tmp_path / "experiments" / "v00" / "experiment.py"
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text("print('hi')")

        results_path = script_path.parent / "results.txt"

        run_result = RunResult(success=True, stdout="R2=0.82", return_code=0)

        with (
            patch.object(orchestrator, "_run_analyst", new_callable=AsyncMock, return_value={}),
            patch.object(orchestrator, "_run_scientist_plan", new_callable=AsyncMock, return_value=plan),
            patch.object(orchestrator, "_run_coder", new_callable=AsyncMock, return_value=script_path),
            patch.object(orchestrator, "_validate_script", new_callable=AsyncMock, return_value=True),
            patch.object(orchestrator, "_run_experiment", new_callable=AsyncMock, return_value=run_result) as mock_run,
            patch.object(orchestrator, "_apply_criteria_updates"),
            patch("auto_scientist.orchestrator.summarize_results", new_callable=AsyncMock, return_value="Good results") as mock_sr,
            patch("auto_scientist.orchestrator.print_summary"),
        ):
            # Simulate _run_experiment writing results.txt
            async def run_and_write(*args, **kwargs):
                results_path.write_text("R2=0.82")
                return run_result
            mock_run.side_effect = run_and_write
            await orchestrator._run_iteration_body()

        mock_sr.assert_called_once()

    @pytest.mark.asyncio
    async def test_summary_failure_does_not_break(self, orchestrator, tmp_path):
        """run_with_summaries handles summary errors internally; pipeline completes."""
        orchestrator.output_dir.mkdir(parents=True, exist_ok=True)
        orchestrator.summary_model = "gpt-4o-mini"
        orchestrator.state.phase = "iteration"
        orchestrator.state.iteration = 0

        plan = {"should_stop": False, "hypothesis": "test"}
        script_path = tmp_path / "experiments" / "v00" / "experiment.py"
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text("print('hi')")
        run_result = RunResult(success=True, stdout="ok", return_code=0)

        async def rws_passthrough(coro_fn, agent_name, model, buf, **kwargs):
            return await coro_fn(buf)

        with (
            patch("auto_scientist.agents.analyst.run_analyst", new_callable=AsyncMock, return_value={}),
            patch("auto_scientist.agents.scientist.run_scientist", new_callable=AsyncMock, return_value=plan),
            patch("auto_scientist.agents.coder.run_coder", new_callable=AsyncMock, return_value=script_path),
            patch.object(orchestrator, "_validate_script", new_callable=AsyncMock, return_value=True),
            patch.object(orchestrator, "_run_experiment", new_callable=AsyncMock, return_value=run_result),
            patch.object(orchestrator, "_apply_criteria_updates"),
            patch("auto_scientist.orchestrator.run_with_summaries", new_callable=AsyncMock, side_effect=rws_passthrough),
        ):
            await orchestrator._run_iteration_body()

        assert orchestrator.state.iteration == 1
