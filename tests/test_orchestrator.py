"""Tests for the orchestrator state machine."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from auto_scientist.config import DomainConfig, SuccessCriterion
from auto_scientist.model_config import AgentModelConfig, ModelConfig, ReasoningConfig
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
        name="test",
        description="Test domain",
        data_paths=["data.csv"],
    )
    # Domain knowledge now lives in state, not config
    base_state.domain_knowledge = "test knowledge"
    return o


class TestOrchestratorInit:
    def test_defaults(self, base_state, tmp_path):
        o = Orchestrator(state=base_state, data_path=tmp_path, output_dir=tmp_path)
        assert o.max_iterations == 20
        assert len(o.model_config.critics) == 2
        assert o.debate_rounds == 2
        assert o.max_consecutive_failures == 5

    def test_custom_values(self, base_state, tmp_path):
        mc = ModelConfig(
            defaults=AgentModelConfig(model="claude-sonnet-4-6"),
            critics=[AgentModelConfig(provider="openai", model="gpt-4o")],
        )
        o = Orchestrator(
            state=base_state,
            data_path=tmp_path,
            output_dir=tmp_path,
            max_iterations=5,
            model_config=mc,
            debate_rounds=3,
            max_consecutive_failures=2,
        )
        assert o.max_iterations == 5
        assert len(o.model_config.critics) == 1
        assert o.debate_rounds == 3


class TestValidatePrerequisites:
    """Pre-flight checks: directories, API keys, Claude CLI."""

    _SENTINEL = object()

    @pytest.fixture(autouse=True)
    def _skip_model_validation(self, monkeypatch):
        """Skip model name and reasoning API validation in prerequisite tests."""
        monkeypatch.setattr(
            "auto_scientist.orchestrator._validate_model_names", lambda mc: [],
        )
        monkeypatch.setattr(
            "auto_scientist.orchestrator._validate_reasoning_configs", lambda mc: [],
        )

    def _make_orchestrator(self, tmp_path, state=None, data_path=_SENTINEL, mc=None):
        s = state or ExperimentState(domain="test", goal="g", phase="ingestion")
        dp = tmp_path / "data.csv" if data_path is self._SENTINEL else data_path
        return Orchestrator(
            state=s,
            data_path=dp,
            output_dir=tmp_path / "out",
            model_config=mc or ModelConfig(
                defaults=AgentModelConfig(model="claude-sonnet-4-6"),
                critics=[],
            ),
        )

    @staticmethod
    def _mock_auth_ok(provider):
        """All providers authenticate successfully."""
        return None

    @staticmethod
    def _mock_auth_fail(fail_provider):
        """Return a factory that fails for a specific provider."""
        _names = {"anthropic": "Anthropic", "openai": "OpenAI", "google": "Google"}
        def _check(provider):
            if provider == fail_provider:
                name = _names.get(provider, provider)
                return f"{name} SDK authentication failed: missing credentials"
            return None
        return _check

    def test_passes_when_everything_present(self, tmp_path, monkeypatch):
        data = tmp_path / "data.csv"
        data.write_text("x")
        monkeypatch.setattr("auto_scientist.orchestrator._check_provider_auth", self._mock_auth_ok)
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/claude")
        o = self._make_orchestrator(tmp_path, data_path=data)
        o._validate_prerequisites()  # should not raise

    def test_missing_data_path(self, tmp_path, monkeypatch):
        monkeypatch.setattr("auto_scientist.orchestrator._check_provider_auth", self._mock_auth_ok)
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/claude")
        o = self._make_orchestrator(tmp_path)
        with pytest.raises(RuntimeError, match="Data path does not exist"):
            o._validate_prerequisites()

    def test_none_data_path(self, tmp_path, monkeypatch):
        monkeypatch.setattr("auto_scientist.orchestrator._check_provider_auth", self._mock_auth_ok)
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/claude")
        o = self._make_orchestrator(tmp_path, data_path=None)
        # data_path=None is only a problem during ingestion
        o.state.phase = "ingestion"
        with pytest.raises(RuntimeError, match="--data is required"):
            o._validate_prerequisites()

    def test_data_path_not_checked_after_ingestion(self, tmp_path, monkeypatch):
        monkeypatch.setattr("auto_scientist.orchestrator._check_provider_auth", self._mock_auth_ok)
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/claude")
        state = ExperimentState(domain="t", goal="g", phase="iteration")
        o = self._make_orchestrator(tmp_path, state=state, data_path=None)
        o._validate_prerequisites()  # should not raise

    def test_missing_anthropic_auth(self, tmp_path, monkeypatch):
        data = tmp_path / "data.csv"
        data.write_text("x")
        monkeypatch.setattr(
            "auto_scientist.orchestrator._check_provider_auth",
            self._mock_auth_fail("anthropic"),
        )
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/claude")
        o = self._make_orchestrator(tmp_path, data_path=data)
        with pytest.raises(RuntimeError, match="Anthropic SDK authentication failed"):
            o._validate_prerequisites()

    def test_missing_openai_auth_when_summarizer_set(self, tmp_path, monkeypatch):
        data = tmp_path / "data.csv"
        data.write_text("x")
        monkeypatch.setattr(
            "auto_scientist.orchestrator._check_provider_auth",
            self._mock_auth_fail("openai"),
        )
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/claude")
        mc = ModelConfig(
            defaults=AgentModelConfig(model="claude-sonnet-4-6"),
            summarizer=AgentModelConfig(provider="openai", model="gpt-5.4-nano"),
            critics=[],
        )
        o = self._make_orchestrator(tmp_path, data_path=data, mc=mc)
        with pytest.raises(RuntimeError, match="OpenAI SDK authentication failed"):
            o._validate_prerequisites()

    def test_missing_google_key_for_critic(self, tmp_path, monkeypatch):
        data = tmp_path / "data.csv"
        data.write_text("x")
        monkeypatch.setattr(
            "auto_scientist.orchestrator._check_provider_auth",
            self._mock_auth_fail("google"),
        )
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/claude")
        mc = ModelConfig(
            defaults=AgentModelConfig(model="claude-sonnet-4-6"),
            critics=[AgentModelConfig(provider="google", model="gemini-2.5-pro")],
        )
        o = self._make_orchestrator(tmp_path, data_path=data, mc=mc)
        with pytest.raises(RuntimeError, match="Google SDK authentication failed"):
            o._validate_prerequisites()

    def test_multiple_errors_reported_at_once(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "auto_scientist.orchestrator._check_provider_auth",
            self._mock_auth_fail("anthropic"),
        )
        monkeypatch.setattr("shutil.which", lambda name: None)
        o = self._make_orchestrator(tmp_path)  # data.csv doesn't exist
        with pytest.raises(RuntimeError, match="Data path does not exist") as exc_info:
            o._validate_prerequisites()
        msg = str(exc_info.value)
        assert "Anthropic SDK authentication failed" in msg
        assert "Claude Code CLI not found" in msg

    def test_missing_claude_cli(self, tmp_path, monkeypatch):
        data = tmp_path / "data.csv"
        data.write_text("x")
        monkeypatch.setattr("auto_scientist.orchestrator._check_provider_auth", self._mock_auth_ok)
        monkeypatch.setattr("shutil.which", lambda name: None)
        o = self._make_orchestrator(tmp_path, data_path=data)
        with pytest.raises(RuntimeError, match="Claude Code CLI not found"):
            o._validate_prerequisites()

    def test_non_anthropic_sdk_agent_rejected(self, tmp_path, monkeypatch):
        """SDK agents (analyst, coder, etc.) must use Anthropic models."""
        data = tmp_path / "data.csv"
        data.write_text("x")
        monkeypatch.setattr("auto_scientist.orchestrator._check_provider_auth", self._mock_auth_ok)
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/claude")
        mc = ModelConfig(
            defaults=AgentModelConfig(provider="google", model="gemini-3.1-flash-lite-preview"),
            critics=[],
        )
        o = self._make_orchestrator(tmp_path, data_path=data, mc=mc)
        with pytest.raises(RuntimeError, match="require provider 'anthropic'"):
            o._validate_prerequisites()

    def test_non_anthropic_critic_allowed(self, tmp_path, monkeypatch):
        """Critics can use non-Anthropic models."""
        data = tmp_path / "data.csv"
        data.write_text("x")
        monkeypatch.setattr("auto_scientist.orchestrator._check_provider_auth", self._mock_auth_ok)
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/claude")
        mc = ModelConfig(
            defaults=AgentModelConfig(model="claude-sonnet-4-6"),
            critics=[AgentModelConfig(provider="google", model="gemini-2.5-pro")],
        )
        o = self._make_orchestrator(tmp_path, data_path=data, mc=mc)
        o._validate_prerequisites()  # should not raise


class TestValidateModelNames:
    """Test model name validation against provider APIs."""

    def test_invalid_model_returns_error(self, monkeypatch):
        monkeypatch.setattr(
            "auto_scientist.orchestrator._check_model_exists",
            lambda provider, model: "not found" if model == "bad-model" else None,
        )
        from auto_scientist.orchestrator import _validate_model_names

        mc = ModelConfig(
            defaults=AgentModelConfig(model="bad-model"),
            critics=[],
        )
        errors = _validate_model_names(mc)
        assert len(errors) == 1
        assert "bad-model" in errors[0]
        assert "not found" in errors[0]

    def test_valid_models_return_no_errors(self, monkeypatch):
        monkeypatch.setattr(
            "auto_scientist.orchestrator._check_model_exists",
            lambda provider, model: None,
        )
        from auto_scientist.orchestrator import _validate_model_names

        mc = ModelConfig(
            defaults=AgentModelConfig(model="claude-sonnet-4-6"),
            critics=[AgentModelConfig(provider="openai", model="gpt-5.4")],
        )
        errors = _validate_model_names(mc)
        assert errors == []

    def test_deduplicates_same_model(self, monkeypatch):
        """Same model used by multiple agents should only be checked once."""
        call_count = 0

        def mock_check(provider, model):
            nonlocal call_count
            call_count += 1
            return None

        monkeypatch.setattr("auto_scientist.orchestrator._check_model_exists", mock_check)
        from auto_scientist.orchestrator import _validate_model_names

        mc = ModelConfig(
            defaults=AgentModelConfig(model="claude-sonnet-4-6"),
            critics=[],
        )
        _validate_model_names(mc)
        # All 5 SDK agents use defaults, so only 1 unique check
        assert call_count == 1


class TestValidateReasoningConfigs:
    """Test reasoning config validation against provider constraints."""

    def test_valid_anthropic_reasoning_returns_no_errors(self):
        from auto_scientist.orchestrator import _validate_reasoning_configs

        mc = ModelConfig(
            defaults=AgentModelConfig(model="claude-sonnet-4-6", reasoning=ReasoningConfig(level="high")),
            critics=[],
        )
        errors = _validate_reasoning_configs(mc)
        assert errors == []

    def test_valid_openai_reasoning_returns_no_errors(self):
        from auto_scientist.orchestrator import _validate_reasoning_configs

        mc = ModelConfig(
            defaults=AgentModelConfig(model="claude-sonnet-4-6"),
            critics=[AgentModelConfig(provider="openai", model="gpt-5.4", reasoning=ReasoningConfig(level="high"))],
        )
        errors = _validate_reasoning_configs(mc)
        assert errors == []

    def test_valid_google_3x_reasoning_returns_no_errors(self):
        from auto_scientist.orchestrator import _validate_reasoning_configs

        mc = ModelConfig(
            defaults=AgentModelConfig(model="claude-sonnet-4-6"),
            critics=[AgentModelConfig(provider="google", model="gemini-3.1-pro-preview", reasoning=ReasoningConfig(level="high"))],
        )
        errors = _validate_reasoning_configs(mc)
        assert errors == []

    def test_google_3x_medium_only_flash(self):
        """MEDIUM thinkingLevel is only valid for Gemini 3 Flash models."""
        from auto_scientist.orchestrator import _validate_reasoning_configs

        mc = ModelConfig(
            defaults=AgentModelConfig(model="claude-sonnet-4-6"),
            critics=[AgentModelConfig(provider="google", model="gemini-3.1-pro-preview", reasoning=ReasoningConfig(level="medium"))],
        )
        errors = _validate_reasoning_configs(mc)
        assert len(errors) == 1
        assert "MEDIUM" in errors[0]
        assert "Flash" in errors[0]

    def test_google_3x_minimal_only_flash(self):
        """MINIMAL thinkingLevel is only valid for Gemini 3 Flash models."""
        from auto_scientist.orchestrator import _validate_reasoning_configs

        mc = ModelConfig(
            defaults=AgentModelConfig(model="claude-sonnet-4-6"),
            critics=[AgentModelConfig(provider="google", model="gemini-3-flash-preview", reasoning=ReasoningConfig(level="minimal"))],
        )
        errors = _validate_reasoning_configs(mc)
        assert errors == []

    def test_default_and_off_skip_validation(self):
        from auto_scientist.orchestrator import _validate_reasoning_configs

        mc = ModelConfig(
            defaults=AgentModelConfig(model="claude-sonnet-4-6", reasoning=ReasoningConfig(level="default")),
            critics=[AgentModelConfig(provider="openai", model="gpt-5.4", reasoning=ReasoningConfig(level="off"))],
        )
        errors = _validate_reasoning_configs(mc)
        assert errors == []

    def test_all_builtin_presets_pass_validation(self):
        from auto_scientist.orchestrator import _validate_reasoning_configs

        for preset_name in ["default", "fast", "high", "max"]:
            mc = ModelConfig.builtin_preset(preset_name)
            errors = _validate_reasoning_configs(mc)
            assert errors == [], f"Preset '{preset_name}' has reasoning errors: {errors}"

    def test_anthropic_budget_too_low_returns_error(self):
        from auto_scientist.orchestrator import _validate_reasoning_configs

        mc = ModelConfig(
            defaults=AgentModelConfig(model="claude-sonnet-4-6"),
            analyst=AgentModelConfig(model="claude-sonnet-4-6", reasoning=ReasoningConfig(level="high", budget=50)),
            critics=[],
        )
        errors = _validate_reasoning_configs(mc)
        assert len(errors) == 1
        assert "budget_tokens" in errors[0]
        assert "below" in errors[0]

    def test_anthropic_budget_too_high_returns_error(self):
        from auto_scientist.orchestrator import _validate_reasoning_configs

        mc = ModelConfig(
            defaults=AgentModelConfig(model="claude-sonnet-4-6"),
            analyst=AgentModelConfig(model="claude-sonnet-4-6", reasoning=ReasoningConfig(level="high", budget=200_000)),
            critics=[],
        )
        errors = _validate_reasoning_configs(mc)
        assert len(errors) == 1
        assert "budget_tokens" in errors[0]
        assert "exceeds" in errors[0]


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
        notebook = tmp_path / "experiments" / "lab_notebook.xml"
        notebook.parent.mkdir(parents=True, exist_ok=True)
        notebook.write_text("<lab_notebook/>")
        assert orchestrator._notebook_content() == "<lab_notebook/>"

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
            domain="test",
            goal="g",
            phase="ingestion",
            data_path=str(tmp_path / "raw.csv"),
        )
        o = Orchestrator(
            state=state,
            data_path=tmp_path / "raw.csv",
            output_dir=tmp_path / "experiments",
        )
        result = await o._run_ingestion()

        assert result == canonical
        mock_ingestor.assert_called_once()


class TestPhaseTransitions:
    @pytest.mark.asyncio
    async def test_max_iterations_triggers_report(self, tmp_path):
        state = ExperimentState(
            domain="test",
            goal="g",
            phase="iteration",
            iteration=20,
        )
        o = Orchestrator(
            state=state,
            data_path=tmp_path,
            output_dir=tmp_path,
            max_iterations=20,
        )
        o.config = DomainConfig(name="t", description="d", data_paths=[])

        with (
            patch.object(o, "_validate_prerequisites"),
            patch.object(o, "_run_report", new_callable=AsyncMock),
        ):
            await o.run()

        assert state.phase == "stopped"

    @pytest.mark.asyncio
    async def test_consecutive_failures_triggers_report(self, tmp_path):
        state = ExperimentState(
            domain="test",
            goal="g",
            phase="iteration",
            consecutive_failures=5,
        )
        o = Orchestrator(
            state=state,
            data_path=tmp_path,
            output_dir=tmp_path,
            max_consecutive_failures=5,
        )
        o.config = DomainConfig(name="t", description="d", data_paths=[])

        with (
            patch.object(o, "_validate_prerequisites"),
            patch.object(o, "_run_report", new_callable=AsyncMock),
        ):
            await o.run()

        assert state.phase == "stopped"

    @pytest.mark.asyncio
    async def test_ingestion_transitions_to_iteration(self, tmp_path):
        """After ingestion, phase should be 'iteration' (not 'discovery')."""
        state = ExperimentState(
            domain="test",
            goal="g",
            phase="ingestion",
            data_path=str(tmp_path / "raw.csv"),
        )
        o = Orchestrator(
            state=state,
            data_path=tmp_path / "raw.csv",
            output_dir=tmp_path / "experiments",
            max_iterations=0,
        )

        canonical = tmp_path / "experiments" / "data"
        canonical.mkdir(parents=True, exist_ok=True)

        config_path = tmp_path / "experiments" / "domain_config.json"
        config_path.write_text('{"name":"t","description":"d","data_paths":[]}')

        with (
            patch.object(o, "_validate_prerequisites"),
            patch.object(o, "_run_ingestion", new_callable=AsyncMock, return_value=canonical),
            patch.object(o, "_run_report", new_callable=AsyncMock),
        ):
            await o.run()

        # Should have gone ingestion -> iteration (hit max) -> report -> stopped
        assert state.phase == "stopped"

    @pytest.mark.asyncio
    async def test_resume_from_iteration_skips_ingestion(self, tmp_path):
        state = ExperimentState(
            domain="test",
            goal="g",
            phase="iteration",
            iteration=20,
        )
        o = Orchestrator(
            state=state,
            data_path=tmp_path,
            output_dir=tmp_path,
            max_iterations=20,
        )
        o.config = DomainConfig(name="t", description="d", data_paths=[])

        with (
            patch.object(o, "_validate_prerequisites"),
            patch.object(o, "_run_report", new_callable=AsyncMock) as mock_report,
        ):
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

        analysis = {"observations": ["200 rows"]}

        with patch.object(
            orchestrator,
            "_run_analyst_initial",
            new_callable=AsyncMock,
            return_value=analysis,
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
        orchestrator.model_config.critics = [AgentModelConfig(provider="openai", model="gpt-4o")]

        plan = {"should_stop": False, "hypothesis": "explore"}
        script_path = tmp_path / "experiments" / "v00" / "experiment.py"
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text("print('hi')")
        run_result = RunResult(success=True, stdout="ok", return_code=0)

        with (
            patch.object(
                orchestrator,
                "_run_analyst",
                new_callable=AsyncMock,
                return_value={"success_score": None},
            ),
            patch.object(
                orchestrator, "_run_scientist_plan", new_callable=AsyncMock, return_value=plan
            ),
            patch.object(
                orchestrator,
                "_run_debate",
                new_callable=AsyncMock,
            ) as mock_debate,
            patch.object(
                orchestrator,
                "_run_scientist_revision",
                new_callable=AsyncMock,
            ) as mock_revision,
            patch.object(
                orchestrator, "_run_coder", new_callable=AsyncMock, return_value=script_path
            ),
            patch.object(orchestrator, "_read_run_result", return_value=run_result),
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
                orchestrator,
                "_run_analyst",
                new_callable=AsyncMock,
                return_value={},
            ),
            patch.object(
                orchestrator,
                "_run_scientist_plan",
                new_callable=AsyncMock,
                return_value=plan,
            ),
            patch.object(
                orchestrator,
                "_run_debate",
                new_callable=AsyncMock,
            ),
            patch.object(
                orchestrator,
                "_run_scientist_revision",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch.object(
                orchestrator,
                "_run_coder",
                new_callable=AsyncMock,
                return_value=script_path,
            ),
            patch.object(
                orchestrator,
                "_read_run_result",
                return_value=run_result,
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

    def test_invalid_condition_returns_none(self, orchestrator):
        raw = {"name": "x", "description": "d", "metric_key": "x", "condition": "between 0 and 1"}
        assert Orchestrator._parse_criterion(raw) is None

    def test_empty_condition_returns_none(self, orchestrator):
        raw = {"name": "x", "description": "d", "metric_key": "x", "condition": ""}
        assert Orchestrator._parse_criterion(raw) is None

    def test_missing_condition_key_returns_none(self, orchestrator):
        raw = {"name": "x", "description": "d", "metric_key": "x"}
        assert Orchestrator._parse_criterion(raw) is None

    def test_non_numeric_condition_returns_none(self, orchestrator):
        raw = {
            "name": "Normal residuals",
            "description": "d",
            "metric_key": "x",
            "condition": "approximately normal",
        }
        assert Orchestrator._parse_criterion(raw) is None

    def test_preserves_name_description(self, orchestrator):
        raw = {
            "name": "Accuracy",
            "description": "Model accuracy",
            "metric_key": "acc",
            "condition": "> 0.9",
        }
        sc = Orchestrator._parse_criterion(raw)
        assert sc.name == "Accuracy"
        assert sc.description == "Model accuracy"
        assert sc.metric_key == "acc"


class TestApplyCriteriaUpdates:
    """Test _apply_criteria_updates with various plan shapes."""

    def test_top_level_criteria_populates_state(self, orchestrator):
        plan = {
            "top_level_criteria": [
                {
                    "name": "RMSE",
                    "description": "low error",
                    "metric_key": "rmse",
                    "condition": "< 0.5",
                },
                {
                    "name": "R-squared",
                    "description": "high fit",
                    "metric_key": "r2",
                    "condition": "> 0.95",
                },
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
                    {"name": "R2", "description": "fit", "metric_key": "r2", "condition": "> 0.90"},
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

    def test_filters_out_unmeasurable_criteria(self, orchestrator):
        """Criteria without numeric targets should be filtered out."""
        plan = {
            "top_level_criteria": [
                {
                    "name": "RMSE",
                    "description": "low error",
                    "metric_key": "rmse",
                    "condition": "< 0.5",
                },
                {
                    "name": "Normal residuals",
                    "description": "bad",
                    "metric_key": "x",
                    "condition": "approximately normal",
                },
                {
                    "name": "R2",
                    "description": "high fit",
                    "metric_key": "r2",
                    "condition": "> 0.95",
                },
            ],
        }
        orchestrator.state.iteration = 1
        orchestrator._apply_criteria_updates(plan)

        assert len(orchestrator.state.success_criteria) == 2
        names = [c.name for c in orchestrator.state.success_criteria]
        assert "RMSE" in names
        assert "R2" in names
        assert "Normal residuals" not in names

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
                {"name": "Error", "description": "low", "metric_key": "err", "condition": "< 500"},
            ],
        }
        orchestrator.state.iteration = 1
        orchestrator._apply_criteria_updates(plan)
        assert orchestrator.state.success_criteria[0].target_max == 500.0
        assert orchestrator.state.success_criteria[0].target_min is None

    def test_condition_parsing_greater_than(self, orchestrator):
        plan = {
            "top_level_criteria": [
                {
                    "name": "Accuracy",
                    "description": "high",
                    "metric_key": "acc",
                    "condition": "> 0.95",
                },
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
            version="v01",
            iteration=1,
            script_path=str(tmp_path / "s.py"),
            results_path=str(tmp_path / "results.txt"),
        )
        orchestrator.state.versions = [latest]
        (tmp_path / "results.txt").write_text("data")

        captured_kwargs = {}

        async def capture_analyst(**kwargs):
            captured_kwargs.update(kwargs)
            return {"observations": []}

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
            orchestrator,
            "_run_analyst_initial",
            new_callable=AsyncMock,
            return_value=analysis,
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
                version="v00",
                iteration=0,
                script_path="/tmp/s.py",
                results_path=str(tmp_path / "results.txt"),
            ),
        ]
        (tmp_path / "results.txt").write_text("data")

        plan = {"should_stop": True, "stop_reason": "goal reached"}

        with (
            patch.object(
                orchestrator,
                "_run_analyst",
                new_callable=AsyncMock,
                return_value={},
            ),
            patch.object(
                orchestrator,
                "_run_scientist_plan",
                new_callable=AsyncMock,
                return_value=plan,
            ),
            patch.object(orchestrator, "_apply_criteria_updates"),
        ):
            await orchestrator._run_iteration_body()

        assert orchestrator.state.phase == "report"

    @pytest.mark.asyncio
    async def test_no_critics_skips_debate(self, orchestrator, tmp_path):
        orchestrator.output_dir.mkdir(parents=True, exist_ok=True)
        orchestrator.model_config.critics = []
        orchestrator.state.phase = "iteration"
        orchestrator.state.iteration = 1
        orchestrator.state.versions = [
            VersionEntry(
                version="v00",
                iteration=0,
                script_path="/tmp/s.py",
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
                orchestrator,
                "_run_analyst",
                new_callable=AsyncMock,
                return_value={},
            ),
            patch.object(
                orchestrator,
                "_run_scientist_plan",
                new_callable=AsyncMock,
                return_value=plan,
            ),
            patch.object(
                orchestrator,
                "_run_debate",
                new_callable=AsyncMock,
                return_value=None,
            ) as mock_debate,
            patch.object(
                orchestrator,
                "_run_scientist_revision",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch.object(
                orchestrator,
                "_run_coder",
                new_callable=AsyncMock,
                return_value=script_path,
            ),
            patch.object(
                orchestrator,
                "_read_run_result",
                return_value=run_result,
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
            patch.object(
                orchestrator, "_run_analyst", new_callable=AsyncMock, return_value=analysis
            ),
            patch.object(
                orchestrator, "_run_scientist_plan", new_callable=AsyncMock, return_value=plan
            ),
            patch.object(
                orchestrator, "_run_coder", new_callable=AsyncMock, return_value=script_path
            ),
            patch.object(orchestrator, "_read_run_result", return_value=run_result),
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
                version="v00",
                iteration=0,
                script_path="/tmp/s.py",
                results_path=str(tmp_path / "results.txt"),
            ),
        ]
        (tmp_path / "results.txt").write_text("data")

        plan = {"should_stop": True, "stop_reason": "criteria met"}

        with (
            patch.object(orchestrator, "_run_analyst", new_callable=AsyncMock, return_value={}),
            patch.object(
                orchestrator, "_run_scientist_plan", new_callable=AsyncMock, return_value=plan
            ),
            patch.object(orchestrator, "_apply_criteria_updates"),
        ):
            await orchestrator._run_iteration_body()

        assert orchestrator.state.phase == "report"

    @pytest.mark.asyncio
    async def test_missing_run_result_records_failure(self, orchestrator, tmp_path):
        """When coder produces script but no run_result.json, iteration is a failure."""
        orchestrator.output_dir.mkdir(parents=True, exist_ok=True)
        orchestrator.state.phase = "iteration"
        orchestrator.state.iteration = 0

        plan = {"should_stop": False, "hypothesis": "test"}
        script_path = tmp_path / "experiments" / "v00" / "experiment.py"
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text("print('hi')")
        # No run_result.json -> _read_run_result returns failure

        with (
            patch.object(orchestrator, "_run_analyst", new_callable=AsyncMock, return_value={}),
            patch.object(
                orchestrator, "_run_scientist_plan", new_callable=AsyncMock, return_value=plan
            ),
            patch.object(
                orchestrator, "_run_coder", new_callable=AsyncMock, return_value=script_path
            ),
            patch.object(orchestrator, "_apply_criteria_updates"),
        ):
            await orchestrator._run_iteration_body()

        assert orchestrator.state.consecutive_failures == 1
        assert orchestrator.state.iteration == 1

    @pytest.mark.asyncio
    async def test_coder_returns_none_records_failure(self, orchestrator, tmp_path):
        """When coder returns None, iteration records failure and returns early."""
        orchestrator.output_dir.mkdir(parents=True, exist_ok=True)
        orchestrator.state.phase = "iteration"
        orchestrator.state.iteration = 0

        plan = {"should_stop": False, "hypothesis": "test"}

        with (
            patch.object(
                orchestrator,
                "_run_analyst",
                new_callable=AsyncMock,
                return_value={},
            ),
            patch.object(
                orchestrator,
                "_run_scientist_plan",
                new_callable=AsyncMock,
                return_value=plan,
            ),
            patch.object(
                orchestrator,
                "_run_coder",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch.object(orchestrator, "_apply_criteria_updates"),
        ):
            await orchestrator._run_iteration_body()

        assert orchestrator.state.consecutive_failures == 1
        assert orchestrator.state.iteration == 1
        assert len(orchestrator.state.versions) == 1
        assert orchestrator.state.versions[0].status == "failed"
        assert orchestrator.state.versions[0].script_path == ""

    @pytest.mark.asyncio
    async def test_iteration_summary_prints_artifacts(self, orchestrator, tmp_path):
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
            patch.object(orchestrator, "_run_analyst", new_callable=AsyncMock, return_value={}),
            patch.object(
                orchestrator, "_run_scientist_plan", new_callable=AsyncMock, return_value=plan
            ),
            patch.object(
                orchestrator, "_run_coder", new_callable=AsyncMock, return_value=script_path
            ),
            patch.object(orchestrator, "_read_run_result", return_value=run_result),
            patch.object(orchestrator, "_apply_criteria_updates"),
        ):
            await orchestrator._run_iteration_body()

        # Iteration summary is now in _history (no static prints)
        history = orchestrator._live._history
        assert len(history) > 0


class TestComputeScore:
    """Test _compute_score deterministic scoring from criteria_results."""

    def test_all_required_pass_returns_100(self):
        criteria = [
            SuccessCriterion(
                name="RMSE", description="d", metric_key="rmse", target_max=0.5, required=True
            ),
            SuccessCriterion(
                name="R2", description="d", metric_key="r2", target_min=0.9, required=True
            ),
        ]
        results = [
            {"name": "RMSE", "status": "pass"},
            {"name": "R2", "status": "pass"},
        ]
        assert Orchestrator._compute_score(results, criteria) == 100

    def test_no_required_pass_returns_0(self):
        criteria = [
            SuccessCriterion(
                name="RMSE", description="d", metric_key="rmse", target_max=0.5, required=True
            ),
            SuccessCriterion(
                name="R2", description="d", metric_key="r2", target_min=0.9, required=True
            ),
        ]
        results = [
            {"name": "RMSE", "status": "fail"},
            {"name": "R2", "status": "fail"},
        ]
        assert Orchestrator._compute_score(results, criteria) == 0

    def test_mixed_returns_proportional(self):
        criteria = [
            SuccessCriterion(
                name="A", description="d", metric_key="a", target_min=0.5, required=True
            ),
            SuccessCriterion(
                name="B", description="d", metric_key="b", target_max=10, required=True
            ),
            SuccessCriterion(
                name="C", description="d", metric_key="c", target_min=0.1, required=True
            ),
        ]
        results = [
            {"name": "A", "status": "pass"},
            {"name": "B", "status": "fail"},
            {"name": "C", "status": "pass"},
        ]
        assert Orchestrator._compute_score(results, criteria) == 67  # round(2/3 * 100)

    def test_optional_criteria_ignored(self):
        criteria = [
            SuccessCriterion(
                name="A", description="d", metric_key="a", target_min=0.5, required=True
            ),
            SuccessCriterion(
                name="B", description="d", metric_key="b", target_max=10, required=False
            ),
        ]
        results = [
            {"name": "A", "status": "pass"},
            {"name": "B", "status": "fail"},
        ]
        # Only 1 required, it passes -> 100
        assert Orchestrator._compute_score(results, criteria) == 100

    def test_no_criteria_returns_0(self):
        assert Orchestrator._compute_score([], []) == 0

    def test_no_results_returns_0(self):
        criteria = [
            SuccessCriterion(
                name="A", description="d", metric_key="a", target_min=0.5, required=True
            ),
        ]
        assert Orchestrator._compute_score([], criteria) == 0

    def test_unable_to_measure_counts_as_fail(self):
        criteria = [
            SuccessCriterion(
                name="A", description="d", metric_key="a", target_min=0.5, required=True
            ),
        ]
        results = [
            {"name": "A", "status": "unable_to_measure"},
        ]
        assert Orchestrator._compute_score(results, criteria) == 0


class TestScoreLatest:
    """Test _score_latest updates version score and best tracking."""

    def test_scores_latest_version(self, orchestrator):
        orchestrator.state.success_criteria = [
            SuccessCriterion(
                name="RMSE", description="d", metric_key="rmse", target_max=0.5, required=True
            ),
        ]
        latest = VersionEntry(version="v01", iteration=1, script_path="/tmp/s.py")
        orchestrator.state.versions = [latest]

        analysis = {"criteria_results": [{"name": "RMSE", "status": "pass"}]}
        orchestrator._score_latest(analysis)

        assert latest.score == 100
        assert orchestrator.state.best_score == 100
        assert orchestrator.state.best_version == "v01"

    def test_no_versions_is_noop(self, orchestrator):
        orchestrator._score_latest({"criteria_results": []})
        assert orchestrator.state.best_score == 0

    def test_none_analysis_scores_zero(self, orchestrator):
        orchestrator.state.success_criteria = [
            SuccessCriterion(
                name="A", description="d", metric_key="a", target_min=0.5, required=True
            ),
        ]
        latest = VersionEntry(version="v01", iteration=1, script_path="/tmp/s.py")
        orchestrator.state.versions = [latest]

        orchestrator._score_latest(None)
        assert latest.score == 0

    def test_scores_after_criteria_defined(self, orchestrator):
        """Score should work even when criteria are defined in the same iteration."""
        latest = VersionEntry(version="v01", iteration=1, script_path="/tmp/s.py")
        orchestrator.state.versions = [latest]

        # Criteria defined by Scientist in same iteration
        orchestrator.state.success_criteria = [
            SuccessCriterion(
                name="R2", description="d", metric_key="r2", target_min=0.9, required=True
            ),
            SuccessCriterion(
                name="RMSE", description="d", metric_key="rmse", target_max=0.5, required=True
            ),
        ]

        analysis = {
            "criteria_results": [
                {"name": "R2", "status": "pass"},
                {"name": "RMSE", "status": "fail"},
            ],
        }
        orchestrator._score_latest(analysis)
        assert latest.score == 50


class TestRunAnalystInitial:
    @pytest.mark.asyncio
    async def test_calls_run_analyst_with_data_dir(self, orchestrator, tmp_path):
        orchestrator.output_dir.mkdir(parents=True, exist_ok=True)
        orchestrator.data_path = tmp_path / "data"

        captured_kwargs = {}

        async def capture_analyst(**kwargs):
            captured_kwargs.update(kwargs)
            return {"observations": []}

        with patch("auto_scientist.agents.analyst.run_analyst", side_effect=capture_analyst):
            result = await orchestrator._run_analyst_initial()

        assert captured_kwargs["results_path"] is None
        assert captured_kwargs["data_dir"] == tmp_path / "data"
        assert "success_score" not in result

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
            VersionEntry(version="v00", iteration=0, script_path="/tmp/s.py", results_path=None),
        ]

        result = await orchestrator._run_analyst()
        assert result is None

    @pytest.mark.asyncio
    async def test_analyst_returns_criteria_results(self, orchestrator, tmp_path):
        """_run_analyst returns analysis with criteria_results for later scoring."""
        orchestrator.output_dir.mkdir(parents=True, exist_ok=True)

        results_path = tmp_path / "results.txt"
        results_path.write_text("data")
        latest = VersionEntry(
            version="v01",
            iteration=1,
            script_path=str(tmp_path / "s.py"),
            results_path=str(results_path),
        )
        orchestrator.state.versions = [latest]

        async def fake_analyst(**kwargs):
            return {
                "criteria_results": [{"name": "RMSE", "status": "pass"}],
            }

        with patch("auto_scientist.agents.analyst.run_analyst", side_effect=fake_analyst):
            result = await orchestrator._run_analyst()

        assert result["criteria_results"] == [{"name": "RMSE", "status": "pass"}]

    @pytest.mark.asyncio
    async def test_error_returns_none(self, orchestrator, tmp_path):
        orchestrator.output_dir.mkdir(parents=True, exist_ok=True)

        results_path = tmp_path / "results.txt"
        results_path.write_text("data")
        orchestrator.state.versions = [
            VersionEntry(
                version="v01",
                iteration=1,
                script_path=str(tmp_path / "s.py"),
                results_path=str(results_path),
            ),
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
        plan = {
            "hypothesis": "test",
            "strategy": "incremental",
            "notebook_entry": "Test plan\n\nNarrative",
        }

        with patch(
            "auto_scientist.agents.scientist.run_scientist",
            new_callable=AsyncMock,
            return_value=plan,
        ):
            result = await orchestrator._run_scientist_plan({"observations": []})

        assert result["hypothesis"] == "test"
        notebook = orchestrator.output_dir / "lab_notebook.xml"
        assert notebook.exists()
        content = notebook.read_text()
        assert "<entry" in content
        assert "<title>Test plan</title>" in content

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
        orchestrator.model_config.critics = []
        result = await orchestrator._run_debate({"hypothesis": "test"})
        assert result is None

    @pytest.mark.asyncio
    async def test_no_plan_returns_none(self, orchestrator):
        orchestrator.model_config.critics = [AgentModelConfig(provider="openai", model="gpt-4o")]
        result = await orchestrator._run_debate(None)
        assert result is None

    @pytest.mark.asyncio
    async def test_calls_run_debate_with_args(self, orchestrator, tmp_path):
        orchestrator.output_dir.mkdir(parents=True, exist_ok=True)
        orchestrator.model_config.critics = [AgentModelConfig(provider="openai", model="gpt-4o")]
        orchestrator.model_config.summarizer = None  # disable summarizer to test run_debate path
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
        assert len(call_kwargs["critic_configs"]) == 1
        assert call_kwargs["critic_configs"][0].model == "gpt-4o"
        assert call_kwargs["domain_knowledge"] == "test knowledge"
        assert call_kwargs["plot_paths"] == []  # no versions => no plots
        assert isinstance(call_kwargs["message_buffers"], dict)

    @pytest.mark.asyncio
    async def test_calls_run_single_critic_debate_with_summaries(self, orchestrator, tmp_path):
        """When summarizer is enabled, each critic gets its own summarizer."""
        orchestrator.output_dir.mkdir(parents=True, exist_ok=True)
        orchestrator.model_config.critics = [AgentModelConfig(provider="openai", model="gpt-4o")]
        plan = {"hypothesis": "test"}

        critique = {"model": "openai:gpt-4o", "critique": "looks good", "transcript": []}

        with (
            patch(
                "auto_scientist.agents.critic.run_single_critic_debate",
                new_callable=AsyncMock,
                return_value=critique,
            ) as mock_single,
            patch(
                "auto_scientist.summarizer._query_summary",
                new_callable=AsyncMock,
                return_value="Summarizing...",
            ),
        ):
            result = await orchestrator._run_debate(plan)

        assert result == [critique]
        call_kwargs = mock_single.call_args.kwargs
        assert call_kwargs["config"].model == "gpt-4o"

    @pytest.mark.asyncio
    async def test_passes_plot_paths_from_version_dir(self, orchestrator, tmp_path):
        """When versions exist with PNGs, plot_paths are gathered and passed."""
        orchestrator.output_dir.mkdir(parents=True, exist_ok=True)
        orchestrator.model_config.critics = [AgentModelConfig(provider="openai", model="gpt-4o")]
        orchestrator.model_config.summarizer = None  # disable summarizer to test run_debate path

        # Create a version directory with plot PNGs
        version_dir = orchestrator.output_dir / "v01"
        version_dir.mkdir()
        script_path = version_dir / "experiment.py"
        script_path.write_text("pass")
        (version_dir / "loss_curve.png").write_bytes(b"\x89PNG")
        (version_dir / "scatter.png").write_bytes(b"\x89PNG")

        orchestrator.state.versions = [
            VersionEntry(
                version="v01",
                iteration=1,
                script_path=str(script_path),
                results_path=str(version_dir / "results.txt"),
            )
        ]

        critique = [{"model": "openai:gpt-4o", "critique": "ok", "transcript": []}]

        with patch(
            "auto_scientist.agents.critic.run_debate",
            new_callable=AsyncMock,
            return_value=critique,
        ) as mock_debate:
            await orchestrator._run_debate({"hypothesis": "test"})

        call_kwargs = mock_debate.call_args.kwargs
        plot_paths = call_kwargs["plot_paths"]
        assert len(plot_paths) == 2
        plot_names = sorted(p.name for p in plot_paths)
        assert plot_names == ["loss_curve.png", "scatter.png"]


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
            {
                "model": "openai:gpt-4o",
                "critique": "weak",
                "transcript": [{"role": "critic", "content": "weak"}],
            },
        ]
        revised = {"hypothesis": "revised", "notebook_entry": "Revised plan\n\nAdjusted approach"}

        with patch(
            "auto_scientist.agents.scientist.run_scientist_revision",
            new_callable=AsyncMock,
            return_value=revised,
        ):
            result = await orchestrator._run_scientist_revision(plan, debate_result, {})

        assert result["hypothesis"] == "revised"
        notebook = orchestrator.output_dir / "lab_notebook.xml"
        content = notebook.read_text()
        assert "<title>Revised plan</title>" in content
        assert 'source="revision"' in content

    @pytest.mark.asyncio
    async def test_error_returns_none(self, orchestrator, tmp_path):
        orchestrator.output_dir.mkdir(parents=True, exist_ok=True)
        plan = {"hypothesis": "original"}
        debate_result = [
            {
                "model": "openai:gpt-4o",
                "critique": "weak",
                "transcript": [{"role": "critic", "content": "weak"}],
            },
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
            VersionEntry(
                version="v00",
                iteration=0,
                script_path=str(tmp_path / "v00" / "experiment.py"),
            ),
        ]

        captured_kwargs = {}

        async def capture_coder(**kwargs):
            captured_kwargs.update(kwargs)
            return tmp_path / "v01" / "experiment.py"

        with patch(
            "auto_scientist.agents.coder.run_coder",
            side_effect=capture_coder,
        ):
            await orchestrator._run_coder({"hypothesis": "test"})

        assert str(captured_kwargs["previous_script"]) == str(
            tmp_path / "v00" / "experiment.py"
        )

    @pytest.mark.asyncio
    async def test_forwards_run_config_from_domain(self, orchestrator, tmp_path):
        """Orchestrator passes run_timeout_minutes and run_command to coder."""
        orchestrator.output_dir.mkdir(parents=True, exist_ok=True)
        orchestrator.config = DomainConfig(
            name="t",
            description="d",
            data_paths=[],
            run_command="python {script_path}",
            run_timeout_minutes=60,
        )

        captured_kwargs = {}

        async def capture_coder(**kwargs):
            captured_kwargs.update(kwargs)
            return tmp_path / "v00" / "experiment.py"

        with (
            patch(
                "auto_scientist.agents.coder.run_coder",
                side_effect=capture_coder,
            ),
            patch("shutil.which", return_value="/usr/bin/python"),
        ):
            await orchestrator._run_coder({"hypothesis": "test"})

        assert captured_kwargs["run_timeout_minutes"] == 60
        # Orchestrator resolves the executable to an absolute path
        assert captured_kwargs["run_command"] == "/usr/bin/python {script_path}"

    @pytest.mark.asyncio
    async def test_run_config_defaults_without_config(self, orchestrator, tmp_path):
        """Without DomainConfig, uses default run values."""
        orchestrator.output_dir.mkdir(parents=True, exist_ok=True)
        orchestrator.config = None

        captured_kwargs = {}

        async def capture_coder(**kwargs):
            captured_kwargs.update(kwargs)
            return tmp_path / "v00" / "experiment.py"

        with (
            patch(
                "auto_scientist.agents.coder.run_coder",
                side_effect=capture_coder,
            ),
            patch("shutil.which", return_value="/usr/local/bin/uv"),
        ):
            await orchestrator._run_coder({"hypothesis": "test"})

        assert captured_kwargs["run_timeout_minutes"] == 120
        # Orchestrator resolves the executable to an absolute path
        assert captured_kwargs["run_command"] == "/usr/local/bin/uv run {script_path}"

    @pytest.mark.asyncio
    async def test_run_command_warns_when_exe_not_found(self, orchestrator, tmp_path):
        """When shutil.which returns None, logs a warning and passes command through."""
        orchestrator.output_dir.mkdir(parents=True, exist_ok=True)
        orchestrator.config = None

        captured_kwargs = {}

        async def capture_coder(**kwargs):
            captured_kwargs.update(kwargs)
            return tmp_path / "v00" / "experiment.py"

        with (
            patch(
                "auto_scientist.agents.coder.run_coder",
                side_effect=capture_coder,
            ),
            patch("shutil.which", return_value=None),
        ):
            await orchestrator._run_coder({"hypothesis": "test"})

        # Falls through with original command when which returns None
        assert captured_kwargs["run_command"] == "uv run {script_path}"


class TestReadRunResult:
    def test_happy_path(self, orchestrator, tmp_path):
        """Reads valid run_result.json + results.txt into RunResult."""
        version_dir = tmp_path / "v01"
        version_dir.mkdir()
        (version_dir / "run_result.json").write_text(
            '{"success": true, "return_code": 0, "timed_out": false, "error": null, "attempts": 1}'
        )
        (version_dir / "results.txt").write_text("output data")
        (version_dir / "plot.png").write_text("fake png")

        result = orchestrator._read_run_result(version_dir)

        assert result.success is True
        assert result.return_code == 0
        assert result.timed_out is False
        assert result.stdout == "output data"
        assert any("plot.png" in f for f in result.output_files)

    def test_missing_run_result_json(self, orchestrator, tmp_path):
        """Returns failure RunResult when run_result.json is missing."""
        version_dir = tmp_path / "v01"
        version_dir.mkdir()

        result = orchestrator._read_run_result(version_dir)

        assert result.success is False
        assert "run_result.json" in result.stderr

    def test_malformed_json(self, orchestrator, tmp_path):
        """Returns failure RunResult when run_result.json is malformed."""
        version_dir = tmp_path / "v01"
        version_dir.mkdir()
        (version_dir / "run_result.json").write_text("not valid json{}")

        result = orchestrator._read_run_result(version_dir)

        assert result.success is False
        assert result.stderr  # should contain error message

    def test_reads_stderr_txt(self, orchestrator, tmp_path):
        """Includes stderr.txt content in RunResult.stderr."""
        version_dir = tmp_path / "v01"
        version_dir.mkdir()
        (version_dir / "run_result.json").write_text(
            '{"success": false, "return_code": 1, "timed_out": false, '
            '"error": "import error", "attempts": 2}'
        )
        (version_dir / "stderr.txt").write_text("Traceback: ImportError")

        result = orchestrator._read_run_result(version_dir)

        assert result.success is False
        assert "import error" in result.stderr
        assert "Traceback" in result.stderr

    def test_discovers_output_files(self, orchestrator, tmp_path):
        """Discovers experiment outputs, excludes infra files."""
        version_dir = tmp_path / "v01"
        version_dir.mkdir()
        (version_dir / "run_result.json").write_text(
            '{"success": true, "return_code": 0, "timed_out": false, '
            '"error": null, "attempts": 1}'
        )
        (version_dir / "experiment.py").write_text("print('hi')")
        (version_dir / "plot.png").write_text("fake")
        (version_dir / "results.txt").write_text("output")
        (version_dir / "data.csv").write_text("a,b")
        (version_dir / "meta.json").write_text("{}")
        # Infra files that should be excluded
        (version_dir / "exitcode.txt").write_text("0")
        (version_dir / "stderr.txt").write_text("")

        result = orchestrator._read_run_result(version_dir)

        filenames = {Path(f).name for f in result.output_files}
        # Experiment outputs included
        assert "plot.png" in filenames
        assert "results.txt" in filenames
        assert "data.csv" in filenames
        assert "meta.json" in filenames
        # Infra files excluded
        assert "run_result.json" not in filenames
        assert "exitcode.txt" not in filenames
        assert "stderr.txt" not in filenames
        # Script not included
        assert "experiment.py" not in filenames

    def test_timed_out(self, orchestrator, tmp_path):
        """Correctly maps timed_out field."""
        version_dir = tmp_path / "v01"
        version_dir.mkdir()
        (version_dir / "run_result.json").write_text(
            '{"success": false, "return_code": 124, "timed_out": true, '
            '"error": "Script timed out", "attempts": 1}'
        )

        result = orchestrator._read_run_result(version_dir)

        assert result.success is False
        assert result.timed_out is True
        assert result.return_code == 124


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
    async def test_error_does_not_raise(self, orchestrator, tmp_path):
        orchestrator.output_dir.mkdir(parents=True, exist_ok=True)

        with patch(
            "auto_scientist.agents.report.run_report",
            new_callable=AsyncMock,
            side_effect=RuntimeError("Report failed"),
        ):
            await orchestrator._run_report()

        # Error is handled internally (logged + panel error state), not raised
        # The test passes if no exception propagates


class TestRunFullOrchestration:
    @pytest.mark.asyncio
    async def test_run_prints_critics(self, tmp_path, capsys):
        state = ExperimentState(
            domain="test",
            goal="g",
            phase="iteration",
            iteration=20,
        )
        mc = ModelConfig(
            defaults=AgentModelConfig(model="claude-sonnet-4-6"),
            critics=[AgentModelConfig(provider="openai", model="gpt-4o")],
        )
        o = Orchestrator(
            state=state,
            data_path=tmp_path,
            output_dir=tmp_path,
            max_iterations=20,
            model_config=mc,
        )
        o.config = DomainConfig(name="t", description="d", data_paths=[])

        with (
            patch.object(o, "_validate_prerequisites"),
            patch.object(o, "_run_report", new_callable=AsyncMock),
        ):
            await o.run()

        captured = capsys.readouterr()
        assert "openai:gpt-4o" in captured.out

    @pytest.mark.asyncio
    async def test_run_iteration_loop_with_schedule(self, tmp_path):
        state = ExperimentState(
            domain="test",
            goal="g",
            phase="iteration",
            iteration=0,
        )
        o = Orchestrator(
            state=state,
            data_path=tmp_path,
            output_dir=tmp_path,
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
            patch.object(o, "_validate_prerequisites"),
            patch("auto_scientist.orchestrator.wait_for_window", new_callable=AsyncMock),
            patch.object(o, "_run_analyst", new_callable=AsyncMock, return_value={}),
            patch.object(o, "_run_scientist_plan", new_callable=AsyncMock, return_value=plan),
            patch.object(o, "_run_coder", new_callable=AsyncMock, return_value=script_path),
            patch.object(o, "_read_run_result", return_value=run_result),
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
            domain="test",
            goal="g",
            phase="ingestion",
        )
        o = Orchestrator(
            state=state,
            data_path=tmp_path / "raw.csv",
            output_dir=tmp_path / "experiments",
            max_iterations=0,
        )

        canonical = tmp_path / "experiments" / "data"
        canonical.mkdir(parents=True, exist_ok=True)
        (canonical / "clean.csv").write_text("a,b\n1,2\n")

        config_path = tmp_path / "experiments" / "domain_config.json"
        config_path.write_text('{"name":"mydom","description":"test","data_paths":["clean.csv"]}')

        with (
            patch.object(o, "_validate_prerequisites"),
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
    async def test_ingestion_prints_data_files(self, tmp_path):
        state = ExperimentState(
            domain="test",
            goal="g",
            phase="ingestion",
        )
        o = Orchestrator(
            state=state,
            data_path=tmp_path / "raw.csv",
            output_dir=tmp_path / "experiments",
            max_iterations=0,
        )

        canonical = tmp_path / "experiments" / "data"
        canonical.mkdir(parents=True, exist_ok=True)
        (canonical / "data.csv").write_text("a,b\n1,2\n")

        with (
            patch.object(o, "_validate_prerequisites"),
            patch(
                "auto_scientist.agents.ingestor.run_ingestor",
                new_callable=AsyncMock,
                return_value=canonical,
            ),
            patch.object(o, "_run_report", new_callable=AsyncMock),
        ):
            await o.run()

        # Ingestor panel is in history with data file info in done_summary
        from auto_scientist.console import AgentPanel

        history_panels = [p for p in o._live._history if isinstance(p, AgentPanel)]
        assert any("data.csv" in p.done_summary for p in history_panels)


class TestSummaryIntegration:
    def test_default_has_summarizer(self, base_state, tmp_path):
        o = Orchestrator(state=base_state, data_path=tmp_path, output_dir=tmp_path)
        assert o.model_config.summarizer is not None

    def test_no_summarizer_when_none(self, base_state, tmp_path):
        mc = ModelConfig(defaults=AgentModelConfig(model="claude-sonnet-4-6"))
        o = Orchestrator(
            state=base_state,
            data_path=tmp_path,
            output_dir=tmp_path,
            model_config=mc,
        )
        assert o.model_config.summarizer is None

    @pytest.mark.asyncio
    async def test_no_summary_when_summarizer_none(self, tmp_path):
        """When summarizer is None, run_with_summaries is never called."""
        state = ExperimentState(
            domain="test",
            goal="g",
            phase="iteration",
            iteration=0,
        )
        mc = ModelConfig(defaults=AgentModelConfig(model="claude-sonnet-4-6"))
        o = Orchestrator(
            state=state,
            data_path=tmp_path,
            output_dir=tmp_path,
            max_iterations=1,
            model_config=mc,
        )
        o.config = DomainConfig(name="t", description="d", data_paths=[])
        o.output_dir.mkdir(parents=True, exist_ok=True)

        plan = {"should_stop": False, "hypothesis": "test"}
        script_path = tmp_path / "v00" / "experiment.py"
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text("print('hi')")
        run_result = RunResult(success=True, stdout="ok", return_code=0)

        with (
            patch.object(o, "_validate_prerequisites"),
            patch("auto_scientist.orchestrator.wait_for_window", new_callable=AsyncMock),
            patch.object(o, "_run_analyst", new_callable=AsyncMock, return_value={}),
            patch.object(o, "_run_scientist_plan", new_callable=AsyncMock, return_value=plan),
            patch.object(o, "_run_coder", new_callable=AsyncMock, return_value=script_path),
            patch.object(o, "_read_run_result", return_value=run_result),
            patch.object(o, "_apply_criteria_updates"),
            patch.object(o, "_run_report", new_callable=AsyncMock),
            patch(
                "auto_scientist.orchestrator.run_with_summaries", new_callable=AsyncMock
            ) as mock_rws,
        ):
            await o.run()

        mock_rws.assert_not_called()

    @pytest.mark.asyncio
    async def test_summaries_in_iteration(self, tmp_path):
        """When summarizer is set, run_with_summaries is called for agent steps."""
        state = ExperimentState(
            domain="test",
            goal="g",
            phase="iteration",
            iteration=0,
        )
        mc = ModelConfig.builtin_preset("default")
        o = Orchestrator(
            state=state,
            data_path=tmp_path,
            output_dir=tmp_path,
            max_iterations=1,
            model_config=mc,
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
            patch.object(o, "_validate_prerequisites"),
            patch("auto_scientist.orchestrator.wait_for_window", new_callable=AsyncMock),
            patch(
                "auto_scientist.agents.analyst.run_analyst", new_callable=AsyncMock, return_value={}
            ),
            patch(
                "auto_scientist.agents.scientist.run_scientist",
                new_callable=AsyncMock,
                return_value=plan,
            ),
            patch(
                "auto_scientist.agents.coder.run_coder",
                new_callable=AsyncMock,
                return_value=script_path,
            ),
            patch.object(o, "_read_run_result", return_value=run_result),
            patch.object(o, "_apply_criteria_updates"),
            patch.object(o, "_run_report", new_callable=AsyncMock),
            patch(
                "auto_scientist.orchestrator.run_with_summaries", new_callable=AsyncMock
            ) as mock_rws,
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
        orchestrator.model_config.summarizer = AgentModelConfig(provider="openai", model="gpt-4o-mini")
        orchestrator.state.phase = "iteration"
        orchestrator.state.iteration = 0

        plan = {"should_stop": False, "hypothesis": "test"}
        script_path = tmp_path / "experiments" / "v00" / "experiment.py"
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text("print('hi')")

        results_path = script_path.parent / "results.txt"

        run_result = RunResult(success=True, stdout="R2=0.82", return_code=0)

        # Pre-create results.txt (written by the Coder agent now)
        results_path.write_text("R2=0.82")

        with (
            patch.object(orchestrator, "_run_analyst", new_callable=AsyncMock, return_value={}),
            patch.object(
                orchestrator, "_run_scientist_plan", new_callable=AsyncMock, return_value=plan
            ),
            patch.object(
                orchestrator, "_run_coder", new_callable=AsyncMock, return_value=script_path
            ),
            patch.object(orchestrator, "_read_run_result", return_value=run_result),
            patch.object(orchestrator, "_apply_criteria_updates"),
            patch(
                "auto_scientist.orchestrator.summarize_results",
                new_callable=AsyncMock,
                return_value="Good results",
            ) as mock_sr,
        ):
            await orchestrator._run_iteration_body()

        mock_sr.assert_called_once()

    @pytest.mark.asyncio
    async def test_summary_failure_does_not_break(self, orchestrator, tmp_path):
        """run_with_summaries handles summary errors internally; pipeline completes."""
        orchestrator.output_dir.mkdir(parents=True, exist_ok=True)
        orchestrator.model_config.summarizer = AgentModelConfig(provider="openai", model="gpt-4o-mini")
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
            patch(
                "auto_scientist.agents.analyst.run_analyst", new_callable=AsyncMock, return_value={}
            ),
            patch(
                "auto_scientist.agents.scientist.run_scientist",
                new_callable=AsyncMock,
                return_value=plan,
            ),
            patch(
                "auto_scientist.agents.coder.run_coder",
                new_callable=AsyncMock,
                return_value=script_path,
            ),
            patch.object(orchestrator, "_read_run_result", return_value=run_result),
            patch.object(orchestrator, "_apply_criteria_updates"),
            patch(
                "auto_scientist.orchestrator.run_with_summaries",
                new_callable=AsyncMock,
                side_effect=rws_passthrough,
            ),
        ):
            await orchestrator._run_iteration_body()

        assert orchestrator.state.iteration == 1
