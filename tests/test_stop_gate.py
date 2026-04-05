"""Tests for the stop gate agents: completeness assessment, stop debate, and stop revision."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from auto_scientist.agent_result import AgentResult
from auto_scientist.agents.debate_models import CriticOutput, DebateResult
from auto_scientist.model_config import AgentModelConfig
from auto_scientist.sdk_utils import OutputValidationError

# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------


def _valid_critic_json(
    concerns: list | None = None,
    overall: str = "Stop looks premature.",
) -> str:
    """Build a valid CriticOutput JSON string for stop debate tests."""
    obj = {
        "concerns": concerns
        or [
            {
                "claim": "Coverage of sub-question 2 is shallow",
                "severity": "high",
                "confidence": "medium",
                "category": "methodology",
            }
        ],
        "alternative_hypotheses": ["Try nonlinear relationship"],
        "overall_assessment": overall,
    }
    return json.dumps(obj)


def _valid_assessment_dict(
    recommendation: str = "continue",
    overall_coverage: str = "partial",
) -> dict:
    """Build a valid CompletenessAssessmentOutput dict."""
    return {
        "sub_questions": [
            {
                "question": "What drives outlet clarity?",
                "coverage": "shallow",
                "evidence": ["R²=0.3"],
                "gaps": ["only linear tested"],
            }
        ],
        "overall_coverage": overall_coverage,
        "recommendation": recommendation,
    }


def _valid_scientist_plan_dict(should_stop: bool = False) -> dict:
    """Build a valid ScientistPlanOutput dict."""
    return {
        "hypothesis": "Nonlinear effects explain remaining variance",
        "strategy": "incremental",
        "changes": [
            {
                "what": "add polynomial terms",
                "why": "test nonlinearity",
                "how": "degree=2",
                "priority": 1,
            }
        ],
        "expected_impact": "Higher R²",
        "should_stop": should_stop,
        "stop_reason": "Investigation complete" if should_stop else None,
        "notebook_entry": "## Stop revision\nContinuing with nonlinear test.",
    }


def _pad(text: str) -> str:
    """Pad a string to minimum length to satisfy response length checks."""
    min_len = 50
    if len(text) >= min_len:
        return text
    return text + " " * (min_len - len(text))


def _critic_result(text: str) -> AgentResult:
    """Create an AgentResult for use as a mock return value."""
    return AgentResult(text=_pad(text), input_tokens=10, output_tokens=5)


def _structured_critic_result(
    concerns: list | None = None,
    overall: str = "Stop looks premature.",
) -> AgentResult:
    """Create an AgentResult containing valid CriticOutput JSON."""
    return AgentResult(
        text=_pad(_valid_critic_json(concerns, overall)),
        input_tokens=10,
        output_tokens=5,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def openai_config():
    return AgentModelConfig(provider="openai", model="gpt-5.4")


@pytest.fixture
def anthropic_config():
    return AgentModelConfig(provider="anthropic", model="claude-sonnet-4-6")


@pytest.fixture
def valid_assessment():
    return _valid_assessment_dict()


@pytest.fixture
def stop_notebook(tmp_path):
    """Create a temporary notebook file."""
    nb = tmp_path / "notebook.md"
    nb.write_text("# Lab Notebook\n## Iteration 1\nResult: R²=0.85")
    return nb


# ---------------------------------------------------------------------------
# run_completeness_assessment
# ---------------------------------------------------------------------------

COLLECT_TEXT_PATH = "auto_scientist.agents.stop_gate.collect_text_from_query"
VALIDATE_JSON_PATH = "auto_scientist.agents.stop_gate.validate_json_output"


class TestRunCompletenessAssessment:
    """Tests for run_completeness_assessment()."""

    @pytest.mark.asyncio
    async def test_returns_validated_dict_on_success(self, stop_notebook):
        """Happy path: collect_text_from_query returns valid JSON, dict is returned."""
        from auto_scientist.agents.stop_gate import run_completeness_assessment

        assessment = _valid_assessment_dict()
        raw_json = json.dumps(assessment)

        with patch(COLLECT_TEXT_PATH, new_callable=AsyncMock, return_value=(raw_json, {}, None)):
            result = await run_completeness_assessment(
                goal="find factors driving water clarity",
                stop_reason="All sub-questions answered",
                notebook_path=stop_notebook,
            )

        assert result["overall_coverage"] == "partial"
        assert result["recommendation"] == "continue"
        assert len(result["sub_questions"]) == 1

    @pytest.mark.asyncio
    async def test_returns_stop_recommendation_when_thorough(self, stop_notebook):
        """When coverage is thorough, recommendation=stop is preserved."""
        from auto_scientist.agents.stop_gate import run_completeness_assessment

        assessment = _valid_assessment_dict(recommendation="stop", overall_coverage="thorough")
        assessment["sub_questions"][0]["coverage"] = "thorough"
        raw_json = json.dumps(assessment)

        with patch(COLLECT_TEXT_PATH, new_callable=AsyncMock, return_value=(raw_json, {}, None)):
            result = await run_completeness_assessment(
                goal="discover mineral composition patterns",
                stop_reason="Full coverage achieved",
                notebook_path=stop_notebook,
            )

        assert result["recommendation"] == "stop"
        assert result["overall_coverage"] == "thorough"

    @pytest.mark.asyncio
    async def test_retries_on_validation_error(self, stop_notebook):
        """When the first response fails validation, a second attempt is made."""
        from auto_scientist.agents.stop_gate import run_completeness_assessment

        valid_json = json.dumps(_valid_assessment_dict())
        invalid_response = ("not json at all {{", {}, None)
        valid_response = (valid_json, {}, None)

        with patch(
            COLLECT_TEXT_PATH,
            new_callable=AsyncMock,
            side_effect=[invalid_response, valid_response],
        ):
            result = await run_completeness_assessment(
                goal="test goal",
                stop_reason="seemed done",
                notebook_path=stop_notebook,
            )

        assert "overall_coverage" in result

    @pytest.mark.asyncio
    async def test_retries_on_sdk_error(self, stop_notebook):
        """When the SDK raises an exception, assessment retries."""
        from auto_scientist.agents.stop_gate import run_completeness_assessment

        valid_json = json.dumps(_valid_assessment_dict())
        valid_response = (valid_json, {}, None)

        with patch(
            COLLECT_TEXT_PATH,
            new_callable=AsyncMock,
            side_effect=[RuntimeError("SDK timeout"), valid_response],
        ):
            result = await run_completeness_assessment(
                goal="test goal",
                stop_reason="seemed done",
                notebook_path=stop_notebook,
            )

        assert "overall_coverage" in result

    @pytest.mark.asyncio
    async def test_raises_after_max_sdk_errors(self, stop_notebook):
        """After MAX_ATTEMPTS SDK errors, the last exception propagates."""
        from auto_scientist.agents.stop_gate import run_completeness_assessment

        with (
            patch(
                COLLECT_TEXT_PATH,
                new_callable=AsyncMock,
                side_effect=RuntimeError("API down"),
            ),
            pytest.raises(RuntimeError, match="API down"),
        ):
            await run_completeness_assessment(
                goal="test goal",
                stop_reason="seemed done",
                notebook_path=stop_notebook,
            )

    @pytest.mark.asyncio
    async def test_raises_after_max_validation_errors(self, stop_notebook):
        """After MAX_ATTEMPTS validation failures, OutputValidationError propagates."""
        from auto_scientist.agents.stop_gate import run_completeness_assessment

        with (
            patch(
                COLLECT_TEXT_PATH,
                new_callable=AsyncMock,
                return_value=("not json", {}, None),
            ),
            pytest.raises(OutputValidationError),
        ):
            await run_completeness_assessment(
                goal="test goal",
                stop_reason="seemed done",
                notebook_path=stop_notebook,
            )

    @pytest.mark.asyncio
    async def test_missing_notebook_uses_empty_string(self, tmp_path):
        """When the notebook path does not exist, empty string is used."""
        from auto_scientist.agents.stop_gate import run_completeness_assessment

        valid_json = json.dumps(_valid_assessment_dict())

        with patch(
            COLLECT_TEXT_PATH,
            new_callable=AsyncMock,
            return_value=(valid_json, {}, None),
        ) as mock_collect:
            await run_completeness_assessment(
                goal="test goal",
                stop_reason="seemed done",
                notebook_path=tmp_path / "nonexistent.md",
            )

        # The prompt should have been called (notebook content is empty)
        mock_collect.assert_called_once()
        prompt = mock_collect.call_args[0][0]
        assert "(empty notebook)" in prompt

    @pytest.mark.asyncio
    async def test_goal_and_stop_reason_in_prompt(self, stop_notebook):
        """Goal and stop_reason are injected into the assessment prompt."""
        from auto_scientist.agents.stop_gate import run_completeness_assessment

        valid_json = json.dumps(_valid_assessment_dict())

        with patch(
            COLLECT_TEXT_PATH,
            new_callable=AsyncMock,
            return_value=(valid_json, {}, None),
        ) as mock_collect:
            await run_completeness_assessment(
                goal="discover alloy fatigue mechanism",
                stop_reason="All sub-questions addressed",
                notebook_path=stop_notebook,
            )

        prompt = mock_collect.call_args[0][0]
        assert "discover alloy fatigue mechanism" in prompt
        assert "All sub-questions addressed" in prompt

    @pytest.mark.asyncio
    async def test_domain_knowledge_in_prompt(self, stop_notebook):
        """domain_knowledge is injected into the assessment prompt."""
        from auto_scientist.agents.stop_gate import run_completeness_assessment

        valid_json = json.dumps(_valid_assessment_dict())

        with patch(
            COLLECT_TEXT_PATH,
            new_callable=AsyncMock,
            return_value=(valid_json, {}, None),
        ) as mock_collect:
            await run_completeness_assessment(
                goal="test",
                stop_reason="done",
                notebook_path=stop_notebook,
                domain_knowledge="SpO2 pulse oximetry physiology",
            )

        prompt = mock_collect.call_args[0][0]
        assert "SpO2 pulse oximetry physiology" in prompt


# ---------------------------------------------------------------------------
# _query_stop_agent
# ---------------------------------------------------------------------------

QUERY_CRITIC_PATH = "auto_scientist.agents.critic._query_critic"


class TestQueryStopAgent:
    """Tests for _query_stop_agent() - the provider-aware dispatch helper."""

    @pytest.mark.asyncio
    async def test_delegates_to_query_critic(self, openai_config):
        """Delegates to _query_critic and returns a validated model instance."""
        from auto_scientist.agents.stop_gate import _query_stop_agent

        valid_json = _valid_critic_json()
        mock_result = AgentResult(text=_pad(valid_json), input_tokens=10, output_tokens=5)

        with patch(QUERY_CRITIC_PATH, new_callable=AsyncMock, return_value=mock_result):
            parsed, result = await _query_stop_agent(
                config=openai_config,
                user_prompt="Evaluate this stop decision",
                system_prompt="You are a stop gate critic.",
                output_model=CriticOutput,
                label="Stop Critic (test)",
            )

        assert isinstance(parsed, CriticOutput)
        assert result is mock_result

    @pytest.mark.asyncio
    async def test_retry_on_output_validation_error(self, openai_config):
        """Retries when the first response fails validation."""
        from auto_scientist.agents.stop_gate import _query_stop_agent

        invalid_result = AgentResult(text=_pad("not json"), input_tokens=5, output_tokens=2)
        valid_result = AgentResult(
            text=_pad(_valid_critic_json()), input_tokens=10, output_tokens=5
        )

        with patch(
            QUERY_CRITIC_PATH,
            new_callable=AsyncMock,
            side_effect=[invalid_result, valid_result],
        ):
            parsed, result = await _query_stop_agent(
                config=openai_config,
                user_prompt="Evaluate stop",
                system_prompt="You are a critic.",
                output_model=CriticOutput,
                label="Stop Critic (retry test)",
            )

        assert isinstance(parsed, CriticOutput)

    @pytest.mark.asyncio
    async def test_retry_on_sdk_error(self, openai_config):
        """Retries when _query_critic raises an exception."""
        from auto_scientist.agents.stop_gate import _query_stop_agent

        valid_result = AgentResult(
            text=_pad(_valid_critic_json()), input_tokens=10, output_tokens=5
        )

        with patch(
            QUERY_CRITIC_PATH,
            new_callable=AsyncMock,
            side_effect=[RuntimeError("Rate limit"), valid_result],
        ):
            parsed, result = await _query_stop_agent(
                config=openai_config,
                user_prompt="Evaluate stop",
                system_prompt="You are a critic.",
                output_model=CriticOutput,
                label="Stop Critic (sdk error test)",
            )

        assert isinstance(parsed, CriticOutput)

    @pytest.mark.asyncio
    async def test_raises_after_max_retries_on_sdk_error(self, openai_config):
        """After MAX_ATTEMPTS SDK errors, the last exception propagates."""
        from auto_scientist.agents.stop_gate import _query_stop_agent

        with (
            patch(
                QUERY_CRITIC_PATH,
                new_callable=AsyncMock,
                side_effect=ConnectionError("API unreachable"),
            ),
            pytest.raises(ConnectionError, match="API unreachable"),
        ):
            await _query_stop_agent(
                config=openai_config,
                user_prompt="Evaluate stop",
                system_prompt="You are a critic.",
                output_model=CriticOutput,
                label="Stop Critic (exhausted)",
            )

    @pytest.mark.asyncio
    async def test_raises_after_max_retries_on_validation_error(self, openai_config):
        """After MAX_ATTEMPTS validation failures, OutputValidationError propagates."""
        from auto_scientist.agents.stop_gate import _query_stop_agent

        invalid_result = AgentResult(text=_pad("not json"), input_tokens=5, output_tokens=2)

        with (
            patch(
                QUERY_CRITIC_PATH,
                new_callable=AsyncMock,
                return_value=invalid_result,
            ),
            pytest.raises(OutputValidationError),
        ):
            await _query_stop_agent(
                config=openai_config,
                user_prompt="Evaluate stop",
                system_prompt="You are a critic.",
                output_model=CriticOutput,
                label="Stop Critic (validation exhausted)",
            )

    @pytest.mark.asyncio
    async def test_passes_system_prompt_to_query_critic(self, openai_config):
        """system_prompt is forwarded as a keyword argument to _query_critic."""
        from auto_scientist.agents.stop_gate import _query_stop_agent

        valid_result = AgentResult(
            text=_pad(_valid_critic_json()), input_tokens=10, output_tokens=5
        )

        with patch(QUERY_CRITIC_PATH, new_callable=AsyncMock, return_value=valid_result) as mock_qc:
            await _query_stop_agent(
                config=openai_config,
                user_prompt="Evaluate stop",
                system_prompt="You are a stop critic persona.",
                output_model=CriticOutput,
                label="test",
            )

        assert mock_qc.call_args.kwargs["system_prompt"] == "You are a stop critic persona."


# ---------------------------------------------------------------------------
# run_single_stop_debate
# ---------------------------------------------------------------------------


class TestRunSingleStopDebate:
    """Tests for run_single_stop_debate() - one critic persona's challenge."""

    @pytest.mark.asyncio
    async def test_returns_debate_result(self, openai_config, valid_assessment):
        """Happy path: returns a DebateResult with one round."""
        from auto_scientist.agents.stop_gate import run_single_stop_debate

        valid_result = AgentResult(
            text=_pad(_valid_critic_json()), input_tokens=10, output_tokens=5
        )

        with patch(QUERY_CRITIC_PATH, new_callable=AsyncMock, return_value=valid_result):
            result = await run_single_stop_debate(
                config=openai_config,
                stop_reason="All criteria met",
                completeness_assessment=valid_assessment,
                notebook_content="# Lab Notebook",
                goal="discover alloy fatigue mechanism",
            )

        assert isinstance(result, DebateResult)

    @pytest.mark.asyncio
    async def test_critic_output_is_parsed(self, openai_config, valid_assessment):
        """The critic's response is parsed into a CriticOutput instance."""
        from auto_scientist.agents.stop_gate import run_single_stop_debate

        concerns = [
            {
                "claim": "Sub-question on nonlinear effects unexplored",
                "severity": "high",
                "confidence": "high",
                "category": "falsification",
            }
        ]
        valid_result = AgentResult(
            text=_pad(_valid_critic_json(concerns=concerns, overall="Needs more work")),
            input_tokens=10,
            output_tokens=5,
        )

        with patch(QUERY_CRITIC_PATH, new_callable=AsyncMock, return_value=valid_result):
            result = await run_single_stop_debate(
                config=openai_config,
                stop_reason="done",
                completeness_assessment=valid_assessment,
                notebook_content="",
            )

        critic_out = result.critic_output
        assert isinstance(critic_out, CriticOutput)
        assert len(critic_out.concerns) == 1
        assert critic_out.concerns[0].claim == "Sub-question on nonlinear effects unexplored"
        assert critic_out.overall_assessment == "Needs more work"

    @pytest.mark.asyncio
    async def test_persona_name_in_result(self, openai_config, valid_assessment):
        """The persona name is recorded in the DebateResult."""
        from auto_scientist.agents.stop_gate import run_single_stop_debate

        valid_result = AgentResult(
            text=_pad(_valid_critic_json()), input_tokens=10, output_tokens=5
        )
        persona = {"name": "Completeness Auditor", "system_text": "<persona>Auditor</persona>"}

        with patch(QUERY_CRITIC_PATH, new_callable=AsyncMock, return_value=valid_result):
            result = await run_single_stop_debate(
                config=openai_config,
                stop_reason="done",
                completeness_assessment=valid_assessment,
                notebook_content="",
                persona=persona,
            )

        assert result.persona == "Completeness Auditor"

    @pytest.mark.asyncio
    async def test_critic_model_label_in_result(self, openai_config, valid_assessment):
        """The critic model label (provider:model) is stored in DebateResult."""
        from auto_scientist.agents.stop_gate import run_single_stop_debate

        valid_result = AgentResult(
            text=_pad(_valid_critic_json()), input_tokens=10, output_tokens=5
        )

        with patch(QUERY_CRITIC_PATH, new_callable=AsyncMock, return_value=valid_result):
            result = await run_single_stop_debate(
                config=openai_config,
                stop_reason="done",
                completeness_assessment=valid_assessment,
                notebook_content="",
            )

        assert result.critic_model == "openai:gpt-5.4"

    @pytest.mark.asyncio
    async def test_token_counts_from_critic_result(self, openai_config, valid_assessment):
        """Token counts are taken directly from the critic AgentResult."""
        from auto_scientist.agents.stop_gate import run_single_stop_debate

        valid_result = AgentResult(
            text=_pad(_valid_critic_json()),
            input_tokens=42,
            output_tokens=17,
            thinking_tokens=5,
        )

        with patch(QUERY_CRITIC_PATH, new_callable=AsyncMock, return_value=valid_result):
            result = await run_single_stop_debate(
                config=openai_config,
                stop_reason="done",
                completeness_assessment=valid_assessment,
                notebook_content="",
            )

        assert result.input_tokens == 42
        assert result.output_tokens == 17
        assert result.thinking_tokens == 5

    @pytest.mark.asyncio
    async def test_raw_transcript_has_critic_role(self, openai_config, valid_assessment):
        """Raw transcript entry has role='critic'."""
        from auto_scientist.agents.stop_gate import run_single_stop_debate

        valid_result = AgentResult(
            text=_pad(_valid_critic_json()), input_tokens=10, output_tokens=5
        )

        with patch(QUERY_CRITIC_PATH, new_callable=AsyncMock, return_value=valid_result):
            result = await run_single_stop_debate(
                config=openai_config,
                stop_reason="done",
                completeness_assessment=valid_assessment,
                notebook_content="",
            )

        assert len(result.raw_transcript) == 1
        assert result.raw_transcript[0]["role"] == "critic"

    @pytest.mark.asyncio
    async def test_completeness_assessment_in_prompt(self, openai_config, valid_assessment):
        """The completeness assessment JSON is included in the critic's user prompt."""
        from auto_scientist.agents.stop_gate import run_single_stop_debate

        valid_result = AgentResult(
            text=_pad(_valid_critic_json()), input_tokens=10, output_tokens=5
        )

        with patch(QUERY_CRITIC_PATH, new_callable=AsyncMock, return_value=valid_result) as mock_qc:
            await run_single_stop_debate(
                config=openai_config,
                stop_reason="done",
                completeness_assessment=valid_assessment,
                notebook_content="# Lab Notebook",
            )

        # The user_prompt is the second positional arg to _query_critic via _query_stop_agent
        # _query_stop_agent calls _query_critic(config, effective_prompt, ...)
        user_prompt = mock_qc.call_args[0][1]
        assert "outlet clarity" in user_prompt  # from valid_assessment sub_question

    @pytest.mark.asyncio
    async def test_default_persona_used_when_none_provided(self, openai_config, valid_assessment):
        """When no persona is given, defaults to Generic persona."""
        from auto_scientist.agents.stop_gate import run_single_stop_debate

        valid_result = AgentResult(
            text=_pad(_valid_critic_json()), input_tokens=10, output_tokens=5
        )

        with patch(QUERY_CRITIC_PATH, new_callable=AsyncMock, return_value=valid_result):
            result = await run_single_stop_debate(
                config=openai_config,
                stop_reason="done",
                completeness_assessment=valid_assessment,
                notebook_content="",
                persona=None,
            )

        assert result.persona == "Generic"


# ---------------------------------------------------------------------------
# run_scientist_stop_revision
# ---------------------------------------------------------------------------


class TestRunScientistStopRevision:
    """Tests for run_scientist_stop_revision() - scientist revises after stop debate."""

    @pytest.mark.asyncio
    async def test_returns_plan_dict_when_stop_withdrawn(self, stop_notebook, valid_assessment):
        """Returns a ScientistPlanOutput-compatible dict with should_stop=False."""
        from auto_scientist.agents.stop_gate import run_scientist_stop_revision

        plan = _valid_scientist_plan_dict(should_stop=False)
        raw_json = json.dumps(plan)

        with patch(COLLECT_TEXT_PATH, new_callable=AsyncMock, return_value=(raw_json, {}, None)):
            result = await run_scientist_stop_revision(
                stop_reason="seemed complete",
                completeness_assessment=valid_assessment,
                concern_ledger=[],
                analysis={"key_metrics": {"r2": 0.85}},
                notebook_path=stop_notebook,
                version="v03",
                goal="discover alloy fatigue",
            )

        assert result["should_stop"] is False
        assert result["hypothesis"] == "Nonlinear effects explain remaining variance"

    @pytest.mark.asyncio
    async def test_returns_plan_dict_when_stop_upheld(self, stop_notebook, valid_assessment):
        """Returns a dict with should_stop=True when the scientist maintains its decision."""
        from auto_scientist.agents.stop_gate import run_scientist_stop_revision

        plan = _valid_scientist_plan_dict(should_stop=True)
        raw_json = json.dumps(plan)

        with patch(COLLECT_TEXT_PATH, new_callable=AsyncMock, return_value=(raw_json, {}, None)):
            result = await run_scientist_stop_revision(
                stop_reason="investigation complete",
                completeness_assessment={
                    "sub_questions": [
                        {
                            "question": "Main question",
                            "coverage": "thorough",
                            "evidence": ["R²=0.97"],
                            "gaps": [],
                        }
                    ],
                    "overall_coverage": "thorough",
                    "recommendation": "stop",
                },
                concern_ledger=[],
                analysis={},
                notebook_path=stop_notebook,
                version="v05",
            )

        assert result["should_stop"] is True
        assert result["stop_reason"] == "Investigation complete"

    @pytest.mark.asyncio
    async def test_retries_on_validation_error(self, stop_notebook, valid_assessment):
        """When the first response fails schema validation, a second attempt is made."""
        from auto_scientist.agents.stop_gate import run_scientist_stop_revision

        valid_json = json.dumps(_valid_scientist_plan_dict())
        invalid_response = ("malformed {{", {}, None)
        valid_response = (valid_json, {}, None)

        with patch(
            COLLECT_TEXT_PATH,
            new_callable=AsyncMock,
            side_effect=[invalid_response, valid_response],
        ):
            result = await run_scientist_stop_revision(
                stop_reason="seemed done",
                completeness_assessment=valid_assessment,
                concern_ledger=[],
                analysis={},
                notebook_path=stop_notebook,
                version="v01",
            )

        assert "hypothesis" in result

    @pytest.mark.asyncio
    async def test_retries_on_sdk_error(self, stop_notebook, valid_assessment):
        """When the SDK raises an exception, revision retries."""
        from auto_scientist.agents.stop_gate import run_scientist_stop_revision

        valid_json = json.dumps(_valid_scientist_plan_dict())
        valid_response = (valid_json, {}, None)

        with patch(
            COLLECT_TEXT_PATH,
            new_callable=AsyncMock,
            side_effect=[OSError("network blip"), valid_response],
        ):
            result = await run_scientist_stop_revision(
                stop_reason="seemed done",
                completeness_assessment=valid_assessment,
                concern_ledger=[],
                analysis={},
                notebook_path=stop_notebook,
                version="v01",
            )

        assert "hypothesis" in result

    @pytest.mark.asyncio
    async def test_raises_after_max_sdk_errors(self, stop_notebook, valid_assessment):
        """After MAX_ATTEMPTS SDK errors, the last exception propagates."""
        from auto_scientist.agents.stop_gate import run_scientist_stop_revision

        with (
            patch(
                COLLECT_TEXT_PATH,
                new_callable=AsyncMock,
                side_effect=ConnectionError("API down"),
            ),
            pytest.raises(ConnectionError, match="API down"),
        ):
            await run_scientist_stop_revision(
                stop_reason="seemed done",
                completeness_assessment=valid_assessment,
                concern_ledger=[],
                analysis={},
                notebook_path=stop_notebook,
                version="v01",
            )

    @pytest.mark.asyncio
    async def test_raises_after_max_validation_errors(self, stop_notebook, valid_assessment):
        """After MAX_ATTEMPTS validation failures, OutputValidationError propagates."""
        from auto_scientist.agents.stop_gate import run_scientist_stop_revision

        with (
            patch(
                COLLECT_TEXT_PATH,
                new_callable=AsyncMock,
                return_value=("not json", {}, None),
            ),
            pytest.raises(OutputValidationError),
        ):
            await run_scientist_stop_revision(
                stop_reason="seemed done",
                completeness_assessment=valid_assessment,
                concern_ledger=[],
                analysis={},
                notebook_path=stop_notebook,
                version="v01",
            )

    @pytest.mark.asyncio
    async def test_goal_in_prompt(self, stop_notebook, valid_assessment):
        """Goal is injected into the stop revision prompt."""
        from auto_scientist.agents.stop_gate import run_scientist_stop_revision

        valid_json = json.dumps(_valid_scientist_plan_dict())

        with patch(
            COLLECT_TEXT_PATH,
            new_callable=AsyncMock,
            return_value=(valid_json, {}, None),
        ) as mock_collect:
            await run_scientist_stop_revision(
                stop_reason="done",
                completeness_assessment=valid_assessment,
                concern_ledger=[],
                analysis={},
                notebook_path=stop_notebook,
                version="v02",
                goal="understand alloy failure modes",
            )

        prompt = mock_collect.call_args[0][0]
        assert "understand alloy failure modes" in prompt

    @pytest.mark.asyncio
    async def test_concern_ledger_in_prompt(self, stop_notebook, valid_assessment):
        """When a concern ledger is provided, it appears in the revision prompt."""
        from auto_scientist.agents.stop_gate import run_scientist_stop_revision

        valid_json = json.dumps(_valid_scientist_plan_dict())
        concern_ledger = [
            {
                "claim": "Nonlinear effects not tested",
                "severity": "high",
                "confidence": "high",
                "category": "falsification",
                "persona": "Completeness Auditor",
                "critic_model": "openai:gpt-5.4",
            }
        ]

        with patch(
            COLLECT_TEXT_PATH,
            new_callable=AsyncMock,
            return_value=(valid_json, {}, None),
        ) as mock_collect:
            await run_scientist_stop_revision(
                stop_reason="done",
                completeness_assessment=valid_assessment,
                concern_ledger=concern_ledger,
                analysis={},
                notebook_path=stop_notebook,
                version="v02",
            )

        prompt = mock_collect.call_args[0][0]
        assert "Nonlinear effects not tested" in prompt

    @pytest.mark.asyncio
    async def test_missing_notebook_uses_empty_string(self, tmp_path, valid_assessment):
        """When the notebook does not exist, empty notebook placeholder is used."""
        from auto_scientist.agents.stop_gate import run_scientist_stop_revision

        valid_json = json.dumps(_valid_scientist_plan_dict())

        with patch(
            COLLECT_TEXT_PATH,
            new_callable=AsyncMock,
            return_value=(valid_json, {}, None),
        ) as mock_collect:
            await run_scientist_stop_revision(
                stop_reason="done",
                completeness_assessment=valid_assessment,
                concern_ledger=[],
                analysis={},
                notebook_path=tmp_path / "nonexistent.md",
                version="v01",
            )

        prompt = mock_collect.call_args[0][0]
        assert "(empty notebook)" in prompt


class TestStopGatePromptBuilders:
    def test_assessment_system_includes_schema_example_and_tool_guidance(self):
        from auto_scientist.prompts.stop_gate import build_assessment_system

        system = build_assessment_system("gpt")

        assert "Tool calls are allowed before the final JSON response." in system
        assert '"sub_questions"' in system
        assert '"recommendation": "continue"' in system

    def test_stop_revision_example_in_system_not_user(self):
        from auto_scientist.prompts.stop_gate import (
            STOP_REVISION_USER,
            build_stop_revision_system,
        )

        user = STOP_REVISION_USER.format(
            goal="goal",
            domain_knowledge="dk",
            notebook_content="nb",
            analysis_json="{}",
            prediction_history="ph",
            stop_reason="done",
            completeness_assessment="{}",
            concern_ledger="[]",
            version="v02",
            plan_schema="{}",
        )
        system = build_stop_revision_system("claude")

        # Example moved from user to system prompt
        assert "Example" not in user
        assert "Example (withdrawal):" in system
        assert "Untested nonlinear response forms" in system
