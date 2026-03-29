"""Tests for the critic debate loop with personas and structured output."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from auto_scientist.agent_result import AgentResult
from auto_scientist.agents.critic import (
    _build_critic_prompt,
    _build_scientist_debate_user_prompt,
    run_debate,
    run_single_critic_debate,
)
from auto_scientist.agents.debate_models import CriticOutput, DebateResult, ScientistDefense
from auto_scientist.model_config import AgentModelConfig, ReasoningConfig

# Mock paths for direct API calls (OpenAI/Google) and SDK (Anthropic)
OPENAI_PATH = "auto_scientist.agents.critic.query_openai"
GOOGLE_PATH = "auto_scientist.agents.critic.query_google"
SDK_PATH = "auto_scientist.agents.critic.collect_text_from_query"


def _valid_critic_json(
    concerns: list | None = None,
    overall: str = "Plan has issues.",
) -> str:
    """Build a valid CriticOutput JSON string."""
    obj = {
        "concerns": concerns
        or [
            {
                "claim": "Data quality issue",
                "severity": "high",
                "confidence": "medium",
                "category": "methodology",
            }
        ],
        "alternative_hypotheses": ["Try log-transform"],
        "overall_assessment": overall,
    }
    return json.dumps(obj)


def _valid_defense_json(
    responses: list | None = None,
) -> str:
    """Build a valid ScientistDefense JSON string."""
    obj = {
        "responses": responses
        or [
            {
                "concern": "Data quality issue",
                "verdict": "accepted",
                "reasoning": "Valid point, will fix.",
            }
        ],
        "additional_points": "",
    }
    return json.dumps(obj)


def _pad(text: str) -> str:
    """Pad a response to minimum length for tests."""
    min_len = 50
    if len(text) >= min_len:
        return text
    return text + " " + "x" * (min_len - len(text) - 1)


def _critic_result(text: str) -> AgentResult:
    """Create an AgentResult from text for critic mock returns."""
    return AgentResult(text=_pad(text), input_tokens=10, output_tokens=5)


def _structured_critic_result(
    concerns: list | None = None,
    overall: str = "Plan has issues.",
) -> AgentResult:
    """Create an AgentResult containing valid CriticOutput JSON."""
    return AgentResult(
        text=_pad(_valid_critic_json(concerns, overall)),
        input_tokens=10,
        output_tokens=5,
    )


def _structured_defense_result(
    responses: list | None = None,
) -> AgentResult:
    """Create an AgentResult containing valid ScientistDefense JSON."""
    return AgentResult(
        text=_pad(_valid_defense_json(responses)),
        input_tokens=10,
        output_tokens=5,
    )


# Short aliases for mock returns used in many tests
_CR = _structured_critic_result  # critic result
_DR = _structured_defense_result  # defense result


def _sdk_mock(
    text: str = "",
    input_tokens: int = 10,
    output_tokens: int = 5,
) -> AsyncMock:
    """Create an AsyncMock for collect_text_from_query (SDK path).

    Returns text directly (not AgentResult) and sets last_usage attribute.
    """
    mock = AsyncMock(return_value=text)
    mock.last_usage = {"input_tokens": input_tokens, "output_tokens": output_tokens}
    return mock


def _sdk_critic_mock(**kwargs) -> AsyncMock:
    """SDK mock returning valid CriticOutput JSON."""
    return _sdk_mock(text=_pad(_valid_critic_json(**kwargs)))


def _sdk_defense_mock(**kwargs) -> AsyncMock:
    """SDK mock returning valid ScientistDefense JSON."""
    return _sdk_mock(text=_pad(_valid_defense_json(**kwargs)))


@pytest.fixture(autouse=True)
def _no_stagger_delay():
    """Disable stagger delays in tests to keep them fast."""
    with patch("auto_scientist.agents.critic._STAGGER_DELAY_SECONDS", 0.0):
        yield


@pytest.fixture
def plan():
    return {
        "hypothesis": "Adjusting learning rate will improve convergence",
        "strategy": "incremental",
        "changes": [
            {
                "what": "Reduce learning rate",
                "why": "Current rate causes oscillation",
                "how": "Set lr=0.001",
                "priority": 1,
            }
        ],
        "expected_impact": "Smoother convergence, better final score",
        "should_stop": False,
        "stop_reason": None,
        "notebook_entry": "Learning rate adjustment\n\nReducing lr for convergence",
    }


@pytest.fixture
def openai_critic():
    return AgentModelConfig(provider="openai", model="gpt-4o")


@pytest.fixture
def google_critic():
    return AgentModelConfig(provider="google", model="gemini-2.5-pro")


@pytest.fixture
def two_critics(openai_critic, google_critic):
    return [openai_critic, google_critic]


@pytest.fixture
def base_kwargs(plan, two_critics):
    return {
        "critic_configs": two_critics,
        "plan": plan,
        "notebook_content": "# Lab Notebook\nEntry 1",
    }


class TestBuildCriticPrompt:
    def test_returns_tuple(self):
        result = _build_critic_prompt({"h": "p"}, "", "")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_includes_plan_json_in_user(self):
        plan = {"hypothesis": "test plan"}
        _system, user = _build_critic_prompt(plan, "", "")
        assert "test plan" in user
        assert "<plan>" in user

    def test_empty_defense_no_tag(self):
        _system, user = _build_critic_prompt({"h": "p"}, "", "", scientist_defense="")
        assert "<scientist_defense>" not in user

    def test_with_defense_includes_tag(self):
        _system, user = _build_critic_prompt(
            {"h": "p"},
            "",
            "",
            scientist_defense="I disagree because...",
        )
        assert "<scientist_defense>" in user
        assert "I disagree because..." in user

    def test_fallback_for_empty_values(self):
        _system, user = _build_critic_prompt({"h": "p"}, "", "")
        assert "(empty)" in user or "(none provided)" in user

    def test_persona_injected_into_system(self):
        persona = "<persona>\nYou are the Methodologist.\n</persona>"
        system, _user = _build_critic_prompt({"h": "p"}, "", "", persona_text=persona)
        assert "<persona>" in system
        assert "Methodologist" in system

    def test_output_format_in_system(self):
        system, _user = _build_critic_prompt({"h": "p"}, "", "")
        assert "<output_format>" in system
        assert "concerns" in system
        assert "severity" in system

    def test_no_persona_leaves_empty(self):
        system, _user = _build_critic_prompt({"h": "p"}, "", "")
        assert "<persona>" not in system


class TestRunDebate:
    @pytest.mark.asyncio
    async def test_empty_configs_returns_empty(self, plan):
        result = await run_debate(
            critic_configs=[],
            plan=plan,
            notebook_content="",
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_always_4_debates_with_2_critics(self, base_kwargs):
        """On iteration 1+, runs 4 debates (one per persona), regardless of critic count."""
        with (
            patch(OPENAI_PATH, new_callable=AsyncMock, return_value=_CR()),
            patch(GOOGLE_PATH, new_callable=AsyncMock, return_value=_CR()),
        ):
            results = await run_debate(**base_kwargs, iteration=1, max_rounds=1)

        assert len(results) == 4
        persona_names = [r.persona for r in results]
        assert "Methodologist" in persona_names
        assert "Trajectory Critic" in persona_names
        assert "Falsification Expert" in persona_names
        assert "Evidence Auditor" in persona_names

    @pytest.mark.asyncio
    async def test_returns_debate_result_objects(self, base_kwargs):
        """Results are DebateResult instances with structured data."""
        with (
            patch(OPENAI_PATH, new_callable=AsyncMock, return_value=_CR()),
            patch(GOOGLE_PATH, new_callable=AsyncMock, return_value=_CR()),
        ):
            results = await run_debate(**base_kwargs, max_rounds=1)

        for r in results:
            assert isinstance(r, DebateResult)
            assert len(r.rounds) >= 1
            assert isinstance(r.rounds[0].critic_output, CriticOutput)
            assert len(r.raw_transcript) >= 1

    @pytest.mark.asyncio
    async def test_iteration_0_runs_only_subset_personas(self, base_kwargs):
        """Iteration 0 runs only Methodologist and Falsification Expert."""
        with (
            patch(OPENAI_PATH, new_callable=AsyncMock, return_value=_CR()),
            patch(GOOGLE_PATH, new_callable=AsyncMock, return_value=_CR()),
        ):
            results = await run_debate(**base_kwargs, iteration=0, max_rounds=1)

        assert len(results) == 2
        persona_names = {r.persona for r in results}
        assert persona_names == {"Methodologist", "Falsification Expert"}

    @pytest.mark.asyncio
    async def test_iteration_1_runs_all_personas(self, base_kwargs):
        """Iteration 1+ runs all four personas."""
        with (
            patch(OPENAI_PATH, new_callable=AsyncMock, return_value=_CR()),
            patch(GOOGLE_PATH, new_callable=AsyncMock, return_value=_CR()),
        ):
            results = await run_debate(**base_kwargs, iteration=1, max_rounds=1)

        assert len(results) == 4
        persona_names = {r.persona for r in results}
        assert persona_names == {
            "Methodologist",
            "Trajectory Critic",
            "Falsification Expert",
            "Evidence Auditor",
        }

    @pytest.mark.asyncio
    async def test_persona_rotation_across_iterations(self, base_kwargs):
        """Different iterations assign different models to the same persona."""
        with (
            patch(OPENAI_PATH, new_callable=AsyncMock, return_value=_CR()),
            patch(GOOGLE_PATH, new_callable=AsyncMock, return_value=_CR()),
        ):
            results_iter0 = await run_debate(**base_kwargs, iteration=0)
            results_iter1 = await run_debate(**base_kwargs, iteration=1)

        # Methodologist should use different models across iterations
        meth_0 = next(r for r in results_iter0 if r.persona == "Methodologist")
        meth_1 = next(r for r in results_iter1 if r.persona == "Methodologist")
        assert meth_0.critic_model != meth_1.critic_model

    @pytest.mark.asyncio
    async def test_single_round_no_scientist_defense(self, base_kwargs):
        """With max_rounds=1, no scientist defense is produced."""
        mock_sdk = _sdk_defense_mock()
        with (
            patch(OPENAI_PATH, new_callable=AsyncMock, return_value=_CR()),
            patch(GOOGLE_PATH, new_callable=AsyncMock, return_value=_CR()),
            patch(SDK_PATH, mock_sdk),
        ):
            results = await run_debate(**base_kwargs, max_rounds=1)

        for r in results:
            assert r.rounds[0].scientist_defense is None
        # SDK (scientist) should not be called in single-round mode
        mock_sdk.assert_not_called()

    @pytest.mark.asyncio
    async def test_two_rounds_has_scientist_defense(self, base_kwargs):
        """With max_rounds=2, scientist defense is included."""
        with (
            patch(OPENAI_PATH, new_callable=AsyncMock, return_value=_CR()),
            patch(GOOGLE_PATH, new_callable=AsyncMock, return_value=_CR()),
            patch(SDK_PATH, _sdk_defense_mock()),
        ):
            results = await run_debate(**base_kwargs, max_rounds=2)

        for r in results:
            # First round has both critic and defense
            assert r.rounds[0].scientist_defense is not None
            assert isinstance(r.rounds[0].scientist_defense, ScientistDefense)
            # Last round is critic only
            assert r.rounds[-1].scientist_defense is None

    @pytest.mark.asyncio
    async def test_raw_transcript_preserved(self, base_kwargs):
        """Raw transcript is kept alongside structured data."""
        with (
            patch(OPENAI_PATH, new_callable=AsyncMock, return_value=_CR()),
            patch(GOOGLE_PATH, new_callable=AsyncMock, return_value=_CR()),
        ):
            results = await run_debate(**base_kwargs, max_rounds=1)

        for r in results:
            assert len(r.raw_transcript) == 1
            assert r.raw_transcript[0]["role"] == "critic"

    @pytest.mark.asyncio
    async def test_plan_in_critic_prompt(self, base_kwargs):
        """Critic prompt includes the scientist's plan."""
        with (
            patch(OPENAI_PATH, new_callable=AsyncMock, return_value=_CR()) as mock_openai,
            patch(GOOGLE_PATH, new_callable=AsyncMock, return_value=_CR()),
        ):
            await run_debate(**base_kwargs, max_rounds=1)

        critic_prompt = mock_openai.call_args[0][1]
        assert "<plan>" in critic_prompt
        assert "Adjusting learning rate" in critic_prompt

    @pytest.mark.asyncio
    async def test_web_search_enabled_for_critic(self, base_kwargs):
        """Critic calls pass web_search=True."""
        with (
            patch(OPENAI_PATH, new_callable=AsyncMock, return_value=_CR()) as mock_openai,
            patch(GOOGLE_PATH, new_callable=AsyncMock, return_value=_CR()),
        ):
            await run_debate(**base_kwargs, max_rounds=1)

        for call in mock_openai.call_args_list:
            assert call.kwargs.get("web_search") is True

    @pytest.mark.asyncio
    async def test_web_search_enabled_for_scientist(self, base_kwargs):
        """Scientist SDK calls include WebSearch in allowed_tools."""
        mock_sdk = _sdk_defense_mock()
        with (
            patch(OPENAI_PATH, new_callable=AsyncMock, return_value=_CR()),
            patch(GOOGLE_PATH, new_callable=AsyncMock, return_value=_CR()),
            patch(SDK_PATH, mock_sdk),
        ):
            await run_debate(**base_kwargs, max_rounds=2)

        # Verify ClaudeCodeOptions passed to collect_text_from_query has WebSearch
        for call in mock_sdk.call_args_list:
            options = call[0][1]  # second positional arg is ClaudeCodeOptions
            assert "WebSearch" in options.allowed_tools

    @pytest.mark.asyncio
    async def test_unknown_provider_raises_when_all_fail(self, plan):
        """Unknown provider causes all debates to fail, raising RuntimeError."""
        bad_config = AgentModelConfig.model_validate({"provider": "openai", "model": "model"})
        object.__setattr__(bad_config, "provider", "unknown")
        with pytest.raises(RuntimeError, match="All .* critic debates failed"):
            await run_debate(
                critic_configs=[bad_config],
                plan=plan,
                notebook_content="",
            )

    @pytest.mark.asyncio
    async def test_anthropic_critic_dispatches_correctly(self, plan):
        critic = AgentModelConfig(provider="anthropic", model="claude-sonnet-4-6")
        with patch(SDK_PATH, _sdk_critic_mock(overall="Anthropic critique")):
            result = await run_debate(
                critic_configs=[critic],
                plan=plan,
                notebook_content="",
                max_rounds=1,
                iteration=1,
            )

        assert len(result) == 4
        # All 4 debates use the same model (only 1 critic configured)
        for r in result:
            assert r.critic_model == "anthropic:claude-sonnet-4-6"

    @pytest.mark.asyncio
    async def test_reasoning_passed_to_critic(self, plan):
        """Critic reasoning config is forwarded to model client."""
        critic = AgentModelConfig(
            provider="openai",
            model="o4-mini",
            reasoning=ReasoningConfig(level="high"),
        )
        with patch(
            OPENAI_PATH,
            new_callable=AsyncMock,
            return_value=_CR(),
        ) as mock_openai:
            await run_debate(
                critic_configs=[critic],
                plan=plan,
                notebook_content="",
                max_rounds=1,
            )

        assert mock_openai.call_args.kwargs["reasoning"].level == "high"

    @pytest.mark.asyncio
    async def test_token_counts_accumulated(self, base_kwargs):
        """Token counts are accumulated across rounds."""
        with (
            patch(OPENAI_PATH, new_callable=AsyncMock, return_value=_CR()),
            patch(GOOGLE_PATH, new_callable=AsyncMock, return_value=_CR()),
            patch(SDK_PATH, _sdk_defense_mock()),
        ):
            results = await run_debate(**base_kwargs, max_rounds=2)

        for r in results:
            # 2 critic calls (10 each) + 1 scientist call (10) = 30 input tokens
            assert r.input_tokens == 30
            # 2 critic calls (5 each) + 1 scientist call (5) = 15 output tokens
            assert r.output_tokens == 15


class TestCriticRetry:
    @pytest.mark.asyncio
    async def test_retry_on_empty_critic_response(self, plan, two_critics):
        """Empty critic response triggers retry."""
        valid = _structured_critic_result()
        with (
            patch(
                OPENAI_PATH,
                new_callable=AsyncMock,
                # Trajectory Critic: empty -> retry -> valid. Evidence Auditor: valid.
                side_effect=[AgentResult(text=""), valid, valid],
            ) as mock_openai,
            patch(GOOGLE_PATH, new_callable=AsyncMock, return_value=valid),
        ):
            result = await run_debate(
                critic_configs=two_critics,
                plan=plan,
                notebook_content="",
                max_rounds=1,
                iteration=1,
            )

        assert len(result) == 4
        # OpenAI called: 2 for Trajectory Critic (retry) + 1 for Evidence Auditor
        assert mock_openai.call_count == 3

    @pytest.mark.asyncio
    async def test_sdk_error_captured_gracefully(self, plan, two_critics):
        """Errors in individual debates don't crash the whole run."""
        with (
            patch(OPENAI_PATH, new_callable=AsyncMock, side_effect=RuntimeError("API error")),
            patch(GOOGLE_PATH, new_callable=AsyncMock, return_value=_CR()),
        ):
            result = await run_debate(
                critic_configs=two_critics,
                plan=plan,
                notebook_content="",
                max_rounds=1,
            )
        # Some debates may fail, but we get results from successful ones
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_all_debates_fail_raises_runtime_error(self, plan, two_critics):
        """When all debates fail, run_debate raises RuntimeError."""
        with (
            patch(OPENAI_PATH, new_callable=AsyncMock, side_effect=RuntimeError("API error")),
            patch(GOOGLE_PATH, new_callable=AsyncMock, side_effect=RuntimeError("API error")),
            pytest.raises(RuntimeError, match="All .* critic debates failed"),
        ):
            await run_debate(
                critic_configs=two_critics,
                plan=plan,
                notebook_content="",
                max_rounds=1,
            )


class TestCriticValidation:
    @pytest.mark.asyncio
    async def test_valid_json_parsed_to_critic_output(self, plan):
        """Valid structured JSON is parsed into CriticOutput."""
        critic = AgentModelConfig(provider="openai", model="gpt-4o")
        with patch(
            OPENAI_PATH,
            new_callable=AsyncMock,
            return_value=_CR(),
        ):
            result = await run_single_critic_debate(
                config=critic,
                plan=plan,
                notebook_content="",
                max_rounds=1,
            )

        assert isinstance(result.rounds[0].critic_output, CriticOutput)
        assert len(result.rounds[0].critic_output.concerns) == 1
        assert result.rounds[0].critic_output.concerns[0].claim == "Data quality issue"

    @pytest.mark.asyncio
    async def test_invalid_json_preserves_raw_text_as_concern(self, plan):
        """Invalid JSON falls back with raw text preserved as a PARSE ERROR concern."""
        critic = AgentModelConfig(provider="openai", model="gpt-4o")
        with patch(
            OPENAI_PATH,
            new_callable=AsyncMock,
            return_value=_critic_result("Not valid JSON at all, just prose critique."),
        ):
            result = await run_single_critic_debate(
                config=critic,
                plan=plan,
                notebook_content="",
                max_rounds=1,
            )

        # Should not crash; raw text preserved as a concern
        co = result.rounds[0].critic_output
        assert isinstance(co, CriticOutput)
        assert len(co.concerns) == 1
        assert "[SYNTHETIC - PARSE ERROR]" in co.concerns[0].claim
        assert co.concerns[0].severity == "low"
        assert co.concerns[0].category == "other"


class TestScientistDefenseValidation:
    @pytest.mark.asyncio
    async def test_valid_defense_parsed(self, plan):
        """Valid defense JSON is parsed into ScientistDefense."""
        critic = AgentModelConfig(provider="openai", model="gpt-4o")
        with (
            patch(OPENAI_PATH, new_callable=AsyncMock, return_value=_CR()),
            patch(SDK_PATH, _sdk_defense_mock()),
        ):
            result = await run_single_critic_debate(
                config=critic,
                plan=plan,
                notebook_content="",
                max_rounds=2,
            )

        defense = result.rounds[0].scientist_defense
        assert isinstance(defense, ScientistDefense)
        assert len(defense.responses) == 1
        assert defense.responses[0].verdict == "accepted"

    @pytest.mark.asyncio
    async def test_invalid_defense_falls_back(self, plan):
        """Invalid defense JSON falls back to empty responses."""
        critic = AgentModelConfig(provider="openai", model="gpt-4o")
        with (
            patch(OPENAI_PATH, new_callable=AsyncMock, return_value=_CR()),
            patch(
                SDK_PATH,
                _sdk_mock(text=_pad("Not JSON, just a prose defense.")),
            ),
        ):
            result = await run_single_critic_debate(
                config=critic,
                plan=plan,
                notebook_content="",
                max_rounds=2,
            )

        defense = result.rounds[0].scientist_defense
        assert isinstance(defense, ScientistDefense)
        assert defense.responses == []


class TestBuildCriticPromptContext:
    def test_analysis_json_included(self):
        _system, user = _build_critic_prompt(
            {"h": "p"},
            "",
            "",
            analysis_json='{"key_metrics": {"rmse": 0.52}}',
        )
        assert "<analysis>" in user
        assert "rmse" in user

    def test_prediction_history_included(self):
        _system, user = _build_critic_prompt(
            {"h": "p"},
            "",
            "",
            prediction_history="[1.0] CONFIRMED: polynomial fits well",
        )
        assert "<prediction_history>" in user
        assert "polynomial fits well" in user

    def test_empty_analysis_shows_fallback(self):
        _system, user = _build_critic_prompt({"h": "p"}, "", "")
        assert "(no analysis yet)" in user

    def test_empty_prediction_history_shows_fallback(self):
        _system, user = _build_critic_prompt({"h": "p"}, "", "")
        assert "(no prediction history yet)" in user


class TestBuildScientistDebatePromptContext:
    def test_analysis_json_included(self):
        prompt = _build_scientist_debate_user_prompt(
            {"h": "p"},
            "",
            "",
            critique="test",
            analysis_json='{"key_metrics": {"r2": 0.97}}',
        )
        assert "<analysis>" in prompt
        assert "r2" in prompt

    def test_prediction_history_included(self):
        prompt = _build_scientist_debate_user_prompt(
            {"h": "p"},
            "",
            "",
            critique="test",
            prediction_history="[2.0] REFUTED: linear model insufficient",
        )
        assert "<prediction_history>" in prompt
        assert "linear model insufficient" in prompt


class TestBuildScientistDebatePromptStructured:
    def test_critic_persona_in_prompt(self):
        prompt = _build_scientist_debate_user_prompt(
            {"h": "p"},
            "",
            "",
            critique="test",
            critic_persona="Methodologist",
        )
        assert "<critic_persona>" in prompt
        assert "Methodologist" in prompt

    def test_default_critic_persona(self):
        prompt = _build_scientist_debate_user_prompt(
            {"h": "p"},
            "",
            "",
            critique="test",
        )
        assert "<critic_persona>" in prompt
        assert "(generic critic)" in prompt


class TestRunDebateWithContext:
    @pytest.mark.asyncio
    async def test_analysis_json_in_critic_prompt(self, plan, two_critics):
        """When analysis_json is provided, critic prompt includes <analysis> section."""
        with (
            patch(OPENAI_PATH, new_callable=AsyncMock, return_value=_CR()) as mock_openai,
            patch(GOOGLE_PATH, new_callable=AsyncMock, return_value=_CR()),
        ):
            await run_debate(
                critic_configs=two_critics,
                plan=plan,
                notebook_content="",
                max_rounds=1,
                analysis_json='{"key_metrics": {"rmse": 0.52}}',
            )

        critic_prompt = mock_openai.call_args[0][1]
        assert "<analysis>" in critic_prompt
        assert "rmse" in critic_prompt

    @pytest.mark.asyncio
    async def test_prediction_history_in_critic_prompt(self, plan, two_critics):
        """When prediction_history is provided, critic prompt includes it."""
        with (
            patch(OPENAI_PATH, new_callable=AsyncMock, return_value=_CR()) as mock_openai,
            patch(GOOGLE_PATH, new_callable=AsyncMock, return_value=_CR()),
        ):
            await run_debate(
                critic_configs=two_critics,
                plan=plan,
                notebook_content="",
                max_rounds=1,
                prediction_history="[1.0] CONFIRMED: polynomial fits well",
            )

        critic_prompt = mock_openai.call_args[0][1]
        assert "<prediction_history>" in critic_prompt
        assert "polynomial fits well" in critic_prompt

    @pytest.mark.asyncio
    async def test_analysis_in_scientist_debate_prompt(self, plan, two_critics):
        """Scientist debate defense prompt includes analysis and prediction history."""
        mock_sdk = _sdk_defense_mock()
        with (
            patch(OPENAI_PATH, new_callable=AsyncMock, return_value=_CR()),
            patch(GOOGLE_PATH, new_callable=AsyncMock, return_value=_CR()),
            patch(SDK_PATH, mock_sdk),
        ):
            await run_debate(
                critic_configs=two_critics,
                plan=plan,
                notebook_content="",
                max_rounds=2,
                analysis_json='{"key_metrics": {"r2": 0.97}}',
                prediction_history="[2.0] REFUTED: linear model",
            )

        # First positional arg to collect_text_from_query is the prompt
        scientist_prompt = mock_sdk.call_args_list[0][0][0]
        assert "<analysis>" in scientist_prompt
        assert "r2" in scientist_prompt
        assert "<prediction_history>" in scientist_prompt
        assert "linear model" in scientist_prompt


class TestPersonas:
    def test_personas_has_four_entries(self):
        from auto_scientist.prompts.critic import PERSONAS

        assert len(PERSONAS) == 4

    def test_each_persona_has_name_and_system_text(self):
        from auto_scientist.prompts.critic import PERSONAS

        for persona in PERSONAS:
            assert "name" in persona
            assert "system_text" in persona
            assert isinstance(persona["name"], str)
            assert "<persona>" in persona["system_text"]

    def test_persona_names(self):
        from auto_scientist.prompts.critic import PERSONAS

        names = [p["name"] for p in PERSONAS]
        assert "Methodologist" in names
        assert "Trajectory Critic" in names
        assert "Falsification Expert" in names
        assert "Evidence Auditor" in names

    def test_iteration_0_personas_is_subset_of_persona_names(self):
        from auto_scientist.prompts.critic import ITERATION_0_PERSONAS, PERSONAS

        all_names = {p["name"] for p in PERSONAS}
        assert all_names >= ITERATION_0_PERSONAS

    def test_trajectory_critic_has_instructions(self):
        from auto_scientist.prompts.critic import PERSONAS

        tc = next(p for p in PERSONAS if p["name"] == "Trajectory Critic")
        assert "instructions" in tc
        assert "<instructions>" in tc["instructions"]

    def test_default_instructions_constant_exists(self):
        from auto_scientist.prompts.critic import DEFAULT_CRITIC_INSTRUCTIONS

        assert "<instructions>" in DEFAULT_CRITIC_INSTRUCTIONS

    def test_model_rotation_two_models(self):
        from auto_scientist.prompts.critic import get_model_index_for_debate

        # 2 models, 3 personas, iteration 0
        assert get_model_index_for_debate(0, 0, 2) == 0
        assert get_model_index_for_debate(1, 0, 2) == 1
        assert get_model_index_for_debate(2, 0, 2) == 0

        # iteration 1: shift by 1
        assert get_model_index_for_debate(0, 1, 2) == 1
        assert get_model_index_for_debate(1, 1, 2) == 0
        assert get_model_index_for_debate(2, 1, 2) == 1

    def test_model_rotation_one_model(self):
        from auto_scientist.prompts.critic import get_model_index_for_debate

        assert get_model_index_for_debate(0, 0, 1) == 0
        assert get_model_index_for_debate(1, 5, 1) == 0

    def test_model_rotation_three_models(self):
        from auto_scientist.prompts.critic import get_model_index_for_debate

        assert get_model_index_for_debate(0, 0, 3) == 0
        assert get_model_index_for_debate(1, 0, 3) == 1
        assert get_model_index_for_debate(2, 0, 3) == 2


class TestResponseSchemaPassthrough:
    """Verify response_schema is passed through _query_critic to providers."""

    @pytest.mark.asyncio
    async def test_critic_passes_response_schema_to_provider(self, plan):
        """Critic structured query passes response_schema=CriticOutput to provider."""
        critic = AgentModelConfig(provider="openai", model="gpt-4o")
        with patch(
            OPENAI_PATH,
            new_callable=AsyncMock,
            return_value=_CR(),
        ) as mock_openai:
            await run_single_critic_debate(
                config=critic,
                plan=plan,
                notebook_content="",
                max_rounds=1,
            )

        assert mock_openai.call_args.kwargs.get("response_schema") is CriticOutput

    @pytest.mark.asyncio
    async def test_scientist_defense_uses_sdk_with_correct_model(self, plan):
        """Scientist defense (Anthropic) uses SDK with correct model in options."""
        critic = AgentModelConfig(provider="openai", model="gpt-4o")
        mock_sdk = _sdk_defense_mock()
        with (
            patch(OPENAI_PATH, new_callable=AsyncMock, return_value=_CR()),
            patch(SDK_PATH, mock_sdk),
        ):
            await run_single_critic_debate(
                config=critic,
                plan=plan,
                notebook_content="",
                max_rounds=2,
            )

        # Verify ClaudeCodeOptions has the default scientist model
        options = mock_sdk.call_args[0][1]
        assert options.model == "claude-sonnet-4-6"


class TestGoalInPrompts:
    """Verify that the goal placeholder is present and populated in critic/debate prompts."""

    def test_goal_in_critic_prompt(self):
        _system, user = _build_critic_prompt(
            {"hypothesis": "test"},
            "",
            "",
            goal="discover causal relationships",
        )
        assert "discover causal relationships" in user
        assert "<goal>" in user

    def test_goal_in_scientist_debate_prompt(self):
        prompt = _build_scientist_debate_user_prompt(
            {"hypothesis": "test"},
            "",
            "",
            goal="optimize alloy compositions",
        )
        assert "optimize alloy compositions" in prompt
        assert "<goal>" in prompt


class TestAnthropicSDKPath:
    """Verify Anthropic critics use Claude Code SDK instead of direct API."""

    @pytest.mark.asyncio
    async def test_anthropic_critic_uses_sdk(self, plan):
        """Anthropic critics call collect_text_from_query, not query_anthropic."""
        critic = AgentModelConfig(provider="anthropic", model="claude-sonnet-4-6")
        valid_json = _valid_critic_json()

        mock_sdk = AsyncMock(return_value=_pad(valid_json))
        # SDK splits input tokens across cache buckets
        mock_sdk.last_usage = {
            "input_tokens": 20,
            "cache_creation_input_tokens": 30,
            "cache_read_input_tokens": 50,
            "output_tokens": 45,
        }

        with patch(SDK_PATH, mock_sdk):
            result = await run_single_critic_debate(
                config=critic,
                plan=plan,
                notebook_content="",
                max_rounds=1,
            )

        mock_sdk.assert_called()
        assert isinstance(result, DebateResult)
        # Token count sums all cache buckets: 20 + 30 + 50 = 100
        assert result.input_tokens == 100
        assert result.output_tokens == 45

    @pytest.mark.asyncio
    async def test_anthropic_sdk_passes_system_prompt(self, plan):
        """Anthropic SDK path passes system_prompt to ClaudeCodeOptions."""
        critic = AgentModelConfig(provider="anthropic", model="claude-sonnet-4-6")
        mock_sdk = _sdk_critic_mock()

        with patch(SDK_PATH, mock_sdk):
            await run_single_critic_debate(
                config=critic,
                plan=plan,
                notebook_content="",
                max_rounds=1,
            )

        options = mock_sdk.call_args[0][1]
        assert options.system_prompt
        assert "<output_format>" in options.system_prompt

    @pytest.mark.asyncio
    async def test_anthropic_scientist_defense_uses_sdk(self, plan):
        """Scientist-in-debate with Anthropic config uses SDK too."""
        critic = AgentModelConfig(provider="openai", model="gpt-4o")
        scientist = AgentModelConfig(provider="anthropic", model="claude-sonnet-4-6")

        mock_sdk = AsyncMock(return_value=_pad(_valid_defense_json()))
        mock_sdk.last_usage = {"input_tokens": 80, "output_tokens": 40}

        with (
            patch(OPENAI_PATH, new_callable=AsyncMock, return_value=_CR()),
            patch(SDK_PATH, mock_sdk),
        ):
            result = await run_single_critic_debate(
                config=critic,
                plan=plan,
                notebook_content="",
                max_rounds=2,
                scientist_config=scientist,
            )

        mock_sdk.assert_called()
        assert result.rounds[0].scientist_defense is not None
