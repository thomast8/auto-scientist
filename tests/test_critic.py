"""Tests for the critic debate loop with personas and structured output."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from auto_scientist.agent_result import AgentResult
from auto_scientist.agents.critic import (
    _build_critic_prompt,
    _build_critic_tools_and_mcp,
    run_debate,
    run_single_critic_debate,
)
from auto_scientist.agents.debate_models import CriticOutput, DebateResult
from auto_scientist.agents.prediction_tool import PREDICTION_SPEC
from auto_scientist.model_config import AgentModelConfig, ReasoningConfig
from auto_scientist.prompts.critic import PREDICTION_PERSONAS
from auto_scientist.state import PredictionRecord

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


# Short alias for mock returns used in many tests
_CR = _structured_critic_result


def _sdk_mock(
    text: str = "",
    input_tokens: int = 10,
    output_tokens: int = 5,
) -> AsyncMock:
    """Create an AsyncMock for collect_text_from_query (SDK path).

    Returns (text, usage) tuple matching the real function signature.
    Also sets last_usage attribute for backward compat with orchestrator.
    """
    usage = {"input_tokens": input_tokens, "output_tokens": output_tokens}
    mock = AsyncMock(return_value=(text, usage))
    mock.last_usage = usage
    return mock


def _sdk_critic_mock(**kwargs) -> AsyncMock:
    """SDK mock returning valid CriticOutput JSON."""
    return _sdk_mock(text=_pad(_valid_critic_json(**kwargs)))


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
    return AgentModelConfig(provider="openai", model="gpt-4o", mode="api")


@pytest.fixture
def google_critic():
    return AgentModelConfig(provider="google", model="gemini-2.5-pro", mode="api")


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
        """On iteration 1+, runs 4 critiques (one per persona), regardless of critic count."""
        with (
            patch(OPENAI_PATH, new_callable=AsyncMock, return_value=_CR()),
            patch(GOOGLE_PATH, new_callable=AsyncMock, return_value=_CR()),
        ):
            results = await run_debate(**base_kwargs, iteration=1)

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
            results = await run_debate(**base_kwargs)

        for r in results:
            assert isinstance(r, DebateResult)
            assert isinstance(r.critic_output, CriticOutput)
            assert len(r.raw_transcript) >= 1

    @pytest.mark.asyncio
    async def test_iteration_0_runs_only_subset_personas(self, base_kwargs):
        """Iteration 0 runs only Methodologist and Falsification Expert."""
        with (
            patch(OPENAI_PATH, new_callable=AsyncMock, return_value=_CR()),
            patch(GOOGLE_PATH, new_callable=AsyncMock, return_value=_CR()),
        ):
            results = await run_debate(**base_kwargs, iteration=0)

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
            results = await run_debate(**base_kwargs, iteration=1)

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
            patch("auto_scientist.agents.critic.random.shuffle"),
        ):
            results_iter0 = await run_debate(**base_kwargs, iteration=0)
            results_iter1 = await run_debate(**base_kwargs, iteration=1)

        meth_0 = next(r for r in results_iter0 if r.persona == "Methodologist")
        meth_1 = next(r for r in results_iter1 if r.persona == "Methodologist")
        assert meth_0.critic_model != meth_1.critic_model

    @pytest.mark.asyncio
    async def test_raw_transcript_preserved(self, base_kwargs):
        """Raw transcript is kept alongside structured data."""
        with (
            patch(OPENAI_PATH, new_callable=AsyncMock, return_value=_CR()),
            patch(GOOGLE_PATH, new_callable=AsyncMock, return_value=_CR()),
        ):
            results = await run_debate(**base_kwargs)

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
            await run_debate(**base_kwargs)

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
            await run_debate(**base_kwargs)

        for call in mock_openai.call_args_list:
            assert call.kwargs.get("web_search") is True

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
                iteration=1,
            )

        assert len(result) == 4
        for r in result:
            assert r.critic_model == "anthropic:claude-sonnet-4-6"

    @pytest.mark.asyncio
    async def test_reasoning_passed_to_critic(self, plan):
        """Critic reasoning config is forwarded to model client."""
        critic = AgentModelConfig(
            provider="openai",
            model="o4-mini",
            reasoning=ReasoningConfig(level="high"),
            mode="api",
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
            )

        assert mock_openai.call_args.kwargs["reasoning"].level == "high"

    @pytest.mark.asyncio
    async def test_token_counts(self, base_kwargs):
        """Token counts are set from the critic call."""
        with (
            patch(OPENAI_PATH, new_callable=AsyncMock, return_value=_CR()),
            patch(GOOGLE_PATH, new_callable=AsyncMock, return_value=_CR()),
        ):
            results = await run_debate(**base_kwargs)

        for r in results:
            assert r.input_tokens == 10
            assert r.output_tokens == 5


class TestCriticRetry:
    @pytest.mark.asyncio
    async def test_retry_on_empty_critic_response(self, plan, two_critics):
        """Empty critic response triggers retry."""
        valid = _structured_critic_result()
        with (
            patch(
                OPENAI_PATH,
                new_callable=AsyncMock,
                side_effect=[AgentResult(text=""), valid, valid],
            ) as mock_openai,
            patch(GOOGLE_PATH, new_callable=AsyncMock, return_value=valid),
        ):
            result = await run_debate(
                critic_configs=two_critics,
                plan=plan,
                notebook_content="",
                iteration=1,
            )

        assert len(result) == 4
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
            )
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
            )


class TestCriticValidation:
    @pytest.mark.asyncio
    async def test_valid_json_parsed_to_critic_output(self, plan):
        """Valid structured JSON is parsed into CriticOutput."""
        critic = AgentModelConfig(provider="openai", model="gpt-4o", mode="api")
        with patch(
            OPENAI_PATH,
            new_callable=AsyncMock,
            return_value=_CR(),
        ):
            result = await run_single_critic_debate(
                config=critic,
                plan=plan,
                notebook_content="",
            )

        assert isinstance(result.critic_output, CriticOutput)
        assert len(result.critic_output.concerns) == 1
        assert result.critic_output.concerns[0].claim == "Data quality issue"

    @pytest.mark.asyncio
    async def test_invalid_json_preserves_raw_text_as_concern(self, plan):
        """Invalid JSON falls back with raw text preserved as a PARSE ERROR concern."""
        critic = AgentModelConfig(provider="openai", model="gpt-4o", mode="api")
        with patch(
            OPENAI_PATH,
            new_callable=AsyncMock,
            return_value=_critic_result("Not valid JSON at all, just prose critique."),
        ):
            result = await run_single_critic_debate(
                config=critic,
                plan=plan,
                notebook_content="",
            )

        co = result.critic_output
        assert isinstance(co, CriticOutput)
        assert len(co.concerns) == 1
        assert "[SYNTHETIC - PARSE ERROR]" in co.concerns[0].claim
        assert co.concerns[0].severity == "high"
        assert co.concerns[0].category == "other"


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
            has_predictions=True,
        )
        assert "<prediction_history>" in user
        assert "polynomial fits well" in user

    def test_empty_analysis_shows_fallback(self):
        _system, user = _build_critic_prompt({"h": "p"}, "", "")
        assert "(no analysis yet)" in user

    def test_empty_prediction_history_shows_fallback(self):
        _system, user = _build_critic_prompt({"h": "p"}, "", "", has_predictions=True)
        assert "(no prediction history yet)" in user

    def test_no_prediction_history_when_disabled(self):
        system, user = _build_critic_prompt(
            {"h": "p"},
            "",
            "",
            prediction_history="[1.0] CONFIRMED: polynomial fits well",
            has_predictions=False,
        )
        assert "<prediction_history>" not in user
        assert "polynomial fits well" not in user
        assert "read_predictions" not in system
        assert "prediction history" not in system


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
                analysis_json='{"key_metrics": {"rmse": 0.52}}',
            )

        critic_prompt = mock_openai.call_args[0][1]
        assert "<analysis>" in critic_prompt
        assert "rmse" in critic_prompt

    @pytest.mark.asyncio
    async def test_prediction_history_in_critic_prompt(self, plan):
        """When prediction_history is provided, prediction persona's prompt includes it."""
        critic = AgentModelConfig(provider="openai", model="gpt-4o", mode="api")
        with patch(OPENAI_PATH, new_callable=AsyncMock, return_value=_CR()) as mock_openai:
            await run_single_critic_debate(
                config=critic,
                plan=plan,
                notebook_content="",
                prediction_history="[1.0] CONFIRMED: polynomial fits well",
                persona={"name": "Evidence Auditor", "system_text": ""},
            )

        critic_prompt = mock_openai.call_args[0][1]
        assert "<prediction_history>" in critic_prompt
        assert "polynomial fits well" in critic_prompt


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

        assert get_model_index_for_debate(0, 0, 2) == 0
        assert get_model_index_for_debate(1, 0, 2) == 1
        assert get_model_index_for_debate(2, 0, 2) == 0

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
        critic = AgentModelConfig(provider="openai", model="gpt-4o", mode="api")
        with patch(
            OPENAI_PATH,
            new_callable=AsyncMock,
            return_value=_CR(),
        ) as mock_openai:
            await run_single_critic_debate(
                config=critic,
                plan=plan,
                notebook_content="",
            )

        assert mock_openai.call_args.kwargs.get("response_schema") is CriticOutput


class TestGoalInPrompts:
    """Verify that the goal placeholder is present and populated in critic prompts."""

    def test_goal_in_critic_prompt(self):
        _system, user = _build_critic_prompt(
            {"hypothesis": "test"},
            "",
            "",
            goal="discover causal relationships",
        )
        assert "discover causal relationships" in user
        assert "<goal>" in user


class TestAnthropicSDKPath:
    """Verify Anthropic critics use Claude Code SDK instead of direct API."""

    @pytest.mark.asyncio
    async def test_anthropic_critic_uses_sdk(self, plan):
        """Anthropic critics call collect_text_from_query, not query_anthropic."""
        critic = AgentModelConfig(provider="anthropic", model="claude-sonnet-4-6")
        valid_json = _valid_critic_json()

        sdk_usage = {
            "input_tokens": 20,
            "cache_creation_input_tokens": 30,
            "cache_read_input_tokens": 50,
            "output_tokens": 45,
        }
        mock_sdk = AsyncMock(return_value=(_pad(valid_json), sdk_usage))
        mock_sdk.last_usage = sdk_usage

        with patch(SDK_PATH, mock_sdk):
            result = await run_single_critic_debate(
                config=critic,
                plan=plan,
                notebook_content="",
            )

        mock_sdk.assert_called()
        assert isinstance(result, DebateResult)
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
            )

        options = mock_sdk.call_args[0][1]
        assert options.system_prompt
        assert "<output_format>" in options.system_prompt


class TestBuildCriticToolsAndMcp:
    """Verify _build_critic_tools_and_mcp builds correct tools and MCP config."""

    def test_no_records_returns_base_tools_only(self):
        tools, mcp_servers = _build_critic_tools_and_mcp(None)
        assert tools == ["WebSearch"]
        assert mcp_servers == {}

    def test_empty_records_returns_base_tools_only(self):
        tools, mcp_servers = _build_critic_tools_and_mcp([])
        assert tools == ["WebSearch"]
        assert mcp_servers == {}

    def test_with_records_adds_mcp_tool(self):
        records = [
            PredictionRecord(
                prediction="noise is additive",
                diagnostic="check residuals",
                if_confirmed="add noise term",
                if_refuted="look elsewhere",
                iteration_prescribed=1,
            ),
        ]
        tools, mcp_servers = _build_critic_tools_and_mcp(records)
        assert PREDICTION_SPEC.mcp_tool_name in tools
        assert "WebSearch" in tools
        assert "predictions" in mcp_servers
        assert mcp_servers["predictions"]["type"] == "stdio"


class TestCriticMcpIntegration:
    """Verify MCP servers are passed to SDK options when prediction records exist."""

    _PREDICTION_PERSONA = {"name": "Evidence Auditor", "system_text": ""}

    @pytest.mark.asyncio
    async def test_sdk_receives_mcp_servers_with_records(self, plan):
        """SDK path receives MCP servers when prediction_history_records is provided."""
        critic = AgentModelConfig(provider="anthropic", model="claude-sonnet-4-6")
        records = [
            PredictionRecord(
                prediction="noise is additive",
                diagnostic="check residuals",
                if_confirmed="add noise term",
                if_refuted="look elsewhere",
                iteration_prescribed=1,
            ),
        ]
        mock_sdk = _sdk_critic_mock()

        with patch(SDK_PATH, mock_sdk):
            await run_single_critic_debate(
                config=critic,
                plan=plan,
                notebook_content="",
                prediction_history_records=records,
                persona=self._PREDICTION_PERSONA,
            )

        options = mock_sdk.call_args[0][1]
        assert "predictions" in options.mcp_servers
        assert PREDICTION_SPEC.mcp_tool_name in options.allowed_tools

    @pytest.mark.asyncio
    async def test_sdk_no_mcp_without_records(self, plan):
        """SDK path has empty MCP servers when no prediction records."""
        critic = AgentModelConfig(provider="anthropic", model="claude-sonnet-4-6")
        mock_sdk = _sdk_critic_mock()

        with patch(SDK_PATH, mock_sdk):
            await run_single_critic_debate(
                config=critic,
                plan=plan,
                notebook_content="",
            )

        options = mock_sdk.call_args[0][1]
        assert options.mcp_servers == {}
        assert PREDICTION_SPEC.mcp_tool_name not in options.allowed_tools

    @pytest.mark.asyncio
    async def test_sdk_max_turns_includes_toolsearch_bump(self, plan):
        """SDK path uses max_turns=11 (10 base + 1 for ToolSearch to resolve WebSearch)."""
        critic = AgentModelConfig(provider="anthropic", model="claude-sonnet-4-6")
        mock_sdk = _sdk_critic_mock()

        with patch(SDK_PATH, mock_sdk):
            await run_single_critic_debate(
                config=critic,
                plan=plan,
                notebook_content="",
            )

        options = mock_sdk.call_args[0][1]
        assert options.max_turns == 11
        assert "ToolSearch" in options.allowed_tools

    @pytest.mark.asyncio
    async def test_api_mode_ignores_records(self, plan):
        """Direct API mode (OpenAI) ignores prediction records (no MCP available)."""
        critic = AgentModelConfig(provider="openai", model="gpt-4o", mode="api")
        records = [
            PredictionRecord(
                prediction="noise is additive",
                diagnostic="check residuals",
                if_confirmed="add noise term",
                if_refuted="look elsewhere",
                iteration_prescribed=1,
            ),
        ]
        with patch(OPENAI_PATH, new_callable=AsyncMock, return_value=_CR()) as mock_openai:
            result = await run_single_critic_debate(
                config=critic,
                plan=plan,
                notebook_content="",
                prediction_history_records=records,
                persona=self._PREDICTION_PERSONA,
            )

        assert isinstance(result, DebateResult)
        # Direct API gets no mcp_servers kwarg
        assert "mcp_servers" not in mock_openai.call_args.kwargs

    @pytest.mark.asyncio
    async def test_sdk_with_records_uses_compact_tree_in_prompt(self, plan):
        """SDK mode with records uses compact tree from records, not raw text."""
        critic = AgentModelConfig(provider="anthropic", model="claude-sonnet-4-6")
        records = [
            PredictionRecord(
                prediction="noise is additive",
                diagnostic="check residuals",
                if_confirmed="add noise term",
                if_refuted="look elsewhere",
                iteration_prescribed=1,
            ),
        ]
        mock_sdk = _sdk_critic_mock()

        with patch(SDK_PATH, mock_sdk):
            await run_single_critic_debate(
                config=critic,
                plan=plan,
                notebook_content="",
                prediction_history="RAW TEXT FALLBACK",
                prediction_history_records=records,
                persona=self._PREDICTION_PERSONA,
            )

        # The prompt should contain compact tree from records, not the raw text
        prompt = mock_sdk.call_args[0][0]
        assert "PREDICTION TREE" in prompt
        assert "noise is additive" in prompt
        assert "RAW TEXT FALLBACK" not in prompt

    @pytest.mark.asyncio
    async def test_api_mode_keeps_text_in_prompt(self, plan):
        """API mode without MCP keeps the full prediction text in the prompt."""
        critic = AgentModelConfig(provider="openai", model="gpt-4o", mode="api")
        with patch(OPENAI_PATH, new_callable=AsyncMock, return_value=_CR()) as mock_openai:
            await run_single_critic_debate(
                config=critic,
                plan=plan,
                notebook_content="",
                prediction_history="[1.0] CONFIRMED: polynomial fits well",
                persona=self._PREDICTION_PERSONA,
            )

        prompt = mock_openai.call_args[0][1]
        assert "polynomial fits well" in prompt


class TestPersonaPredictionAccess:
    """Verify that only prediction personas get prediction history and MCP tools."""

    def test_prediction_personas_constant(self):
        assert {"Trajectory Critic", "Evidence Auditor"} == PREDICTION_PERSONAS

    @pytest.mark.asyncio
    async def test_prediction_persona_gets_mcp_and_history(self, plan):
        """Trajectory Critic (a prediction persona) gets MCP server + prediction history."""
        from auto_scientist.prompts.critic import PERSONAS

        critic = AgentModelConfig(provider="anthropic", model="claude-sonnet-4-6")
        tc_persona = next(p for p in PERSONAS if p["name"] == "Trajectory Critic")
        records = [
            PredictionRecord(
                prediction="noise is additive",
                diagnostic="check residuals",
                if_confirmed="add noise term",
                if_refuted="look elsewhere",
                iteration_prescribed=1,
            ),
        ]
        mock_sdk = _sdk_critic_mock()

        with patch(SDK_PATH, mock_sdk):
            await run_single_critic_debate(
                config=critic,
                plan=plan,
                notebook_content="",
                persona=tc_persona,
                prediction_history_records=records,
            )

        options = mock_sdk.call_args[0][1]
        assert "predictions" in options.mcp_servers
        assert PREDICTION_SPEC.mcp_tool_name in options.allowed_tools
        prompt = mock_sdk.call_args[0][0]
        assert "<prediction_history>" in prompt
        assert "noise is additive" in prompt

    @pytest.mark.asyncio
    async def test_non_prediction_persona_skips_mcp_and_history(self, plan):
        """Methodologist (not a prediction persona) gets no MCP and no prediction history."""
        from auto_scientist.prompts.critic import PERSONAS

        critic = AgentModelConfig(provider="anthropic", model="claude-sonnet-4-6")
        meth_persona = next(p for p in PERSONAS if p["name"] == "Methodologist")
        records = [
            PredictionRecord(
                prediction="noise is additive",
                diagnostic="check residuals",
                if_confirmed="add noise term",
                if_refuted="look elsewhere",
                iteration_prescribed=1,
            ),
        ]
        mock_sdk = _sdk_critic_mock()

        with patch(SDK_PATH, mock_sdk):
            await run_single_critic_debate(
                config=critic,
                plan=plan,
                notebook_content="",
                persona=meth_persona,
                prediction_history_records=records,
            )

        options = mock_sdk.call_args[0][1]
        assert options.mcp_servers == {}
        assert PREDICTION_SPEC.mcp_tool_name not in options.allowed_tools
        prompt = mock_sdk.call_args[0][0]
        assert "<prediction_history>" not in prompt

    @pytest.mark.asyncio
    async def test_non_prediction_persona_no_tool_in_system(self, plan):
        """Non-prediction persona's system prompt does not mention read_predictions."""
        from auto_scientist.prompts.critic import PERSONAS

        critic = AgentModelConfig(provider="anthropic", model="claude-sonnet-4-6")
        fals_persona = next(p for p in PERSONAS if p["name"] == "Falsification Expert")
        mock_sdk = _sdk_critic_mock()

        with patch(SDK_PATH, mock_sdk):
            await run_single_critic_debate(
                config=critic,
                plan=plan,
                notebook_content="",
                persona=fals_persona,
            )

        options = mock_sdk.call_args[0][1]
        assert "read_predictions" not in options.system_prompt
