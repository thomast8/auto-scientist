"""Tests for the critic debate loop."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from auto_scientist.agent_result import AgentResult
from auto_scientist.agents.critic import (
    MIN_RESPONSE_LENGTH,
    _build_critic_prompt,
    _build_scientist_debate_user_prompt,
    run_debate,
)
from auto_scientist.images import ImageData
from auto_scientist.model_config import AgentModelConfig, ReasoningConfig

SCIENTIST_SDK_PATH = "auto_scientist.agents.critic.collect_text_from_query"


def _pad(text: str) -> str:
    """Pad a response to meet MIN_RESPONSE_LENGTH for tests."""
    if len(text) >= MIN_RESPONSE_LENGTH:
        return text
    return text + " " + "x" * (MIN_RESPONSE_LENGTH - len(text) - 1)


def _critic_result(text: str) -> AgentResult:
    """Create an AgentResult from padded text for critic mock returns."""
    return AgentResult(text=_pad(text), input_tokens=10, output_tokens=5)


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
        "success_criteria": [
            {
                "name": "Convergence improves",
                "description": "Final loss should decrease with lower learning rate",
                "metric_key": "final_loss_decreased",
                "condition": "== true",
            }
        ],
    }


@pytest.fixture
def openai_critic():
    return AgentModelConfig(provider="openai", model="gpt-4o")


@pytest.fixture
def base_kwargs(plan, openai_critic):
    return {
        "critic_configs": [openai_critic],
        "plan": plan,
        "notebook_content": "# Lab Notebook\nEntry 1",
    }


class TestBuildCriticPrompt:
    def test_includes_plan_json(self):
        plan = {"hypothesis": "test plan"}
        prompt = _build_critic_prompt(plan, "", "")
        assert "test plan" in prompt
        assert "<plan>" in prompt

    def test_empty_defense_no_tag(self):
        prompt = _build_critic_prompt({"h": "p"}, "", "", scientist_defense="")
        assert "<scientist_defense>" not in prompt

    def test_with_defense_includes_tag(self):
        prompt = _build_critic_prompt(
            {"h": "p"}, "", "", scientist_defense="I disagree because...",
        )
        assert "<scientist_defense>" in prompt
        assert "I disagree because..." in prompt

    def test_fallback_for_empty_values(self):
        prompt = _build_critic_prompt({"h": "p"}, "", "")
        assert "(empty)" in prompt or "(none provided)" in prompt


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
    async def test_single_round_no_scientist_response(self, base_kwargs):
        """With max_rounds=1, only the critic is called, no scientist response."""
        with (
            patch(
                "auto_scientist.agents.critic.query_openai",
                new_callable=AsyncMock,
                return_value=_critic_result("Initial critique"),
            ) as mock_openai,
            patch(
                SCIENTIST_SDK_PATH,
                new_callable=AsyncMock,
            ) as mock_scientist,
        ):
            result = await run_debate(**base_kwargs, max_rounds=1)

        assert len(result) == 1
        assert result[0]["model"] == "openai:gpt-4o"
        assert "Initial critique" in result[0]["critique"]
        assert len(result[0]["transcript"]) == 1
        assert result[0]["transcript"][0]["role"] == "critic"
        mock_openai.assert_called_once()
        mock_scientist.assert_not_called()

    @pytest.mark.asyncio
    async def test_two_rounds_calls_scientist_then_refines(self, base_kwargs):
        """With max_rounds=2, critic -> scientist -> critic refinement."""
        with (
            patch(
                "auto_scientist.agents.critic.query_openai",
                new_callable=AsyncMock,
                side_effect=[_critic_result("Initial critique"), _critic_result("Refined critique")],
            ) as mock_openai,
            patch(
                SCIENTIST_SDK_PATH,
                new_callable=AsyncMock,
                return_value=_pad("Scientist response"),
            ) as mock_scientist,
        ):
            result = await run_debate(**base_kwargs, max_rounds=2)

        assert len(result) == 1
        assert "Refined critique" in result[0]["critique"]
        assert mock_openai.call_count == 2
        mock_scientist.assert_called_once()

        # Scientist prompt should contain the initial critique
        scientist_prompt = mock_scientist.call_args[0][0]
        assert "Initial critique" in scientist_prompt

        # Round 2 critic prompt should contain the scientist's defense
        round2_prompt = mock_openai.call_args_list[1][0][1]
        assert "Scientist response" in round2_prompt

    @pytest.mark.asyncio
    async def test_three_rounds(self, base_kwargs):
        """With max_rounds=3, two scientist-response-refinement cycles."""
        with (
            patch(
                "auto_scientist.agents.critic.query_openai",
                new_callable=AsyncMock,
                side_effect=[_critic_result("Critique R1"), _critic_result("Critique R2"), _critic_result("Critique R3")],
            ) as mock_openai,
            patch(
                SCIENTIST_SDK_PATH,
                new_callable=AsyncMock,
                side_effect=[_pad("Scientist R1"), _pad("Scientist R2")],
            ) as mock_scientist,
        ):
            result = await run_debate(**base_kwargs, max_rounds=3)

        assert "Critique R3" in result[0]["critique"]
        assert mock_openai.call_count == 3
        assert mock_scientist.call_count == 2

    @pytest.mark.asyncio
    async def test_debate_returns_transcript(self, base_kwargs):
        """Debate returns transcript with all rounds."""
        with (
            patch(
                "auto_scientist.agents.critic.query_openai",
                new_callable=AsyncMock,
                side_effect=[_critic_result("Critique R1"), _critic_result("Critique R2")],
            ),
            patch(
                SCIENTIST_SDK_PATH,
                new_callable=AsyncMock,
                return_value=_pad("Scientist response"),
            ),
        ):
            result = await run_debate(**base_kwargs, max_rounds=2)

        assert len(result) == 1
        transcript = result[0]["transcript"]
        assert len(transcript) == 3  # critic, scientist, critic
        assert transcript[0]["role"] == "critic"
        assert "Critique R1" in transcript[0]["content"]
        assert transcript[1]["role"] == "scientist"
        assert "Scientist response" in transcript[1]["content"]
        assert transcript[2]["role"] == "critic"
        assert "Critique R2" in transcript[2]["content"]

    @pytest.mark.asyncio
    async def test_multiple_critics(self, plan):
        """Each critic runs its own independent debate."""
        critics = [
            AgentModelConfig(provider="openai", model="gpt-4o"),
            AgentModelConfig(provider="google", model="gemini-2.5-pro"),
        ]
        with (
            patch(
                "auto_scientist.agents.critic.query_openai",
                new_callable=AsyncMock,
                side_effect=[_critic_result("OAI initial"), _critic_result("OAI refined")],
            ),
            patch(
                "auto_scientist.agents.critic.query_google",
                new_callable=AsyncMock,
                side_effect=[_critic_result("Google initial"), _critic_result("Google refined")],
            ),
            patch(
                SCIENTIST_SDK_PATH,
                new_callable=AsyncMock,
                side_effect=[_pad("Scientist for OAI"), _pad("Scientist for Google")],
            ),
        ):
            result = await run_debate(
                critic_configs=critics,
                plan=plan,
                notebook_content="",
                max_rounds=2,
            )

        assert len(result) == 2
        assert result[0]["model"] == "openai:gpt-4o"
        assert "OAI refined" in result[0]["critique"]
        assert result[1]["model"] == "google:gemini-2.5-pro"
        assert "Google refined" in result[1]["critique"]

    @pytest.mark.asyncio
    async def test_plan_in_critic_prompt(self, base_kwargs):
        """Critic prompt includes the scientist's plan."""
        with (
            patch(
                "auto_scientist.agents.critic.query_openai",
                new_callable=AsyncMock,
                return_value=_critic_result("Critique of plan"),
            ) as mock_openai,
        ):
            await run_debate(**base_kwargs, max_rounds=1)

        critic_prompt = mock_openai.call_args[0][1]
        assert "<plan>" in critic_prompt
        assert "Adjusting learning rate" in critic_prompt

    @pytest.mark.asyncio
    async def test_plan_in_scientist_prompt(self, base_kwargs):
        """Scientist response prompt includes the plan."""
        with (
            patch(
                "auto_scientist.agents.critic.query_openai",
                new_callable=AsyncMock,
                side_effect=[_critic_result("Critique"), _critic_result("Refined")],
            ),
            patch(
                SCIENTIST_SDK_PATH,
                new_callable=AsyncMock,
                return_value=_pad("Response"),
            ) as mock_scientist,
        ):
            await run_debate(**base_kwargs, max_rounds=2)

        scientist_prompt = mock_scientist.call_args[0][0]
        assert "<plan>" in scientist_prompt
        assert "Adjusting learning rate" in scientist_prompt

    @pytest.mark.asyncio
    async def test_criteria_in_critic_prompt(self, base_kwargs):
        """Critic prompt includes success criteria from the plan."""
        with patch(
            "auto_scientist.agents.critic.query_openai",
            new_callable=AsyncMock,
            return_value=_critic_result("Critique"),
        ) as mock_openai:
            await run_debate(**base_kwargs, max_rounds=1)

        critic_prompt = mock_openai.call_args[0][1]
        assert "success_criteria" in critic_prompt
        assert "Convergence improves" in critic_prompt

    @pytest.mark.asyncio
    async def test_no_analysis_or_script_in_prompts(self, base_kwargs):
        """Neither the critic nor scientist sees analysis JSON or script content."""
        with (
            patch(
                "auto_scientist.agents.critic.query_openai",
                new_callable=AsyncMock,
                side_effect=[_critic_result("Critique"), _critic_result("Refined")],
            ) as mock_openai,
            patch(
                SCIENTIST_SDK_PATH,
                new_callable=AsyncMock,
                return_value=_pad("Response"),
            ) as mock_scientist,
        ):
            await run_debate(**base_kwargs, max_rounds=2)

        critic_prompt = mock_openai.call_args_list[0][0][1]
        scientist_prompt = mock_scientist.call_args[0][0]

        assert "Latest Analysis" not in critic_prompt
        assert "Current Script" not in scientist_prompt

    @pytest.mark.asyncio
    async def test_web_search_enabled(self, base_kwargs):
        """Critic calls pass web_search=True; scientist gets WebSearch tool via SDK."""
        with (
            patch(
                "auto_scientist.agents.critic.query_openai",
                new_callable=AsyncMock,
                side_effect=[_critic_result("Critique"), _critic_result("Refined")],
            ) as mock_openai,
            patch(
                SCIENTIST_SDK_PATH,
                new_callable=AsyncMock,
                return_value=_pad("Response"),
            ) as mock_scientist,
        ):
            await run_debate(**base_kwargs, max_rounds=2)

        for call in mock_openai.call_args_list:
            assert call.kwargs.get("web_search") is True

        # Scientist uses SDK with WebSearch tool (passed via ClaudeCodeOptions)
        options = mock_scientist.call_args[0][1]
        assert "WebSearch" in options.allowed_tools

    @pytest.mark.asyncio
    async def test_symmetric_context(self, base_kwargs):
        """Critic and scientist receive the same context (symmetric)."""
        with (
            patch(
                "auto_scientist.agents.critic.query_openai",
                new_callable=AsyncMock,
                side_effect=[_critic_result("Critique"), _critic_result("Refined")],
            ) as mock_openai,
            patch(
                SCIENTIST_SDK_PATH,
                new_callable=AsyncMock,
                return_value=_pad("Response"),
            ) as mock_scientist,
        ):
            await run_debate(**base_kwargs, max_rounds=2)

        critic_prompt = mock_openai.call_args_list[0][0][1]
        scientist_prompt = mock_scientist.call_args[0][0]

        assert "<notebook>" in critic_prompt
        assert "<notebook>" in scientist_prompt

    @pytest.mark.asyncio
    async def test_no_compressed_history_in_prompts(self, base_kwargs):
        """Prompts should not contain compressed history sections."""
        with (
            patch(
                "auto_scientist.agents.critic.query_openai",
                new_callable=AsyncMock,
                side_effect=[_critic_result("Critique"), _critic_result("Refined")],
            ) as mock_openai,
            patch(
                SCIENTIST_SDK_PATH,
                new_callable=AsyncMock,
                return_value=_pad("Response"),
            ) as mock_scientist,
        ):
            await run_debate(**base_kwargs, max_rounds=2)

        critic_prompt = mock_openai.call_args_list[0][0][1]
        scientist_prompt = mock_scientist.call_args[0][0]

        assert "Experiment History" not in critic_prompt
        assert "Experiment History" not in scientist_prompt

    @pytest.mark.asyncio
    async def test_custom_scientist_config(self, base_kwargs):
        """Scientist uses the specified model from config via SDK options."""
        scientist = AgentModelConfig(model="claude-haiku-4-5-20251001")
        with (
            patch(
                "auto_scientist.agents.critic.query_openai",
                new_callable=AsyncMock,
                side_effect=[_critic_result("Critique"), _critic_result("Refined")],
            ),
            patch(
                SCIENTIST_SDK_PATH,
                new_callable=AsyncMock,
                return_value=_pad("Response"),
            ) as mock_scientist,
        ):
            await run_debate(
                **base_kwargs,
                max_rounds=2,
                scientist_config=scientist,
            )

        options = mock_scientist.call_args[0][1]
        assert options.model == "claude-haiku-4-5-20251001"

    @pytest.mark.asyncio
    async def test_unknown_provider_returns_empty(self, plan):
        """Unknown provider is captured as a failed debate, returns empty results."""
        bad_config = AgentModelConfig.model_validate(
            {"provider": "openai", "model": "model"}
        )
        # Monkeypatch provider to bypass Pydantic validation
        object.__setattr__(bad_config, "provider", "unknown")
        result = await run_debate(
            critic_configs=[bad_config],
            plan=plan,
            notebook_content="",
        )
        assert result == []  # error captured, no successful debates

    @pytest.mark.asyncio
    async def test_anthropic_critic_dispatches_correctly(self, plan):
        critic = AgentModelConfig(provider="anthropic", model="claude-sonnet-4-6")
        with patch(
            "auto_scientist.agents.critic.query_anthropic",
            new_callable=AsyncMock,
            return_value=_critic_result("Anthropic critique"),
        ) as mock_anthropic:
            result = await run_debate(
                critic_configs=[critic],
                plan=plan,
                notebook_content="",
                max_rounds=1,
            )

        assert len(result) == 1
        assert "Anthropic critique" in result[0]["critique"]
        assert mock_anthropic.call_count == 1

    @pytest.mark.asyncio
    async def test_reasoning_passed_to_critic(self, plan):
        """Critic reasoning config is forwarded to model client."""
        critic = AgentModelConfig(
            provider="openai", model="o4-mini",
            reasoning=ReasoningConfig(level="high"),
        )
        with patch(
            "auto_scientist.agents.critic.query_openai",
            new_callable=AsyncMock,
            return_value=_critic_result("Critique"),
        ) as mock_openai:
            await run_debate(
                critic_configs=[critic],
                plan=plan,
                notebook_content="",
                max_rounds=1,
            )

        assert mock_openai.call_args.kwargs["reasoning"].level == "high"


class TestCriticRetry:
    @pytest.mark.asyncio
    async def test_retry_on_empty_critic_response(self, base_kwargs):
        """Empty critic response triggers retry."""
        with (
            patch(
                "auto_scientist.agents.critic.query_openai",
                new_callable=AsyncMock,
                side_effect=[AgentResult(text=""), _critic_result("Valid critique")],
            ) as mock_openai,
        ):
            result = await run_debate(**base_kwargs, max_rounds=1)

        assert "Valid critique" in result[0]["critique"]
        assert mock_openai.call_count == 2

    @pytest.mark.asyncio
    async def test_sdk_error_propagates_from_scientist(self, base_kwargs):
        """SDK error in scientist debate response propagates up."""
        with (
            patch(
                "auto_scientist.agents.critic.query_openai",
                new_callable=AsyncMock,
                side_effect=[_critic_result("Critique"), _critic_result("Refined")],
            ),
            patch(
                SCIENTIST_SDK_PATH,
                new_callable=AsyncMock,
                side_effect=RuntimeError("SDK error"),
            ),
        ):
            result = await run_debate(**base_kwargs, max_rounds=2)
            assert result == []  # error captured, no successful debates

    @pytest.mark.asyncio
    async def test_exhausted_retries_uses_whatever_we_have(self, base_kwargs):
        """If all retries return empty/short, use what we have."""
        with (
            patch(
                "auto_scientist.agents.critic.query_openai",
                new_callable=AsyncMock,
                return_value=AgentResult(text=""),
            ),
        ):
            result = await run_debate(**base_kwargs, max_rounds=1)

        assert result[0]["critique"] == ""

    @pytest.mark.asyncio
    async def test_retry_on_too_short_response(self, base_kwargs):
        """Response shorter than MIN_RESPONSE_LENGTH triggers retry."""
        with (
            patch(
                "auto_scientist.agents.critic.query_openai",
                new_callable=AsyncMock,
                side_effect=[AgentResult(text="OK"), AgentResult(text="This is a substantive critique that addresses the hypothesis, strategy, and provides alternative approaches to consider.")],
            ) as mock_openai,
        ):
            result = await run_debate(**base_kwargs, max_rounds=1)

        assert mock_openai.call_count == 2
        assert "substantive" in result[0]["critique"]


# Minimal valid 1x1 PNG for test fixtures
TINY_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
    b"\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00"
    b"\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
)


class TestBuildCriticPromptPlots:
    def test_has_plots_adds_section(self):
        prompt = _build_critic_prompt({"h": "p"}, "", "", has_plots=True)
        assert "<plots_attached>" in prompt

    def test_no_plots_omits_section(self):
        prompt = _build_critic_prompt({"h": "p"}, "", "", has_plots=False)
        assert "<plots_attached>" not in prompt

    def test_default_no_plots_section(self):
        prompt = _build_critic_prompt({"h": "p"}, "", "")
        assert "<plots_attached>" not in prompt


class TestBuildScientistDebatePromptPlots:
    def test_has_plots_adds_section(self):
        prompt = _build_scientist_debate_user_prompt(
            {"h": "p"}, "", "", critique="test", has_plots=True,
        )
        assert "<plots_attached>" in prompt

    def test_no_plots_omits_section(self):
        prompt = _build_scientist_debate_user_prompt(
            {"h": "p"}, "", "", critique="test", has_plots=False,
        )
        assert "<plots_attached>" not in prompt


class TestRunDebateWithPlots:
    @pytest.mark.asyncio
    async def test_images_forwarded_to_critic(self, plan):
        """When plot_paths are provided, encoded images are forwarded to critic."""
        critic = AgentModelConfig(provider="openai", model="gpt-4o")
        with (
            patch(
                "auto_scientist.agents.critic.query_openai",
                new_callable=AsyncMock,
                return_value=_critic_result("Critique with plots"),
            ) as mock_openai,
            patch(
                "auto_scientist.agents.critic.encode_images_from_paths",
                return_value=[ImageData(data="abc", media_type="image/png")],
            ) as mock_encode,
        ):
            await run_debate(
                critic_configs=[critic],
                plan=plan,
                notebook_content="",
                max_rounds=1,
                plot_paths=[Path("/fake/plot.png")],
            )

        mock_encode.assert_called_once_with([Path("/fake/plot.png")])
        call_kwargs = mock_openai.call_args.kwargs
        assert call_kwargs["images"] == [ImageData(data="abc", media_type="image/png")]

    @pytest.mark.asyncio
    async def test_no_plots_passes_no_images(self, plan):
        """Without plot_paths, images kwarg is empty list."""
        critic = AgentModelConfig(provider="openai", model="gpt-4o")
        with patch(
            "auto_scientist.agents.critic.query_openai",
            new_callable=AsyncMock,
            return_value=_critic_result("Critique"),
        ) as mock_openai:
            await run_debate(
                critic_configs=[critic],
                plan=plan,
                notebook_content="",
                max_rounds=1,
            )

        call_kwargs = mock_openai.call_args.kwargs
        assert call_kwargs.get("images") == []

    @pytest.mark.asyncio
    async def test_scientist_gets_read_tool_with_plots(self, base_kwargs):
        """When plots are provided, scientist gets Read tool access."""
        with (
            patch(
                "auto_scientist.agents.critic.query_openai",
                new_callable=AsyncMock,
                side_effect=[_critic_result("Critique"), _critic_result("Refined")],
            ),
            patch(
                SCIENTIST_SDK_PATH,
                new_callable=AsyncMock,
                return_value=_pad("Response"),
            ) as mock_scientist,
            patch(
                "auto_scientist.agents.critic.encode_images_from_paths",
                return_value=[ImageData(data="abc", media_type="image/png")],
            ),
        ):
            await run_debate(
                **base_kwargs,
                max_rounds=2,
                plot_paths=[Path("/fake/plot.png")],
            )

        options = mock_scientist.call_args[0][1]
        assert "Read" in options.allowed_tools

    @pytest.mark.asyncio
    async def test_scientist_prompt_lists_plot_paths(self, base_kwargs):
        """When plots are provided, scientist prompt includes plot file paths."""
        with (
            patch(
                "auto_scientist.agents.critic.query_openai",
                new_callable=AsyncMock,
                side_effect=[_critic_result("Critique"), _critic_result("Refined")],
            ),
            patch(
                SCIENTIST_SDK_PATH,
                new_callable=AsyncMock,
                return_value=_pad("Response"),
            ) as mock_scientist,
            patch(
                "auto_scientist.agents.critic.encode_images_from_paths",
                return_value=[ImageData(data="abc", media_type="image/png")],
            ),
        ):
            await run_debate(
                **base_kwargs,
                max_rounds=2,
                plot_paths=[Path("/fake/plot.png")],
            )

        scientist_prompt = mock_scientist.call_args[0][0]
        assert "/fake/plot.png" in scientist_prompt

    @pytest.mark.asyncio
    async def test_critic_prompt_has_plots_section(self, base_kwargs):
        """When plots are provided, critic prompt includes <plots_attached> section."""
        with (
            patch(
                "auto_scientist.agents.critic.query_openai",
                new_callable=AsyncMock,
                return_value=_critic_result("Critique"),
            ) as mock_openai,
            patch(
                "auto_scientist.agents.critic.encode_images_from_paths",
                return_value=[ImageData(data="abc", media_type="image/png")],
            ),
        ):
            await run_debate(
                **base_kwargs,
                max_rounds=1,
                plot_paths=[Path("/fake/plot.png")],
            )

        critic_prompt = mock_openai.call_args[0][1]
        assert "<plots_attached>" in critic_prompt
