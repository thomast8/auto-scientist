"""Tests for the critic debate loop."""

from unittest.mock import AsyncMock, patch

import pytest

from auto_scientist.agents.critic import (
    parse_critic_spec,
    run_critic,
    run_debate,
)


class TestParseCriticSpec:
    def test_valid_spec(self):
        assert parse_critic_spec("openai:gpt-4o") == ("openai", "gpt-4o")
        assert parse_critic_spec("google:gemini-2.5-pro") == ("google", "gemini-2.5-pro")
        assert parse_critic_spec("anthropic:claude-sonnet-4-6") == (
            "anthropic",
            "claude-sonnet-4-6",
        )

    def test_invalid_spec(self):
        with pytest.raises(ValueError, match="Invalid critic spec"):
            parse_critic_spec("no-colon")

    def test_spec_with_multiple_colons(self):
        provider, model = parse_critic_spec("openai:ft:gpt-4o:custom")
        assert provider == "openai"
        assert model == "ft:gpt-4o:custom"


@pytest.fixture
def analysis():
    return {"score": 5, "what_worked": "baseline correction"}


@pytest.fixture
def base_kwargs(analysis):
    return {
        "critic_specs": ["openai:gpt-4o"],
        "analysis": analysis,
        "compressed_history": "v1: initial model",
        "notebook_content": "# Lab Notebook\nEntry 1",
    }


class TestRunDebate:
    @pytest.mark.asyncio
    async def test_empty_specs_returns_empty(self, analysis):
        result = await run_debate(
            critic_specs=[],
            analysis=analysis,
            compressed_history="",
            notebook_content="",
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_single_round_no_defender(self, base_kwargs):
        """With max_rounds=1, only the critic is called, no defender."""
        with (
            patch(
                "auto_scientist.agents.critic.query_openai",
                new_callable=AsyncMock,
                return_value="Initial critique",
            ) as mock_openai,
            patch(
                "auto_scientist.agents.critic.query_anthropic",
                new_callable=AsyncMock,
            ) as mock_anthropic,
        ):
            result = await run_debate(**base_kwargs, max_rounds=1)

        assert len(result) == 1
        assert result[0]["model"] == "openai:gpt-4o"
        assert result[0]["critique"] == "Initial critique"
        mock_openai.assert_called_once()
        mock_anthropic.assert_not_called()

    @pytest.mark.asyncio
    async def test_two_rounds_calls_defender_then_refines(self, base_kwargs):
        """With max_rounds=2, critic -> defender -> critic refinement."""
        with (
            patch(
                "auto_scientist.agents.critic.query_openai",
                new_callable=AsyncMock,
                side_effect=["Initial critique", "Refined critique"],
            ) as mock_openai,
            patch(
                "auto_scientist.agents.critic.query_anthropic",
                new_callable=AsyncMock,
                return_value="Defense response",
            ) as mock_anthropic,
        ):
            result = await run_debate(**base_kwargs, max_rounds=2)

        assert len(result) == 1
        assert result[0]["critique"] == "Refined critique"
        assert mock_openai.call_count == 2
        mock_anthropic.assert_called_once()

        # Defender prompt should contain the initial critique
        defender_prompt = mock_anthropic.call_args[0][1]
        assert "Initial critique" in defender_prompt

        # Refinement prompt should contain the defense
        refinement_prompt = mock_openai.call_args_list[1][0][1]
        assert "Defense response" in refinement_prompt

    @pytest.mark.asyncio
    async def test_three_rounds(self, base_kwargs):
        """With max_rounds=3, two defense-refinement cycles."""
        with (
            patch(
                "auto_scientist.agents.critic.query_openai",
                new_callable=AsyncMock,
                side_effect=["Critique R1", "Critique R2", "Critique R3"],
            ) as mock_openai,
            patch(
                "auto_scientist.agents.critic.query_anthropic",
                new_callable=AsyncMock,
                side_effect=["Defense R1", "Defense R2"],
            ) as mock_anthropic,
        ):
            result = await run_debate(**base_kwargs, max_rounds=3)

        assert result[0]["critique"] == "Critique R3"
        assert mock_openai.call_count == 3
        assert mock_anthropic.call_count == 2

    @pytest.mark.asyncio
    async def test_multiple_critics(self, analysis):
        """Each critic runs its own independent debate."""
        with (
            patch(
                "auto_scientist.agents.critic.query_openai",
                new_callable=AsyncMock,
                side_effect=["OAI initial", "OAI refined"],
            ),
            patch(
                "auto_scientist.agents.critic.query_google",
                new_callable=AsyncMock,
                side_effect=["Google initial", "Google refined"],
            ),
            patch(
                "auto_scientist.agents.critic.query_anthropic",
                new_callable=AsyncMock,
                side_effect=["Defense for OAI", "Defense for Google"],
            ),
        ):
            result = await run_debate(
                critic_specs=["openai:gpt-4o", "google:gemini-2.5-pro"],
                analysis=analysis,
                compressed_history="",
                notebook_content="",
                max_rounds=2,
            )

        assert len(result) == 2
        assert result[0] == {"model": "openai:gpt-4o", "critique": "OAI refined"}
        assert result[1] == {"model": "google:gemini-2.5-pro", "critique": "Google refined"}

    @pytest.mark.asyncio
    async def test_script_content_in_defender_prompt(self, base_kwargs):
        """Defender prompt includes script content when provided."""
        with (
            patch(
                "auto_scientist.agents.critic.query_openai",
                new_callable=AsyncMock,
                side_effect=["Critique", "Refined"],
            ),
            patch(
                "auto_scientist.agents.critic.query_anthropic",
                new_callable=AsyncMock,
                return_value="Defense",
            ) as mock_anthropic,
        ):
            await run_debate(
                **base_kwargs,
                script_content="import numpy as np\n# experiment code",
                max_rounds=2,
            )

        defender_prompt = mock_anthropic.call_args[0][1]
        assert "import numpy as np" in defender_prompt

    @pytest.mark.asyncio
    async def test_custom_defender_model(self, base_kwargs):
        """Defender uses the specified model."""
        with (
            patch(
                "auto_scientist.agents.critic.query_openai",
                new_callable=AsyncMock,
                side_effect=["Critique", "Refined"],
            ),
            patch(
                "auto_scientist.agents.critic.query_anthropic",
                new_callable=AsyncMock,
                return_value="Defense",
            ) as mock_anthropic,
        ):
            await run_debate(
                **base_kwargs,
                max_rounds=2,
                defender_model="claude-haiku-4-5-20251001",
            )

        assert mock_anthropic.call_args[0][0] == "claude-haiku-4-5-20251001"

    @pytest.mark.asyncio
    async def test_unknown_provider_raises(self, analysis):
        with pytest.raises(ValueError, match="Unknown critic provider"):
            await run_debate(
                critic_specs=["unknown:model"],
                analysis=analysis,
                compressed_history="",
                notebook_content="",
            )


class TestRunCriticBackwardCompat:
    @pytest.mark.asyncio
    async def test_run_critic_calls_run_debate_with_rounds_1(self, analysis):
        """run_critic is equivalent to run_debate with max_rounds=1."""
        with patch(
            "auto_scientist.agents.critic.query_openai",
            new_callable=AsyncMock,
            return_value="Single-pass critique",
        ) as mock_openai, patch(
            "auto_scientist.agents.critic.query_anthropic",
            new_callable=AsyncMock,
        ) as mock_anthropic:
            result = await run_critic(
                critic_specs=["openai:gpt-4o"],
                analysis=analysis,
                compressed_history="history",
                notebook_content="notebook",
            )

        assert len(result) == 1
        assert result[0]["critique"] == "Single-pass critique"
        mock_openai.assert_called_once()
        mock_anthropic.assert_not_called()
