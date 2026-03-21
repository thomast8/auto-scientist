"""Tests for the critic debate loop."""

from unittest.mock import AsyncMock, patch

import pytest

from auto_scientist.agents.critic import (
    _build_critic_prompt,
    parse_critic_spec,
    run_critic,
    run_debate,
)


class TestParseCriticSpec:
    def test_valid_openai(self):
        assert parse_critic_spec("openai:gpt-4o") == ("openai", "gpt-4o")

    def test_valid_google(self):
        assert parse_critic_spec("google:gemini-2.0-flash") == ("google", "gemini-2.0-flash")

    def test_valid_anthropic(self):
        assert parse_critic_spec("anthropic:claude-sonnet-4-6") == (
            "anthropic",
            "claude-sonnet-4-6",
        )

    def test_model_with_colons(self):
        provider, model = parse_critic_spec("openai:ft:gpt-4o:org:id")
        assert provider == "openai"
        assert model == "ft:gpt-4o:org:id"

    def test_no_colon_raises(self):
        with pytest.raises(ValueError, match="Invalid critic spec"):
            parse_critic_spec("gpt4o")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            parse_critic_spec("")


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
def base_kwargs(plan):
    return {
        "critic_specs": ["openai:gpt-4o"],
        "plan": plan,
        "notebook_content": "# Lab Notebook\nEntry 1",
    }


class TestRunDebate:
    @pytest.mark.asyncio
    async def test_empty_specs_returns_empty(self, plan):
        result = await run_debate(
            critic_specs=[],
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
        assert len(result[0]["transcript"]) == 1
        assert result[0]["transcript"][0]["role"] == "critic"
        mock_openai.assert_called_once()
        mock_anthropic.assert_not_called()

    @pytest.mark.asyncio
    async def test_two_rounds_calls_scientist_then_refines(self, base_kwargs):
        """With max_rounds=2, critic -> scientist -> critic refinement."""
        with (
            patch(
                "auto_scientist.agents.critic.query_openai",
                new_callable=AsyncMock,
                side_effect=["Initial critique", "Refined critique"],
            ) as mock_openai,
            patch(
                "auto_scientist.agents.critic.query_anthropic",
                new_callable=AsyncMock,
                return_value="Scientist response",
            ) as mock_anthropic,
        ):
            result = await run_debate(**base_kwargs, max_rounds=2)

        assert len(result) == 1
        assert result[0]["critique"] == "Refined critique"
        assert mock_openai.call_count == 2
        mock_anthropic.assert_called_once()

        # Scientist prompt should contain the initial critique
        scientist_prompt = mock_anthropic.call_args[0][1]
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
                side_effect=["Critique R1", "Critique R2", "Critique R3"],
            ) as mock_openai,
            patch(
                "auto_scientist.agents.critic.query_anthropic",
                new_callable=AsyncMock,
                side_effect=["Scientist R1", "Scientist R2"],
            ) as mock_anthropic,
        ):
            result = await run_debate(**base_kwargs, max_rounds=3)

        assert result[0]["critique"] == "Critique R3"
        assert mock_openai.call_count == 3
        assert mock_anthropic.call_count == 2

    @pytest.mark.asyncio
    async def test_debate_returns_transcript(self, base_kwargs):
        """Debate returns transcript with all rounds."""
        with (
            patch(
                "auto_scientist.agents.critic.query_openai",
                new_callable=AsyncMock,
                side_effect=["Critique R1", "Critique R2"],
            ),
            patch(
                "auto_scientist.agents.critic.query_anthropic",
                new_callable=AsyncMock,
                return_value="Scientist response",
            ),
        ):
            result = await run_debate(**base_kwargs, max_rounds=2)

        assert len(result) == 1
        transcript = result[0]["transcript"]
        assert len(transcript) == 3  # critic, scientist, critic
        assert transcript[0]["role"] == "critic"
        assert transcript[0]["content"] == "Critique R1"
        assert transcript[1]["role"] == "scientist"
        assert transcript[1]["content"] == "Scientist response"
        assert transcript[2]["role"] == "critic"
        assert transcript[2]["content"] == "Critique R2"

    @pytest.mark.asyncio
    async def test_multiple_critics(self, plan):
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
                side_effect=["Scientist for OAI", "Scientist for Google"],
            ),
        ):
            result = await run_debate(
                critic_specs=["openai:gpt-4o", "google:gemini-2.5-pro"],
                plan=plan,
                notebook_content="",
                max_rounds=2,
            )

        assert len(result) == 2
        assert result[0]["model"] == "openai:gpt-4o"
        assert result[0]["critique"] == "OAI refined"
        assert result[1]["model"] == "google:gemini-2.5-pro"
        assert result[1]["critique"] == "Google refined"

    @pytest.mark.asyncio
    async def test_plan_in_critic_prompt(self, base_kwargs):
        """Critic prompt includes the scientist's plan."""
        with (
            patch(
                "auto_scientist.agents.critic.query_openai",
                new_callable=AsyncMock,
                return_value="Critique of plan",
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
                side_effect=["Critique", "Refined"],
            ),
            patch(
                "auto_scientist.agents.critic.query_anthropic",
                new_callable=AsyncMock,
                return_value="Response",
            ) as mock_anthropic,
        ):
            await run_debate(**base_kwargs, max_rounds=2)

        scientist_prompt = mock_anthropic.call_args[0][1]
        assert "<plan>" in scientist_prompt
        assert "Adjusting learning rate" in scientist_prompt

    @pytest.mark.asyncio
    async def test_criteria_in_critic_prompt(self, base_kwargs):
        """Critic prompt includes success criteria from the plan."""
        with patch(
            "auto_scientist.agents.critic.query_openai",
            new_callable=AsyncMock,
            return_value="Critique",
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
                side_effect=["Critique", "Refined"],
            ) as mock_openai,
            patch(
                "auto_scientist.agents.critic.query_anthropic",
                new_callable=AsyncMock,
                return_value="Response",
            ) as mock_anthropic,
        ):
            await run_debate(**base_kwargs, max_rounds=2)

        critic_prompt = mock_openai.call_args_list[0][0][1]
        scientist_prompt = mock_anthropic.call_args[0][1]

        # Neither side should see "Latest Analysis" or "Current Script" sections
        assert "Latest Analysis" not in critic_prompt
        assert "Current Script" not in scientist_prompt

    @pytest.mark.asyncio
    async def test_web_search_enabled(self, base_kwargs):
        """Critic and scientist calls pass web_search=True."""
        with (
            patch(
                "auto_scientist.agents.critic.query_openai",
                new_callable=AsyncMock,
                side_effect=["Critique", "Refined"],
            ) as mock_openai,
            patch(
                "auto_scientist.agents.critic.query_anthropic",
                new_callable=AsyncMock,
                return_value="Response",
            ) as mock_anthropic,
        ):
            await run_debate(**base_kwargs, max_rounds=2)

        # Critic calls should have web_search=True
        for call in mock_openai.call_args_list:
            assert call.kwargs.get("web_search") is True

        # Scientist call should have web_search=True
        assert mock_anthropic.call_args.kwargs.get("web_search") is True

    @pytest.mark.asyncio
    async def test_symmetric_context(self, base_kwargs):
        """Critic and scientist receive the same context (symmetric)."""
        with (
            patch(
                "auto_scientist.agents.critic.query_openai",
                new_callable=AsyncMock,
                side_effect=["Critique", "Refined"],
            ) as mock_openai,
            patch(
                "auto_scientist.agents.critic.query_anthropic",
                new_callable=AsyncMock,
                return_value="Response",
            ) as mock_anthropic,
        ):
            await run_debate(**base_kwargs, max_rounds=2)

        critic_prompt = mock_openai.call_args_list[0][0][1]
        scientist_prompt = mock_anthropic.call_args[0][1]

        # Both should see notebook and domain knowledge
        assert "<notebook>" in critic_prompt
        assert "<notebook>" in scientist_prompt

    @pytest.mark.asyncio
    async def test_no_compressed_history_in_prompts(self, base_kwargs):
        """Prompts should not contain compressed history sections."""
        with (
            patch(
                "auto_scientist.agents.critic.query_openai",
                new_callable=AsyncMock,
                side_effect=["Critique", "Refined"],
            ) as mock_openai,
            patch(
                "auto_scientist.agents.critic.query_anthropic",
                new_callable=AsyncMock,
                return_value="Response",
            ) as mock_anthropic,
        ):
            await run_debate(**base_kwargs, max_rounds=2)

        critic_prompt = mock_openai.call_args_list[0][0][1]
        scientist_prompt = mock_anthropic.call_args[0][1]

        assert "Experiment History" not in critic_prompt
        assert "Experiment History" not in scientist_prompt

    @pytest.mark.asyncio
    async def test_custom_scientist_model(self, base_kwargs):
        """Scientist uses the specified model."""
        with (
            patch(
                "auto_scientist.agents.critic.query_openai",
                new_callable=AsyncMock,
                side_effect=["Critique", "Refined"],
            ),
            patch(
                "auto_scientist.agents.critic.query_anthropic",
                new_callable=AsyncMock,
                return_value="Response",
            ) as mock_anthropic,
        ):
            await run_debate(
                **base_kwargs,
                max_rounds=2,
                scientist_model="claude-haiku-4-5-20251001",
            )

        assert mock_anthropic.call_args[0][0] == "claude-haiku-4-5-20251001"

    @pytest.mark.asyncio
    async def test_unknown_provider_raises(self, plan):
        with pytest.raises(ValueError, match="Unknown critic provider"):
            await run_debate(
                critic_specs=["unknown:model"],
                plan=plan,
                notebook_content="",
            )

    @pytest.mark.asyncio
    async def test_anthropic_critic_dispatches_correctly(self, plan):
        with patch(
            "auto_scientist.agents.critic.query_anthropic",
            new_callable=AsyncMock,
            return_value="Anthropic critique",
        ) as mock_anthropic:
            result = await run_debate(
                critic_specs=["anthropic:claude-sonnet-4-6"],
                plan=plan,
                notebook_content="",
                max_rounds=1,
            )

        assert len(result) == 1
        assert result[0]["critique"] == "Anthropic critique"
        # query_anthropic called for the critic (not scientist response)
        assert mock_anthropic.call_count == 1


class TestRunDebateStreaming:
    @pytest.mark.asyncio
    async def test_on_token_factory_called_with_labels(self, base_kwargs):
        """Factory is called with correct labels for critic and scientist."""
        labels_seen = []

        def factory(label):
            labels_seen.append(label)
            return lambda token: None

        with (
            patch(
                "auto_scientist.agents.critic.query_openai",
                new_callable=AsyncMock,
                side_effect=["Critique R1", "Critique R2"],
            ),
            patch(
                "auto_scientist.agents.critic.query_anthropic",
                new_callable=AsyncMock,
                return_value="Scientist response",
            ),
        ):
            await run_debate(**base_kwargs, max_rounds=2, on_token_factory=factory)

        assert len(labels_seen) == 3  # critic R1, scientist R1, critic R2
        assert "Critic" in labels_seen[0]
        assert "Scientist" in labels_seen[1]
        assert "Critic" in labels_seen[2]

    @pytest.mark.asyncio
    async def test_on_token_passed_to_model_clients(self, base_kwargs):
        """The callback from factory is passed as on_token to model clients."""
        def mock_callback(token):
            pass

        def factory(label):
            return mock_callback

        with (
            patch(
                "auto_scientist.agents.critic.query_openai",
                new_callable=AsyncMock,
                return_value="Critique",
            ) as mock_openai,
            patch(
                "auto_scientist.agents.critic.query_anthropic",
                new_callable=AsyncMock,
                return_value="Response",
            ) as mock_anthropic,
        ):
            await run_debate(**base_kwargs, max_rounds=2, on_token_factory=factory)

        # Verify on_token was passed to both model clients
        for call in mock_openai.call_args_list:
            assert call.kwargs.get("on_token") is mock_callback
        assert mock_anthropic.call_args.kwargs.get("on_token") is mock_callback

    @pytest.mark.asyncio
    async def test_no_factory_passes_none(self, base_kwargs):
        """Without a factory, on_token=None is passed (or not passed)."""
        with (
            patch(
                "auto_scientist.agents.critic.query_openai",
                new_callable=AsyncMock,
                return_value="Critique",
            ) as mock_openai,
        ):
            await run_debate(**base_kwargs, max_rounds=1)

        assert mock_openai.call_args.kwargs.get("on_token") is None


class TestRunCriticBackwardCompat:
    @pytest.mark.asyncio
    async def test_run_critic_calls_run_debate_with_rounds_1(self, plan):
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
                plan=plan,
                notebook_content="notebook",
            )

        assert len(result) == 1
        assert result[0]["critique"] == "Single-pass critique"
        mock_openai.assert_called_once()
        mock_anthropic.assert_not_called()
