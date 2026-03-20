"""Tests for the summarizer module."""

from unittest.mock import AsyncMock, patch

import pytest

from auto_scientist.summarizer import (
    SUMMARY_PROMPTS,
    summarize_agent_output,
    summarize_results,
)


class TestSummaryPrompts:
    def test_all_agent_types_present(self):
        expected = {
            "Ingestor", "Analyst", "Scientist", "Scientist Revision",
            "Debate", "Coder", "Results", "Report",
        }
        assert set(SUMMARY_PROMPTS.keys()) == expected

    def test_prompts_are_nonempty_strings(self):
        for name, prompt in SUMMARY_PROMPTS.items():
            assert isinstance(prompt, str), f"{name} prompt is not a string"
            assert len(prompt) > 10, f"{name} prompt is too short"


class TestSummarizeAgentOutput:
    @pytest.mark.asyncio
    @patch("auto_scientist.summarizer.query_openai", new_callable=AsyncMock)
    async def test_ingestor_prompt(self, mock_query):
        mock_query.return_value = "Processing files"
        await summarize_agent_output("Ingestor", "raw output", "gpt-4o-mini")
        prompt_arg = mock_query.call_args[0][1]
        assert "Ingestor" in prompt_arg or "files" in prompt_arg.lower() or "transform" in prompt_arg.lower()

    @pytest.mark.asyncio
    @patch("auto_scientist.summarizer.query_openai", new_callable=AsyncMock)
    async def test_analyst_prompt(self, mock_query):
        mock_query.return_value = "Key metrics found"
        await summarize_agent_output("Analyst", "analysis data", "gpt-4o-mini")
        prompt_arg = mock_query.call_args[0][1]
        assert "metric" in prompt_arg.lower() or "finding" in prompt_arg.lower()

    @pytest.mark.asyncio
    @patch("auto_scientist.summarizer.query_openai", new_callable=AsyncMock)
    async def test_scientist_prompt(self, mock_query):
        mock_query.return_value = "Hypothesis formed"
        await summarize_agent_output("Scientist", "plan data", "gpt-4o-mini")
        prompt_arg = mock_query.call_args[0][1]
        assert "hypothes" in prompt_arg.lower() or "strategy" in prompt_arg.lower()

    @pytest.mark.asyncio
    @patch("auto_scientist.summarizer.query_openai", new_callable=AsyncMock)
    async def test_scientist_revision_prompt(self, mock_query):
        mock_query.return_value = "Plan revised"
        await summarize_agent_output("Scientist Revision", "revision data", "gpt-4o-mini")
        prompt_arg = mock_query.call_args[0][1]
        assert "chang" in prompt_arg.lower() or "revis" in prompt_arg.lower()

    @pytest.mark.asyncio
    @patch("auto_scientist.summarizer.query_openai", new_callable=AsyncMock)
    async def test_debate_prompt(self, mock_query):
        mock_query.return_value = "Challenge identified"
        await summarize_agent_output("Debate", "debate text", "gpt-4o-mini")
        prompt_arg = mock_query.call_args[0][1]
        assert "challeng" in prompt_arg.lower() or "position" in prompt_arg.lower()

    @pytest.mark.asyncio
    @patch("auto_scientist.summarizer.query_openai", new_callable=AsyncMock)
    async def test_coder_prompt(self, mock_query):
        mock_query.return_value = "Writing script"
        await summarize_agent_output("Coder", "code output", "gpt-4o-mini")
        prompt_arg = mock_query.call_args[0][1]
        assert "approach" in prompt_arg.lower() or "code" in prompt_arg.lower()

    @pytest.mark.asyncio
    @patch("auto_scientist.summarizer.query_openai", new_callable=AsyncMock)
    async def test_results_prompt(self, mock_query):
        mock_query.return_value = "Key outcomes"
        await summarize_agent_output("Results", "results text", "gpt-4o-mini")
        prompt_arg = mock_query.call_args[0][1]
        assert "metric" in prompt_arg.lower() or "outcome" in prompt_arg.lower()

    @pytest.mark.asyncio
    @patch("auto_scientist.summarizer.query_openai", new_callable=AsyncMock)
    async def test_report_prompt(self, mock_query):
        mock_query.return_value = "Report summary"
        await summarize_agent_output("Report", "report text", "gpt-4o-mini")
        prompt_arg = mock_query.call_args[0][1]
        assert "finding" in prompt_arg.lower() or "result" in prompt_arg.lower()

    @pytest.mark.asyncio
    @patch("auto_scientist.summarizer.query_openai", new_callable=AsyncMock)
    async def test_max_tokens_150(self, mock_query):
        mock_query.return_value = "summary"
        await summarize_agent_output("Analyst", "data", "gpt-4o-mini")
        assert mock_query.call_args.kwargs["max_tokens"] == 150

    @pytest.mark.asyncio
    @patch("auto_scientist.summarizer.query_openai", new_callable=AsyncMock)
    async def test_returns_response(self, mock_query):
        mock_query.return_value = "the summary"
        result = await summarize_agent_output("Analyst", "data", "gpt-4o-mini")
        assert result == "the summary"

    @pytest.mark.asyncio
    @patch("auto_scientist.summarizer.query_openai", new_callable=AsyncMock)
    async def test_exception_returns_empty(self, mock_query):
        mock_query.side_effect = RuntimeError("API error")
        result = await summarize_agent_output("Analyst", "data", "gpt-4o-mini")
        assert result == ""

    @pytest.mark.asyncio
    async def test_none_output_returns_empty(self):
        result = await summarize_agent_output("Analyst", None, "gpt-4o-mini")
        assert result == ""

    @pytest.mark.asyncio
    async def test_empty_output_returns_empty(self):
        result = await summarize_agent_output("Analyst", "", "gpt-4o-mini")
        assert result == ""

    @pytest.mark.asyncio
    @patch("auto_scientist.summarizer.query_openai", new_callable=AsyncMock)
    async def test_uses_progress_prefix(self, mock_query):
        mock_query.return_value = "in progress"
        await summarize_agent_output("Analyst", "data", "gpt-4o-mini", progress=True)
        prompt_arg = mock_query.call_args[0][1]
        assert "currently" in prompt_arg.lower() or "doing" in prompt_arg.lower()

    @pytest.mark.asyncio
    @patch("auto_scientist.summarizer.query_openai", new_callable=AsyncMock)
    async def test_uses_final_prefix(self, mock_query):
        mock_query.return_value = "done"
        await summarize_agent_output("Analyst", "data", "gpt-4o-mini", progress=False)
        prompt_arg = mock_query.call_args[0][1]
        assert "accomplish" in prompt_arg.lower() or "did" in prompt_arg.lower()


class TestSummarizeResults:
    @pytest.mark.asyncio
    @patch("auto_scientist.summarizer.query_openai", new_callable=AsyncMock)
    async def test_prompt_focuses_on_metrics(self, mock_query):
        mock_query.return_value = "R2=0.82"
        await summarize_results("R2=0.82, RMSE=0.15", "gpt-4o-mini")
        prompt_arg = mock_query.call_args[0][1]
        assert "numeric" in prompt_arg.lower() or "outcome" in prompt_arg.lower()

    @pytest.mark.asyncio
    @patch("auto_scientist.summarizer.query_openai", new_callable=AsyncMock)
    async def test_returns_response(self, mock_query):
        mock_query.return_value = "good results"
        result = await summarize_results("R2=0.82", "gpt-4o-mini")
        assert result == "good results"

    @pytest.mark.asyncio
    @patch("auto_scientist.summarizer.query_openai", new_callable=AsyncMock)
    async def test_exception_returns_empty(self, mock_query):
        mock_query.side_effect = RuntimeError("API error")
        result = await summarize_results("data", "gpt-4o-mini")
        assert result == ""

    @pytest.mark.asyncio
    @patch("auto_scientist.summarizer.query_openai", new_callable=AsyncMock)
    async def test_max_tokens_150(self, mock_query):
        mock_query.return_value = "summary"
        await summarize_results("data", "gpt-4o-mini")
        assert mock_query.call_args.kwargs["max_tokens"] == 150
