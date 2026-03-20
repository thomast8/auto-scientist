"""Tests for the summarizer module."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from auto_scientist.summarizer import (
    SUMMARY_PROMPTS,
    run_with_summaries,
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
    @patch("auto_scientist.summarizer._query_summary", new_callable=AsyncMock)
    async def test_ingestor_prompt(self, mock_query):
        mock_query.return_value = "Processing files"
        await summarize_agent_output("Ingestor", "raw output", "gpt-4o-mini")
        prompt_arg = mock_query.call_args[0][1]
        assert "Ingestor" in prompt_arg or "files" in prompt_arg.lower() or "transform" in prompt_arg.lower()

    @pytest.mark.asyncio
    @patch("auto_scientist.summarizer._query_summary", new_callable=AsyncMock)
    async def test_analyst_prompt(self, mock_query):
        mock_query.return_value = "Key metrics found"
        await summarize_agent_output("Analyst", "analysis data", "gpt-4o-mini")
        prompt_arg = mock_query.call_args[0][1]
        assert "metric" in prompt_arg.lower() or "finding" in prompt_arg.lower()

    @pytest.mark.asyncio
    @patch("auto_scientist.summarizer._query_summary", new_callable=AsyncMock)
    async def test_scientist_prompt(self, mock_query):
        mock_query.return_value = "Hypothesis formed"
        await summarize_agent_output("Scientist", "plan data", "gpt-4o-mini")
        prompt_arg = mock_query.call_args[0][1]
        assert "hypothes" in prompt_arg.lower() or "strategy" in prompt_arg.lower()

    @pytest.mark.asyncio
    @patch("auto_scientist.summarizer._query_summary", new_callable=AsyncMock)
    async def test_scientist_revision_prompt(self, mock_query):
        mock_query.return_value = "Plan revised"
        await summarize_agent_output("Scientist Revision", "revision data", "gpt-4o-mini")
        prompt_arg = mock_query.call_args[0][1]
        assert "chang" in prompt_arg.lower() or "revis" in prompt_arg.lower()

    @pytest.mark.asyncio
    @patch("auto_scientist.summarizer._query_summary", new_callable=AsyncMock)
    async def test_debate_prompt(self, mock_query):
        mock_query.return_value = "Challenge identified"
        await summarize_agent_output("Debate", "debate text", "gpt-4o-mini")
        prompt_arg = mock_query.call_args[0][1]
        assert "challeng" in prompt_arg.lower() or "position" in prompt_arg.lower()

    @pytest.mark.asyncio
    @patch("auto_scientist.summarizer._query_summary", new_callable=AsyncMock)
    async def test_coder_prompt(self, mock_query):
        mock_query.return_value = "Writing script"
        await summarize_agent_output("Coder", "code output", "gpt-4o-mini")
        prompt_arg = mock_query.call_args[0][1]
        assert "approach" in prompt_arg.lower() or "code" in prompt_arg.lower()

    @pytest.mark.asyncio
    @patch("auto_scientist.summarizer._query_summary", new_callable=AsyncMock)
    async def test_results_prompt(self, mock_query):
        mock_query.return_value = "Key outcomes"
        await summarize_agent_output("Results", "results text", "gpt-4o-mini")
        prompt_arg = mock_query.call_args[0][1]
        assert "metric" in prompt_arg.lower() or "outcome" in prompt_arg.lower()

    @pytest.mark.asyncio
    @patch("auto_scientist.summarizer._query_summary", new_callable=AsyncMock)
    async def test_report_prompt(self, mock_query):
        mock_query.return_value = "Report summary"
        await summarize_agent_output("Report", "report text", "gpt-4o-mini")
        prompt_arg = mock_query.call_args[0][1]
        assert "finding" in prompt_arg.lower() or "result" in prompt_arg.lower()

    @pytest.mark.asyncio
    @patch("auto_scientist.summarizer._query_summary", new_callable=AsyncMock)
    async def test_passes_model(self, mock_query):
        mock_query.return_value = "summary"
        await summarize_agent_output("Analyst", "data", "gpt-4o-mini")
        assert mock_query.call_args[0][0] == "gpt-4o-mini"

    @pytest.mark.asyncio
    @patch("auto_scientist.summarizer._query_summary", new_callable=AsyncMock)
    async def test_returns_response(self, mock_query):
        mock_query.return_value = "the summary"
        result = await summarize_agent_output("Analyst", "data", "gpt-4o-mini")
        assert result == "the summary"

    @pytest.mark.asyncio
    @patch("auto_scientist.summarizer._query_summary", new_callable=AsyncMock)
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
    @patch("auto_scientist.summarizer._query_summary", new_callable=AsyncMock)
    async def test_uses_progress_prefix(self, mock_query):
        mock_query.return_value = "in progress"
        await summarize_agent_output("Analyst", "data", "gpt-4o-mini", progress=True)
        instructions_arg = mock_query.call_args[0][1]
        assert "currently" in instructions_arg.lower()

    @pytest.mark.asyncio
    @patch("auto_scientist.summarizer._query_summary", new_callable=AsyncMock)
    async def test_uses_final_prefix(self, mock_query):
        mock_query.return_value = "done"
        await summarize_agent_output("Analyst", "data", "gpt-4o-mini", progress=False)
        instructions_arg = mock_query.call_args[0][1]
        assert "accomplish" in instructions_arg.lower()


class TestSummarizeResults:
    @pytest.mark.asyncio
    @patch("auto_scientist.summarizer._query_summary", new_callable=AsyncMock)
    async def test_prompt_focuses_on_metrics(self, mock_query):
        mock_query.return_value = "R2=0.82"
        await summarize_results("R2=0.82, RMSE=0.15", "gpt-4o-mini")
        prompt_arg = mock_query.call_args[0][1]
        assert "numeric" in prompt_arg.lower() or "outcome" in prompt_arg.lower()

    @pytest.mark.asyncio
    @patch("auto_scientist.summarizer._query_summary", new_callable=AsyncMock)
    async def test_returns_response(self, mock_query):
        mock_query.return_value = "good results"
        result = await summarize_results("R2=0.82", "gpt-4o-mini")
        assert result == "good results"

    @pytest.mark.asyncio
    @patch("auto_scientist.summarizer._query_summary", new_callable=AsyncMock)
    async def test_exception_returns_empty(self, mock_query):
        mock_query.side_effect = RuntimeError("API error")
        result = await summarize_results("data", "gpt-4o-mini")
        assert result == ""

    @pytest.mark.asyncio
    @patch("auto_scientist.summarizer._query_summary", new_callable=AsyncMock)
    async def test_passes_model(self, mock_query):
        mock_query.return_value = "summary"
        await summarize_results("data", "gpt-4o-mini")
        assert mock_query.call_args[0][0] == "gpt-4o-mini"


class TestRunWithSummaries:
    @pytest.mark.asyncio
    async def test_periodic_polls_fire(self):
        """Fake coroutine that takes ~0.6s with 0.2s interval should get >= 2 polls."""
        buf: list[str] = []

        async def slow_coro(message_buffer):
            for i in range(3):
                message_buffer.append(f"chunk {i}")
                await asyncio.sleep(0.2)
            return "done"

        with (
            patch("auto_scientist.summarizer.summarize_agent_output", new_callable=AsyncMock, return_value="progress"),
            patch("auto_scientist.summarizer.print_summary") as mock_print,
        ):
            result = await run_with_summaries(
                slow_coro, "Analyst", "gpt-4o-mini", buf, interval=0.2,
            )
            assert result == "done"
            periodic_calls = [
                c for c in mock_print.call_args_list
                if c[1].get("label") != "done"
            ]
            assert len(periodic_calls) >= 2

    @pytest.mark.asyncio
    async def test_final_summary_printed(self):
        buf: list[str] = []

        async def fast_coro(message_buffer):
            message_buffer.append("output")
            return 42

        with (
            patch("auto_scientist.summarizer.summarize_agent_output", new_callable=AsyncMock, return_value="final result"),
            patch("auto_scientist.summarizer.print_summary") as mock_print,
        ):
            result = await run_with_summaries(
                fast_coro, "Coder", "gpt-4o-mini", buf, interval=100,
            )
            assert result == 42
            final_calls = [
                c for c in mock_print.call_args_list
                if c[1].get("label") == "done"
            ]
            assert len(final_calls) == 1

    @pytest.mark.asyncio
    async def test_empty_buffer_skips_poll(self):
        """If nothing is added to buffer, periodic summarizer should not be called."""
        buf: list[str] = []

        async def empty_coro(message_buffer):
            await asyncio.sleep(0.3)
            return "done"

        with (
            patch("auto_scientist.summarizer.summarize_agent_output", new_callable=AsyncMock, return_value="summary") as mock_summarize,
            patch("auto_scientist.summarizer.print_summary"),
        ):
            await run_with_summaries(
                empty_coro, "Analyst", "gpt-4o-mini", buf, interval=0.1,
            )
            periodic_calls = [
                c for c in mock_summarize.call_args_list
                if c[1].get("progress") is True
            ]
            assert len(periodic_calls) == 0

    @pytest.mark.asyncio
    async def test_incremental_only_new_content(self):
        """Each periodic poll should only send new entries since last poll."""
        captured_outputs = []

        async def capture_summarize(agent, output, model, **kwargs):
            captured_outputs.append(output)
            return "summary"

        buf: list[str] = []

        async def staged_coro(message_buffer):
            message_buffer.append("first")
            await asyncio.sleep(0.25)
            message_buffer.append("second")
            await asyncio.sleep(0.25)
            return "done"

        with (
            patch("auto_scientist.summarizer.summarize_agent_output", new_callable=AsyncMock, side_effect=capture_summarize),
            patch("auto_scientist.summarizer.print_summary"),
        ):
            await run_with_summaries(
                staged_coro, "Analyst", "gpt-4o-mini", buf, interval=0.2,
            )
            progress_outputs = [
                o for i, o in enumerate(captured_outputs)
                if i < len(captured_outputs) - 1
            ]
            if len(progress_outputs) >= 2:
                assert "first" not in progress_outputs[1]

    @pytest.mark.asyncio
    async def test_exception_in_coro_still_cleans_up(self):
        buf: list[str] = []

        async def failing_coro(message_buffer):
            message_buffer.append("before crash")
            raise ValueError("boom")

        with (
            patch("auto_scientist.summarizer.summarize_agent_output", new_callable=AsyncMock, return_value="summary"),
            patch("auto_scientist.summarizer.print_summary"),
        ):
            with pytest.raises(ValueError, match="boom"):
                await run_with_summaries(
                    failing_coro, "Coder", "gpt-4o-mini", buf, interval=100,
                )

    @pytest.mark.asyncio
    async def test_summary_failure_does_not_crash(self):
        """If summarizer raises, the coroutine should still complete."""
        buf: list[str] = []

        async def coro(message_buffer):
            message_buffer.append("data")
            return "result"

        with (
            patch("auto_scientist.summarizer.summarize_agent_output", new_callable=AsyncMock, side_effect=RuntimeError("API down")),
            patch("auto_scientist.summarizer.print_summary"),
        ):
            result = await run_with_summaries(
                coro, "Analyst", "gpt-4o-mini", buf, interval=100,
            )
            assert result == "result"
