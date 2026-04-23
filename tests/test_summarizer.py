"""Tests for the summarizer module."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from auto_core.summarizer import (
    SUMMARY_PROMPTS,
    run_with_summaries,
    summarize_agent_output,
    summarize_results,
)


class TestSummaryPrompts:
    def test_all_agent_types_present(self):
        expected = {
            "Ingestor",
            "Analyst",
            "Scientist",
            "Scientist Revision",
            "Debate",
            "Coder",
            "Results",
            "Report",
            "Completeness Assessment",
            "Stop Debate",
            "Stop Revision",
        }
        assert set(SUMMARY_PROMPTS.keys()) == expected

    def test_prompts_are_nonempty_strings(self):
        for name, prompt in SUMMARY_PROMPTS.items():
            assert isinstance(prompt, str), f"{name} prompt is not a string"
            assert len(prompt) > 10, f"{name} prompt is too short"


class TestSummarizeAgentOutput:
    @pytest.mark.asyncio
    @patch("auto_core.summarizer._query_summary", new_callable=AsyncMock)
    async def test_ingestor_prompt(self, mock_query):
        mock_query.return_value = "Processing files"
        await summarize_agent_output("Ingestor", "raw output", "gpt-4o-mini")
        prompt_arg = mock_query.call_args[0][1]
        assert (
            "Ingestor" in prompt_arg
            or "files" in prompt_arg.lower()
            or "transform" in prompt_arg.lower()
        )

    @pytest.mark.asyncio
    @patch("auto_core.summarizer._query_summary", new_callable=AsyncMock)
    async def test_analyst_prompt(self, mock_query):
        mock_query.return_value = "Key metrics found"
        await summarize_agent_output("Analyst", "analysis data", "gpt-4o-mini")
        prompt_arg = mock_query.call_args[0][1]
        assert "metric" in prompt_arg.lower() or "finding" in prompt_arg.lower()

    @pytest.mark.asyncio
    @patch("auto_core.summarizer._query_summary", new_callable=AsyncMock)
    async def test_scientist_prompt(self, mock_query):
        mock_query.return_value = "Hypothesis formed"
        await summarize_agent_output("Scientist", "plan data", "gpt-4o-mini")
        prompt_arg = mock_query.call_args[0][1]
        assert "hypothes" in prompt_arg.lower() or "strategy" in prompt_arg.lower()

    @pytest.mark.asyncio
    @patch("auto_core.summarizer._query_summary", new_callable=AsyncMock)
    async def test_scientist_revision_prompt(self, mock_query):
        mock_query.return_value = "Plan revised"
        await summarize_agent_output("Scientist Revision", "revision data", "gpt-4o-mini")
        prompt_arg = mock_query.call_args[0][1]
        assert "chang" in prompt_arg.lower() or "revis" in prompt_arg.lower()

    @pytest.mark.asyncio
    @patch("auto_core.summarizer._query_summary", new_callable=AsyncMock)
    async def test_debate_prompt(self, mock_query):
        mock_query.return_value = "Challenge identified"
        await summarize_agent_output("Debate", "debate text", "gpt-4o-mini")
        prompt_arg = mock_query.call_args[0][1]
        assert "challeng" in prompt_arg.lower() or "position" in prompt_arg.lower()

    @pytest.mark.asyncio
    @patch("auto_core.summarizer._query_summary", new_callable=AsyncMock)
    async def test_coder_prompt(self, mock_query):
        mock_query.return_value = "Writing script"
        await summarize_agent_output("Coder", "code output", "gpt-4o-mini")
        prompt_arg = mock_query.call_args[0][1]
        assert "approach" in prompt_arg.lower() or "code" in prompt_arg.lower()

    @pytest.mark.asyncio
    @patch("auto_core.summarizer._query_summary", new_callable=AsyncMock)
    async def test_results_prompt(self, mock_query):
        mock_query.return_value = "Key outcomes"
        await summarize_agent_output("Results", "results text", "gpt-4o-mini")
        prompt_arg = mock_query.call_args[0][1]
        assert "metric" in prompt_arg.lower() or "outcome" in prompt_arg.lower()

    @pytest.mark.asyncio
    @patch("auto_core.summarizer._query_summary", new_callable=AsyncMock)
    async def test_report_prompt(self, mock_query):
        mock_query.return_value = "Report summary"
        await summarize_agent_output("Report", "report text", "gpt-4o-mini")
        prompt_arg = mock_query.call_args[0][1]
        assert "finding" in prompt_arg.lower() or "result" in prompt_arg.lower()

    @pytest.mark.asyncio
    @patch("auto_core.summarizer._query_summary", new_callable=AsyncMock)
    async def test_passes_model(self, mock_query):
        mock_query.return_value = "summary"
        await summarize_agent_output("Analyst", "data", "gpt-4o-mini")
        assert mock_query.call_args[0][0] == "gpt-4o-mini"

    @pytest.mark.asyncio
    @patch("auto_core.summarizer._query_summary", new_callable=AsyncMock)
    async def test_returns_response(self, mock_query):
        mock_query.return_value = "the summary"
        result = await summarize_agent_output("Analyst", "data", "gpt-4o-mini")
        assert result == "the summary"

    @pytest.mark.asyncio
    @patch("auto_core.summarizer._query_summary", new_callable=AsyncMock)
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
    @patch("auto_core.summarizer._query_summary", new_callable=AsyncMock)
    async def test_uses_progress_prefix(self, mock_query):
        mock_query.return_value = "in progress"
        await summarize_agent_output("Analyst", "data", "gpt-4o-mini", progress=True)
        instructions_arg = mock_query.call_args[0][1]
        assert "-ing" in instructions_arg

    @pytest.mark.asyncio
    @patch("auto_core.summarizer._query_summary", new_callable=AsyncMock)
    async def test_uses_final_prefix(self, mock_query):
        mock_query.return_value = "done"
        await summarize_agent_output("Analyst", "data", "gpt-4o-mini", progress=False)
        instructions_arg = mock_query.call_args[0][1]
        assert "past tense" in instructions_arg.lower()


class TestSummarizeResults:
    @pytest.mark.asyncio
    @patch("auto_core.summarizer._query_summary", new_callable=AsyncMock)
    async def test_prompt_focuses_on_metrics(self, mock_query):
        mock_query.return_value = "R2=0.82"
        await summarize_results("R2=0.82, RMSE=0.15", "gpt-4o-mini")
        prompt_arg = mock_query.call_args[0][1]
        assert "numeric" in prompt_arg.lower() or "outcome" in prompt_arg.lower()

    @pytest.mark.asyncio
    @patch("auto_core.summarizer._query_summary", new_callable=AsyncMock)
    async def test_returns_response(self, mock_query):
        mock_query.return_value = "good results"
        result = await summarize_results("R2=0.82", "gpt-4o-mini")
        assert result == "good results"

    @pytest.mark.asyncio
    @patch("auto_core.summarizer._query_summary", new_callable=AsyncMock)
    async def test_exception_returns_empty(self, mock_query):
        mock_query.side_effect = RuntimeError("API error")
        result = await summarize_results("data", "gpt-4o-mini")
        assert result == ""

    @pytest.mark.asyncio
    @patch("auto_core.summarizer._query_summary", new_callable=AsyncMock)
    async def test_passes_model(self, mock_query):
        mock_query.return_value = "summary"
        await summarize_results("data", "gpt-4o-mini")
        assert mock_query.call_args[0][0] == "gpt-4o-mini"


class TestRunWithSummaries:
    @pytest.mark.asyncio
    async def test_periodic_polls_fire(self):
        """Fake coroutine that takes ~0.6s with 0.2s interval should get >= 2 polls."""
        buf: list[str] = []
        collector: list[tuple[str, str, str]] = []

        async def slow_coro(message_buffer):
            for i in range(3):
                message_buffer.append(f"chunk {i}")
                await asyncio.sleep(0.2)
            return "done"

        with patch(
            "auto_core.summarizer.summarize_agent_output",
            new_callable=AsyncMock,
            return_value="progress",
        ):
            result = await run_with_summaries(
                slow_coro,
                "Analyst",
                "gpt-4o-mini",
                buf,
                interval=0.2,
                summary_collector=collector,
            )
            assert result == "done"
            periodic = [c for c in collector if not c[2].endswith("done")]
            assert len(periodic) >= 2

    @pytest.mark.asyncio
    async def test_final_summary_collected(self):
        """Final summary recaps progress summaries, not raw buffer."""
        buf: list[str] = []
        collector: list[tuple[str, str, str]] = []

        async def slow_coro(message_buffer):
            message_buffer.append("output")
            await asyncio.sleep(0.3)
            return 42

        with patch(
            "auto_core.summarizer.summarize_agent_output",
            new_callable=AsyncMock,
            return_value="summary",
        ):
            result = await run_with_summaries(
                slow_coro,
                "Coder",
                "gpt-4o-mini",
                buf,
                interval=0.1,
                summary_collector=collector,
            )
            assert result == 42
            final = [c for c in collector if c[2].endswith("done")]
            assert len(final) == 1

    @pytest.mark.asyncio
    async def test_empty_buffer_skips_poll(self):
        """If nothing is added to buffer, periodic summarizer should not be called."""
        buf: list[str] = []

        async def empty_coro(message_buffer):
            await asyncio.sleep(0.3)
            return "done"

        with (
            patch(
                "auto_core.summarizer.summarize_agent_output",
                new_callable=AsyncMock,
                return_value="summary",
            ) as mock_summarize,
        ):
            await run_with_summaries(
                empty_coro,
                "Analyst",
                "gpt-4o-mini",
                buf,
                interval=0.1,
            )
            periodic_calls = [
                c for c in mock_summarize.call_args_list if c[1].get("progress") is True
            ]
            assert len(periodic_calls) == 0

    @pytest.mark.asyncio
    async def test_stale_buffer_skips_summarize(self):
        """When buffer hasn't changed between polls, no summarize call fires."""
        progress_calls = 0

        async def count_summarize(agent, output, model, **kwargs):
            nonlocal progress_calls
            if kwargs.get("progress"):
                progress_calls += 1
            return "summary"

        buf: list[str] = []

        async def staged_coro(message_buffer):
            message_buffer.append("first")
            await asyncio.sleep(0.15)
            # Buffer stays stale for the remaining 0.35s
            await asyncio.sleep(0.35)
            return "done"

        with (
            patch(
                "auto_core.summarizer.summarize_agent_output",
                new_callable=AsyncMock,
                side_effect=count_summarize,
            ),
        ):
            await run_with_summaries(
                staged_coro,
                "Analyst",
                "gpt-4o-mini",
                buf,
                interval=0.1,
            )
            # Only 1 progress call (when "first" appeared), stale polls are skipped
            assert progress_calls == 1

    @pytest.mark.asyncio
    async def test_exception_in_coro_still_cleans_up(self):
        buf: list[str] = []

        async def failing_coro(message_buffer):
            message_buffer.append("before crash")
            raise ValueError("boom")

        with (
            patch(
                "auto_core.summarizer.summarize_agent_output",
                new_callable=AsyncMock,
                return_value="summary",
            ),
            pytest.raises(ValueError, match="boom"),
        ):
            await run_with_summaries(
                failing_coro,
                "Coder",
                "gpt-4o-mini",
                buf,
                interval=100,
            )

    @pytest.mark.asyncio
    async def test_summary_failure_does_not_crash(self):
        """If summarizer raises, the coroutine should still complete."""
        buf: list[str] = []

        async def coro(message_buffer):
            message_buffer.append("data")
            return "result"

        with (
            patch(
                "auto_core.summarizer.summarize_agent_output",
                new_callable=AsyncMock,
                side_effect=RuntimeError("API down"),
            ),
        ):
            result = await run_with_summaries(
                coro,
                "Analyst",
                "gpt-4o-mini",
                buf,
                interval=100,
            )
            assert result == "result"

    @pytest.mark.asyncio
    async def test_stale_buffer_backs_off(self):
        """When buffer hasn't changed, polls should back off exponentially."""
        poll_times: list[float] = []
        loop = asyncio.get_event_loop()
        start = loop.time()

        async def track_summarize(agent, output, model, **kwargs):
            poll_times.append(loop.time() - start)
            return "summary"

        buf: list[str] = ["initial data"]

        async def long_coro(message_buffer):
            # Buffer stays stale the whole time
            await asyncio.sleep(2.0)
            return "done"

        with (
            patch(
                "auto_core.summarizer.summarize_agent_output",
                new_callable=AsyncMock,
                side_effect=track_summarize,
            ),
        ):
            await run_with_summaries(
                long_coro,
                "Coder",
                "gpt-4o-mini",
                buf,
                interval=0.1,
            )

        # With 0.1s base interval and 2.0s runtime, without backoff we'd get ~20 polls.
        # With backoff (0.1, 0.2, 0.4, 0.8, 1.6) we expect around 4-5 polls.
        assert len(poll_times) <= 8, f"Expected backoff to reduce polls, got {len(poll_times)}"

    @pytest.mark.asyncio
    async def test_backoff_resets_on_new_content(self):
        """After backoff, rapid new content resets the poll interval."""
        poll_count = 0

        async def count_summarize(agent, output, model, **kwargs):
            nonlocal poll_count
            if kwargs.get("progress"):
                poll_count += 1
            return "summary"

        buf: list[str] = []

        async def staged_coro(message_buffer):
            # Phase 1: single item then stale (triggers backoff)
            message_buffer.append("chunk1")
            await asyncio.sleep(1.0)
            # Phase 2: rapid content stream (resets backoff, multiple polls)
            for i in range(20):
                message_buffer.append(f"rapid-{i}")
                await asyncio.sleep(0.05)
            return "done"

        with (
            patch(
                "auto_core.summarizer.summarize_agent_output",
                new_callable=AsyncMock,
                side_effect=count_summarize,
            ),
        ):
            await run_with_summaries(
                staged_coro,
                "Coder",
                "gpt-4o-mini",
                buf,
                interval=0.1,
            )

        # Phase 1: 1 poll. Phase 2: ~10 polls at 0.1s interval over 1.0s.
        # If backoff didn't reset, phase 2 would get far fewer polls.
        assert poll_count >= 4, (
            f"Expected backoff reset to produce multiple polls, got {poll_count}"
        )
