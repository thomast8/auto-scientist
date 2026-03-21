"""Tests for the Report agent."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from auto_scientist.agents.report import run_report
from auto_scientist.state import ExperimentState


def test_run_report_is_async():
    assert asyncio.iscoroutinefunction(run_report)


class TestRunReport:
    @pytest.mark.asyncio
    @patch("auto_scientist.agents.report.safe_query")
    async def test_returns_report_content_string(self, mock_query, tmp_path):
        """run_report should return the report text, not a Path."""
        from claude_code_sdk import AssistantMessage, ResultMessage, TextBlock

        assistant_msg = MagicMock(spec=AssistantMessage)
        text_block = MagicMock(spec=TextBlock)
        text_block.text = "# Final Report\n\nThis is the report."
        assistant_msg.content = [text_block]
        result_msg = MagicMock(spec=ResultMessage)

        async def fake_query(**kwargs):
            yield assistant_msg
            yield result_msg

        mock_query.side_effect = fake_query

        state = ExperimentState(
            domain="test", goal="test goal",
            iteration=5, best_version="v03", best_score=85,
        )
        notebook_path = tmp_path / "lab_notebook.xml"
        notebook_path.write_text("# Lab Notebook")

        result = await run_report(
            state=state, notebook_path=notebook_path, output_dir=tmp_path,
        )

        assert isinstance(result, str)
        assert "# Final Report" in result

    @pytest.mark.asyncio
    @patch("auto_scientist.agents.report.safe_query")
    async def test_returns_empty_string_when_no_text(self, mock_query, tmp_path):
        """If the agent produces no text blocks, return empty string."""
        from claude_code_sdk import ResultMessage

        result_msg = MagicMock(spec=ResultMessage)

        async def fake_query(**kwargs):
            yield result_msg

        mock_query.side_effect = fake_query

        state = ExperimentState(domain="test", goal="test goal")
        notebook_path = tmp_path / "lab_notebook.xml"

        result = await run_report(
            state=state, notebook_path=notebook_path, output_dir=tmp_path,
        )
        assert result == ""

    @pytest.mark.asyncio
    @patch("auto_scientist.agents.report.safe_query")
    async def test_prompt_includes_state_fields(self, mock_query, tmp_path):
        from auto_scientist.agents.report import ResultMessage
        result_msg = MagicMock(spec=ResultMessage)

        captured_prompt = {}

        async def fake_query(**kwargs):
            captured_prompt["prompt"] = kwargs.get("prompt", "")
            yield result_msg

        mock_query.side_effect = fake_query

        state = ExperimentState(
            domain="spo2", goal="predict oxygen levels",
            iteration=10, best_version="v07", best_score=92,
        )
        notebook_path = tmp_path / "lab_notebook.xml"
        notebook_path.write_text("# Notebook")

        await run_report(state=state, notebook_path=notebook_path, output_dir=tmp_path)

        prompt = captured_prompt["prompt"]
        assert "spo2" in prompt
        assert "predict oxygen levels" in prompt
        assert "v07" in prompt
        assert "92" in prompt

    @pytest.mark.asyncio
    @patch("auto_scientist.agents.report.safe_query")
    async def test_options_no_write_tool(self, mock_query, tmp_path):
        """Report agent should NOT have Write access - orchestrator writes."""
        from auto_scientist.agents.report import ResultMessage
        result_msg = MagicMock(spec=ResultMessage)

        captured_opts = {}

        async def fake_query(**kwargs):
            captured_opts.update(kwargs)
            yield result_msg

        mock_query.side_effect = fake_query

        state = ExperimentState(domain="test", goal="test goal")
        notebook_path = tmp_path / "lab_notebook.xml"
        notebook_path.write_text("# Notebook")

        await run_report(state=state, notebook_path=notebook_path, output_dir=tmp_path)

        opts = captured_opts["options"]
        assert "Write" not in opts.allowed_tools
        assert "Read" in opts.allowed_tools
        assert "Glob" in opts.allowed_tools
        assert opts.max_turns == 10
        assert opts.permission_mode == "acceptEdits"

    @pytest.mark.asyncio
    @patch("auto_scientist.agents.report.safe_query")
    async def test_missing_notebook_fallback(self, mock_query, tmp_path):
        from auto_scientist.agents.report import ResultMessage
        result_msg = MagicMock(spec=ResultMessage)

        captured_prompt = {}

        async def fake_query(**kwargs):
            captured_prompt["prompt"] = kwargs.get("prompt", "")
            yield result_msg

        mock_query.side_effect = fake_query

        state = ExperimentState(domain="test", goal="test goal")
        notebook_path = tmp_path / "nonexistent_notebook.md"

        await run_report(state=state, notebook_path=notebook_path, output_dir=tmp_path)
        assert "(no notebook)" in captured_prompt["prompt"]


class TestReportPrompt:
    @pytest.mark.asyncio
    @patch("auto_scientist.agents.report.safe_query")
    async def test_prompt_asks_to_output_text_not_write_file(self, mock_query, tmp_path):
        """The prompt should instruct the agent to output the report as text."""
        from auto_scientist.agents.report import ResultMessage
        result_msg = MagicMock(spec=ResultMessage)

        captured_prompt = {}

        async def fake_query(**kwargs):
            captured_prompt["prompt"] = kwargs.get("prompt", "")
            yield result_msg

        mock_query.side_effect = fake_query

        state = ExperimentState(domain="test", goal="test goal")
        notebook_path = tmp_path / "lab_notebook.xml"
        notebook_path.write_text("# Notebook")

        await run_report(state=state, notebook_path=notebook_path, output_dir=tmp_path)

        prompt = captured_prompt["prompt"]
        # Should NOT instruct the agent to write a file
        assert "Write the final report to the file" not in prompt


class TestReportMessageBuffer:
    @pytest.mark.asyncio
    @patch("auto_scientist.agents.report.safe_query")
    async def test_populates_message_buffer(self, mock_query, tmp_path):
        from claude_code_sdk import AssistantMessage, ResultMessage, TextBlock

        assistant_msg = MagicMock(spec=AssistantMessage)
        text_block = MagicMock(spec=TextBlock)
        text_block.text = "Generating report..."
        assistant_msg.content = [text_block]
        result_msg = MagicMock(spec=ResultMessage)

        async def fake_query(**kwargs):
            yield assistant_msg
            yield result_msg

        mock_query.side_effect = fake_query

        state = ExperimentState(domain="test", goal="test goal")
        notebook_path = tmp_path / "lab_notebook.xml"
        notebook_path.write_text("# Notebook")

        buf: list[str] = []
        await run_report(
            state=state, notebook_path=notebook_path, output_dir=tmp_path,
            message_buffer=buf,
        )
        assert len(buf) == 1
        assert "Generating report..." in buf[0]
