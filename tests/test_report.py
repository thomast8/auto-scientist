"""Tests for the Report agent."""

import asyncio
from unittest.mock import MagicMock, patch

import pytest
from claude_code_sdk import AssistantMessage, ResultMessage, TextBlock

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
            iteration=5,
        )
        notebook_path = tmp_path / "lab_notebook.xml"
        notebook_path.write_text("# Lab Notebook")

        result = await run_report(
            state=state, notebook_path=notebook_path, output_dir=tmp_path,
        )

        assert isinstance(result, str)
        assert "# Final Report" in result

    @pytest.mark.asyncio
    @patch("auto_scientist.agents.report.validate_report_structure", return_value=[])
    @patch("auto_scientist.agents.report.safe_query")
    async def test_strips_preamble_before_first_heading(self, mock_query, _mock_validate, tmp_path):
        """Conversational preamble before the first markdown heading is stripped."""
        from claude_code_sdk import AssistantMessage, ResultMessage, TextBlock

        preamble_msg = MagicMock(spec=AssistantMessage)
        preamble_block = MagicMock(spec=TextBlock)
        preamble_block.text = "Let me read the results first."
        preamble_msg.content = [preamble_block]

        report_msg = MagicMock(spec=AssistantMessage)
        report_block = MagicMock(spec=TextBlock)
        # Must exceed MIN_REPORT_LENGTH (100 chars)
        long_content = "Detailed findings about the investigation. " * 3
        report_block.text = f"# Final Report\n\n{long_content}"
        report_msg.content = [report_block]

        result_msg = MagicMock(spec=ResultMessage)

        async def fake_query(**kwargs):
            yield preamble_msg
            yield report_msg
            yield result_msg

        mock_query.side_effect = fake_query

        state = ExperimentState(domain="test", goal="test goal")
        notebook_path = tmp_path / "lab_notebook.xml"
        notebook_path.write_text("# Notebook")

        result = await run_report(
            state=state, notebook_path=notebook_path, output_dir=tmp_path,
        )

        assert result.startswith("# Final Report")
        assert "Let me read" not in result

    @pytest.mark.asyncio
    @patch("auto_scientist.agents.report.safe_query")
    async def test_raises_when_no_text(self, mock_query, tmp_path):
        """If the agent produces no text blocks, raise RuntimeError."""
        from claude_code_sdk import ResultMessage

        result_msg = MagicMock(spec=ResultMessage)

        async def fake_query(**kwargs):
            yield result_msg

        mock_query.side_effect = fake_query

        state = ExperimentState(domain="test", goal="test goal")
        notebook_path = tmp_path / "lab_notebook.xml"

        with pytest.raises(RuntimeError, match="no output"):
            await run_report(
                state=state, notebook_path=notebook_path, output_dir=tmp_path,
            )

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
            iteration=10,
        )
        from auto_scientist.state import VersionEntry
        state.versions.append(VersionEntry(version="v07", iteration=7, script_path="/tmp/s.py"))
        notebook_path = tmp_path / "lab_notebook.xml"
        notebook_path.write_text("# Notebook")

        with pytest.raises(RuntimeError):
            await run_report(state=state, notebook_path=notebook_path, output_dir=tmp_path)

        prompt = captured_prompt["prompt"]
        assert "spo2" in prompt
        assert "predict oxygen levels" in prompt
        assert "v07" in prompt

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

        with pytest.raises(RuntimeError):
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

        with pytest.raises(RuntimeError):
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

        with pytest.raises(RuntimeError):
            await run_report(state=state, notebook_path=notebook_path, output_dir=tmp_path)

        prompt = captured_prompt["prompt"]
        # Should NOT instruct the agent to write a file
        assert "Write the final report to the file" not in prompt


class TestReportMessageBuffer:
    @pytest.mark.asyncio
    @patch("auto_scientist.agents.report.validate_report_structure", return_value=[])
    @patch("auto_scientist.agents.report.safe_query")
    async def test_populates_message_buffer(self, mock_query, _mock_validate, tmp_path):
        assistant_msg = MagicMock(spec=AssistantMessage)
        text_block = MagicMock(spec=TextBlock)
        # Must be >= MIN_REPORT_LENGTH (100) to avoid retry
        text_block.text = "# Final Report\n\n" + "This is a comprehensive report. " * 5
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
        assert "Final Report" in buf[0]


class TestReportRetry:
    @pytest.mark.asyncio
    @patch("auto_scientist.agents.report.safe_query")
    async def test_retry_on_empty_output(self, mock_query, tmp_path):
        """First attempt returns empty, second returns valid report."""
        call_count = 0

        async def fake_query(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                result_msg = MagicMock(spec=ResultMessage)
                yield result_msg
            else:
                assistant_msg = MagicMock(spec=AssistantMessage)
                text_block = MagicMock(spec=TextBlock)
                text_block.text = "# Report\n\nThis is a valid report with enough content."
                assistant_msg.content = [text_block]
                yield assistant_msg
                yield MagicMock(spec=ResultMessage)

        mock_query.side_effect = fake_query

        state = ExperimentState(domain="test", goal="test goal")
        notebook_path = tmp_path / "lab_notebook.xml"
        notebook_path.write_text("# Notebook")

        result = await run_report(
            state=state, notebook_path=notebook_path, output_dir=tmp_path,
        )
        assert "# Report" in result
        assert call_count == 2

    @pytest.mark.asyncio
    @patch("auto_scientist.agents.report.safe_query")
    async def test_exhausts_retries_raises(self, mock_query, tmp_path):
        """All attempts return empty, should raise RuntimeError."""
        async def fake_query(**kwargs):
            yield MagicMock(spec=ResultMessage)

        mock_query.side_effect = fake_query

        state = ExperimentState(domain="test", goal="test goal")
        notebook_path = tmp_path / "lab_notebook.xml"

        with pytest.raises(RuntimeError, match="no output"):
            await run_report(
                state=state, notebook_path=notebook_path, output_dir=tmp_path,
            )


# A report that passes length check but is missing required sections
_INCOMPLETE_REPORT = """\
# Investigation Report

## Executive Summary
This is a summary of the investigation with enough content to pass the length check.

## Results
R² = 0.964 which is pretty good. We tested several approaches and found that
polynomial regression with degree 4 works best for this dataset.
"""

# A complete report with all sections
_COMPLETE_REPORT = """\
## Executive Summary
Summary content here with enough text.

## Problem Statement and Data
We studied the dataset.

## Methodology
Iterative autonomous loop.

## Journey
v01 linear, v02 polynomial.

## Best Approach
Degree-4 polynomial.

## Results
Test R² = 0.964.

## Key Scientific Insights
Nonlinear relationship.

## Limitations
Fails for x > 100.

## Recommended Future Work
Try GP regression.

## Version Comparison Table
| Version | Status | Key Change | Key Metric | Prediction Outcome |
|---------|--------|------------|------------|-------------------|
| v01 | baseline | Linear | R²=0.72 | N/A |
"""


class TestReportStructuralValidation:
    @pytest.mark.asyncio
    @patch("auto_scientist.agents.report.safe_query")
    async def test_incomplete_report_triggers_retry(self, mock_query, tmp_path):
        """Report missing sections triggers session resume retry."""
        call_count = 0

        async def fake_query(**kwargs):
            nonlocal call_count
            call_count += 1

            assistant_msg = MagicMock(spec=AssistantMessage)
            text_block = MagicMock(spec=TextBlock)
            if call_count == 1:
                text_block.text = _INCOMPLETE_REPORT
            else:
                text_block.text = _COMPLETE_REPORT
                # Second call should use session resume
                options = kwargs.get("options")
                assert options is not None
                assert options.resume is not None
            assistant_msg.content = [text_block]

            result_msg = MagicMock(spec=ResultMessage)
            result_msg.session_id = "report-session-456"
            yield assistant_msg
            yield result_msg

        mock_query.side_effect = fake_query

        state = ExperimentState(domain="test", goal="test goal")
        notebook_path = tmp_path / "lab_notebook.xml"
        notebook_path.write_text("# Notebook")

        result = await run_report(
            state=state, notebook_path=notebook_path, output_dir=tmp_path,
        )
        assert call_count == 2
        assert "Executive Summary" in result

    @pytest.mark.asyncio
    @patch("auto_scientist.agents.report.safe_query")
    async def test_correction_prompt_lists_issues(self, mock_query, tmp_path):
        """Correction prompt should list the structural issues found."""
        call_count = 0
        captured_prompts: list[str] = []

        async def fake_query(**kwargs):
            nonlocal call_count
            call_count += 1
            captured_prompts.append(kwargs.get("prompt", ""))

            assistant_msg = MagicMock(spec=AssistantMessage)
            text_block = MagicMock(spec=TextBlock)
            if call_count == 1:
                text_block.text = _INCOMPLETE_REPORT
            else:
                text_block.text = _COMPLETE_REPORT
            assistant_msg.content = [text_block]

            result_msg = MagicMock(spec=ResultMessage)
            result_msg.session_id = "report-session-789"
            yield assistant_msg
            yield result_msg

        mock_query.side_effect = fake_query

        state = ExperimentState(domain="test", goal="test goal")
        notebook_path = tmp_path / "lab_notebook.xml"
        notebook_path.write_text("# Notebook")

        await run_report(
            state=state, notebook_path=notebook_path, output_dir=tmp_path,
        )

        # Second prompt should contain the correction with missing sections
        assert len(captured_prompts) == 2
        correction = captured_prompts[1]
        assert "<validation_error>" in correction
        assert "Missing section" in correction
