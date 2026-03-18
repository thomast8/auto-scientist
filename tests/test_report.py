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
    @patch("auto_scientist.agents.report.query")
    async def test_creates_report_at_expected_path(self, mock_query, tmp_path):
        from auto_scientist.agents.report import ResultMessage
        result_msg = MagicMock(spec=ResultMessage)

        async def fake_query(**kwargs):
            # Simulate agent writing the report
            report_path = tmp_path / "report.md"
            report_path.write_text("# Final Report")
            yield result_msg

        mock_query.side_effect = fake_query

        state = ExperimentState(
            domain="test", goal="test goal",
            iteration=5, best_version="v03", best_score=85,
        )
        notebook_path = tmp_path / "lab_notebook.md"
        notebook_path.write_text("# Lab Notebook")

        result = await run_report(
            state=state, notebook_path=notebook_path, output_dir=tmp_path,
        )

        assert result == tmp_path / "report.md"

    @pytest.mark.asyncio
    @patch("auto_scientist.agents.report.query")
    async def test_raises_when_report_not_created(self, mock_query, tmp_path):
        from auto_scientist.agents.report import ResultMessage
        result_msg = MagicMock(spec=ResultMessage)

        async def fake_query(**kwargs):
            yield result_msg

        mock_query.side_effect = fake_query

        state = ExperimentState(domain="test", goal="test goal")
        notebook_path = tmp_path / "lab_notebook.md"

        with pytest.raises(FileNotFoundError, match="did not create"):
            await run_report(
                state=state, notebook_path=notebook_path, output_dir=tmp_path,
            )

    @pytest.mark.asyncio
    @patch("auto_scientist.agents.report.query")
    async def test_prompt_includes_state_fields(self, mock_query, tmp_path):
        from auto_scientist.agents.report import ResultMessage
        result_msg = MagicMock(spec=ResultMessage)

        captured_prompt = {}

        async def fake_query(**kwargs):
            captured_prompt["prompt"] = kwargs.get("prompt", "")
            report_path = tmp_path / "report.md"
            report_path.write_text("# Report")
            yield result_msg

        mock_query.side_effect = fake_query

        state = ExperimentState(
            domain="spo2", goal="predict oxygen levels",
            iteration=10, best_version="v07", best_score=92,
        )
        notebook_path = tmp_path / "lab_notebook.md"
        notebook_path.write_text("# Notebook")

        await run_report(state=state, notebook_path=notebook_path, output_dir=tmp_path)

        prompt = captured_prompt["prompt"]
        assert "spo2" in prompt
        assert "predict oxygen levels" in prompt
        assert "v07" in prompt
        assert "92" in prompt
