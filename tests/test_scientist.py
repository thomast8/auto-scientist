"""Tests for the Scientist agent."""

import json
from unittest.mock import MagicMock, patch

import pytest

from auto_scientist.agents.scientist import (
    _parse_json_response,
    run_scientist,
    run_scientist_revision,
)


class TestParseJsonResponse:
    """Tests for the pure JSON parsing helper."""

    def test_clean_json(self):
        result = _parse_json_response('{"key": "value"}', "test")
        assert result == {"key": "value"}

    def test_markdown_fenced_json(self):
        raw = '```json\n{"key": "value"}\n```'
        result = _parse_json_response(raw, "test")
        assert result == {"key": "value"}

    def test_markdown_fenced_no_language(self):
        raw = '```\n{"key": "value"}\n```'
        result = _parse_json_response(raw, "test")
        assert result == {"key": "value"}

    def test_invalid_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            _parse_json_response("not json", "test")

    def test_whitespace_stripped(self):
        result = _parse_json_response('  \n{"key": "value"}\n  ', "test")
        assert result == {"key": "value"}


SAMPLE_PLAN = {
    "hypothesis": "test hypothesis",
    "strategy": "incremental",
    "changes": [{"what": "do thing", "why": "because", "how": "like this", "priority": 1}],
    "expected_impact": "improvement",
    "should_stop": False,
    "stop_reason": None,
    "notebook_entry": "## v01",
    "success_criteria": [
        {"name": "metric", "description": "desc", "metric_key": "m", "condition": "> 0.5"}
    ],
}


class TestRunScientist:
    @pytest.mark.asyncio
    @patch("auto_scientist.agents.scientist.query")
    async def test_returns_parsed_plan(self, mock_query, tmp_path):
        from auto_scientist.agents.scientist import ResultMessage
        result_msg = MagicMock(spec=ResultMessage)
        result_msg.result = json.dumps(SAMPLE_PLAN)

        async def fake_query(**kwargs):
            yield result_msg

        mock_query.side_effect = fake_query

        notebook_path = tmp_path / "notebook.md"
        notebook_path.write_text("# Notebook content")

        result = await run_scientist(
            analysis={"success_score": 50},
            notebook_path=notebook_path,
            version="v01",
        )

        assert result["hypothesis"] == "test hypothesis"
        assert result["strategy"] == "incremental"

    @pytest.mark.asyncio
    @patch("auto_scientist.agents.scientist.query")
    async def test_missing_notebook_uses_fallback(self, mock_query, tmp_path):
        from auto_scientist.agents.scientist import ResultMessage
        result_msg = MagicMock(spec=ResultMessage)
        result_msg.result = json.dumps(SAMPLE_PLAN)

        async def fake_query(**kwargs):
            yield result_msg

        mock_query.side_effect = fake_query

        notebook_path = tmp_path / "nonexistent.md"

        result = await run_scientist(
            analysis={}, notebook_path=notebook_path, version="v01",
        )
        assert result["hypothesis"] == "test hypothesis"

    @pytest.mark.asyncio
    @patch("auto_scientist.agents.scientist.query")
    async def test_empty_output_raises(self, mock_query, tmp_path):
        from auto_scientist.agents.scientist import ResultMessage
        result_msg = MagicMock(spec=ResultMessage)
        result_msg.result = ""

        async def fake_query(**kwargs):
            yield result_msg

        mock_query.side_effect = fake_query

        notebook_path = tmp_path / "notebook.md"

        with pytest.raises(RuntimeError, match="returned no output"):
            await run_scientist(
                analysis={}, notebook_path=notebook_path, version="v01",
            )

    @pytest.mark.asyncio
    @patch("auto_scientist.agents.scientist.query")
    async def test_no_tools_configured(self, mock_query, tmp_path):
        """Scientist should have no tools (pure prompt-in/JSON-out)."""
        from auto_scientist.agents.scientist import ResultMessage
        result_msg = MagicMock(spec=ResultMessage)
        result_msg.result = json.dumps(SAMPLE_PLAN)

        captured_options = {}

        async def fake_query(**kwargs):
            captured_options.update(kwargs)
            yield result_msg

        mock_query.side_effect = fake_query

        notebook_path = tmp_path / "notebook.md"
        await run_scientist(analysis={}, notebook_path=notebook_path, version="v01")

        assert captured_options["options"].allowed_tools == []


class TestRunScientistRevision:
    @pytest.mark.asyncio
    @patch("auto_scientist.agents.scientist.query")
    async def test_returns_revised_plan(self, mock_query, tmp_path):
        from auto_scientist.agents.scientist import ResultMessage
        result_msg = MagicMock(spec=ResultMessage)
        result_msg.result = json.dumps(SAMPLE_PLAN)

        async def fake_query(**kwargs):
            yield result_msg

        mock_query.side_effect = fake_query

        notebook_path = tmp_path / "notebook.md"
        notebook_path.write_text("# Notebook")

        transcript = [
            {"role": "critic", "content": "This is weak"},
            {"role": "scientist", "content": "I disagree"},
        ]

        result = await run_scientist_revision(
            original_plan=SAMPLE_PLAN,
            debate_transcript=transcript,
            analysis={"success_score": 50},
            notebook_path=notebook_path,
            version="v01",
        )

        assert result["hypothesis"] == "test hypothesis"

    @pytest.mark.asyncio
    @patch("auto_scientist.agents.scientist.query")
    async def test_empty_output_raises(self, mock_query, tmp_path):
        from auto_scientist.agents.scientist import ResultMessage
        result_msg = MagicMock(spec=ResultMessage)
        result_msg.result = ""

        async def fake_query(**kwargs):
            yield result_msg

        mock_query.side_effect = fake_query

        notebook_path = tmp_path / "notebook.md"

        with pytest.raises(RuntimeError, match="returned no output"):
            await run_scientist_revision(
                original_plan=SAMPLE_PLAN,
                debate_transcript=[],
                analysis={},
                notebook_path=notebook_path,
                version="v01",
            )
