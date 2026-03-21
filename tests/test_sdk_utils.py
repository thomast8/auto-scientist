"""Tests for SDK utility functions (monkey-patch, safe_query, validation)."""

import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from auto_scientist.schemas import AnalystOutput, ScientistPlanOutput
from auto_scientist.sdk_utils import (
    OutputValidationError,
    _tolerant_parse_message,
    collect_text_from_query,
    safe_query,
    validate_json_output,
)


class TestTolerantParseMessage:
    def test_known_type_passes_through(self):
        msg = MagicMock()
        with patch(
            "auto_scientist.sdk_utils._original_parse_message", return_value=msg,
        ):
            result = _tolerant_parse_message({"type": "assistant"})
        assert result is msg

    def test_unknown_type_returns_none(self):
        from claude_code_sdk._errors import MessageParseError

        with patch(
            "auto_scientist.sdk_utils._original_parse_message",
            side_effect=MessageParseError("Unknown message type: rate_limit_event"),
        ):
            result = _tolerant_parse_message({"type": "rate_limit_event"})
        assert result is None

    def test_non_unknown_parse_error_reraises(self):
        from claude_code_sdk._errors import MessageParseError

        with patch(
            "auto_scientist.sdk_utils._original_parse_message",
            side_effect=MessageParseError("Malformed JSON payload"),
        ):
            with pytest.raises(MessageParseError, match="Malformed JSON payload"):
                _tolerant_parse_message({"type": "bad"})

    def test_logs_skipped_type(self, caplog):
        from claude_code_sdk._errors import MessageParseError

        with patch(
            "auto_scientist.sdk_utils._original_parse_message",
            side_effect=MessageParseError("Unknown message type: foo_event"),
        ):
            with caplog.at_level(logging.DEBUG, logger="auto_scientist.sdk_utils"):
                _tolerant_parse_message({"type": "foo_event"})

        assert "foo_event" in caplog.text


class TestSafeQuery:
    @pytest.mark.asyncio
    async def test_yields_non_none_messages(self):
        msg1, msg2 = MagicMock(), MagicMock()

        async def fake_query(**kwargs):
            for item in [msg1, None, msg2]:
                yield item

        with patch("auto_scientist.sdk_utils.query", side_effect=fake_query):
            results = [m async for m in safe_query(prompt="hi", options=MagicMock())]

        assert results == [msg1, msg2]

    @pytest.mark.asyncio
    async def test_empty_stream_yields_nothing(self):
        async def fake_query(**kwargs):
            return
            yield  # noqa: RET504 - make it an async generator

        with patch("auto_scientist.sdk_utils.query", side_effect=fake_query):
            results = [m async for m in safe_query(prompt="hi", options=MagicMock())]

        assert results == []

    @pytest.mark.asyncio
    async def test_all_none_yields_nothing(self):
        async def fake_query(**kwargs):
            yield None
            yield None

        with patch("auto_scientist.sdk_utils.query", side_effect=fake_query):
            results = [m async for m in safe_query(prompt="hi", options=MagicMock())]

        assert results == []

    @pytest.mark.asyncio
    async def test_passes_prompt_and_options(self):
        opts = MagicMock()

        async def fake_query(**kwargs):
            return
            yield  # noqa: RET504

        with patch("auto_scientist.sdk_utils.query", side_effect=fake_query) as mock_q:
            async for _ in safe_query(prompt="test prompt", options=opts):
                pass

        mock_q.assert_called_once_with(prompt="test prompt", options=opts)


# ---------------------------------------------------------------------------
# OutputValidationError
# ---------------------------------------------------------------------------

class TestOutputValidationError:
    def test_stores_attributes(self):
        inner = ValueError("bad field")
        e = OutputValidationError(
            raw_output='{"bad": true}', validation_error=inner, agent_name="Analyst",
        )
        assert e.raw_output == '{"bad": true}'
        assert e.validation_error is inner
        assert e.agent_name == "Analyst"

    def test_correction_prompt_contains_error(self):
        inner = ValueError("missing field 'observations'")
        e = OutputValidationError(
            raw_output='{"bad": true}', validation_error=inner, agent_name="Analyst",
        )
        prompt = e.correction_prompt()
        assert "<validation_error>" in prompt
        assert "missing field" in prompt
        assert '{"bad": true}' in prompt

    def test_correction_prompt_truncates_long_output(self):
        long_output = "x" * 1000
        e = OutputValidationError(
            raw_output=long_output, validation_error=ValueError("err"), agent_name="Test",
        )
        prompt = e.correction_prompt()
        assert len(prompt) < len(long_output) + 500  # correction text + truncated output


# ---------------------------------------------------------------------------
# validate_json_output
# ---------------------------------------------------------------------------

class TestValidateJsonOutput:
    def test_valid_json_valid_schema(self):
        data = {
            "criteria_results": [],
            "key_metrics": {},
            "improvements": [],
            "regressions": [],
            "observations": ["ok"],
            "iteration_criteria_results": [],
        }
        raw = json.dumps(data)
        result = validate_json_output(raw, AnalystOutput, "Analyst")
        assert isinstance(result, dict)
        assert result["observations"] == ["ok"]

    def test_strips_markdown_fencing(self):
        data = {
            "criteria_results": [],
            "key_metrics": {},
            "improvements": [],
            "regressions": [],
            "observations": [],
            "iteration_criteria_results": [],
        }
        raw = f"```json\n{json.dumps(data)}\n```"
        result = validate_json_output(raw, AnalystOutput, "Analyst")
        assert isinstance(result, dict)

    def test_invalid_json_raises(self):
        with pytest.raises(OutputValidationError) as exc_info:
            validate_json_output("not json at all", AnalystOutput, "Analyst")
        assert exc_info.value.agent_name == "Analyst"

    def test_valid_json_invalid_schema_raises(self):
        raw = json.dumps({"hypothesis": "test"})  # missing required fields
        with pytest.raises(OutputValidationError) as exc_info:
            validate_json_output(raw, ScientistPlanOutput, "Scientist")
        assert exc_info.value.agent_name == "Scientist"

    def test_extra_fields_tolerated(self):
        data = {
            "criteria_results": [],
            "key_metrics": {},
            "improvements": [],
            "regressions": [],
            "observations": [],
            "iteration_criteria_results": [],
            "llm_reasoning": "should be ignored",
        }
        result = validate_json_output(json.dumps(data), AnalystOutput, "Analyst")
        assert "llm_reasoning" not in result


# ---------------------------------------------------------------------------
# collect_text_from_query
# ---------------------------------------------------------------------------

class TestCollectTextFromQuery:
    @pytest.mark.asyncio
    async def test_collects_result_text(self):
        """Prefers ResultMessage.result over assistant text."""
        from claude_code_sdk import AssistantMessage, ResultMessage, TextBlock

        result_msg = MagicMock(spec=ResultMessage)
        result_msg.result = '{"answer": 42}'

        assistant_msg = MagicMock(spec=AssistantMessage)
        text_block = MagicMock(spec=TextBlock)
        text_block.text = "intermediate text"
        assistant_msg.content = [text_block]

        async def fake_query(**kwargs):
            yield assistant_msg
            yield result_msg

        with patch("auto_scientist.sdk_utils.query", side_effect=fake_query):
            raw = await collect_text_from_query("prompt", MagicMock())
        assert raw == '{"answer": 42}'

    @pytest.mark.asyncio
    async def test_falls_back_to_assistant_text(self):
        """When no ResultMessage.result, falls back to concatenated assistant text."""
        from claude_code_sdk import AssistantMessage, ResultMessage, TextBlock

        result_msg = MagicMock(spec=ResultMessage)
        result_msg.result = ""

        assistant_msg = MagicMock(spec=AssistantMessage)
        text_block = MagicMock(spec=TextBlock)
        text_block.text = '{"answer": 42}'
        assistant_msg.content = [text_block]

        async def fake_query(**kwargs):
            yield assistant_msg
            yield result_msg

        with patch("auto_scientist.sdk_utils.query", side_effect=fake_query):
            raw = await collect_text_from_query("prompt", MagicMock())
        assert raw == '{"answer": 42}'

    @pytest.mark.asyncio
    async def test_empty_output_raises(self):
        from claude_code_sdk import ResultMessage

        result_msg = MagicMock(spec=ResultMessage)
        result_msg.result = ""

        async def fake_query(**kwargs):
            yield result_msg

        with patch("auto_scientist.sdk_utils.query", side_effect=fake_query):
            with pytest.raises(RuntimeError, match="returned no output"):
                await collect_text_from_query("prompt", MagicMock(), agent_name="Analyst")

    @pytest.mark.asyncio
    async def test_populates_message_buffer(self):
        from claude_code_sdk import AssistantMessage, ResultMessage, TextBlock

        result_msg = MagicMock(spec=ResultMessage)
        result_msg.result = "result"

        assistant_msg = MagicMock(spec=AssistantMessage)
        text_block = MagicMock(spec=TextBlock)
        text_block.text = "block text"
        assistant_msg.content = [text_block]

        async def fake_query(**kwargs):
            yield assistant_msg
            yield result_msg

        buffer: list[str] = []
        with patch("auto_scientist.sdk_utils.query", side_effect=fake_query):
            with patch("auto_scientist.sdk_utils.append_block_to_buffer") as mock_append:
                await collect_text_from_query("prompt", MagicMock(), message_buffer=buffer)
                mock_append.assert_called_once_with(text_block, buffer)
