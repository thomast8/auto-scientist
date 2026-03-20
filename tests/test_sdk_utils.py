"""Tests for SDK utility functions (monkey-patch and safe_query)."""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from auto_scientist.sdk_utils import _tolerant_parse_message, safe_query


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
