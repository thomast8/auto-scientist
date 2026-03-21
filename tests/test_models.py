"""Tests for LLM model client wrappers."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from auto_scientist.model_config import ReasoningConfig
from auto_scientist.models.anthropic_client import query_anthropic
from auto_scientist.models.google_client import query_google
from auto_scientist.models.openai_client import query_openai


class TestQueryOpenAIStreaming:
    @pytest.mark.asyncio
    @patch("auto_scientist.models.openai_client.AsyncOpenAI")
    async def test_streaming_calls_on_token(self, mock_cls):
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client

        chunk1 = MagicMock()
        chunk1.choices = [MagicMock(delta=MagicMock(content="hel"))]
        chunk2 = MagicMock()
        chunk2.choices = [MagicMock(delta=MagicMock(content="lo"))]
        chunk3 = MagicMock()
        chunk3.choices = [MagicMock(delta=MagicMock(content=None))]

        async def fake_stream(**kwargs):
            for c in [chunk1, chunk2, chunk3]:
                yield c

        mock_client.chat.completions.create.return_value = fake_stream()

        tokens = []
        result = await query_openai("gpt-5.4", "test", on_token=tokens.append)

        assert result == "hello"
        assert tokens == ["hel", "lo"]

    @pytest.mark.asyncio
    @patch("auto_scientist.models.openai_client.AsyncOpenAI")
    async def test_streaming_web_search(self, mock_cls):
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client

        event1 = MagicMock(type="response.output_text.delta", delta="search")
        event2 = MagicMock(type="response.output_text.delta", delta="ed")
        event_other = MagicMock(type="response.web_search_call.in_progress")

        async def fake_stream(**kwargs):
            for e in [event_other, event1, event2]:
                yield e

        mock_client.responses.create.return_value = fake_stream()

        tokens = []
        result = await query_openai("gpt-5.4", "q", web_search=True, on_token=tokens.append)

        assert result == "searched"
        assert tokens == ["search", "ed"]

    @pytest.mark.asyncio
    @patch("auto_scientist.models.openai_client.AsyncOpenAI")
    async def test_no_on_token_uses_non_streaming(self, mock_cls):
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="hello"))]
        mock_client.chat.completions.create.return_value = mock_response

        result = await query_openai("gpt-5.4", "test")

        assert result == "hello"
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert "stream" not in call_kwargs


class TestQueryOpenAI:
    @pytest.mark.asyncio
    @patch("auto_scientist.models.openai_client.AsyncOpenAI")
    async def test_standard_call(self, mock_cls):
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="hello"))]
        mock_client.chat.completions.create.return_value = mock_response

        result = await query_openai("gpt-5.4", "test prompt")

        assert result == "hello"
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "gpt-5.4"
        assert call_kwargs["messages"][0]["content"] == "test prompt"

    @pytest.mark.asyncio
    @patch("auto_scientist.models.openai_client.AsyncOpenAI")
    async def test_web_search_uses_responses_api(self, mock_cls):
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        mock_response = MagicMock(output_text="searched result")
        mock_client.responses.create.return_value = mock_response

        result = await query_openai("gpt-5.4", "search this", web_search=True)

        assert result == "searched result"
        mock_client.responses.create.assert_called_once()
        call_kwargs = mock_client.responses.create.call_args.kwargs
        assert any(t["type"] == "web_search_preview" for t in call_kwargs["tools"])

    @pytest.mark.asyncio
    @patch("auto_scientist.models.openai_client.AsyncOpenAI")
    async def test_empty_response_returns_empty_string(self, mock_cls):
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=None))]
        mock_client.chat.completions.create.return_value = mock_response

        result = await query_openai("gpt-5.4", "prompt")

        assert result == ""


class TestQueryOpenAIMaxTokens:
    @pytest.mark.asyncio
    @patch("auto_scientist.models.openai_client.AsyncOpenAI")
    async def test_custom_max_tokens(self, mock_cls):
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="ok"))]
        mock_client.chat.completions.create.return_value = mock_response

        await query_openai("gpt-5.4", "test", max_tokens=150)

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["max_tokens"] == 150

    @pytest.mark.asyncio
    @patch("auto_scientist.models.openai_client.AsyncOpenAI")
    async def test_default_max_tokens(self, mock_cls):
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="ok"))]
        mock_client.chat.completions.create.return_value = mock_response

        await query_openai("gpt-5.4", "test")

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["max_tokens"] == 4096

    @pytest.mark.asyncio
    @patch("auto_scientist.models.openai_client.AsyncOpenAI")
    async def test_custom_max_tokens_streaming(self, mock_cls):
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client

        chunk = MagicMock()
        chunk.choices = [MagicMock(delta=MagicMock(content="ok"))]

        async def fake_stream(**kwargs):
            yield chunk

        mock_client.chat.completions.create.return_value = fake_stream()

        await query_openai("gpt-5.4", "test", max_tokens=200, on_token=lambda t: None)

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["max_tokens"] == 200


class TestQueryGoogleStreaming:
    @pytest.mark.asyncio
    @patch("auto_scientist.models.google_client.genai")
    async def test_streaming_calls_on_token(self, mock_genai):
        async def fake_stream():
            for text in ["goo", "gle"]:
                yield MagicMock(text=text)

        mock_genai.Client.return_value.aio.models.generate_content_stream = AsyncMock(
            return_value=fake_stream()
        )

        tokens = []
        result = await query_google("gemini-3.1-pro-preview", "test", on_token=tokens.append)

        assert result == "google"
        assert tokens == ["goo", "gle"]

    @pytest.mark.asyncio
    @patch("auto_scientist.models.google_client.genai")
    async def test_streaming_skips_none_text(self, mock_genai):
        async def fake_stream():
            yield MagicMock(text=None)
            yield MagicMock(text="data")

        mock_genai.Client.return_value.aio.models.generate_content_stream = AsyncMock(
            return_value=fake_stream()
        )

        tokens = []
        result = await query_google("gemini-3.1-pro-preview", "test", on_token=tokens.append)

        assert result == "data"
        assert tokens == ["data"]

    @pytest.mark.asyncio
    @patch("auto_scientist.models.google_client.genai")
    async def test_no_on_token_uses_non_streaming(self, mock_genai):
        mock_response = MagicMock(text="response")
        mock_genai.Client.return_value.aio.models.generate_content = AsyncMock(
            return_value=mock_response
        )

        result = await query_google("gemini-3.1-pro-preview", "test")

        assert result == "response"
        mock_genai.Client.return_value.aio.models.generate_content.assert_called_once()


class TestQueryGoogle:
    @pytest.mark.asyncio
    @patch("auto_scientist.models.google_client.genai")
    async def test_standard_call(self, mock_genai):
        mock_response = MagicMock(text="google response")
        mock_genai.Client.return_value.aio.models.generate_content = AsyncMock(
            return_value=mock_response
        )

        result = await query_google("gemini-3.1-pro-preview", "test prompt")

        assert result == "google response"

    @pytest.mark.asyncio
    @patch("auto_scientist.models.google_client.genai")
    async def test_web_search_adds_config(self, mock_genai):
        mock_response = MagicMock(text="searched")
        mock_genai.Client.return_value.aio.models.generate_content = AsyncMock(
            return_value=mock_response
        )

        result = await query_google("gemini-3.1-pro-preview", "search", web_search=True)

        assert result == "searched"
        call_kwargs = mock_genai.Client.return_value.aio.models.generate_content.call_args.kwargs
        assert call_kwargs["config"] is not None

    @pytest.mark.asyncio
    @patch("auto_scientist.models.google_client.genai")
    async def test_empty_response_returns_empty_string(self, mock_genai):
        mock_response = MagicMock(text=None)
        mock_genai.Client.return_value.aio.models.generate_content = AsyncMock(
            return_value=mock_response
        )

        result = await query_google("gemini-3.1-pro-preview", "prompt")

        assert result == ""


def _make_anthropic_stream_mock(chunks):
    """Create a mock that behaves like client.messages.stream() context manager."""
    async def fake_text_stream():
        for chunk in chunks:
            yield chunk

    mock_stream = MagicMock()
    mock_stream.text_stream = fake_text_stream()
    mock_stream.__aenter__ = AsyncMock(return_value=mock_stream)
    mock_stream.__aexit__ = AsyncMock(return_value=False)
    return mock_stream


class TestQueryAnthropicStreaming:
    @pytest.mark.asyncio
    @patch("auto_scientist.models.anthropic_client.AsyncAnthropic")
    async def test_streaming_calls_on_token(self, mock_cls):
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock()
        mock_cls.return_value = mock_client
        mock_client.messages.stream.return_value = _make_anthropic_stream_mock(
            ["anthro", "pic"]
        )

        tokens = []
        result = await query_anthropic("claude-sonnet-4-6", "test", on_token=tokens.append)

        assert result == "anthropic"
        assert tokens == ["anthro", "pic"]
        mock_client.messages.stream.assert_called_once()

    @pytest.mark.asyncio
    @patch("auto_scientist.models.anthropic_client.AsyncAnthropic")
    async def test_streaming_with_web_search(self, mock_cls):
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock()
        mock_cls.return_value = mock_client
        mock_client.messages.stream.return_value = _make_anthropic_stream_mock(["result"])

        tokens = []
        result = await query_anthropic(
            "claude-sonnet-4-6", "q", web_search=True, on_token=tokens.append
        )

        assert result == "result"
        call_kwargs = mock_client.messages.stream.call_args.kwargs
        assert any(t["type"] == "web_search_20250305" for t in call_kwargs["tools"])

    @pytest.mark.asyncio
    @patch("auto_scientist.models.anthropic_client.AsyncAnthropic")
    async def test_no_on_token_uses_non_streaming(self, mock_cls):
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock()
        mock_cls.return_value = mock_client
        mock_block = MagicMock(text="hello")
        mock_response = MagicMock(content=[mock_block])
        mock_client.messages.create.return_value = mock_response

        result = await query_anthropic("claude-sonnet-4-6", "test")

        assert result == "hello"
        mock_client.messages.create.assert_called_once()
        mock_client.messages.stream.assert_not_called()


class TestQueryAnthropic:
    @pytest.mark.asyncio
    @patch("auto_scientist.models.anthropic_client.AsyncAnthropic")
    async def test_standard_call(self, mock_cls):
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        mock_block = MagicMock(text="anthropic response")
        mock_response = MagicMock(content=[mock_block])
        mock_client.messages.create.return_value = mock_response

        result = await query_anthropic("claude-sonnet-4-6", "test prompt")

        assert result == "anthropic response"
        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["model"] == "claude-sonnet-4-6"
        assert "tools" not in call_kwargs

    @pytest.mark.asyncio
    @patch("auto_scientist.models.anthropic_client.AsyncAnthropic")
    async def test_web_search_adds_tool(self, mock_cls):
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        mock_block = MagicMock(text="searched")
        mock_response = MagicMock(content=[mock_block])
        mock_client.messages.create.return_value = mock_response

        result = await query_anthropic("claude-sonnet-4-6", "search", web_search=True)

        assert result == "searched"
        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert any(t["type"] == "web_search_20250305" for t in call_kwargs["tools"])

    @pytest.mark.asyncio
    @patch("auto_scientist.models.anthropic_client.AsyncAnthropic")
    async def test_empty_response_returns_empty_string(self, mock_cls):
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        # No text attribute on blocks
        mock_block = MagicMock(spec=[])
        mock_response = MagicMock(content=[mock_block])
        mock_client.messages.create.return_value = mock_response

        result = await query_anthropic("claude-sonnet-4-6", "prompt")

        assert result == ""

    @pytest.mark.asyncio
    @patch("auto_scientist.models.anthropic_client.AsyncAnthropic")
    async def test_multiple_text_blocks_joined(self, mock_cls):
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        block1 = MagicMock(text="part1")
        block2 = MagicMock(text="part2")
        mock_response = MagicMock(content=[block1, block2])
        mock_client.messages.create.return_value = mock_response

        result = await query_anthropic("claude-sonnet-4-6", "prompt")

        assert result == "part1\npart2"


# ── Anthropic reasoning tests ────────────────────────────────────────────────


class TestQueryAnthropicReasoning:
    @pytest.mark.asyncio
    @patch("auto_scientist.models.anthropic_client.AsyncAnthropic")
    async def test_no_reasoning_omits_thinking(self, mock_cls):
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        mock_response = MagicMock(content=[MagicMock(text="ok")])
        mock_client.messages.create.return_value = mock_response

        await query_anthropic("claude-sonnet-4-6", "test")

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert "thinking" not in call_kwargs

    @pytest.mark.asyncio
    @patch("auto_scientist.models.anthropic_client.AsyncAnthropic")
    async def test_off_reasoning_omits_thinking(self, mock_cls):
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        mock_response = MagicMock(content=[MagicMock(text="ok")])
        mock_client.messages.create.return_value = mock_response

        await query_anthropic("claude-sonnet-4-6", "test", reasoning=ReasoningConfig(level="off"))

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert "thinking" not in call_kwargs

    @pytest.mark.asyncio
    @patch("auto_scientist.models.anthropic_client.AsyncAnthropic")
    async def test_default_reasoning_omits_thinking(self, mock_cls):
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        mock_response = MagicMock(content=[MagicMock(text="ok")])
        mock_client.messages.create.return_value = mock_response

        await query_anthropic("claude-sonnet-4-6", "test", reasoning=ReasoningConfig(level="default"))

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert "thinking" not in call_kwargs

    @pytest.mark.asyncio
    @patch("auto_scientist.models.anthropic_client.AsyncAnthropic")
    async def test_high_reasoning_with_default_budget(self, mock_cls):
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        mock_response = MagicMock(content=[MagicMock(text="ok")])
        mock_client.messages.create.return_value = mock_response

        await query_anthropic("claude-sonnet-4-6", "test", reasoning=ReasoningConfig(level="high"))

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["thinking"]["type"] == "enabled"
        assert call_kwargs["thinking"]["budget_tokens"] == 16384

    @pytest.mark.asyncio
    @patch("auto_scientist.models.anthropic_client.AsyncAnthropic")
    async def test_budget_override(self, mock_cls):
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        mock_response = MagicMock(content=[MagicMock(text="ok")])
        mock_client.messages.create.return_value = mock_response

        await query_anthropic(
            "claude-sonnet-4-6", "test",
            reasoning=ReasoningConfig(level="high", budget=8000),
        )

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["thinking"]["budget_tokens"] == 8000

    @pytest.mark.asyncio
    @patch("auto_scientist.models.anthropic_client.AsyncAnthropic")
    async def test_max_tokens_increased_for_thinking(self, mock_cls):
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        mock_response = MagicMock(content=[MagicMock(text="ok")])
        mock_client.messages.create.return_value = mock_response

        await query_anthropic("claude-sonnet-4-6", "test", reasoning=ReasoningConfig(level="high"))

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["max_tokens"] >= 16384 + 4096


# ── OpenAI reasoning tests ───────────────────────────────────────────────────


class TestQueryOpenAIReasoning:
    @pytest.mark.asyncio
    @patch("auto_scientist.models.openai_client.AsyncOpenAI")
    async def test_no_reasoning_omits_effort(self, mock_cls):
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="ok"))]
        mock_client.chat.completions.create.return_value = mock_response

        await query_openai("gpt-5.4", "test")

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert "reasoning_effort" not in call_kwargs

    @pytest.mark.asyncio
    @patch("auto_scientist.models.openai_client.AsyncOpenAI")
    async def test_high_reasoning_chat_completions(self, mock_cls):
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="ok"))]
        mock_client.chat.completions.create.return_value = mock_response

        await query_openai("o3", "test", reasoning=ReasoningConfig(level="high"))

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["reasoning_effort"] == "high"
        assert "max_completion_tokens" in call_kwargs
        assert "max_tokens" not in call_kwargs

    @pytest.mark.asyncio
    @patch("auto_scientist.models.openai_client.AsyncOpenAI")
    async def test_high_reasoning_responses_api(self, mock_cls):
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        mock_response = MagicMock(output_text="ok")
        mock_client.responses.create.return_value = mock_response

        await query_openai(
            "o3", "test",
            web_search=True,
            reasoning=ReasoningConfig(level="high"),
        )

        call_kwargs = mock_client.responses.create.call_args.kwargs
        assert call_kwargs["reasoning"] == {"effort": "high"}

    @pytest.mark.asyncio
    @patch("auto_scientist.models.openai_client.AsyncOpenAI")
    async def test_off_reasoning_omits_effort(self, mock_cls):
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="ok"))]
        mock_client.chat.completions.create.return_value = mock_response

        await query_openai("gpt-5.4", "test", reasoning=ReasoningConfig(level="off"))

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert "reasoning_effort" not in call_kwargs


# ── Google reasoning tests ───────────────────────────────────────────────────


class TestQueryGoogleReasoning:
    @pytest.mark.asyncio
    @patch("auto_scientist.models.google_client.genai")
    async def test_no_reasoning_omits_thinking(self, mock_genai):
        mock_response = MagicMock(text="ok")
        mock_genai.Client.return_value.aio.models.generate_content = AsyncMock(
            return_value=mock_response
        )

        await query_google("gemini-2.5-pro", "test")

        call_kwargs = mock_genai.Client.return_value.aio.models.generate_content.call_args.kwargs
        config = call_kwargs.get("config")
        assert config is None or getattr(config, "thinking_config", None) is None

    @pytest.mark.asyncio
    @patch("auto_scientist.models.google_client.genai")
    @patch("auto_scientist.models.google_client.types")
    async def test_high_reasoning_gemini_25(self, mock_types, mock_genai):
        mock_response = MagicMock(text="ok")
        mock_genai.Client.return_value.aio.models.generate_content = AsyncMock(
            return_value=mock_response
        )

        await query_google(
            "gemini-2.5-pro", "test",
            reasoning=ReasoningConfig(level="high"),
        )

        # Should use ThinkingConfig with thinking_budget
        mock_types.ThinkingConfig.assert_called_once_with(thinking_budget=16384)

    @pytest.mark.asyncio
    @patch("auto_scientist.models.google_client.genai")
    @patch("auto_scientist.models.google_client.types")
    async def test_high_reasoning_gemini_3(self, mock_types, mock_genai):
        mock_response = MagicMock(text="ok")
        mock_genai.Client.return_value.aio.models.generate_content = AsyncMock(
            return_value=mock_response
        )

        await query_google(
            "gemini-3-flash", "test",
            reasoning=ReasoningConfig(level="high"),
        )

        # Should use ThinkingConfig with thinking_level
        mock_types.ThinkingConfig.assert_called_once()
        call_kwargs = mock_types.ThinkingConfig.call_args.kwargs
        assert "thinking_level" in call_kwargs

    @pytest.mark.asyncio
    @patch("auto_scientist.models.google_client.genai")
    @patch("auto_scientist.models.google_client.types")
    async def test_budget_override_gemini_25(self, mock_types, mock_genai):
        mock_response = MagicMock(text="ok")
        mock_genai.Client.return_value.aio.models.generate_content = AsyncMock(
            return_value=mock_response
        )

        await query_google(
            "gemini-2.5-flash", "test",
            reasoning=ReasoningConfig(level="high", budget=8192),
        )

        mock_types.ThinkingConfig.assert_called_once_with(thinking_budget=8192)
