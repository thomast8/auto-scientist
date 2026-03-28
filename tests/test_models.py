"""Tests for LLM model client wrappers."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from auto_scientist.images import ImageData
from auto_scientist.model_config import ReasoningConfig
from auto_scientist.models.anthropic_client import query_anthropic
from auto_scientist.models.google_client import query_google
from auto_scientist.models.openai_client import _make_strict_schema, query_openai
from auto_scientist.schemas import ScientistPlanOutput

FAKE_IMAGE = ImageData(data="aW1hZ2VieXRlcw==", media_type="image/png")


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

        assert result.text == "hello"
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

        assert result.text == "searched"
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

        assert result.text == "hello"
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
        mock_response.usage = MagicMock(prompt_tokens=50, completion_tokens=20)
        mock_client.chat.completions.create.return_value = mock_response

        result = await query_openai("gpt-5.4", "test prompt")

        assert result.text == "hello"
        assert result.input_tokens == 50
        assert result.output_tokens == 20
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

        assert result.text == "searched result"
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

        assert result.text == ""


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

        assert result.text == "google"
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

        assert result.text == "data"
        assert tokens == ["data"]

    @pytest.mark.asyncio
    @patch("auto_scientist.models.google_client.genai")
    async def test_no_on_token_uses_non_streaming(self, mock_genai):
        mock_response = MagicMock(text="response")
        mock_genai.Client.return_value.aio.models.generate_content = AsyncMock(
            return_value=mock_response
        )

        result = await query_google("gemini-3.1-pro-preview", "test")

        assert result.text == "response"
        mock_genai.Client.return_value.aio.models.generate_content.assert_called_once()


class TestQueryGoogle:
    @pytest.mark.asyncio
    @patch("auto_scientist.models.google_client.genai")
    async def test_standard_call(self, mock_genai):
        mock_response = MagicMock(text="google response")
        mock_response.usage_metadata = MagicMock(prompt_token_count=80, candidates_token_count=30)
        mock_genai.Client.return_value.aio.models.generate_content = AsyncMock(
            return_value=mock_response
        )

        result = await query_google("gemini-3.1-pro-preview", "test prompt")

        assert result.text == "google response"
        assert result.input_tokens == 80
        assert result.output_tokens == 30

    @pytest.mark.asyncio
    @patch("auto_scientist.models.google_client.genai")
    async def test_web_search_adds_config(self, mock_genai):
        mock_response = MagicMock(text="searched")
        mock_genai.Client.return_value.aio.models.generate_content = AsyncMock(
            return_value=mock_response
        )

        result = await query_google("gemini-3.1-pro-preview", "search", web_search=True)

        assert result.text == "searched"
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

        assert result.text == ""


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
        mock_client.messages.stream.return_value = _make_anthropic_stream_mock(["anthro", "pic"])

        tokens = []
        result = await query_anthropic("claude-sonnet-4-6", "test", on_token=tokens.append)

        assert result.text == "anthropic"
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

        assert result.text == "result"
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

        assert result.text == "hello"
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
        mock_response.usage = MagicMock(input_tokens=120, output_tokens=45)
        mock_client.messages.create.return_value = mock_response

        result = await query_anthropic("claude-sonnet-4-6", "test prompt")

        assert result.text == "anthropic response"
        assert result.input_tokens == 120
        assert result.output_tokens == 45
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

        assert result.text == "searched"
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

        assert result.text == ""

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

        assert result.text == "part1\npart2"


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
            "claude-sonnet-4-6",
            "test",
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
            "o3",
            "test",
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
            "gemini-2.5-pro",
            "test",
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
            "gemini-3-flash",
            "test",
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
            "gemini-2.5-flash",
            "test",
            reasoning=ReasoningConfig(level="high", budget=8192),
        )

        mock_types.ThinkingConfig.assert_called_once_with(thinking_budget=8192)


# ── Structured output tests ─────────────────────────────────────────────────


class TestAnthropicStructuredOutput:
    @pytest.mark.asyncio
    @patch("auto_scientist.models.anthropic_client.AsyncAnthropic")
    async def test_response_schema_uses_tool_use(self, mock_cls):
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client

        # Simulate tool_use response
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.name = "submit_response"
        tool_block.input = {"hypothesis": "test", "strategy": "incremental"}
        mock_response = MagicMock(content=[tool_block])
        mock_client.messages.create.return_value = mock_response

        result = await query_anthropic(
            "claude-sonnet-4-6",
            "plan something",
            response_schema=ScientistPlanOutput,
        )

        call_kwargs = mock_client.messages.create.call_args.kwargs
        # Should have a tool defined with the schema
        tools = call_kwargs.get("tools", [])
        assert any(t.get("name") == "submit_response" for t in tools)
        # Should force tool choice
        assert call_kwargs.get("tool_choice", {}).get("type") == "tool"
        # Result should be JSON string of tool input
        parsed = json.loads(result.text)
        assert parsed["hypothesis"] == "test"

    @pytest.mark.asyncio
    @patch("auto_scientist.models.anthropic_client.AsyncAnthropic")
    async def test_system_prompt_passed(self, mock_cls):
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        mock_response = MagicMock(content=[MagicMock(text="ok")])
        mock_client.messages.create.return_value = mock_response

        await query_anthropic(
            "claude-sonnet-4-6",
            "test",
            system_prompt="You are a scientist.",
        )

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs.get("system") == "You are a scientist."

    @pytest.mark.asyncio
    @patch("auto_scientist.models.anthropic_client.AsyncAnthropic")
    async def test_response_schema_with_web_search_no_forced_tool_choice(self, mock_cls):
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client

        # Simulate tool_use response with submit_response
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.name = "submit_response"
        tool_block.input = {"hypothesis": "test", "strategy": "incremental"}
        mock_response = MagicMock(content=[tool_block])
        mock_client.messages.create.return_value = mock_response

        result = await query_anthropic(
            "claude-sonnet-4-6",
            "plan something",
            web_search=True,
            response_schema=ScientistPlanOutput,
        )

        call_kwargs = mock_client.messages.create.call_args.kwargs
        tools = call_kwargs.get("tools", [])
        # Should have both web_search and submit_response tools
        assert any(t.get("type") == "web_search_20250305" for t in tools)
        assert any(t.get("name") == "submit_response" for t in tools)
        # Should NOT force tool_choice when web_search is also enabled
        assert "tool_choice" not in call_kwargs
        # Should still extract the submit_response tool input
        parsed = json.loads(result.text)
        assert parsed["hypothesis"] == "test"

    @pytest.mark.asyncio
    @patch("auto_scientist.models.anthropic_client.AsyncAnthropic")
    async def test_response_schema_extracts_submit_from_mixed_blocks(self, mock_cls):
        """When response has web_search + submit_response blocks, extract submit_response."""
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client

        web_block = MagicMock()
        web_block.type = "tool_use"
        web_block.name = "web_search"
        web_block.input = {"query": "test"}
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "some reasoning"
        submit_block = MagicMock()
        submit_block.type = "tool_use"
        submit_block.name = "submit_response"
        submit_block.input = {"hypothesis": "found it", "strategy": "exploratory"}
        mock_response = MagicMock(content=[web_block, text_block, submit_block])
        mock_client.messages.create.return_value = mock_response

        result = await query_anthropic(
            "claude-sonnet-4-6",
            "test",
            web_search=True,
            response_schema=ScientistPlanOutput,
        )

        parsed = json.loads(result.text)
        assert parsed["hypothesis"] == "found it"

    @pytest.mark.asyncio
    @patch("auto_scientist.models.anthropic_client.AsyncAnthropic")
    async def test_response_schema_ignores_non_submit_tool_use(self, mock_cls):
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client

        # Response has a tool_use block that is NOT submit_response
        other_tool_block = MagicMock()
        other_tool_block.type = "tool_use"
        other_tool_block.name = "web_search"
        other_tool_block.input = {"query": "test"}
        mock_response = MagicMock(content=[other_tool_block])
        mock_client.messages.create.return_value = mock_response

        with pytest.raises(RuntimeError, match="did not call submit_response"):
            await query_anthropic(
                "claude-sonnet-4-6",
                "test",
                response_schema=ScientistPlanOutput,
            )


class TestMakeStrictSchema:
    def test_adds_additional_properties_to_objects(self):
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
        }
        result = _make_strict_schema(schema)
        assert result["additionalProperties"] is False

    def test_recursive_into_defs(self):
        schema = {
            "$defs": {
                "Item": {
                    "type": "object",
                    "properties": {"id": {"type": "integer"}},
                },
            },
            "type": "object",
            "properties": {"items": {"type": "array"}},
        }
        result = _make_strict_schema(schema)
        assert result["additionalProperties"] is False
        assert result["$defs"]["Item"]["additionalProperties"] is False

    def test_preserves_existing_additional_properties(self):
        schema = {
            "type": "object",
            "additionalProperties": True,
            "properties": {},
        }
        result = _make_strict_schema(schema)
        assert result["additionalProperties"] is True

    def test_non_object_types_unchanged(self):
        schema = {"type": "string"}
        result = _make_strict_schema(schema)
        assert "additionalProperties" not in result


class TestOpenAIStructuredOutputWithWebSearch:
    @pytest.mark.asyncio
    @patch("auto_scientist.models.openai_client.AsyncOpenAI")
    async def test_response_schema_with_web_search_uses_text_format(self, mock_cls):
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        mock_response = MagicMock(output_text='{"hypothesis": "test"}')
        mock_client.responses.create.return_value = mock_response

        result = await query_openai(
            "gpt-5.4",
            "plan something",
            web_search=True,
            response_schema=ScientistPlanOutput,
        )

        call_kwargs = mock_client.responses.create.call_args.kwargs
        # Should use Responses API with text.format for structured output
        text_format = call_kwargs.get("text", {}).get("format", {})
        assert text_format.get("type") == "json_schema"
        assert "schema" in text_format
        # Schema should have additionalProperties: false for OpenAI strict mode
        schema = text_format["schema"]
        assert schema.get("additionalProperties") is False
        # Should still have web search tool
        assert any(t["type"] == "web_search_preview" for t in call_kwargs["tools"])
        assert result.text == '{"hypothesis": "test"}'


class TestOpenAIStructuredOutput:
    @pytest.mark.asyncio
    @patch("auto_scientist.models.openai_client.AsyncOpenAI")
    async def test_response_schema_uses_response_format(self, mock_cls):
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content='{"hypothesis": "test"}'))]
        mock_client.chat.completions.create.return_value = mock_response

        result = await query_openai(
            "gpt-5.4",
            "plan something",
            response_schema=ScientistPlanOutput,
        )

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        rf = call_kwargs.get("response_format")
        assert rf is not None
        assert rf["type"] == "json_schema"
        assert result.text == '{"hypothesis": "test"}'

    @pytest.mark.asyncio
    @patch("auto_scientist.models.openai_client.AsyncOpenAI")
    async def test_system_prompt_passed(self, mock_cls):
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="ok"))]
        mock_client.chat.completions.create.return_value = mock_response

        await query_openai(
            "gpt-5.4",
            "test",
            system_prompt="You are a scientist.",
        )

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        messages = call_kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a scientist."


class TestGoogleStructuredOutput:
    @pytest.mark.asyncio
    @patch("auto_scientist.models.google_client.genai")
    async def test_response_schema_adds_config(self, mock_genai):
        mock_response = MagicMock(text='{"hypothesis": "test"}')
        mock_genai.Client.return_value.aio.models.generate_content = AsyncMock(
            return_value=mock_response
        )

        result = await query_google(
            "gemini-2.5-pro",
            "plan something",
            response_schema=ScientistPlanOutput,
        )

        call_kwargs = mock_genai.Client.return_value.aio.models.generate_content.call_args.kwargs
        config = call_kwargs.get("config")
        assert config is not None
        assert result.text == '{"hypothesis": "test"}'

    @pytest.mark.asyncio
    @patch("auto_scientist.models.google_client.types")
    @patch("auto_scientist.models.google_client.genai")
    async def test_response_schema_with_web_search_coexist(self, mock_genai, mock_types):
        mock_response = MagicMock(text='{"hypothesis": "test"}')
        mock_genai.Client.return_value.aio.models.generate_content = AsyncMock(
            return_value=mock_response
        )

        await query_google(
            "gemini-2.5-pro",
            "plan something",
            web_search=True,
            response_schema=ScientistPlanOutput,
        )

        # Verify GenerateContentConfig was called with both tools and response_schema
        config_call = mock_types.GenerateContentConfig.call_args
        config_kwargs = config_call.kwargs
        assert "tools" in config_kwargs
        assert "response_schema" in config_kwargs
        assert config_kwargs["response_mime_type"] == "application/json"

    @pytest.mark.asyncio
    @patch("auto_scientist.models.google_client.genai")
    async def test_system_prompt_passed(self, mock_genai):
        mock_response = MagicMock(text="ok")
        mock_genai.Client.return_value.aio.models.generate_content = AsyncMock(
            return_value=mock_response
        )

        await query_google(
            "gemini-2.5-pro",
            "test",
            system_prompt="You are a scientist.",
        )

        call_kwargs = mock_genai.Client.return_value.aio.models.generate_content.call_args.kwargs
        config = call_kwargs.get("config")
        assert config is not None


# ── Multimodal image tests ─────────────────────────────────────────────────


class TestAnthropicImages:
    @pytest.mark.asyncio
    @patch("auto_scientist.models.anthropic_client.AsyncAnthropic")
    async def test_images_builds_content_blocks(self, mock_cls):
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        mock_response = MagicMock(content=[MagicMock(text="ok")])
        mock_client.messages.create.return_value = mock_response

        await query_anthropic("claude-sonnet-4-6", "describe", images=[FAKE_IMAGE])

        call_kwargs = mock_client.messages.create.call_args.kwargs
        content = call_kwargs["messages"][0]["content"]
        assert isinstance(content, list)
        assert content[0] == {"type": "text", "text": "describe"}
        assert content[1]["type"] == "image"
        assert content[1]["source"]["type"] == "base64"
        assert content[1]["source"]["data"] == FAKE_IMAGE.data

    @pytest.mark.asyncio
    @patch("auto_scientist.models.anthropic_client.AsyncAnthropic")
    async def test_no_images_keeps_string_content(self, mock_cls):
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        mock_response = MagicMock(content=[MagicMock(text="ok")])
        mock_client.messages.create.return_value = mock_response

        await query_anthropic("claude-sonnet-4-6", "test prompt")

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["messages"][0]["content"] == "test prompt"

    @pytest.mark.asyncio
    @patch("auto_scientist.models.anthropic_client.AsyncAnthropic")
    async def test_empty_images_keeps_string_content(self, mock_cls):
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        mock_response = MagicMock(content=[MagicMock(text="ok")])
        mock_client.messages.create.return_value = mock_response

        await query_anthropic("claude-sonnet-4-6", "test", images=[])

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["messages"][0]["content"] == "test"


class TestOpenAIImages:
    @pytest.mark.asyncio
    @patch("auto_scientist.models.openai_client.AsyncOpenAI")
    async def test_images_chat_completions(self, mock_cls):
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="ok"))]
        mock_client.chat.completions.create.return_value = mock_response

        await query_openai("gpt-5.4", "describe", images=[FAKE_IMAGE])

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        content = call_kwargs["messages"][-1]["content"]
        assert isinstance(content, list)
        assert content[0] == {"type": "text", "text": "describe"}
        assert content[1]["type"] == "image_url"
        assert "data:image/png;base64," in content[1]["image_url"]["url"]

    @pytest.mark.asyncio
    @patch("auto_scientist.models.openai_client.AsyncOpenAI")
    async def test_images_responses_api(self, mock_cls):
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        mock_response = MagicMock(output_text="ok")
        mock_client.responses.create.return_value = mock_response

        await query_openai("gpt-5.4", "describe", web_search=True, images=[FAKE_IMAGE])

        call_kwargs = mock_client.responses.create.call_args.kwargs
        inp = call_kwargs["input"]
        assert isinstance(inp, list)
        content = inp[0]["content"]
        assert content[0] == {"type": "input_text", "text": "describe"}
        assert content[1]["type"] == "input_image"

    @pytest.mark.asyncio
    @patch("auto_scientist.models.openai_client.AsyncOpenAI")
    async def test_no_images_keeps_string(self, mock_cls):
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="ok"))]
        mock_client.chat.completions.create.return_value = mock_response

        await query_openai("gpt-5.4", "test prompt")

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["messages"][-1]["content"] == "test prompt"


class TestGoogleImages:
    @pytest.mark.asyncio
    @patch("auto_scientist.models.google_client.genai")
    @patch("auto_scientist.models.google_client.types")
    async def test_images_builds_contents_list(self, mock_types, mock_genai):
        mock_response = MagicMock(text="ok")
        mock_genai.Client.return_value.aio.models.generate_content = AsyncMock(
            return_value=mock_response
        )
        mock_part = MagicMock()
        mock_types.Part.from_bytes.return_value = mock_part

        await query_google("gemini-2.5-pro", "describe", images=[FAKE_IMAGE])

        call_kwargs = mock_genai.Client.return_value.aio.models.generate_content.call_args.kwargs
        contents = call_kwargs["contents"]
        assert isinstance(contents, list)
        assert contents[0] == "describe"
        assert contents[1] is mock_part
        mock_types.Part.from_bytes.assert_called_once()

    @pytest.mark.asyncio
    @patch("auto_scientist.models.google_client.genai")
    async def test_no_images_keeps_string(self, mock_genai):
        mock_response = MagicMock(text="ok")
        mock_genai.Client.return_value.aio.models.generate_content = AsyncMock(
            return_value=mock_response
        )

        await query_google("gemini-2.5-pro", "test prompt")

        call_kwargs = mock_genai.Client.return_value.aio.models.generate_content.call_args.kwargs
        assert call_kwargs["contents"] == "test prompt"
