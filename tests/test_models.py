"""Tests for LLM model client wrappers."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from auto_scientist.models.anthropic_client import query_anthropic
from auto_scientist.models.google_client import query_google
from auto_scientist.models.openai_client import query_openai


class TestQueryOpenAI:
    @pytest.mark.asyncio
    @patch("auto_scientist.models.openai_client.AsyncOpenAI")
    async def test_standard_call(self, mock_cls):
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="hello"))]
        mock_client.chat.completions.create.return_value = mock_response

        result = await query_openai("gpt-4o", "test prompt")

        assert result == "hello"
        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4o"
        assert call_kwargs["messages"][0]["content"] == "test prompt"

    @pytest.mark.asyncio
    @patch("auto_scientist.models.openai_client.AsyncOpenAI")
    async def test_web_search_uses_responses_api(self, mock_cls):
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        mock_response = MagicMock(output_text="searched result")
        mock_client.responses.create.return_value = mock_response

        result = await query_openai("gpt-4o", "search this", web_search=True)

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

        result = await query_openai("gpt-4o", "prompt")

        assert result == ""


class TestQueryGoogle:
    @pytest.mark.asyncio
    @patch("auto_scientist.models.google_client.genai")
    async def test_standard_call(self, mock_genai):
        mock_response = MagicMock(text="google response")
        mock_genai.Client.return_value.aio.models.generate_content = AsyncMock(
            return_value=mock_response
        )

        result = await query_google("gemini-2.5-pro", "test prompt")

        assert result == "google response"

    @pytest.mark.asyncio
    @patch("auto_scientist.models.google_client.genai")
    async def test_web_search_adds_config(self, mock_genai):
        mock_response = MagicMock(text="searched")
        mock_genai.Client.return_value.aio.models.generate_content = AsyncMock(
            return_value=mock_response
        )

        result = await query_google("gemini-2.5-pro", "search", web_search=True)

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

        result = await query_google("gemini-2.5-pro", "prompt")

        assert result == ""


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
