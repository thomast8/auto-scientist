"""Tests for the Ingestor agent module."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from auto_scientist.agents.ingestor import run_ingestor


def test_run_ingestor_is_async():
    """Verify the function exists and is async."""
    assert asyncio.iscoroutinefunction(run_ingestor)


class TestRunIngestorToolSelection:
    """Verify correct tools are passed based on interactive flag."""

    @pytest.mark.asyncio
    @patch("auto_scientist.agents.ingestor.safe_query")
    async def test_interactive_mode_includes_ask_user(self, mock_query, tmp_path):
        """In interactive mode, AskUserQuestion tool should be included."""
        raw_data = tmp_path / "data.csv"
        raw_data.write_text("a,b\n1,2\n")
        output_dir = tmp_path / "experiments"
        output_dir.mkdir()
        data_dir = output_dir / "data"
        data_dir.mkdir()
        (data_dir / "output.csv").write_text("a,b\n1,2\n")

        mock_query.return_value = AsyncMock(__aiter__=lambda self: self, __anext__=AsyncMock(side_effect=StopAsyncIteration))

        await run_ingestor(raw_data, output_dir, "test goal", interactive=True)

        call_kwargs = mock_query.call_args
        options = call_kwargs.kwargs["options"]
        assert "AskUserQuestion" in options.allowed_tools

    @pytest.mark.asyncio
    @patch("auto_scientist.agents.ingestor.safe_query")
    async def test_autonomous_mode_excludes_ask_user(self, mock_query, tmp_path):
        """In autonomous mode, AskUserQuestion tool should NOT be included."""
        raw_data = tmp_path / "data.csv"
        raw_data.write_text("a,b\n1,2\n")
        output_dir = tmp_path / "experiments"
        output_dir.mkdir()
        data_dir = output_dir / "data"
        data_dir.mkdir()
        (data_dir / "output.csv").write_text("a,b\n1,2\n")

        mock_query.return_value = AsyncMock(__aiter__=lambda self: self, __anext__=AsyncMock(side_effect=StopAsyncIteration))

        await run_ingestor(raw_data, output_dir, "test goal", interactive=False)

        call_kwargs = mock_query.call_args
        options = call_kwargs.kwargs["options"]
        assert "AskUserQuestion" not in options.allowed_tools


class TestRunIngestorConfigPath:
    """Verify config_path parameter is accepted and forwarded to the prompt."""

    @pytest.mark.asyncio
    @patch("auto_scientist.agents.ingestor.safe_query")
    async def test_accepts_config_path_param(self, mock_query, tmp_path):
        """run_ingestor should accept a config_path parameter."""
        raw_data = tmp_path / "data.csv"
        raw_data.write_text("a,b\n1,2\n")
        output_dir = tmp_path / "experiments"
        output_dir.mkdir()
        data_dir = output_dir / "data"
        data_dir.mkdir()
        (data_dir / "output.csv").write_text("a,b\n1,2\n")

        mock_query.return_value = AsyncMock(
            __aiter__=lambda self: self,
            __anext__=AsyncMock(side_effect=StopAsyncIteration),
        )

        config_path = output_dir / "domain_config.json"
        await run_ingestor(
            raw_data, output_dir, "test goal",
            config_path=config_path,
        )
        mock_query.assert_called_once()

    @pytest.mark.asyncio
    @patch("auto_scientist.agents.ingestor.safe_query")
    async def test_config_path_in_prompt(self, mock_query, tmp_path):
        """When config_path is provided, it should appear in the prompt."""
        raw_data = tmp_path / "data.csv"
        raw_data.write_text("a,b\n1,2\n")
        output_dir = tmp_path / "experiments"
        output_dir.mkdir()
        data_dir = output_dir / "data"
        data_dir.mkdir()
        (data_dir / "output.csv").write_text("a,b\n1,2\n")

        mock_query.return_value = AsyncMock(
            __aiter__=lambda self: self,
            __anext__=AsyncMock(side_effect=StopAsyncIteration),
        )

        config_path = output_dir / "domain_config.json"
        await run_ingestor(
            raw_data, output_dir, "test goal",
            config_path=config_path,
        )

        call_kwargs = mock_query.call_args
        prompt = call_kwargs.kwargs["prompt"]
        assert str(config_path) in prompt


class TestRunIngestorValidation:
    """Verify error handling when agent produces no output."""

    @pytest.mark.asyncio
    @patch("auto_scientist.agents.ingestor.safe_query")
    async def test_raises_when_no_data_produced(self, mock_query, tmp_path):
        """Should raise FileNotFoundError if data dir is empty after agent runs."""
        raw_data = tmp_path / "data.csv"
        raw_data.write_text("a,b\n1,2\n")
        output_dir = tmp_path / "experiments"
        output_dir.mkdir()

        mock_query.return_value = AsyncMock(__aiter__=lambda self: self, __anext__=AsyncMock(side_effect=StopAsyncIteration))

        with pytest.raises(FileNotFoundError, match="did not produce any data files"):
            await run_ingestor(raw_data, output_dir, "test goal")


class TestRunIngestorMessageProcessing:
    @pytest.mark.asyncio
    @patch("auto_scientist.agents.ingestor.safe_query")
    async def test_assistant_message_printed(self, mock_query, tmp_path, capsys):
        """AssistantMessage text blocks should be printed to stdout."""
        from claude_code_sdk import AssistantMessage, ResultMessage, TextBlock

        raw_data = tmp_path / "data.csv"
        raw_data.write_text("a,b\n1,2\n")
        output_dir = tmp_path / "experiments"
        output_dir.mkdir()
        data_dir = output_dir / "data"
        data_dir.mkdir()
        (data_dir / "output.csv").write_text("a,b\n1,2\n")

        assistant_msg = MagicMock(spec=AssistantMessage)
        text_block = MagicMock(spec=TextBlock)
        text_block.text = "Processing your data files..."
        assistant_msg.content = [text_block]

        result_msg = MagicMock(spec=ResultMessage)

        async def fake_query(**kwargs):
            yield assistant_msg
            yield result_msg

        mock_query.side_effect = fake_query

        await run_ingestor(raw_data, output_dir, "test goal")

        captured = capsys.readouterr()
        assert "Processing your data" in captured.out


class TestRunIngestorMessageBuffer:
    @pytest.mark.asyncio
    @patch("auto_scientist.agents.ingestor.safe_query")
    async def test_populates_buffer_instead_of_printing(self, mock_query, tmp_path, capsys):
        """When message_buffer is provided, text goes to buffer, not stdout."""
        from claude_code_sdk import AssistantMessage, ResultMessage, TextBlock

        raw_data = tmp_path / "data.csv"
        raw_data.write_text("a,b\n1,2\n")
        output_dir = tmp_path / "experiments"
        output_dir.mkdir()
        data_dir = output_dir / "data"
        data_dir.mkdir()
        (data_dir / "output.csv").write_text("a,b\n1,2\n")

        assistant_msg = MagicMock(spec=AssistantMessage)
        text_block = MagicMock(spec=TextBlock)
        text_block.text = "Processing your data files..."
        assistant_msg.content = [text_block]

        result_msg = MagicMock(spec=ResultMessage)

        async def fake_query(**kwargs):
            yield assistant_msg
            yield result_msg

        mock_query.side_effect = fake_query

        buf: list[str] = []
        await run_ingestor(raw_data, output_dir, "test goal", message_buffer=buf)

        assert len(buf) == 1
        assert "Processing your data" in buf[0]
        captured = capsys.readouterr()
        assert "[ingestor]" not in captured.out


class TestRunIngestorToolBlockBuffer:
    @pytest.mark.asyncio
    @patch("auto_scientist.agents.ingestor.safe_query")
    async def test_tool_use_captured_in_buffer(self, mock_query, tmp_path):
        from claude_code_sdk import AssistantMessage, ResultMessage, ToolUseBlock

        raw_data = tmp_path / "data.csv"
        raw_data.write_text("a,b\n1,2\n")
        output_dir = tmp_path / "experiments"
        output_dir.mkdir()
        data_dir = output_dir / "data"
        data_dir.mkdir()
        (data_dir / "output.csv").write_text("a,b\n1,2\n")

        tool_block = MagicMock(spec=ToolUseBlock)
        tool_block.name = "Bash"
        tool_block.input = {"command": "head -5 data.csv"}

        assistant_msg = MagicMock(spec=AssistantMessage)
        assistant_msg.content = [tool_block]
        result_msg = MagicMock(spec=ResultMessage)

        async def fake_query(**kwargs):
            yield assistant_msg
            yield result_msg

        mock_query.side_effect = fake_query

        buf: list[str] = []
        await run_ingestor(raw_data, output_dir, "test goal", message_buffer=buf)
        assert len(buf) >= 1
        assert any("Bash" in entry for entry in buf)


class TestRunIngestorOptions:
    @pytest.mark.asyncio
    @patch("auto_scientist.agents.ingestor.safe_query")
    async def test_options(self, mock_query, tmp_path):
        raw_data = tmp_path / "data.csv"
        raw_data.write_text("a,b\n1,2\n")
        output_dir = tmp_path / "experiments"
        output_dir.mkdir()
        data_dir = output_dir / "data"
        data_dir.mkdir()
        (data_dir / "output.csv").write_text("a,b\n1,2\n")

        mock_query.return_value = AsyncMock(
            __aiter__=lambda self: self,
            __anext__=AsyncMock(side_effect=StopAsyncIteration),
        )

        await run_ingestor(raw_data, output_dir, "test goal")

        call_kwargs = mock_query.call_args
        options = call_kwargs.kwargs["options"]
        assert options.max_turns == 30
        assert options.permission_mode == "acceptEdits"

    @pytest.mark.asyncio
    @patch("auto_scientist.agents.ingestor.safe_query")
    async def test_goal_in_prompt(self, mock_query, tmp_path):
        raw_data = tmp_path / "data.csv"
        raw_data.write_text("a,b\n1,2\n")
        output_dir = tmp_path / "experiments"
        output_dir.mkdir()
        data_dir = output_dir / "data"
        data_dir.mkdir()
        (data_dir / "output.csv").write_text("a,b\n1,2\n")

        mock_query.return_value = AsyncMock(
            __aiter__=lambda self: self,
            __anext__=AsyncMock(side_effect=StopAsyncIteration),
        )

        await run_ingestor(raw_data, output_dir, "predict oxygen levels")

        call_kwargs = mock_query.call_args
        prompt = call_kwargs.kwargs["prompt"]
        assert "predict oxygen levels" in prompt

    @pytest.mark.asyncio
    @patch("auto_scientist.agents.ingestor.safe_query")
    async def test_model_override(self, mock_query, tmp_path):
        raw_data = tmp_path / "data.csv"
        raw_data.write_text("a,b\n1,2\n")
        output_dir = tmp_path / "experiments"
        output_dir.mkdir()
        data_dir = output_dir / "data"
        data_dir.mkdir()
        (data_dir / "output.csv").write_text("a,b\n1,2\n")

        mock_query.return_value = AsyncMock(
            __aiter__=lambda self: self,
            __anext__=AsyncMock(side_effect=StopAsyncIteration),
        )

        await run_ingestor(raw_data, output_dir, "test goal", model="claude-haiku-4-5-20251001")

        call_kwargs = mock_query.call_args
        options = call_kwargs.kwargs["options"]
        assert options.model == "claude-haiku-4-5-20251001"


class TestIngestorRetry:
    @pytest.mark.asyncio
    @patch("auto_scientist.agents.ingestor.safe_query")
    async def test_retry_on_empty_data_dir(self, mock_query, tmp_path):
        """First attempt produces no files, second produces data."""
        raw_data = tmp_path / "data.csv"
        raw_data.write_text("a,b\n1,2\n")
        output_dir = tmp_path / "experiments"
        output_dir.mkdir()
        call_count = 0

        async def fake_query(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                data_dir = output_dir / "data"
                data_dir.mkdir(exist_ok=True)
                (data_dir / "output.csv").write_text("a,b\n1,2\n")
            return
            yield  # make it an async generator

        mock_query.side_effect = fake_query

        result = await run_ingestor(raw_data, output_dir, "test goal")
        assert result == output_dir / "data"
        assert call_count == 2

    @pytest.mark.asyncio
    @patch("auto_scientist.agents.ingestor.safe_query")
    async def test_exhausts_retries_raises(self, mock_query, tmp_path):
        """All attempts produce no files."""
        raw_data = tmp_path / "data.csv"
        raw_data.write_text("a,b\n1,2\n")
        output_dir = tmp_path / "experiments"
        output_dir.mkdir()

        async def fake_query(**kwargs):
            return
            yield  # make it an async generator

        mock_query.side_effect = fake_query

        with pytest.raises(FileNotFoundError, match="did not produce any data files"):
            await run_ingestor(raw_data, output_dir, "test goal")
