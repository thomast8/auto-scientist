"""Tests for the Ingestor agent module."""

import asyncio
from unittest.mock import AsyncMock, patch

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
