"""Tests for the Discovery agent."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from auto_scientist.agents.discovery import run_discovery
from auto_scientist.state import ExperimentState


def test_run_discovery_is_async():
    assert asyncio.iscoroutinefunction(run_discovery)


class TestRunDiscovery:
    @pytest.mark.asyncio
    @patch("auto_scientist.agents.discovery.ClaudeSDKClient")
    async def test_interactive_mode_includes_ask_user(self, mock_client_cls, tmp_path):
        """Interactive mode should add AskUserQuestion to allowed tools."""
        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_client.query = AsyncMock()

        async def empty_iter():
            return
            yield  # make it an async generator

        mock_client.receive_response = MagicMock(return_value=empty_iter())

        # Create expected output files
        config_path = tmp_path / "domain_config.json"
        config_data = {
            "name": "test", "description": "Test domain",
            "data_paths": ["data.csv"],
        }
        config_path.write_text(json.dumps(config_data))

        state = ExperimentState(domain="auto", goal="test goal")

        config = await run_discovery(
            state=state, data_path=tmp_path / "data.csv",
            output_dir=tmp_path, interactive=True,
        )

        # Verify AskUserQuestion was in the options
        options = mock_client_cls.call_args.kwargs["options"]
        assert "AskUserQuestion" in options.allowed_tools

    @pytest.mark.asyncio
    @patch("auto_scientist.agents.discovery.ClaudeSDKClient")
    async def test_missing_config_raises(self, mock_client_cls, tmp_path):
        mock_client = AsyncMock()
        mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        mock_client.query = AsyncMock()

        async def empty_iter():
            return
            yield

        mock_client.receive_response = MagicMock(return_value=empty_iter())

        state = ExperimentState(domain="auto", goal="test goal")

        with pytest.raises(FileNotFoundError, match="domain config"):
            await run_discovery(
                state=state, data_path=tmp_path / "data.csv",
                output_dir=tmp_path,
            )

