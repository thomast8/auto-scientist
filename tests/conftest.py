"""Test configuration: block live SDK subprocesses by default."""

from auto_core.testing import (
    install_claude_sdk_mock,
    install_live_claude_block,
    install_live_codex_block,
)

install_claude_sdk_mock()

import pytest  # noqa: E402

import auto_scientist  # noqa: E402, F401  -- import-time install of scientist registry
from auto_scientist._roles import install_scientist_registry  # noqa: E402


@pytest.fixture(autouse=True)
def _install_scientist_registry():
    """Keep scientist tests isolated from auto-reviewer collection side effects."""
    install_scientist_registry()


@pytest.fixture(autouse=True)
def _block_live_sdks(monkeypatch):
    install_live_claude_block(monkeypatch)
    install_live_codex_block(monkeypatch)


@pytest.fixture(autouse=True)
def _block_summarizer_api(monkeypatch):
    """Prevent real OpenAI API calls from the summarizer in all tests."""

    async def _fake_query(*args, **kwargs):
        return "mocked summary"

    monkeypatch.setattr("auto_core.summarizer._query_summary", _fake_query)
