"""Reviewer test configuration: block live SDK subprocesses by default."""

from auto_core.testing import (
    install_claude_sdk_mock,
    install_live_claude_block,
    install_live_codex_block,
)

install_claude_sdk_mock()

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _block_live_sdks(monkeypatch):
    install_live_claude_block(monkeypatch)
    install_live_codex_block(monkeypatch)
