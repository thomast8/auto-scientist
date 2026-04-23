"""Reviewer test configuration: block live Claude CLI by default."""

from auto_core.testing import install_claude_sdk_mock, install_live_claude_block

install_claude_sdk_mock()

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _block_live_claude_sdk(monkeypatch):
    install_live_claude_block(monkeypatch)
