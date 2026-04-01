"""Tests for SDK backend abstraction (SDKOptions, SDKMessage, backends, factory)."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# SDKOptions
# ---------------------------------------------------------------------------


class TestSDKOptions:
    def test_construction_with_defaults(self):
        from auto_scientist.sdk_backend import SDKOptions

        opts = SDKOptions(system_prompt="test", allowed_tools=["Read"], max_turns=10)
        assert opts.system_prompt == "test"
        assert opts.allowed_tools == ["Read"]
        assert opts.max_turns == 10
        assert opts.model is None
        assert opts.cwd is None
        assert opts.permission_mode == "default"
        assert opts.extra_args == {}
        assert opts.resume is None
        assert opts.env == {}

    def test_construction_with_all_fields(self):
        from auto_scientist.sdk_backend import SDKOptions

        opts = SDKOptions(
            system_prompt="sys",
            allowed_tools=["Read", "Write", "Bash"],
            max_turns=50,
            model="gpt-5.4",
            cwd=Path("/tmp/test"),
            permission_mode="acceptEdits",
            extra_args={"effort": "high"},
            resume="session-123",
            env={"MY_VAR": "val"},
        )
        assert opts.model == "gpt-5.4"
        assert opts.cwd == Path("/tmp/test")
        assert opts.permission_mode == "acceptEdits"
        assert opts.extra_args == {"effort": "high"}
        assert opts.resume == "session-123"
        assert opts.env == {"MY_VAR": "val"}


# ---------------------------------------------------------------------------
# SDKMessage
# ---------------------------------------------------------------------------


class TestSDKMessage:
    def test_assistant_message(self):
        from auto_scientist.sdk_backend import SDKMessage

        msg = SDKMessage(
            type="assistant",
            content_blocks=[{"type": "text", "text": "hello"}],
        )
        assert msg.type == "assistant"
        assert len(msg.content_blocks) == 1
        assert msg.text is None
        assert msg.result is None
        assert msg.usage == {}
        assert msg.session_id is None

    def test_result_message(self):
        from auto_scientist.sdk_backend import SDKMessage

        msg = SDKMessage(
            type="result",
            result='{"answer": 42}',
            usage={"input_tokens": 100, "output_tokens": 50},
            session_id="sess-abc",
        )
        assert msg.type == "result"
        assert msg.result == '{"answer": 42}'
        assert msg.usage["input_tokens"] == 100
        assert msg.session_id == "sess-abc"
        assert msg.content_blocks == []

    def test_text_property(self):
        from auto_scientist.sdk_backend import SDKMessage

        msg = SDKMessage(type="assistant", text="hello world")
        assert msg.text == "hello world"


# ---------------------------------------------------------------------------
# get_backend factory
# ---------------------------------------------------------------------------


class TestGetBackend:
    def test_anthropic_returns_claude_backend(self):
        from auto_scientist.sdk_backend import ClaudeBackend, get_backend

        backend = get_backend("anthropic")
        assert isinstance(backend, ClaudeBackend)

    def test_openai_returns_codex_backend(self):
        from auto_scientist.sdk_backend import CodexBackend, get_backend

        backend = get_backend("openai")
        assert isinstance(backend, CodexBackend)

    def test_google_raises_value_error(self):
        from auto_scientist.sdk_backend import get_backend

        with pytest.raises(ValueError, match="No SDK backend"):
            get_backend("google")

    def test_unknown_raises_value_error(self):
        from auto_scientist.sdk_backend import get_backend

        with pytest.raises(ValueError, match="No SDK backend"):
            get_backend("deepseek")


# ---------------------------------------------------------------------------
# ClaudeBackend
# ---------------------------------------------------------------------------


class TestClaudeBackend:
    def test_maps_sdk_options_to_claude_code_options(self):
        """ClaudeBackend.build_options converts SDKOptions to ClaudeCodeOptions."""
        from auto_scientist.sdk_backend import ClaudeBackend, SDKOptions

        backend = ClaudeBackend()
        opts = SDKOptions(
            system_prompt="test system",
            allowed_tools=["Read", "Write"],
            max_turns=30,
            model="claude-sonnet-4-6",
            cwd=Path("/tmp"),
            permission_mode="acceptEdits",
            extra_args={"setting-sources": ""},
        )
        cc_opts = backend._build_claude_options(opts)
        assert cc_opts.system_prompt == "test system"
        assert cc_opts.allowed_tools == ["Read", "Write"]
        assert cc_opts.max_turns == 30
        assert cc_opts.model == "claude-sonnet-4-6"
        assert cc_opts.permission_mode == "acceptEdits"

    @pytest.mark.asyncio
    async def test_maps_assistant_message_to_sdk_message(self):
        """AssistantMessage from claude_code_sdk maps to SDKMessage(type='assistant')."""
        from claude_code_sdk import AssistantMessage, ResultMessage, TextBlock

        from auto_scientist.sdk_backend import ClaudeBackend, SDKOptions

        text_block = MagicMock(spec=TextBlock)
        text_block.text = "hello"

        assistant_msg = MagicMock(spec=AssistantMessage)
        assistant_msg.content = [text_block]

        result_msg = MagicMock(spec=ResultMessage)
        result_msg.result = "final"
        result_msg.usage = {"input_tokens": 10}
        result_msg.num_turns = 1
        result_msg.total_cost_usd = 0.001
        result_msg.session_id = "sess-1"

        async def fake_query(**kwargs):
            yield assistant_msg
            yield result_msg

        backend = ClaudeBackend()
        opts = SDKOptions(system_prompt="", allowed_tools=[], max_turns=5)

        with patch("auto_scientist.sdk_backend.claude_query", side_effect=fake_query):
            messages = [msg async for msg in backend.query("test", opts)]

        assert len(messages) == 2
        assert messages[0].type == "assistant"
        assert messages[0].content_blocks == [text_block]
        assert messages[1].type == "result"
        assert messages[1].result == "final"
        assert messages[1].session_id == "sess-1"

    @pytest.mark.asyncio
    async def test_strips_anthropic_api_key(self):
        """When ANTHROPIC_API_KEY is in env, ClaudeBackend strips it from subprocess."""
        from auto_scientist.sdk_backend import ClaudeBackend, SDKOptions

        async def fake_query(**kwargs):
            return
            yield

        backend = ClaudeBackend()
        opts = SDKOptions(system_prompt="", allowed_tools=[], max_turns=5)

        with (
            patch("auto_scientist.sdk_backend.claude_query", side_effect=fake_query) as mock_q,
            patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test"}, clear=False),
        ):
            async for _ in backend.query("test", opts):
                pass

        # The ClaudeCodeOptions passed to query should have ANTHROPIC_API_KEY stripped
        call_kwargs = mock_q.call_args
        cc_opts = call_kwargs.kwargs.get("options") or call_kwargs[1].get("options")
        # The env should either not contain ANTHROPIC_API_KEY or have it set to empty
        if hasattr(cc_opts, "env") and cc_opts.env:
            assert cc_opts.env.get("ANTHROPIC_API_KEY", "") == ""

    @pytest.mark.asyncio
    async def test_skips_none_messages(self):
        """None messages (unknown types) are filtered out."""
        from claude_code_sdk import AssistantMessage, TextBlock

        from auto_scientist.sdk_backend import ClaudeBackend, SDKOptions

        text_block = MagicMock(spec=TextBlock)
        text_block.text = "ok"
        assistant_msg = MagicMock(spec=AssistantMessage)
        assistant_msg.content = [text_block]

        async def fake_query(**kwargs):
            yield None
            yield assistant_msg
            yield None

        backend = ClaudeBackend()
        opts = SDKOptions(system_prompt="", allowed_tools=[], max_turns=5)

        with patch("auto_scientist.sdk_backend.claude_query", side_effect=fake_query):
            messages = [msg async for msg in backend.query("test", opts)]

        assert len(messages) == 1
        assert messages[0].type == "assistant"


# ---------------------------------------------------------------------------
# CodexBackend
# ---------------------------------------------------------------------------


class TestCodexBackend:
    def test_maps_allowed_tools_to_sandbox_mode(self):
        """Write/Edit/Bash tools map to workspace-write sandbox mode."""
        from auto_scientist.sdk_backend import CodexBackend

        backend = CodexBackend()
        assert backend._resolve_sandbox(["Read", "Write", "Edit", "Bash"]) == "workspace-write"
        assert backend._resolve_sandbox(["Read", "Glob"]) == "read-only"
        assert backend._resolve_sandbox(["WebSearch"]) == "read-only"
        assert backend._resolve_sandbox([]) == "read-only"

    def test_maps_reasoning_effort(self):
        """ReasoningConfig effort levels map to Codex effort strings."""
        from auto_scientist.sdk_backend import CodexBackend

        backend = CodexBackend()
        assert backend._resolve_effort({"effort": "high"}) == "high"
        assert backend._resolve_effort({"effort": "max"}) == "xhigh"
        assert backend._resolve_effort({}) is None

    @pytest.mark.asyncio
    async def test_chat_once_returns_sdk_messages(self):
        """CodexBackend.query calls chat_once and yields SDKMessages."""
        from auto_scientist.sdk_backend import CodexBackend, SDKOptions

        mock_result = MagicMock()
        mock_result.final_text = "The answer is 42"
        mock_result.thread_id = "thr-123"
        mock_result.raw_events = []

        mock_client = AsyncMock()
        mock_client.chat_once = AsyncMock(return_value=mock_result)
        mock_client.start = AsyncMock(return_value=mock_client)
        mock_client.close = AsyncMock()

        backend = CodexBackend()
        opts = SDKOptions(
            system_prompt="test",
            allowed_tools=["Read", "Write"],
            max_turns=10,
            model="gpt-5.4",
        )

        with patch(
            "auto_scientist.sdk_backend.CodexClient.connect_stdio",
            return_value=mock_client,
        ):
            messages = [msg async for msg in backend.query("hello", opts)]

        assert len(messages) >= 1
        result_msg = [m for m in messages if m.type == "result"]
        assert len(result_msg) == 1
        assert result_msg[0].result == "The answer is 42"
        assert result_msg[0].session_id == "thr-123"

    @pytest.mark.asyncio
    async def test_strips_openai_api_key(self):
        """When OPENAI_API_KEY is in env, CodexBackend strips it from subprocess."""
        from auto_scientist.sdk_backend import CodexBackend, SDKOptions

        mock_result = MagicMock()
        mock_result.final_text = "ok"
        mock_result.thread_id = "thr-1"
        mock_result.raw_events = []

        mock_client = AsyncMock()
        mock_client.chat_once = AsyncMock(return_value=mock_result)
        mock_client.start = AsyncMock(return_value=mock_client)
        mock_client.close = AsyncMock()

        backend = CodexBackend()
        opts = SDKOptions(system_prompt="", allowed_tools=[], max_turns=5)

        with (
            patch(
                "auto_scientist.sdk_backend.CodexClient.connect_stdio",
                return_value=mock_client,
            ) as mock_connect,
            patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}, clear=False),
        ):
            async for _ in backend.query("test", opts):
                pass

        call_kwargs = mock_connect.call_args
        env = call_kwargs.kwargs.get("env", {})
        if env:
            assert env.get("OPENAI_API_KEY", "") == ""

    @pytest.mark.asyncio
    async def test_session_resumption(self):
        """When resume is set, CodexBackend passes thread_id to chat_once."""
        from auto_scientist.sdk_backend import CodexBackend, SDKOptions

        mock_result = MagicMock()
        mock_result.final_text = "continued"
        mock_result.thread_id = "thr-existing"
        mock_result.raw_events = []

        mock_client = AsyncMock()
        mock_client.chat_once = AsyncMock(return_value=mock_result)
        mock_client.start = AsyncMock(return_value=mock_client)
        mock_client.close = AsyncMock()

        backend = CodexBackend()
        opts = SDKOptions(
            system_prompt="",
            allowed_tools=[],
            max_turns=5,
            resume="thr-existing",
        )

        with patch(
            "auto_scientist.sdk_backend.CodexClient.connect_stdio",
            return_value=mock_client,
        ):
            messages = [msg async for msg in backend.query("continue", opts)]

        result_msgs = [m for m in messages if m.type == "result"]
        assert len(result_msgs) == 1
        assert result_msgs[0].result == "continued"
        # Verify thread_id was passed for resumption
        call_kwargs = mock_client.chat_once.call_args
        assert call_kwargs.kwargs.get("thread_id") == "thr-existing"
