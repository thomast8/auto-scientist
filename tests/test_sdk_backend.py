"""Tests for SDK backend abstraction (SDKOptions, SDKMessage, backends, factory)."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# _bare_settings
# ---------------------------------------------------------------------------


class TestBareSettings:
    def test_explicit_api_key_in_env(self):
        """Agent env with ANTHROPIC_API_KEY -> bare mode, no settings."""
        from auto_scientist.sdk_backend import _bare_settings

        use_bare, settings = _bare_settings({"ANTHROPIC_API_KEY": "sk-test"})
        assert use_bare is True
        assert settings is None

    def test_macos_keychain(self):
        """On macOS without API key -> bare mode with apiKeyHelper."""
        from auto_scientist.sdk_backend import _bare_settings

        with patch("auto_scientist.sdk_backend.sys") as mock_sys:
            mock_sys.platform = "darwin"
            use_bare, settings = _bare_settings({})

        assert use_bare is True
        assert settings is not None
        parsed = json.loads(settings)
        assert "apiKeyHelper" in parsed
        assert "security find-generic-password" in parsed["apiKeyHelper"]

    def test_non_macos_with_explicit_env_key(self):
        """On Linux with ANTHROPIC_API_KEY in agent env -> bare, no settings."""
        from auto_scientist.sdk_backend import _bare_settings

        with patch("auto_scientist.sdk_backend.sys") as mock_sys:
            mock_sys.platform = "linux"
            use_bare, settings = _bare_settings({"ANTHROPIC_API_KEY": "sk-test"})

        assert use_bare is True
        assert settings is None

    def test_no_auth_raises(self):
        """No macOS, no API key -> RuntimeError."""
        from auto_scientist.sdk_backend import _bare_settings

        with (
            patch("auto_scientist.sdk_backend.sys") as mock_sys,
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(RuntimeError, match="Cannot resolve Anthropic auth"),
        ):
            mock_sys.platform = "linux"
            _bare_settings({})


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

    def test_mcp_servers_defaults_to_empty(self):
        from auto_scientist.sdk_backend import SDKOptions

        opts = SDKOptions(system_prompt="test", allowed_tools=[], max_turns=5)
        assert opts.mcp_servers == {}

    def test_mcp_servers_stored(self):
        from auto_scientist.sdk_backend import SDKOptions

        mock_server = {"type": "sdk", "name": "predictions", "instance": object()}
        opts = SDKOptions(
            system_prompt="test",
            allowed_tools=[],
            max_turns=5,
            mcp_servers={"predictions": mock_server},
        )
        assert opts.mcp_servers == {"predictions": mock_server}


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
            extra_args={},
        )
        cc_opts = backend._build_claude_options(opts)
        assert cc_opts.system_prompt == "test system"
        assert cc_opts.allowed_tools == ["Read", "Write"]
        assert cc_opts.max_turns == 30
        assert cc_opts.model == "claude-sonnet-4-6"
        assert cc_opts.permission_mode == "acceptEdits"

    def test_passes_mcp_servers_when_present(self):
        from auto_scientist.sdk_backend import ClaudeBackend, SDKOptions

        mock_server = {"type": "sdk", "name": "predictions", "instance": object()}
        backend = ClaudeBackend()
        opts = SDKOptions(
            system_prompt="test",
            allowed_tools=[],
            max_turns=5,
            mcp_servers={"predictions": mock_server},
        )
        with patch("auto_scientist.sdk_backend.ClaudeCodeOptions") as mock_cls:
            backend._build_claude_options(opts)
            call_kwargs = mock_cls.call_args.kwargs
            assert call_kwargs["mcp_servers"] == {"predictions": mock_server}

    def test_omits_mcp_servers_when_empty(self):
        from auto_scientist.sdk_backend import ClaudeBackend, SDKOptions

        backend = ClaudeBackend()
        opts = SDKOptions(system_prompt="test", allowed_tools=[], max_turns=5)
        # When mcp_servers is empty, it should not be passed to ClaudeCodeOptions
        with patch("auto_scientist.sdk_backend.ClaudeCodeOptions") as mock_cls:
            backend._build_claude_options(opts)
            call_kwargs = mock_cls.call_args.kwargs
            assert "mcp_servers" not in call_kwargs

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
    async def test_bare_flag_injected(self):
        """ClaudeBackend injects --bare for subprocess isolation."""
        from auto_scientist.sdk_backend import ClaudeBackend, SDKOptions

        async def fake_query(**kwargs):
            return
            yield

        backend = ClaudeBackend()
        opts = SDKOptions(system_prompt="", allowed_tools=[], max_turns=5)

        with (
            patch("auto_scientist.sdk_backend.claude_query", side_effect=fake_query) as mock_q,
            patch("auto_scientist.sdk_backend._bare_settings", return_value=(True, None)),
        ):
            async for _ in backend.query("test", opts):
                pass

        call_kwargs = mock_q.call_args
        cc_opts = call_kwargs.kwargs.get("options") or call_kwargs[1].get("options")
        assert cc_opts.extra_args.get("bare") is None  # None = boolean flag
        assert "bare" in cc_opts.extra_args

    @pytest.mark.asyncio
    async def test_bare_settings_passed(self):
        """When _bare_settings returns settings JSON, it's passed to ClaudeCodeOptions."""
        from auto_scientist.sdk_backend import ClaudeBackend, SDKOptions

        async def fake_query(**kwargs):
            return
            yield

        backend = ClaudeBackend()
        opts = SDKOptions(system_prompt="", allowed_tools=[], max_turns=5)
        fake_settings = '{"apiKeyHelper": "echo test"}'

        with (
            patch("auto_scientist.sdk_backend.claude_query", side_effect=fake_query) as mock_q,
            patch(
                "auto_scientist.sdk_backend._bare_settings",
                return_value=(True, fake_settings),
            ),
        ):
            async for _ in backend.query("test", opts):
                pass

        call_kwargs = mock_q.call_args
        cc_opts = call_kwargs.kwargs.get("options") or call_kwargs[1].get("options")
        assert cc_opts.settings == fake_settings

    @pytest.mark.asyncio
    async def test_bare_auth_failure_propagates(self):
        """When _bare_settings raises, ClaudeBackend propagates the error."""
        from auto_scientist.sdk_backend import ClaudeBackend, SDKOptions

        backend = ClaudeBackend()
        opts = SDKOptions(system_prompt="", allowed_tools=[], max_turns=5)

        with (
            patch(
                "auto_scientist.sdk_backend._bare_settings",
                side_effect=RuntimeError("no auth"),
            ),
            pytest.raises(RuntimeError, match="no auth"),
        ):
            async for _ in backend.query("test", opts):
                pass

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
        assert backend._resolve_effort({}) == "none"

    @staticmethod
    def _make_mock_step(text: str, thread_id: str = "thr-123") -> MagicMock:
        """Create a mock ConversationStep with text and thread_id."""
        step = MagicMock()
        step.text = text
        step.thread_id = thread_id
        return step

    @staticmethod
    def _make_mock_client(steps: list[MagicMock]) -> AsyncMock:
        """Create a mock CodexClient whose chat() yields the given steps."""
        mock_client = AsyncMock()
        mock_client.start = AsyncMock()
        mock_client.close = AsyncMock()

        async def mock_chat(*args, **kwargs):
            for step in steps:
                yield step

        mock_client.chat = mock_chat
        return mock_client

    @pytest.mark.asyncio
    async def test_chat_streams_sdk_messages(self):
        """CodexBackend.query streams steps from chat() and yields SDKMessages."""
        from auto_scientist.sdk_backend import CodexBackend, SDKOptions

        steps = [self._make_mock_step("The answer is 42", "thr-123")]
        mock_client = self._make_mock_client(steps)

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

        assistant_msgs = [m for m in messages if m.type == "assistant"]
        assert len(assistant_msgs) == 1
        assert assistant_msgs[0].content_blocks[0].text == "The answer is 42"

        result_msgs = [m for m in messages if m.type == "result"]
        assert len(result_msgs) == 1
        assert result_msgs[0].result == "The answer is 42"
        assert result_msgs[0].session_id == "thr-123"
        assert result_msgs[0].usage["num_turns"] == 1

    @pytest.mark.asyncio
    async def test_counts_multiple_steps_as_turns(self):
        """CodexBackend.query counts all ConversationSteps as num_turns."""
        from auto_scientist.sdk_backend import CodexBackend, SDKOptions

        steps = [
            self._make_mock_step("thinking...", "thr-1"),
            self._make_mock_step("running code...", "thr-1"),
            self._make_mock_step("final answer", "thr-1"),
        ]
        mock_client = self._make_mock_client(steps)

        backend = CodexBackend()
        opts = SDKOptions(
            system_prompt="test",
            allowed_tools=["Read"],
            max_turns=10,
            model="gpt-5.4",
        )

        with patch(
            "auto_scientist.sdk_backend.CodexClient.connect_stdio",
            return_value=mock_client,
        ):
            messages = [msg async for msg in backend.query("hello", opts)]

        result_msgs = [m for m in messages if m.type == "result"]
        assert result_msgs[0].usage["num_turns"] == 3

    @pytest.mark.asyncio
    async def test_strips_openai_api_key(self):
        """When OPENAI_API_KEY is in env, CodexBackend strips it from subprocess."""
        from auto_scientist.sdk_backend import CodexBackend, SDKOptions

        steps = [self._make_mock_step("ok", "thr-1")]
        mock_client = self._make_mock_client(steps)

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
        env = call_kwargs.kwargs.get("env")
        assert env is not None, "env must be passed when OPENAI_API_KEY needs stripping"
        assert env["OPENAI_API_KEY"] == ""
        # Must include parent PATH so subprocess can find the codex binary
        assert "PATH" in env

    @pytest.mark.asyncio
    async def test_session_resumption(self):
        """When resume is set, CodexBackend passes thread_id to chat()."""
        from auto_scientist.sdk_backend import CodexBackend, SDKOptions

        steps = [self._make_mock_step("continued", "thr-existing")]

        # Custom mock to capture kwargs
        chat_kwargs_captured = {}
        mock_client = AsyncMock()
        mock_client.start = AsyncMock()
        mock_client.close = AsyncMock()

        async def mock_chat(*args, **kwargs):
            chat_kwargs_captured.update(kwargs)
            for step in steps:
                yield step

        mock_client.chat = mock_chat

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
        assert chat_kwargs_captured.get("thread_id") == "thr-existing"

    def test_writes_codex_mcp_config(self, tmp_path):
        """CodexBackend writes .codex/config.toml for stdio MCP servers."""
        from auto_scientist.sdk_backend import CodexBackend

        backend = CodexBackend()
        mcp_servers = {
            "predictions": {
                "type": "stdio",
                "command": "python3",
                "args": ["/path/to/server.py", "/path/to/data.json"],
            }
        }
        backend._write_codex_mcp_config(mcp_servers, tmp_path)

        config_path = tmp_path / ".codex" / "config.toml"
        assert config_path.exists()
        content = config_path.read_text()
        assert "[mcp_servers.predictions]" in content
        assert 'command = "python3"' in content
        assert "/path/to/server.py" in content
