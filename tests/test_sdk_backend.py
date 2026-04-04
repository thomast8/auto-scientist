"""Tests for SDK backend abstraction (SDKOptions, SDKMessage, backends, factory)."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# _isolation_args
# ---------------------------------------------------------------------------


class TestIsolationConfig:
    def test_returns_setting_sources_empty(self):
        """Isolation disables host settings/hooks/CLAUDE.md via --setting-sources ''."""
        from auto_scientist.sdk_backend import _isolation_config

        cfg = _isolation_config()
        assert cfg.extra_args["setting-sources"] == ""

    def test_disallows_agent_and_skill(self):
        """Agent and Skill tools are blocked to prevent host plugin recursion."""
        from auto_scientist.sdk_backend import _isolation_config

        cfg = _isolation_config()
        assert "Agent" in cfg.extra_args["disallowed-tools"]
        assert "Skill" in cfg.extra_args["disallowed-tools"]

    def test_no_bare_flag(self):
        """Isolation does NOT set --bare (which would strip most tools)."""
        from auto_scientist.sdk_backend import _isolation_config

        cfg = _isolation_config()
        assert "bare" not in cfg.extra_args

    def test_disables_auto_memory(self):
        """Auto-memory is disabled to prevent MEMORY.md leakage."""
        from auto_scientist.sdk_backend import _isolation_config

        cfg = _isolation_config()
        assert cfg.env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] == "1"

    def test_disables_claude_mds(self):
        """CLAUDE.md discovery is disabled via env var."""
        from auto_scientist.sdk_backend import _isolation_config

        cfg = _isolation_config()
        assert cfg.env["CLAUDE_CODE_DISABLE_CLAUDE_MDS"] == "1"


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
        ):
            async for _ in backend.query("test", opts):
                pass

        call_kwargs = mock_q.call_args
        cc_opts = call_kwargs.kwargs.get("options") or call_kwargs[1].get("options")
        # Uses targeted isolation, NOT --bare
        assert "bare" not in cc_opts.extra_args
        assert cc_opts.extra_args.get("setting-sources") == ""
        assert "Agent" in cc_opts.extra_args.get("disallowed-tools", "")
        assert cc_opts.env.get("CLAUDE_CODE_DISABLE_AUTO_MEMORY") == "1"
        assert cc_opts.env.get("CLAUDE_CODE_DISABLE_CLAUDE_MDS") == "1"

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

        assert CodexBackend._resolve_sandbox(["Read", "Write", "Edit", "Bash"]) == "workspace-write"
        assert CodexBackend._resolve_sandbox(["Read", "Glob"]) == "read-only"
        assert CodexBackend._resolve_sandbox(["WebSearch"]) == "read-only"
        assert CodexBackend._resolve_sandbox([]) == "read-only"
        # MCP servers need full access for subprocess spawning
        assert CodexBackend._resolve_sandbox(["WebSearch"], has_mcp=True) == "danger-full-access"
        assert (
            CodexBackend._resolve_sandbox(["Read", "Write"], has_mcp=True) == "danger-full-access"
        )
        # network_access escalates to danger-full-access for pip downloads
        assert (
            CodexBackend._resolve_sandbox(["Read", "Glob"], network_access=True)
            == "danger-full-access"
        )
        assert (
            CodexBackend._resolve_sandbox(
                ["Read", "Write", "Bash"],
                network_access=True,
            )
            == "danger-full-access"
        )

    def test_resolve_disabled_features_critic(self):
        """Critics (WebSearch only) get shell and agent tools disabled."""
        from auto_scientist.sdk_backend import CodexBackend

        disabled = CodexBackend._resolve_disabled_features(
            ["WebSearch", "mcp__predictions__read_predictions"]
        )
        assert "shell_tool" in disabled
        assert "unified_exec" in disabled
        assert "multi_agent" in disabled
        assert "tool_suggest" in disabled

    def test_resolve_disabled_features_coder(self):
        """Coders (shell tools) keep shell enabled but still disable agents."""
        from auto_scientist.sdk_backend import CodexBackend

        disabled = CodexBackend._resolve_disabled_features(
            ["Read", "Write", "Edit", "Bash", "Glob", "Grep"]
        )
        assert "shell_tool" not in disabled
        assert "unified_exec" not in disabled
        assert "multi_agent" in disabled
        assert "tool_suggest" in disabled

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

        await backend.close()

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

        await backend.close()

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
        assert env is not None, "env must always be passed (CODEX_HOME isolation)"
        assert env["OPENAI_API_KEY"] == ""
        # Must include parent PATH so subprocess can find the codex binary
        assert "PATH" in env
        # CODEX_HOME must always be set for isolation
        assert "CODEX_HOME" in env

        await backend.close()

    @pytest.mark.asyncio
    async def test_session_resumption_reuses_client(self):
        """Resume reuses the same client and passes thread_id to chat()."""
        from auto_scientist.sdk_backend import CodexBackend, SDKOptions

        # Track kwargs from each chat() call
        chat_calls: list[dict] = []

        mock_client = AsyncMock()
        mock_client.start = AsyncMock()
        mock_client.close = AsyncMock()

        async def mock_chat(*args, **kwargs):
            chat_calls.append(dict(kwargs))
            for step in [self._make_mock_step("ok", "thr-abc")]:
                yield step

        mock_client.chat = mock_chat

        backend = CodexBackend()

        # First call: fresh (no resume)
        fresh_opts = SDKOptions(system_prompt="sys", allowed_tools=[], max_turns=5)
        with patch(
            "auto_scientist.sdk_backend.CodexClient.connect_stdio",
            return_value=mock_client,
        ) as mock_connect:
            async for _ in backend.query("first", fresh_opts):
                pass

        assert mock_connect.call_count == 1, "First call should create a new client"

        # Second call: resume with session_id from first call
        resume_opts = SDKOptions(
            system_prompt="sys",
            allowed_tools=[],
            max_turns=5,
            resume="thr-abc",
        )
        with patch(
            "auto_scientist.sdk_backend.CodexClient.connect_stdio",
            return_value=mock_client,
        ) as mock_connect:
            messages = [msg async for msg in backend.query("continue", resume_opts)]

        assert mock_connect.call_count == 0, "Resume should reuse existing client, not create new"
        assert len(chat_calls) == 2
        # First call used thread_config (fresh)
        assert "thread_config" in chat_calls[0]
        assert "thread_id" not in chat_calls[0]
        # Second call used thread_id (resume)
        assert chat_calls[1].get("thread_id") == "thr-abc"
        assert "thread_config" not in chat_calls[1]

        result_msgs = [m for m in messages if m.type == "result"]
        assert len(result_msgs) == 1
        assert result_msgs[0].result == "ok"

        await backend.close()

    @pytest.mark.asyncio
    async def test_resume_without_prior_client_falls_back(self):
        """Resume with no live client logs a warning and starts a fresh thread."""
        from auto_scientist.sdk_backend import CodexBackend, SDKOptions

        chat_kwargs_captured: dict = {}
        mock_client = AsyncMock()
        mock_client.start = AsyncMock()
        mock_client.close = AsyncMock()

        async def mock_chat(*args, **kwargs):
            chat_kwargs_captured.update(kwargs)
            for step in [self._make_mock_step("fresh", "thr-new")]:
                yield step

        mock_client.chat = mock_chat

        backend = CodexBackend()
        # Resume requested but no prior call was made
        opts = SDKOptions(
            system_prompt="sys",
            allowed_tools=[],
            max_turns=5,
            resume="thr-gone",
        )

        with patch(
            "auto_scientist.sdk_backend.CodexClient.connect_stdio",
            return_value=mock_client,
        ) as mock_connect:
            messages = [msg async for msg in backend.query("retry", opts)]

        # Should have created a new client (fallback)
        assert mock_connect.call_count == 1
        # Should have used thread_config (fresh), not thread_id
        assert "thread_config" in chat_kwargs_captured
        assert "thread_id" not in chat_kwargs_captured

        result_msgs = [m for m in messages if m.type == "result"]
        assert result_msgs[0].session_id == "thr-new"

        await backend.close()

    @pytest.mark.asyncio
    async def test_close_cleans_up_client_and_home(self):
        """close() shuts down the client and removes the temp directory."""
        from auto_scientist.sdk_backend import CodexBackend, SDKOptions

        steps = [self._make_mock_step("ok", "thr-1")]
        mock_client = self._make_mock_client(steps)

        backend = CodexBackend()
        opts = SDKOptions(system_prompt="", allowed_tools=[], max_turns=5)

        with patch(
            "auto_scientist.sdk_backend.CodexClient.connect_stdio",
            return_value=mock_client,
        ):
            async for _ in backend.query("test", opts):
                pass

        # Client and codex_home should be alive
        assert backend._client is not None
        assert backend._codex_home is not None
        codex_home_path = backend._codex_home
        assert codex_home_path.exists()

        await backend.close()

        assert backend._client is None
        assert backend._codex_home is None
        assert not codex_home_path.exists()
        mock_client.close.assert_called_once()

    def test_writes_codex_home_config(self, tmp_path):
        """CodexBackend writes config.toml with MCP and feature flags."""
        from auto_scientist.sdk_backend import CodexBackend

        mcp_servers = {
            "predictions": {
                "type": "stdio",
                "command": "python3",
                "args": ["/path/to/server.py", "/path/to/data.json"],
            }
        }
        CodexBackend._write_codex_home_config(
            tmp_path,
            mcp_servers=mcp_servers,
            disabled_features=["multi_agent", "tool_suggest"],
        )

        config_path = tmp_path / "config.toml"
        assert config_path.exists(), "config.toml should be written directly in codex_home"
        assert not (tmp_path / ".codex").exists(), "should NOT create .codex/ subdir"
        content = config_path.read_text()
        assert "[mcp_servers.predictions]" in content
        assert 'command = "python3"' in content
        assert "/path/to/server.py" in content
        assert "[features]" in content
        assert "multi_agent = false" in content
        assert "tool_suggest = false" in content

    @pytest.mark.asyncio
    async def test_codex_home_always_isolates(self):
        """CODEX_HOME is always set in the subprocess env for isolation."""
        from auto_scientist.sdk_backend import CodexBackend, SDKOptions

        steps = [self._make_mock_step("ok", "thr-1")]
        mock_client = self._make_mock_client(steps)

        backend = CodexBackend()
        opts = SDKOptions(system_prompt="", allowed_tools=[], max_turns=5)

        with patch(
            "auto_scientist.sdk_backend.CodexClient.connect_stdio",
            return_value=mock_client,
        ) as mock_connect:
            async for _ in backend.query("test", opts):
                pass

        env = mock_connect.call_args.kwargs.get("env")
        assert env is not None, "env must always be passed"
        assert "CODEX_HOME" in env, "CODEX_HOME must be set for isolation"
        assert env["CODEX_HOME"] != str(Path.home() / ".codex"), (
            "CODEX_HOME must NOT point to the real ~/.codex"
        )

        await backend.close()

    @pytest.mark.asyncio
    async def test_codex_home_copies_auth(self, tmp_path):
        """Auth.json is copied to isolated CODEX_HOME for subscription auth."""
        from auto_scientist.sdk_backend import CodexBackend, SDKOptions

        steps = [self._make_mock_step("ok", "thr-1")]
        mock_client = self._make_mock_client(steps)

        backend = CodexBackend()
        opts = SDKOptions(system_prompt="", allowed_tools=[], max_turns=5)

        # Create a fake auth.json in a fake home
        fake_home = tmp_path / "fake_home"
        fake_codex = fake_home / ".codex"
        fake_codex.mkdir(parents=True)
        (fake_codex / "auth.json").write_text('{"auth_mode": "test"}')

        with (
            patch(
                "auto_scientist.sdk_backend.CodexClient.connect_stdio",
                return_value=mock_client,
            ) as mock_connect,
            patch("auto_scientist.sdk_backend.Path.home", return_value=fake_home),
        ):
            async for _ in backend.query("test", opts):
                pass

        codex_home = Path(mock_connect.call_args.kwargs["env"]["CODEX_HOME"])
        auth_copy = codex_home / "auth.json"
        assert auth_copy.exists(), "auth.json must be copied to isolated CODEX_HOME"
        assert auth_copy.read_text() == '{"auth_mode": "test"}'

        await backend.close()

    @pytest.mark.asyncio
    async def test_codex_home_contains_mcp_config(self):
        """When MCP servers are provided, config.toml exists in CODEX_HOME."""
        from auto_scientist.sdk_backend import CodexBackend, SDKOptions

        steps = [self._make_mock_step("ok", "thr-1")]
        mock_client = self._make_mock_client(steps)

        backend = CodexBackend()
        mcp = {
            "predictions": {
                "type": "stdio",
                "command": "python3",
                "args": ["/fake/server.py", "/fake/data.json"],
            }
        }
        opts = SDKOptions(system_prompt="", allowed_tools=[], max_turns=5, mcp_servers=mcp)

        codex_homes_seen: list[str] = []

        def capture_connect(**kwargs):
            codex_homes_seen.append(kwargs.get("env", {}).get("CODEX_HOME", ""))
            return mock_client

        with patch(
            "auto_scientist.sdk_backend.CodexClient.connect_stdio",
            side_effect=capture_connect,
        ):
            async for _ in backend.query("test", opts):
                pass

        assert codex_homes_seen, "connect_stdio must have been called"
        codex_home = Path(codex_homes_seen[0])
        config_path = codex_home / "config.toml"
        assert config_path.exists(), "config.toml must be written in CODEX_HOME"
        content = config_path.read_text()
        assert "[mcp_servers.predictions]" in content

        await backend.close()

    @pytest.mark.asyncio
    async def test_end_to_end_fresh_then_resume(self):
        """Simulates the ingestor/report retry flow: fresh call, then resume."""
        from auto_scientist.sdk_backend import CodexBackend, SDKOptions

        chat_calls: list[dict] = []

        mock_client = AsyncMock()
        mock_client.start = AsyncMock()
        mock_client.close = AsyncMock()

        call_count = 0

        async def mock_chat(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            chat_calls.append(dict(kwargs))
            thread_id = "thr-persist"
            for step in [self._make_mock_step(f"response-{call_count}", thread_id)]:
                yield step

        mock_client.chat = mock_chat

        backend = CodexBackend()

        with patch(
            "auto_scientist.sdk_backend.CodexClient.connect_stdio",
            return_value=mock_client,
        ) as mock_connect:
            # --- First call: fresh ---
            fresh_opts = SDKOptions(
                system_prompt="You are an ingestor.",
                allowed_tools=["Bash", "Read", "Write"],
                max_turns=30,
                model="gpt-5.4-mini",
            )
            session_id = None
            async for msg in backend.query("Ingest this data", fresh_opts):
                if msg.type == "result":
                    session_id = msg.session_id

            assert session_id == "thr-persist"
            assert mock_connect.call_count == 1

            # --- Second call: resume (simulates config validation retry) ---
            resume_opts = SDKOptions(
                system_prompt="You are an ingestor.",
                allowed_tools=["Bash", "Read", "Write"],
                max_turns=10,
                model="gpt-5.4-mini",
                resume=session_id,
            )
            resume_result = None
            async for msg in backend.query("Fix the config", resume_opts):
                if msg.type == "result":
                    resume_result = msg.result

            # Client was reused, not recreated
            assert mock_connect.call_count == 1
            assert resume_result == "response-2"

            # Verify the chat kwargs for each call
            assert "thread_config" in chat_calls[0]
            assert chat_calls[1].get("thread_id") == "thr-persist"

        await backend.close()

    @pytest.mark.asyncio
    async def test_error_mid_turn_cleans_up_client(self):
        """When chat() raises mid-turn, close() tears down client and temp dir."""
        from auto_scientist.sdk_backend import CodexBackend, SDKOptions

        mock_client = AsyncMock()
        mock_client.start = AsyncMock()
        mock_client.close = AsyncMock()

        async def failing_chat(*args, **kwargs):
            yield self._make_mock_step("partial", "thr-fail")
            raise RuntimeError("simulated mid-turn failure")

        mock_client.chat = failing_chat

        backend = CodexBackend()
        opts = SDKOptions(
            system_prompt="test",
            allowed_tools=["Read"],
            max_turns=5,
            model="gpt-5.4",
        )

        with (
            patch(
                "auto_scientist.sdk_backend.CodexClient.connect_stdio",
                return_value=mock_client,
            ),
            pytest.raises(RuntimeError, match="simulated mid-turn failure"),
        ):
            async for _ in backend.query("hello", opts):
                pass

        # Client and codex_home should be cleaned up after error
        assert backend._client is None
        assert backend._codex_home is None
        mock_client.close.assert_called_once()
