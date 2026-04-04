"""Tests for the Coder agent."""

import json
from unittest.mock import MagicMock, patch

import pytest

from auto_scientist.agents.coder import _check_runtime_success, run_coder
from auto_scientist.sdk_backend import SDKMessage


def _text_block(text: str) -> MagicMock:
    """Create a mock TextBlock-like object."""
    block = MagicMock()
    block.text = text
    del block.name
    return block


def _tool_block(name: str, input_data: dict) -> MagicMock:
    """Create a mock ToolUseBlock-like object."""
    block = MagicMock()
    block.name = name
    block.input = input_data
    return block


def _result_msg() -> SDKMessage:
    return SDKMessage(type="result", usage={})


def _assistant_msg(blocks: list) -> SDKMessage:
    return SDKMessage(type="assistant", content_blocks=blocks)


class TestRunCoder:
    @pytest.mark.asyncio
    @patch("auto_scientist.sdk_backend.claude_query")
    async def test_creates_script_at_expected_path(self, mock_query, tmp_path):
        async def fake_query(**kwargs):
            script_path = tmp_path / "v01" / "experiment.py"
            script_path.parent.mkdir(parents=True, exist_ok=True)
            script_path.write_text("print('hello')")
            yield MagicMock(
                spec_set=["result", "usage", "num_turns", "total_cost_usd", "session_id"]
            )

        mock_query.side_effect = fake_query

        result = await run_coder(
            plan={"hypothesis": "test", "changes": []},
            previous_script=tmp_path / "v00" / "experiment.py",
            output_dir=tmp_path,
            version="v01",
        )
        assert result == tmp_path / "v01" / "experiment.py"

    @pytest.mark.asyncio
    @patch("auto_scientist.sdk_backend.claude_query")
    async def test_raises_when_script_not_created(self, mock_query, tmp_path):
        async def fake_query(**kwargs):
            yield MagicMock(
                spec_set=["result", "usage", "num_turns", "total_cost_usd", "session_id"]
            )

        mock_query.side_effect = fake_query

        with pytest.raises(FileNotFoundError, match="did not create"):
            await run_coder(
                plan={"hypothesis": "test", "changes": []},
                previous_script=tmp_path / "v00" / "experiment.py",
                output_dir=tmp_path,
                version="v01",
            )

    @pytest.mark.asyncio
    @patch("auto_scientist.sdk_backend.claude_query")
    async def test_previous_script_exists_uses_has_previous(self, mock_query, tmp_path):
        captured_prompt = {}

        async def fake_query(**kwargs):
            captured_prompt["prompt"] = kwargs.get("prompt", "")
            script_path = tmp_path / "v01" / "experiment.py"
            script_path.parent.mkdir(parents=True, exist_ok=True)
            script_path.write_text("print('v01')")
            yield MagicMock(
                spec_set=["result", "usage", "num_turns", "total_cost_usd", "session_id"]
            )

        mock_query.side_effect = fake_query

        prev_dir = tmp_path / "v00"
        prev_dir.mkdir()
        previous = prev_dir / "experiment.py"
        previous.write_text("print('v00')")

        await run_coder(
            plan={"hypothesis": "test", "changes": []},
            previous_script=previous,
            output_dir=tmp_path,
            version="v01",
        )
        assert str(previous) in captured_prompt["prompt"]

    @pytest.mark.asyncio
    @patch("auto_scientist.sdk_backend.claude_query")
    async def test_no_previous_script_uses_no_previous(self, mock_query, tmp_path):
        captured_prompt = {}

        async def fake_query(**kwargs):
            captured_prompt["prompt"] = kwargs.get("prompt", "")
            script_path = tmp_path / "v01" / "experiment.py"
            script_path.parent.mkdir(parents=True, exist_ok=True)
            script_path.write_text("print('v01')")
            yield MagicMock(
                spec_set=["result", "usage", "num_turns", "total_cost_usd", "session_id"]
            )

        mock_query.side_effect = fake_query

        await run_coder(
            plan={"hypothesis": "test", "changes": []},
            previous_script=tmp_path / "nonexistent" / "experiment.py",
            output_dir=tmp_path,
            version="v01",
        )
        prompt_lower = captured_prompt["prompt"].lower()
        assert (
            "first experiment" in prompt_lower
            or "from scratch" in prompt_lower
            or "no previous" in prompt_lower
        )

    @pytest.mark.asyncio
    @patch("auto_scientist.sdk_backend.claude_query")
    async def test_write_subdirectory_allowed(self, mock_query, tmp_path):
        async def fake_query(**kwargs):
            script_path = tmp_path / "v01" / "experiment.py"
            script_path.parent.mkdir(parents=True, exist_ok=True)
            script_path.write_text("print('nested')")
            yield MagicMock(
                spec_set=["result", "usage", "num_turns", "total_cost_usd", "session_id"]
            )

        mock_query.side_effect = fake_query

        result = await run_coder(
            plan={"hypothesis": "test", "changes": []},
            previous_script=tmp_path / "nonexistent" / "experiment.py",
            output_dir=tmp_path,
            version="v01",
        )
        assert result == tmp_path / "v01" / "experiment.py"

    @pytest.mark.asyncio
    @patch("auto_scientist.sdk_backend.claude_query")
    async def test_options_configuration(self, mock_query, tmp_path):
        captured_opts = {}

        async def fake_query(**kwargs):
            captured_opts.update(kwargs)
            script_path = tmp_path / "v01" / "experiment.py"
            script_path.parent.mkdir(parents=True, exist_ok=True)
            script_path.write_text("print('test')")
            yield MagicMock(
                spec_set=["result", "usage", "num_turns", "total_cost_usd", "session_id"]
            )

        mock_query.side_effect = fake_query

        await run_coder(
            plan={"hypothesis": "test", "changes": []},
            previous_script=tmp_path / "nonexistent" / "experiment.py",
            output_dir=tmp_path,
            version="v01",
        )
        opts = captured_opts["options"]
        assert opts.allowed_tools == ["Read", "Write", "Edit", "Bash", "Glob", "Grep"]
        assert opts.max_turns == 50
        assert opts.permission_mode == "acceptEdits"

    @pytest.mark.asyncio
    @patch("auto_scientist.sdk_backend.claude_query")
    async def test_pep723_instruction_in_prompt(self, mock_query, tmp_path):
        """System prompt instructs coder to declare deps via PEP 723 inline metadata."""
        captured_opts = {}

        async def fake_query(**kwargs):
            captured_opts.update(kwargs)
            script_path = tmp_path / "v01" / "experiment.py"
            script_path.parent.mkdir(parents=True, exist_ok=True)
            script_path.write_text("print('test')")
            yield MagicMock(
                spec_set=["result", "usage", "num_turns", "total_cost_usd", "session_id"]
            )

        mock_query.side_effect = fake_query

        await run_coder(
            plan={"hypothesis": "test", "changes": []},
            previous_script=tmp_path / "nonexistent" / "experiment.py",
            output_dir=tmp_path,
            version="v01",
        )
        system = captured_opts["options"].system_prompt
        assert "# /// script" in system
        assert "uv run" in system

    @pytest.mark.asyncio
    @patch("auto_scientist.sdk_backend.claude_query")
    async def test_run_instructions_in_prompts(self, mock_query, tmp_path):
        """Prompts include run_timeout_minutes and run_command for self-execution."""
        captured_opts = {}

        async def fake_query(**kwargs):
            captured_opts.update(kwargs)
            script_path = tmp_path / "v01" / "experiment.py"
            script_path.parent.mkdir(parents=True, exist_ok=True)
            script_path.write_text("print('test')")
            yield MagicMock(
                spec_set=["result", "usage", "num_turns", "total_cost_usd", "session_id"]
            )

        mock_query.side_effect = fake_query

        await run_coder(
            plan={"hypothesis": "test", "changes": []},
            previous_script=tmp_path / "nonexistent" / "experiment.py",
            output_dir=tmp_path,
            version="v01",
            run_timeout_minutes=60,
            run_command="uv run {script_path}",
        )
        system = captured_opts["options"].system_prompt
        user = captured_opts["prompt"]
        assert "run_result.json" in system
        assert "timed_out" in system
        # run_timeout_minutes appears in the system prompt (Bash tool timeout hint)
        assert "60" in system
        # run_command appears in both system and user prompts (the actual run step)
        assert "uv run" in user
        assert "uv run" in system

    @pytest.mark.asyncio
    @patch("auto_scientist.agents.coder.get_backend")
    async def test_openai_provider_replaces_uv_with_python3(self, mock_get_backend, tmp_path):
        """Codex coder gets python3 instead of uv run for sandbox compatibility."""
        captured = {}

        async def fake_query(prompt, options):
            captured["prompt"] = prompt
            captured["system"] = options.system_prompt
            script_path = tmp_path / "v01" / "experiment.py"
            script_path.parent.mkdir(parents=True, exist_ok=True)
            script_path.write_text("print('ok')")
            yield _result_msg()

        mock_backend = MagicMock()
        mock_backend.query = fake_query
        mock_get_backend.return_value = mock_backend

        await run_coder(
            plan={"hypothesis": "test", "changes": []},
            previous_script=tmp_path / "nonexistent" / "experiment.py",
            output_dir=tmp_path,
            version="v01",
            provider="openai",
        )

        # User prompt should have python3, not uv run
        assert "python3" in captured["prompt"]
        assert "uv run" not in captured["prompt"]
        # System prompt actionable step should also have python3
        assert "python3 {script_path}" in captured["system"]

    @pytest.mark.asyncio
    @patch("auto_scientist.sdk_backend.claude_query")
    async def test_anthropic_provider_keeps_uv_run(self, mock_query, tmp_path):
        """Claude coder keeps uv run (uv works fine on the host)."""
        captured_opts = {}

        async def fake_query(**kwargs):
            captured_opts.update(kwargs)
            script_path = tmp_path / "v01" / "experiment.py"
            script_path.parent.mkdir(parents=True, exist_ok=True)
            script_path.write_text("print('ok')")
            yield MagicMock(
                spec_set=["result", "usage", "num_turns", "total_cost_usd", "session_id"]
            )

        mock_query.side_effect = fake_query

        await run_coder(
            plan={"hypothesis": "test", "changes": []},
            previous_script=tmp_path / "nonexistent" / "experiment.py",
            output_dir=tmp_path,
            version="v01",
            provider="anthropic",
        )

        user = captured_opts["prompt"]
        system = captured_opts["options"].system_prompt
        assert "uv run" in user
        assert "uv run" in system
        assert "python3" not in user

    @pytest.mark.asyncio
    @patch("auto_scientist.agents.coder.get_backend")
    async def test_openai_custom_run_command_without_uv_unchanged(self, mock_get_backend, tmp_path):
        """Custom run_command that doesn't use uv is passed through for Codex."""
        captured = {}

        async def fake_query(prompt, options):
            captured["prompt"] = prompt
            script_path = tmp_path / "v01" / "experiment.py"
            script_path.parent.mkdir(parents=True, exist_ok=True)
            script_path.write_text("print('ok')")
            yield _result_msg()

        mock_backend = MagicMock()
        mock_backend.query = fake_query
        mock_get_backend.return_value = mock_backend

        await run_coder(
            plan={"hypothesis": "test", "changes": []},
            previous_script=tmp_path / "nonexistent" / "experiment.py",
            output_dir=tmp_path,
            version="v01",
            run_command="python3 {script_path}",
            provider="openai",
        )

        # Custom python3 command should pass through unchanged
        assert "python3" in captured["prompt"]

    @pytest.mark.asyncio
    @patch("auto_scientist.agents.coder.get_backend")
    async def test_no_timeout_wrapper_in_prompts(self, mock_get_backend, tmp_path):
        """Run command should not be wrapped with shell-level timeout."""
        captured = {}

        async def fake_query(prompt, options):
            captured["prompt"] = prompt
            captured["system"] = options.system_prompt
            script_path = tmp_path / "v01" / "experiment.py"
            script_path.parent.mkdir(parents=True, exist_ok=True)
            script_path.write_text("print('ok')")
            yield _result_msg()

        mock_backend = MagicMock()
        mock_backend.query = fake_query
        mock_get_backend.return_value = mock_backend

        await run_coder(
            plan={"hypothesis": "test", "changes": []},
            previous_script=tmp_path / "nonexistent" / "experiment.py",
            output_dir=tmp_path,
            version="v01",
            provider="openai",
        )

        # Neither prompt should contain `timeout Nm` shell wrapper
        assert "timeout 120m" not in captured["prompt"]
        assert "timeout 120m" not in captured["system"]


class TestCoderMessageBuffer:
    @pytest.mark.asyncio
    @patch("auto_scientist.sdk_backend.claude_query")
    async def test_populates_message_buffer_with_text(self, mock_query, tmp_path):
        from claude_code_sdk import AssistantMessage, ResultMessage, TextBlock

        assistant_msg = MagicMock(spec=AssistantMessage)
        text_block = MagicMock(spec=TextBlock)
        text_block.text = "Writing experiment script..."
        assistant_msg.content = [text_block]
        result_msg = MagicMock(spec=ResultMessage)
        result_msg.result = ""

        async def fake_query(**kwargs):
            script_path = tmp_path / "v01" / "experiment.py"
            script_path.parent.mkdir(parents=True, exist_ok=True)
            script_path.write_text("print('hi')")
            yield assistant_msg
            yield result_msg

        mock_query.side_effect = fake_query

        buf: list[str] = []
        await run_coder(
            plan={"hypothesis": "test", "changes": []},
            previous_script=tmp_path / "nonexistent" / "experiment.py",
            output_dir=tmp_path,
            version="v01",
            message_buffer=buf,
        )
        assert any("Writing experiment script..." in entry for entry in buf)

    @pytest.mark.asyncio
    @patch("auto_scientist.sdk_backend.claude_query")
    async def test_populates_message_buffer_with_tool_use(self, mock_query, tmp_path):
        from claude_code_sdk import AssistantMessage, ResultMessage, ToolUseBlock

        tool_block = MagicMock(spec=ToolUseBlock)
        tool_block.name = "Write"
        tool_block.input = {"file_path": "/tmp/experiment.py", "content": "print('hi')"}

        assistant_msg = MagicMock(spec=AssistantMessage)
        assistant_msg.content = [tool_block]
        result_msg = MagicMock(spec=ResultMessage)
        result_msg.result = ""

        async def fake_query(**kwargs):
            script_path = tmp_path / "v01" / "experiment.py"
            script_path.parent.mkdir(parents=True, exist_ok=True)
            script_path.write_text("print('hi')")
            yield assistant_msg
            yield result_msg

        mock_query.side_effect = fake_query

        buf: list[str] = []
        await run_coder(
            plan={"hypothesis": "test", "changes": []},
            previous_script=tmp_path / "nonexistent" / "experiment.py",
            output_dir=tmp_path,
            version="v01",
            message_buffer=buf,
        )
        assert len(buf) >= 1
        assert any("Write" in entry for entry in buf)


class TestCoderRetry:
    @pytest.mark.asyncio
    @patch("auto_scientist.sdk_backend.claude_query")
    async def test_retry_on_undeclared_deps(self, mock_query, tmp_path):
        """First attempt has undeclared dep, second fixes it."""
        call_count = 0

        async def fake_query(**kwargs):
            nonlocal call_count
            call_count += 1
            script_path = tmp_path / "v01" / "experiment.py"
            script_path.parent.mkdir(parents=True, exist_ok=True)
            if call_count == 1:
                script_path.write_text(
                    '# /// script\n# dependencies = ["numpy"]\n# ///\n'
                    "import numpy\nimport pandas\nprint('hi')\n"
                )
            else:
                script_path.write_text(
                    '# /// script\n# dependencies = ["numpy", "pandas"]\n# ///\n'
                    "import numpy\nimport pandas\nprint('hi')\n"
                )
            yield MagicMock(
                spec_set=["result", "usage", "num_turns", "total_cost_usd", "session_id"]
            )

        mock_query.side_effect = fake_query

        result = await run_coder(
            plan={"hypothesis": "test", "changes": []},
            previous_script=tmp_path / "nonexistent" / "experiment.py",
            output_dir=tmp_path,
            version="v01",
        )
        assert result == tmp_path / "v01" / "experiment.py"
        assert call_count == 2

    @pytest.mark.asyncio
    @patch("auto_scientist.sdk_backend.claude_query")
    async def test_retry_on_syntax_error(self, mock_query, tmp_path):
        """First attempt produces script with syntax error, second succeeds."""
        call_count = 0

        async def fake_query(**kwargs):
            nonlocal call_count
            call_count += 1
            script_path = tmp_path / "v01" / "experiment.py"
            script_path.parent.mkdir(parents=True, exist_ok=True)
            if call_count == 1:
                script_path.write_text("def broken(\n")
            else:
                script_path.write_text("print('hello')")
            yield MagicMock(
                spec_set=["result", "usage", "num_turns", "total_cost_usd", "session_id"]
            )

        mock_query.side_effect = fake_query

        result = await run_coder(
            plan={"hypothesis": "test", "changes": []},
            previous_script=tmp_path / "nonexistent" / "experiment.py",
            output_dir=tmp_path,
            version="v01",
        )
        assert result == tmp_path / "v01" / "experiment.py"
        assert call_count == 2

    @pytest.mark.asyncio
    @patch("auto_scientist.sdk_backend.claude_query")
    async def test_retry_on_missing_script(self, mock_query, tmp_path):
        """First attempt doesn't create script, second does."""
        call_count = 0

        async def fake_query(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                script_path = tmp_path / "v01" / "experiment.py"
                script_path.parent.mkdir(parents=True, exist_ok=True)
                script_path.write_text("print('hello')")
            yield MagicMock(
                spec_set=["result", "usage", "num_turns", "total_cost_usd", "session_id"]
            )

        mock_query.side_effect = fake_query

        result = await run_coder(
            plan={"hypothesis": "test", "changes": []},
            previous_script=tmp_path / "nonexistent" / "experiment.py",
            output_dir=tmp_path,
            version="v01",
        )
        assert result == tmp_path / "v01" / "experiment.py"
        assert call_count == 2

    @pytest.mark.asyncio
    @patch("auto_scientist.sdk_backend.claude_query")
    async def test_exhausts_retries_raises(self, mock_query, tmp_path):
        """All attempts fail to create script."""

        async def fake_query(**kwargs):
            yield MagicMock(
                spec_set=["result", "usage", "num_turns", "total_cost_usd", "session_id"]
            )

        mock_query.side_effect = fake_query

        with pytest.raises(FileNotFoundError):
            await run_coder(
                plan={"hypothesis": "test", "changes": []},
                previous_script=tmp_path / "nonexistent" / "experiment.py",
                output_dir=tmp_path,
                version="v01",
            )


class TestCheckRuntimeSuccess:
    def test_success_via_run_result(self, tmp_path):
        """run_result.json with success=true returns (True, "")."""
        (tmp_path / "run_result.json").write_text(
            json.dumps({"success": True, "return_code": 0, "timed_out": False, "error": None})
        )
        ok, err = _check_runtime_success(tmp_path)
        assert ok is True
        assert err == ""

    def test_timed_out_treated_as_pass(self, tmp_path):
        """Timeouts should not trigger retry; treated as pass for retry purposes."""
        (tmp_path / "run_result.json").write_text(
            json.dumps({"success": False, "return_code": -1, "timed_out": True, "error": "timeout"})
        )
        ok, err = _check_runtime_success(tmp_path)
        assert ok is True
        assert err == ""

    def test_run_result_failure_returns_error(self, tmp_path):
        """run_result.json with success=false returns the error."""
        (tmp_path / "run_result.json").write_text(
            json.dumps(
                {
                    "success": False,
                    "return_code": 1,
                    "timed_out": False,
                    "error": "KeyError: 'Fenrite'",
                }
            )
        )
        ok, err = _check_runtime_success(tmp_path)
        assert ok is False
        assert "KeyError" in err
        assert "Fenrite" in err

    def test_no_run_result_exitcode_zero_is_success(self, tmp_path):
        """exitcode=0 but no run_result.json: coder forgot, treat as success."""
        (tmp_path / "exitcode.txt").write_text("0\n")
        (tmp_path / "results.txt").write_text("some output")
        ok, err = _check_runtime_success(tmp_path)
        assert ok is True
        # Should write a minimal run_result.json
        assert (tmp_path / "run_result.json").exists()
        data = json.loads((tmp_path / "run_result.json").read_text())
        assert data["success"] is True

    def test_no_run_result_exitcode_nonzero_returns_stderr(self, tmp_path):
        """exitcode=1 and no run_result.json: return stderr as error."""
        (tmp_path / "exitcode.txt").write_text("1\n")
        (tmp_path / "stderr.txt").write_text(
            "Traceback (most recent call last):\n"
            "  File 'experiment.py', line 42\n"
            "KeyError: 'Fenrite'\n"
        )
        ok, err = _check_runtime_success(tmp_path)
        assert ok is False
        assert "KeyError" in err
        assert "Fenrite" in err

    def test_no_artifacts_returns_generic_failure(self, tmp_path):
        """No run_result.json, no exitcode.txt: script was never run."""
        ok, err = _check_runtime_success(tmp_path)
        assert ok is False
        assert "never ran" in err.lower() or "not run" in err.lower() or "no runtime" in err.lower()

    def test_stderr_truncated(self, tmp_path):
        """Long stderr should be truncated to ~3000 chars."""
        (tmp_path / "exitcode.txt").write_text("1\n")
        long_stderr = "x" * 10000
        (tmp_path / "stderr.txt").write_text(long_stderr)
        ok, err = _check_runtime_success(tmp_path)
        assert ok is False
        assert len(err) < 4000  # some overhead for the message wrapper

    def test_malformed_run_result_treated_as_failure(self, tmp_path):
        """Invalid JSON in run_result.json should be treated as failure."""
        (tmp_path / "run_result.json").write_text("{broken json")
        (tmp_path / "exitcode.txt").write_text("1\n")
        (tmp_path / "stderr.txt").write_text("some error\n")
        ok, err = _check_runtime_success(tmp_path)
        assert ok is False
