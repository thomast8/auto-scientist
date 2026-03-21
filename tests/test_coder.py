"""Tests for the Coder agent."""

from unittest.mock import MagicMock, patch

import pytest

from auto_scientist.agents.coder import run_coder


class TestRunCoder:
    @pytest.mark.asyncio
    @patch("auto_scientist.agents.coder.query")
    async def test_creates_script_at_expected_path(self, mock_query, tmp_path):
        from auto_scientist.agents.coder import ResultMessage
        result_msg = MagicMock(spec=ResultMessage)

        async def fake_query(**kwargs):
            # Simulate agent writing the file
            script_path = tmp_path / "v01" / "experiment.py"
            script_path.parent.mkdir(parents=True, exist_ok=True)
            script_path.write_text("print('hello')")
            yield result_msg

        mock_query.side_effect = fake_query

        plan = {"hypothesis": "test", "changes": []}
        previous = tmp_path / "v00" / "experiment.py"

        result = await run_coder(
            plan=plan, previous_script=previous,
            output_dir=tmp_path, version="v01",
        )

        assert result == tmp_path / "v01" / "experiment.py"

    @pytest.mark.asyncio
    @patch("auto_scientist.agents.coder.query")
    async def test_raises_when_script_not_created(self, mock_query, tmp_path):
        from auto_scientist.agents.coder import ResultMessage
        result_msg = MagicMock(spec=ResultMessage)

        async def fake_query(**kwargs):
            yield result_msg

        mock_query.side_effect = fake_query

        plan = {"hypothesis": "test", "changes": []}
        previous = tmp_path / "v00" / "experiment.py"

        with pytest.raises(FileNotFoundError, match="did not create"):
            await run_coder(
                plan=plan, previous_script=previous,
                output_dir=tmp_path, version="v01",
            )

    @pytest.mark.asyncio
    @patch("auto_scientist.agents.coder.query")
    async def test_previous_script_exists_uses_has_previous(self, mock_query, tmp_path):
        from auto_scientist.agents.coder import ResultMessage
        result_msg = MagicMock(spec=ResultMessage)

        captured_prompt = {}

        async def fake_query(**kwargs):
            captured_prompt["prompt"] = kwargs.get("prompt", "")
            script_path = tmp_path / "v01" / "experiment.py"
            script_path.parent.mkdir(parents=True, exist_ok=True)
            script_path.write_text("print('v01')")
            yield result_msg

        mock_query.side_effect = fake_query

        # Create a previous script
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

        # Should reference previous script path in prompt
        assert str(previous) in captured_prompt["prompt"]

    @pytest.mark.asyncio
    @patch("auto_scientist.agents.coder.query")
    async def test_no_previous_script_uses_no_previous(self, mock_query, tmp_path):
        from auto_scientist.agents.coder import ResultMessage
        result_msg = MagicMock(spec=ResultMessage)

        captured_prompt = {}

        async def fake_query(**kwargs):
            captured_prompt["prompt"] = kwargs.get("prompt", "")
            script_path = tmp_path / "v01" / "experiment.py"
            script_path.parent.mkdir(parents=True, exist_ok=True)
            script_path.write_text("print('v01')")
            yield result_msg

        mock_query.side_effect = fake_query

        previous = tmp_path / "nonexistent" / "experiment.py"

        await run_coder(
            plan={"hypothesis": "test", "changes": []},
            previous_script=previous,
            output_dir=tmp_path,
            version="v01",
        )

        # Should contain the "no previous" / "from scratch" text
        prompt_lower = captured_prompt["prompt"].lower()
        assert "first experiment" in prompt_lower or "from scratch" in prompt_lower or "no previous" in prompt_lower

    @pytest.mark.asyncio
    @patch("auto_scientist.agents.coder.query")
    async def test_write_subdirectory_allowed(self, mock_query, tmp_path):
        from auto_scientist.agents.coder import ResultMessage
        result_msg = MagicMock(spec=ResultMessage)

        async def fake_query(**kwargs):
            script_path = tmp_path / "v01" / "experiment.py"
            script_path.parent.mkdir(parents=True, exist_ok=True)
            script_path.write_text("print('nested')")
            yield result_msg

        mock_query.side_effect = fake_query

        result = await run_coder(
            plan={"hypothesis": "test", "changes": []},
            previous_script=tmp_path / "nonexistent" / "experiment.py",
            output_dir=tmp_path, version="v01",
        )
        assert result == tmp_path / "v01" / "experiment.py"

    @pytest.mark.asyncio
    @patch("auto_scientist.agents.coder.query")
    async def test_options_configuration(self, mock_query, tmp_path):
        from auto_scientist.agents.coder import ResultMessage
        result_msg = MagicMock(spec=ResultMessage)

        captured_opts = {}

        async def fake_query(**kwargs):
            captured_opts.update(kwargs)
            script_path = tmp_path / "v01" / "experiment.py"
            script_path.parent.mkdir(parents=True, exist_ok=True)
            script_path.write_text("print('test')")
            yield result_msg

        mock_query.side_effect = fake_query

        await run_coder(
            plan={"hypothesis": "test", "changes": []},
            previous_script=tmp_path / "nonexistent" / "experiment.py",
            output_dir=tmp_path, version="v01",
        )
        opts = captured_opts["options"]
        assert opts.allowed_tools == ["Read", "Write", "Edit", "Bash", "Glob", "Grep"]
        assert opts.max_turns == 50
        assert opts.permission_mode == "acceptEdits"

    @pytest.mark.asyncio
    @patch("auto_scientist.agents.coder.query")
    async def test_pep723_instruction_in_prompt(self, mock_query, tmp_path):
        """System prompt instructs coder to declare deps via PEP 723 inline metadata."""
        from auto_scientist.agents.coder import ResultMessage
        result_msg = MagicMock(spec=ResultMessage)

        captured_opts = {}

        async def fake_query(**kwargs):
            captured_opts.update(kwargs)
            script_path = tmp_path / "v01" / "experiment.py"
            script_path.parent.mkdir(parents=True, exist_ok=True)
            script_path.write_text("print('test')")
            yield result_msg

        mock_query.side_effect = fake_query

        await run_coder(
            plan={"hypothesis": "test", "changes": []},
            previous_script=tmp_path / "nonexistent" / "experiment.py",
            output_dir=tmp_path, version="v01",
        )
        system = captured_opts["options"].system_prompt
        assert "# /// script" in system
        assert "uv run" in system

    @pytest.mark.asyncio
    @patch("auto_scientist.agents.coder.query")
    async def test_run_instructions_in_prompts(self, mock_query, tmp_path):
        """Prompts include run_timeout_minutes and run_command for self-execution."""
        from auto_scientist.agents.coder import ResultMessage
        result_msg = MagicMock(spec=ResultMessage)

        captured_opts = {}

        async def fake_query(**kwargs):
            captured_opts.update(kwargs)
            script_path = tmp_path / "v01" / "experiment.py"
            script_path.parent.mkdir(parents=True, exist_ok=True)
            script_path.write_text("print('test')")
            yield result_msg

        mock_query.side_effect = fake_query

        await run_coder(
            plan={"hypothesis": "test", "changes": []},
            previous_script=tmp_path / "nonexistent" / "experiment.py",
            output_dir=tmp_path, version="v01",
            run_timeout_minutes=60,
            run_command="uv run {script_path}",
        )
        system = captured_opts["options"].system_prompt
        user = captured_opts["prompt"]
        # System prompt should contain run_result.json instructions
        assert "run_result.json" in system
        assert "timed_out" in system
        # User prompt should contain the timeout and run command
        assert "60" in user
        assert "uv run" in user


class TestCoderMessageBuffer:
    @pytest.mark.asyncio
    @patch("auto_scientist.agents.coder.query")
    async def test_populates_message_buffer_with_text(self, mock_query, tmp_path):
        from claude_code_sdk import AssistantMessage, ResultMessage, TextBlock

        assistant_msg = MagicMock(spec=AssistantMessage)
        text_block = MagicMock(spec=TextBlock)
        text_block.text = "Writing experiment script..."
        assistant_msg.content = [text_block]
        result_msg = MagicMock(spec=ResultMessage)

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
            output_dir=tmp_path, version="v01",
            message_buffer=buf,
        )
        assert any("Writing experiment script..." in entry for entry in buf)

    @pytest.mark.asyncio
    @patch("auto_scientist.agents.coder.query")
    async def test_populates_message_buffer_with_tool_use(self, mock_query, tmp_path):
        from claude_code_sdk import AssistantMessage, ResultMessage, ToolUseBlock

        tool_block = MagicMock(spec=ToolUseBlock)
        tool_block.name = "Write"
        tool_block.input = {"file_path": "/tmp/experiment.py", "content": "print('hi')"}

        assistant_msg = MagicMock(spec=AssistantMessage)
        assistant_msg.content = [tool_block]
        result_msg = MagicMock(spec=ResultMessage)

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
            output_dir=tmp_path, version="v01",
            message_buffer=buf,
        )
        assert len(buf) >= 1
        assert any("Write" in entry for entry in buf)
