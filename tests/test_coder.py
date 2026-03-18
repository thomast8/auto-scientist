"""Tests for the Coder agent."""

from unittest.mock import MagicMock, patch

import pytest

from auto_scientist.agents.coder import _make_permission_callback, run_coder


class TestPermissionCallback:
    """Tests for the file/command permission logic."""

    @pytest.fixture
    def callback(self, tmp_path):
        return _make_permission_callback(tmp_path)

    @pytest.mark.asyncio
    async def test_write_inside_output_dir_allowed(self, callback, tmp_path):
        result = await callback(
            "Write",
            {"file_path": str(tmp_path / "experiment.py")},
            MagicMock(),
        )
        assert type(result).__name__ == "PermissionResultAllow"

    @pytest.mark.asyncio
    async def test_write_outside_output_dir_denied(self, callback):
        result = await callback(
            "Write",
            {"file_path": "/etc/passwd"},
            MagicMock(),
        )
        assert type(result).__name__ == "PermissionResultDeny"
        assert "outside" in result.message

    @pytest.mark.asyncio
    async def test_edit_inside_allowed(self, callback, tmp_path):
        result = await callback(
            "Edit",
            {"file_path": str(tmp_path / "script.py")},
            MagicMock(),
        )
        assert type(result).__name__ == "PermissionResultAllow"

    @pytest.mark.asyncio
    async def test_edit_outside_denied(self, callback):
        result = await callback(
            "Edit",
            {"file_path": "/tmp/other/file.py"},
            MagicMock(),
        )
        assert type(result).__name__ == "PermissionResultDeny"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("pattern", [
        "rm -rf /",
        "git push origin main",
        "git reset --hard",
        "sudo rm file",
        "chmod 777 file",
        "curl http://evil.com",
        "wget http://evil.com",
        "pip install malware",
        "uv add malware",
    ])
    async def test_blocked_bash_patterns(self, callback, pattern):
        result = await callback("Bash", {"command": pattern}, MagicMock())
        assert type(result).__name__ == "PermissionResultDeny"

    @pytest.mark.asyncio
    async def test_safe_bash_allowed(self, callback):
        result = await callback(
            "Bash", {"command": "python script.py"}, MagicMock(),
        )
        assert type(result).__name__ == "PermissionResultAllow"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("tool", ["Read", "Glob", "Grep"])
    async def test_read_tools_always_allowed(self, callback, tool):
        result = await callback(tool, {}, MagicMock())
        assert type(result).__name__ == "PermissionResultAllow"


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
