"""Tests for the experiment runner."""

import asyncio
from pathlib import Path

import pytest

from auto_scientist.runner import RunResult, run_experiment, validate_syntax


class TestValidateSyntax:
    def test_valid_script(self, tmp_path):
        script = tmp_path / "valid.py"
        script.write_text("x = 1 + 2\nprint(x)\n")
        valid, error = validate_syntax(script)
        assert valid
        assert error == ""

    def test_invalid_script(self, tmp_path):
        script = tmp_path / "invalid.py"
        script.write_text("def foo(\n")
        valid, error = validate_syntax(script)
        assert not valid
        assert "SyntaxError" in error or "invalid" in error.lower()


class TestRunExperiment:
    @pytest.mark.asyncio
    async def test_successful_run(self, tmp_path):
        script = tmp_path / "test_script.py"
        script.write_text("print('hello from experiment')\n")
        result = await run_experiment(
            script_path=script,
            command_template=f"python {{script_path}}",
            cwd=str(tmp_path),
        )
        assert result.success
        assert "hello from experiment" in result.stdout

    @pytest.mark.asyncio
    async def test_syntax_error_caught(self, tmp_path):
        script = tmp_path / "bad.py"
        script.write_text("def foo(\n")
        result = await run_experiment(
            script_path=script,
            command_template=f"python {{script_path}}",
            cwd=str(tmp_path),
        )
        assert not result.success
        assert "Syntax error" in result.stderr

    @pytest.mark.asyncio
    async def test_runtime_error(self, tmp_path):
        script = tmp_path / "error.py"
        script.write_text("raise ValueError('test error')\n")
        result = await run_experiment(
            script_path=script,
            command_template=f"python {{script_path}}",
            cwd=str(tmp_path),
        )
        assert not result.success
        assert result.return_code != 0

    @pytest.mark.asyncio
    async def test_timeout(self, tmp_path):
        script = tmp_path / "slow.py"
        script.write_text("import time\ntime.sleep(10)\n")
        result = await run_experiment(
            script_path=script,
            command_template=f"python {{script_path}}",
            cwd=str(tmp_path),
            timeout_minutes=0.01,  # ~0.6 seconds
        )
        assert not result.success
        assert result.timed_out

    @pytest.mark.asyncio
    async def test_output_files_discovered(self, tmp_path):
        script = tmp_path / "output.py"
        script.write_text(
            "from pathlib import Path\n"
            f"Path('{tmp_path}/result.txt').write_text('done')\n"
            f"Path('{tmp_path}/plot.png').write_text('fake png')\n"
        )
        result = await run_experiment(
            script_path=script,
            command_template=f"python {{script_path}}",
            cwd=str(tmp_path),
        )
        assert result.success
        assert any("result.txt" in f for f in result.output_files)
        assert any("plot.png" in f for f in result.output_files)

    @pytest.mark.asyncio
    async def test_empty_file_valid_syntax(self, tmp_path):
        script = tmp_path / "empty.py"
        script.write_text("")
        valid, error = validate_syntax(script)
        assert valid

    @pytest.mark.asyncio
    async def test_stderr_captured(self, tmp_path):
        script = tmp_path / "stderr.py"
        script.write_text("import sys\nsys.stderr.write('warning message')\n")
        result = await run_experiment(
            script_path=script,
            command_template=f"python {{script_path}}",
            cwd=str(tmp_path),
        )
        assert result.success
        assert "warning message" in result.stderr

    @pytest.mark.asyncio
    async def test_csv_output_discovered(self, tmp_path):
        script = tmp_path / "csv_out.py"
        script.write_text(
            "from pathlib import Path\n"
            f"Path('{tmp_path}/data.csv').write_text('a,b\\n1,2\\n')\n"
        )
        result = await run_experiment(
            script_path=script,
            command_template=f"python {{script_path}}",
            cwd=str(tmp_path),
        )
        assert result.success
        assert any("data.csv" in f for f in result.output_files)

    @pytest.mark.asyncio
    async def test_nonexistent_script_fails(self, tmp_path):
        script = tmp_path / "nonexistent.py"
        result = await run_experiment(
            script_path=script,
            command_template=f"python {{script_path}}",
            cwd=str(tmp_path),
        )
        assert not result.success


class TestRunResult:
    def test_default_fields(self):
        r = RunResult(success=True)
        assert r.stdout == ""
        assert r.stderr == ""
        assert r.return_code == -1
        assert r.timed_out is False
        assert r.output_files == []

    def test_all_fields(self):
        r = RunResult(
            success=True, stdout="out", stderr="err",
            return_code=0, timed_out=False,
            output_files=["a.txt", "b.png"],
        )
        assert r.stdout == "out"
        assert r.stderr == "err"
        assert r.return_code == 0
        assert r.output_files == ["a.txt", "b.png"]
