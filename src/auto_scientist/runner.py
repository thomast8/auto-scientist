"""Subprocess experiment runner with syntax validation and timeout."""

import asyncio
import os
import subprocess
import sys
from pathlib import Path

from pydantic import BaseModel


class RunResult(BaseModel):
    """Result of running an experiment script."""

    success: bool
    stdout: str = ""
    stderr: str = ""
    return_code: int = -1
    timed_out: bool = False
    output_files: list[str] = []


def validate_syntax(script_path: Path) -> tuple[bool, str]:
    """Run py_compile on the script to check for syntax errors."""
    result = subprocess.run(
        [sys.executable, "-m", "py_compile", str(script_path)],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0, result.stderr


async def run_experiment(
    script_path: Path,
    command_template: str,
    cwd: str = ".",
    timeout_minutes: int = 120,
) -> RunResult:
    """Execute an experiment script as a subprocess.

    Args:
        script_path: Path to the Python script to run.
        command_template: Command template with {script_path} placeholder.
        cwd: Working directory for the subprocess.
        timeout_minutes: Maximum execution time in minutes.

    Returns:
        RunResult with stdout, stderr, success status, and discovered output files.
    """
    # Syntax check first
    valid, syntax_error = validate_syntax(script_path)
    if not valid:
        return RunResult(success=False, stderr=f"Syntax error:\n{syntax_error}")

    cmd = command_template.format(script_path=str(script_path))
    parts = cmd.split()

    env = {**os.environ, "PYTHONUNBUFFERED": "1"}

    try:
        process = await asyncio.create_subprocess_exec(
            *parts,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env=env,
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout_minutes * 60,
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.communicate()
            return RunResult(
                success=False,
                timed_out=True,
                stderr=f"Timed out after {timeout_minutes} minutes",
            )

        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")

        # Discover output files (PNGs, TXTs in the script's directory)
        script_dir = script_path.parent
        output_files = [
            str(f) for f in script_dir.iterdir() if f.suffix in (".png", ".txt", ".csv", ".json")
        ]

        return RunResult(
            success=process.returncode == 0,
            stdout=stdout,
            stderr=stderr,
            return_code=process.returncode or 0,
            output_files=output_files,
        )

    except FileNotFoundError as e:
        return RunResult(success=False, stderr=f"Command not found: {e}")
