"""Coder agent: implements and runs the scientist's plan.

Uses query() (fresh session, reads/writes files via tools).
Tools: Read, Write, Edit, Bash, Glob, Grep.
Input (via prompt): scientist's plan JSON + previous script + run config.
Output: experiment script + run_result.json at {version_dir}/.
max_turns: 50
Safety hooks: block writes outside experiments/ dir, block writes to data files.
"""

import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

from auto_core.retry import QueryResult, ValidationError, agent_retry_loop
from auto_core.sdk_backend import CODEX_SANDBOX_ADDENDUM, SDKOptions, get_backend
from auto_core.sdk_utils import (
    append_block_to_buffer,
    collect_text_from_query,
    prepare_turn_budget,
)

from auto_reviewer.prompts.prober import (
    CODER_HAS_PREVIOUS,
    CODER_NO_PREVIOUS,
    PROBER_USER,
    build_prober_system,
)

logger = logging.getLogger(__name__)

_STDERR_TRUNCATE = 3000


def _check_runtime_success(version_dir: Path) -> tuple[bool, str]:
    """Check whether the coder's experiment script ran successfully.

    Reads run_result.json first, falls back to exitcode.txt/stderr.txt.
    Returns (True, "") on success or (False, error_description) on failure.
    Timeouts are treated as success for retry purposes (they need Scientist
    rethinking, not a coder retry).
    """
    run_result_path = version_dir / "run_result.json"
    exitcode_path = version_dir / "exitcode.txt"
    stderr_path = version_dir / "stderr.txt"

    # Try run_result.json first
    if run_result_path.exists():
        try:
            data = json.loads(run_result_path.read_text())
        except (json.JSONDecodeError, ValueError):
            pass
        else:
            if data.get("timed_out"):
                return True, ""
            if data.get("success"):
                return True, ""
            error = data.get("error") or "script failed (no error message in run_result.json)"
            return False, error

    # Fall back to exitcode.txt
    if exitcode_path.exists():
        try:
            code = int(exitcode_path.read_text().strip())
        except ValueError:
            code = -1

        if code == 0:
            run_result_path.write_text(
                json.dumps({"success": True, "return_code": 0, "timed_out": False, "error": None})
            )
            return True, ""

        stderr = ""
        if stderr_path.exists():
            stderr = stderr_path.read_text()
            if len(stderr) > _STDERR_TRUNCATE:
                stderr = f"...truncated...\n{stderr[-_STDERR_TRUNCATE:]}"
        return False, stderr or f"script exited with code {code} (no stderr captured)"

    return False, "No runtime artifacts found; the script was not run by the coder agent"


def _validate_syntax(script_path: Path) -> tuple[bool, str]:
    """Run py_compile on a script to check for syntax errors."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "py_compile", str(script_path)],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0, result.stderr


def _validate_deps(script_path: Path) -> tuple[bool, str]:
    """Check that every third-party import is covered by PEP 723 deps.

    ensure_deps is science-specific (checks matplotlib / numpy / pandas /
    openpyxl for experiment scripts). Review probes manage their own deps
    via pytest fixtures and the target repo's installed environment, so
    this is a no-op for now.
    """
    return True, ""


async def run_prober(
    plan: dict[str, Any],
    previous_script: Path,
    output_dir: Path,
    version: str,
    domain_knowledge: str = "",
    data_path: str = "",
    model: str | None = None,
    message_buffer: list[str] | None = None,
    run_timeout_minutes: int = 120,
    run_command: str = "uv run {script_path}",
    data_files_listing: str = "",
    provider: str = "anthropic",
) -> Path:
    """Implement the scientist's plan as a runnable experiment script.

    Args:
        plan: Structured plan dict from the Scientist.
        previous_script: Path to the previous version's script.
        output_dir: Base experiments directory.
        version: Version string for the new experiment (e.g., 'v01').
        domain_knowledge: Domain-specific context.
        data_path: Absolute path to the dataset.
        data_files_listing: Pre-computed listing of files in data_path directory.

    Returns:
        Path to the newly created experiment script.
    """
    version_dir = output_dir / version
    version_dir.mkdir(parents=True, exist_ok=True)
    new_script_path = version_dir / "experiment.py"

    # Build the previous script section based on whether one exists
    has_previous = previous_script.exists() and previous_script.name != "null"
    if has_previous:
        previous_script_section = CODER_HAS_PREVIOUS.format(
            previous_script_path=str(previous_script),
        )
    else:
        previous_script_section = CODER_NO_PREVIOUS

    # Codex seatbelt sandbox: uv panics (SCDynamicStore access denied).
    # Replace uv run with python3; keep ensure_deps prefix (it's copied
    # as a local script by the orchestrator).
    if provider == "openai" and "uv run" in run_command:
        run_command = run_command.replace("uv run", "python3", 1)

    prompt_provider = "gpt" if provider == "openai" else "claude"
    system_prompt = build_prober_system(prompt_provider).format(
        data_path=data_path or "(not specified)",
        run_timeout_minutes=run_timeout_minutes,
        run_command=run_command,
    )
    if provider == "openai":
        system_prompt += CODEX_SANDBOX_ADDENDUM

    # Build data files section so coder doesn't need to discover files
    if data_files_listing:
        data_files_section = (
            f"\n<data_files>\n"
            f"Files in the data directory ({data_path}):\n"
            f"{data_files_listing}\n"
            f"</data_files>"
        )
    else:
        data_files_section = ""

    user_prompt = PROBER_USER.format(
        domain_knowledge=domain_knowledge or "(no domain knowledge provided)",
        plan_json=json.dumps(plan, indent=2),
        previous_script_section=previous_script_section,
        new_script_path=str(new_script_path),
        version_dir=str(version_dir),
        version=version,
        run_timeout_minutes=run_timeout_minutes,
        run_command=run_command,
        data_files_section=data_files_section,
    )

    max_turns = 50
    allowed_tools = ["Read", "Write", "Edit", "Bash", "Glob", "Grep"]
    budget = prepare_turn_budget(system_prompt, max_turns, allowed_tools, provider=provider)
    backend = get_backend(provider)
    options = SDKOptions(
        system_prompt=budget.system_prompt,
        allowed_tools=budget.allowed_tools,
        max_turns=budget.max_turns,
        permission_mode="acceptEdits",
        cwd=output_dir,
        model=model,
        extra_args={},
        network_access=provider == "openai",
    )

    async def _query(prompt: str, resume_session_id: str | None) -> QueryResult:
        opts = replace(options, resume=resume_session_id) if resume_session_id else options
        session_id: str | None = None
        async for message in backend.query(prompt=prompt, options=opts):
            if message.type == "assistant":
                if message_buffer is not None:
                    for block in message.content_blocks:
                        append_block_to_buffer(block, message_buffer)
            elif message.type == "result":
                usage = message.usage
                session_id = message.session_id
                collect_text_from_query.last_usage = usage  # type: ignore[attr-defined]
        return QueryResult(raw_output="", session_id=session_id, usage={})

    def _validate(result: QueryResult) -> Path:
        if not new_script_path.exists():
            raise ValidationError(
                "<validation_error>\n"
                f"You did not create the script at {new_script_path}. "
                "Please write the experiment script to that exact path.\n"
                "</validation_error>"
            )
        valid, syntax_error = _validate_syntax(new_script_path)
        if not valid:
            raise ValidationError(
                "<validation_error>\n"
                f"The script at {new_script_path} has a syntax error:\n{syntax_error}\n"
                "Please fix the syntax error and rewrite the script.\n"
                "</validation_error>"
            )

        # Validate: third-party imports must be declared in PEP 723 deps
        deps_ok, deps_error = _validate_deps(new_script_path)
        if not deps_ok:
            raise ValidationError(f"<validation_error>\n{deps_error}\n</validation_error>")

        # Validate: script ran successfully at runtime
        runtime_ok, runtime_error = _check_runtime_success(new_script_path.parent)
        if not runtime_ok:
            raise ValidationError(
                "<runtime_error>\n"
                f"The script at {new_script_path} failed at runtime:\n{runtime_error}\n"
                "The script already exists on disk. Read it, fix the bug, and re-run it.\n"
                "</runtime_error>"
            )

        return new_script_path

    def _on_exhausted(result: QueryResult | None, error: Exception) -> Path:
        if result is None:
            raise error
        if new_script_path.exists():
            return new_script_path
        raise FileNotFoundError(
            f"Coder agent did not create the expected script at {new_script_path}"
        )

    return await agent_retry_loop(
        query_fn=_query,
        validate_fn=_validate,
        prompt=user_prompt,
        agent_name="Coder",
        on_exhausted=_on_exhausted,
    )
