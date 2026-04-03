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
from pathlib import Path
from typing import Any

from auto_scientist.prompts.coder import (
    CODER_HAS_PREVIOUS,
    CODER_NO_PREVIOUS,
    CODER_SYSTEM,
    CODER_USER,
)
from auto_scientist.retry import QueryResult, ValidationError, agent_retry_loop
from auto_scientist.sdk_backend import CODEX_SANDBOX_ADDENDUM, SDKOptions, get_backend
from auto_scientist.sdk_utils import (
    append_block_to_buffer,
    collect_text_from_query,
    with_turn_budget,
)

logger = logging.getLogger(__name__)


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


async def run_coder(
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
    # Replace uv run with python3 BEFORE formatting prompts so both system
    # and user prompts get the corrected command.
    if provider == "openai" and "uv run" in run_command:
        run_command = run_command.replace("uv run", "python3", 1)

    system_prompt = CODER_SYSTEM.format(
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

    user_prompt = CODER_USER.format(
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
    backend = get_backend(provider)
    options = SDKOptions(
        system_prompt=with_turn_budget(system_prompt, max_turns, allowed_tools),
        allowed_tools=allowed_tools,
        max_turns=max_turns,
        permission_mode="acceptEdits",
        cwd=output_dir,
        model=model,
        extra_args={"setting-sources": ""},
    )

    async def _query(prompt: str, resume_session_id: str | None) -> QueryResult:
        opts = options
        if resume_session_id is not None:
            opts = SDKOptions(
                system_prompt=options.system_prompt,
                allowed_tools=options.allowed_tools,
                max_turns=options.max_turns,
                permission_mode=options.permission_mode,
                cwd=options.cwd,
                model=options.model,
                extra_args=options.extra_args,
                resume=resume_session_id,
            )
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
        return new_script_path

    def _on_exhausted(result: QueryResult | None, error: Exception) -> Path:
        # If the script exists but has a syntax error, return it anyway;
        # the runner will catch the error.
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
