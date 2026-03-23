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

from claude_code_sdk import (
    AssistantMessage,
    ClaudeCodeOptions,
    ResultMessage,
    query,
)

from auto_scientist.prompts.coder import (
    CODER_HAS_PREVIOUS,
    CODER_NO_PREVIOUS,
    CODER_SYSTEM,
    CODER_USER,
)
from auto_scientist.sdk_utils import append_block_to_buffer, collect_text_from_query

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 2


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
    top_level_criteria: list[dict[str, Any]] | None = None,
) -> Path:
    """Implement the scientist's plan as a runnable experiment script.

    Args:
        plan: Structured plan dict from the Scientist.
        previous_script: Path to the previous version's script.
        output_dir: Base experiments directory.
        version: Version string for the new experiment (e.g., 'v01').
        domain_knowledge: Domain-specific context.
        data_path: Absolute path to the dataset.
        top_level_criteria: Investigation-wide success criteria from orchestrator.

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

    system_prompt = CODER_SYSTEM.format(
        data_path=data_path or "(not specified)",
        run_timeout_minutes=run_timeout_minutes,
        run_command=run_command,
    )

    # Build top-level criteria section for the prompt
    if top_level_criteria:
        criteria_lines = json.dumps(top_level_criteria, indent=2)
        top_level_section = (
            f"\n<top_level_criteria>\n"
            f"These are the investigation-wide success criteria. Your script "
            f"MUST compute and print these metrics for the best method in the "
            f"TOP-LEVEL SUCCESS CRITERIA section (after the per-iteration "
            f"SUCCESS CRITERIA section). The Analyst uses these to score the "
            f"version.\n{criteria_lines}\n</top_level_criteria>"
        )
    else:
        top_level_section = ""

    user_prompt = CODER_USER.format(
        domain_knowledge=domain_knowledge or "(no domain knowledge provided)",
        plan_json=json.dumps(plan, indent=2),
        previous_script_section=previous_script_section,
        new_script_path=str(new_script_path),
        version=version,
        run_timeout_minutes=run_timeout_minutes,
        run_command=run_command,
        top_level_section=top_level_section,
    )

    options = ClaudeCodeOptions(
        system_prompt=system_prompt,
        allowed_tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep"],
        max_turns=50,
        permission_mode="acceptEdits",
        cwd=output_dir,
        model=model,
        extra_args={"setting-sources": ""},
    )

    correction_hint = ""
    for attempt in range(MAX_ATTEMPTS):
        effective_prompt = user_prompt + correction_hint

        try:
            async for message in query(prompt=effective_prompt, options=options):
                if isinstance(message, AssistantMessage):
                    if message_buffer is not None:
                        for block in message.content:
                            append_block_to_buffer(block, message_buffer)
                elif isinstance(message, ResultMessage):
                    usage = getattr(message, "usage", None) or {}
                    usage["num_turns"] = getattr(message, "num_turns", 0)
                    collect_text_from_query.last_usage = usage  # type: ignore[attr-defined]
        except Exception as e:
            if attempt == MAX_ATTEMPTS - 1:
                raise
            logger.warning(f"Coder attempt {attempt + 1}: SDK error ({e}), retrying")
            continue

        # Validate: script must exist
        if not new_script_path.exists():
            if attempt == MAX_ATTEMPTS - 1:
                raise FileNotFoundError(
                    f"Coder agent did not create the expected script at {new_script_path}"
                )
            correction_hint = (
                "\n\n<validation_error>\n"
                f"You did not create the script at {new_script_path}. "
                "Please write the experiment script to that exact path.\n"
                "</validation_error>"
            )
            logger.warning(f"Coder attempt {attempt + 1}: script not created, retrying")
            continue

        # Validate: script must have valid Python syntax
        valid, syntax_error = _validate_syntax(new_script_path)
        if not valid:
            if attempt == MAX_ATTEMPTS - 1:
                # Return the script anyway; the runner will catch the syntax error
                break
            correction_hint = (
                "\n\n<validation_error>\n"
                f"The script at {new_script_path} has a syntax error:\n{syntax_error}\n"
                "Please fix the syntax error and rewrite the script.\n"
                "</validation_error>"
            )
            logger.warning(f"Coder attempt {attempt + 1}: syntax error, retrying")
            continue

        # All checks passed
        break

    return new_script_path
