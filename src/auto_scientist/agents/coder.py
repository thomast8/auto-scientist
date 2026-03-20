"""Coder agent: implements the scientist's plan as a runnable experiment script.

Uses query() (fresh session, reads/writes files via tools).
Tools: Read, Write, Edit, Bash (for syntax check), Glob, Grep.
Input (via prompt): scientist's plan JSON + previous script.
Output: new experiment script at {version_dir}/experiment.py.
max_turns: 30
Safety hooks: block writes outside experiments/ dir, block writes to data files.
"""

import json
from pathlib import Path
from typing import Any

from claude_code_sdk import (
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

async def run_coder(
    plan: dict[str, Any],
    previous_script: Path,
    output_dir: Path,
    version: str,
    domain_knowledge: str = "",
    data_path: str = "",
    model: str | None = None,
) -> Path:
    """Implement the scientist's plan as a runnable experiment script.

    Args:
        plan: Structured plan dict from the Scientist.
        previous_script: Path to the previous version's script.
        output_dir: Base experiments directory.
        version: Version string for the new experiment (e.g., 'v01').
        domain_knowledge: Domain-specific context.
        data_path: Absolute path to the dataset.

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
    )

    user_prompt = CODER_USER.format(
        domain_knowledge=domain_knowledge or "(no domain knowledge provided)",
        plan_json=json.dumps(plan, indent=2),
        previous_script_section=previous_script_section,
        new_script_path=str(new_script_path),
        version=version,
    )

    options = ClaudeCodeOptions(
        system_prompt=system_prompt,
        allowed_tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep"],
        max_turns=30,
        permission_mode="acceptEdits",
        cwd=output_dir,
        model=model,
    )

    async for message in query(prompt=user_prompt, options=options):
        if isinstance(message, ResultMessage):
            pass  # Agent is done

    # Verify the script was created
    if not new_script_path.exists():
        raise FileNotFoundError(
            f"Coder agent did not create the expected script at {new_script_path}"
        )

    return new_script_path
