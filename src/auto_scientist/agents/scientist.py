"""Scientist agent: implements model changes and updates lab notebook.

Uses query() (fresh session, reads files via tools).
Tools: Read, Write, Edit, Bash (for syntax check), Glob, Grep.
Input (via prompt): analysis JSON + critic's feedback + lab notebook.
Input (via tools): reads previous script, writes new script.
Output: new experiment script + updated lab notebook + hypothesis doc.
max_turns: 30
Safety hooks: block writes outside experiments/ dir, block writes to data files.
"""

from pathlib import Path
from typing import Any


async def run_scientist(
    analysis: dict[str, Any],
    critiques: list[dict[str, str]],
    previous_script: Path,
    notebook_path: Path,
    output_dir: Path,
    version: str,
    domain_knowledge: str = "",
) -> Path:
    """Implement model changes based on analysis and critique.

    Args:
        analysis: Structured analysis JSON from the Analyst.
        critiques: List of critic responses.
        previous_script: Path to the previous version's script.
        notebook_path: Path to the lab notebook (read and append).
        output_dir: Directory for the new version's outputs.
        version: Version string for the new experiment (e.g., 'v8.01').
        domain_knowledge: Domain-specific context.

    Returns:
        Path to the newly created experiment script.
    """
    # TODO: Implement with claude-code-sdk query()
    # 1. Read previous script, lab notebook
    # 2. Synthesize analysis + critique into a hypothesis
    # 3. Write new script with changes
    # 4. Append to lab notebook
    # 5. Return path to new script
    raise NotImplementedError("Scientist agent not yet implemented")
