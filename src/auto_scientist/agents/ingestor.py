"""Ingestor agent: Phase 0 data canonicalization.

Uses claude_code_sdk for a persistent session (may need multi-round human Q&A).
Tools: Bash (data inspection, conversion), Read/Write, Glob, Grep.
When interactive: also AskUserQuestion.
Produces: canonical dataset in {output_dir}/data/.
"""

from pathlib import Path

from claude_code_sdk import (
    AssistantMessage,
    ClaudeCodeOptions,
    TextBlock,
)

from auto_scientist.prompts.ingestor import INGESTOR_SYSTEM, INGESTOR_USER
from auto_scientist.sdk_utils import safe_query


async def run_ingestor(
    raw_data_path: Path,
    output_dir: Path,
    goal: str,
    interactive: bool = False,
    config_path: Path | None = None,
    model: str | None = None,
    message_buffer: list[str] | None = None,
) -> Path:
    """Inspect raw data and produce a canonical dataset.

    Args:
        raw_data_path: Path to raw data file or directory.
        output_dir: Experiment output directory (experiments/).
        goal: The user's investigation goal.
        interactive: If True, agent can ask user questions.
        config_path: Where to write the domain config JSON.

    Returns:
        Path to the canonical data directory (output_dir/data/).
    """
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    notebook_path = output_dir / "lab_notebook.md"

    tools = ["Bash", "Read", "Write", "Glob", "Grep"]
    if interactive:
        tools.append("AskUserQuestion")

    mode = "interactive" if interactive else "autonomous"

    options = ClaudeCodeOptions(
        system_prompt=INGESTOR_SYSTEM,
        allowed_tools=tools,
        max_turns=30,
        permission_mode="acceptEdits",
        cwd=output_dir,
        model=model,
    )

    config_path_str = str(config_path) if config_path else "(not requested)"

    prompt = INGESTOR_USER.format(
        raw_data_path=str(raw_data_path.resolve()),
        goal=goal,
        data_dir=str(data_dir),
        notebook_path=str(notebook_path),
        config_path=config_path_str,
        mode=mode,
    )

    async for msg in safe_query(prompt=prompt, options=options):
        if isinstance(msg, AssistantMessage):
            for block in msg.content:
                if isinstance(block, TextBlock):
                    if message_buffer is not None:
                        message_buffer.append(block.text)
                    else:
                        print(f"  [ingestor] {block.text[:200]}")

    # Verify something was produced in data_dir
    data_files = list(data_dir.iterdir())
    output_files = [f for f in data_files if f.name != "ingest.py"]
    if not output_files:
        raise FileNotFoundError(
            f"Ingestor agent did not produce any data files in {data_dir}"
        )

    return data_dir
