"""Ingestor agent: Phase 0 data canonicalization.

Uses claude_code_sdk for a persistent session (may need multi-round human Q&A).
Tools: Bash (data inspection, conversion), Read/Write, Glob, Grep.
When interactive: also AskUserQuestion.
Produces: canonical dataset in {output_dir}/data/.
"""

import logging
from pathlib import Path

from claude_code_sdk import (
    AssistantMessage,
    ClaudeCodeOptions,
    ResultMessage,
    TextBlock,
)

from auto_scientist.notebook import NOTEBOOK_FILENAME
from auto_scientist.prompts.ingestor import INGESTOR_SYSTEM, INGESTOR_USER
from auto_scientist.sdk_utils import append_block_to_buffer, collect_text_from_query, safe_query

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 2


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
    notebook_path = output_dir / NOTEBOOK_FILENAME

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
        extra_args={"setting-sources": ""},
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

    correction_hint = ""
    last_error: Exception | None = None
    for attempt in range(MAX_ATTEMPTS):
        effective_prompt = prompt + correction_hint

        try:
            async for msg in safe_query(prompt=effective_prompt, options=options):
                if isinstance(msg, ResultMessage):
                    usage = getattr(msg, "usage", None) or {}
                    usage["num_turns"] = getattr(msg, "num_turns", 0)
                    collect_text_from_query.last_usage = usage  # type: ignore[attr-defined]
                elif isinstance(msg, AssistantMessage):
                    for block in msg.content:
                        if message_buffer is not None:
                            append_block_to_buffer(block, message_buffer)
                        elif isinstance(block, TextBlock):
                            print(f"  [ingestor] {block.text[:200]}")
        except Exception as e:
            last_error = e
            if attempt == MAX_ATTEMPTS - 1:
                raise
            logger.warning(f"Ingestor attempt {attempt + 1}: SDK error ({e}), retrying")
            continue

        # Verify something was produced in data_dir
        data_files = list(data_dir.iterdir())
        output_files = [f for f in data_files if f.name != "ingest.py"]
        if output_files:
            return data_dir

        if attempt == MAX_ATTEMPTS - 1:
            raise FileNotFoundError(
                f"Ingestor agent did not produce any data files in {data_dir}"
            )

        correction_hint = (
            "\n\n<validation_error>\n"
            f"No data files were produced in {data_dir}. "
            "You must write at least one canonical data file to the data directory.\n"
            "</validation_error>"
        )
        logger.warning(f"Ingestor attempt {attempt + 1}: no data files, retrying")

    return data_dir  # unreachable
