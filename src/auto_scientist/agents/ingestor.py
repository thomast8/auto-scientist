"""Ingestor agent: Phase 0 data canonicalization.

Uses the SDK backend for a persistent session (may need multi-round human Q&A).
Tools: Bash (data inspection, conversion), Read/Write, Glob, Grep.
When interactive: also AskUserQuestion.
Produces: canonical dataset in {output_dir}/data/.
"""

import json
import logging
from pathlib import Path

from pydantic import ValidationError

from auto_scientist.config import DomainConfig
from auto_scientist.notebook import NOTEBOOK_FILENAME
from auto_scientist.prompts.ingestor import INGESTOR_SYSTEM, INGESTOR_USER
from auto_scientist.sdk_backend import CODEX_SANDBOX_ADDENDUM, SDKOptions, get_backend
from auto_scientist.sdk_utils import (
    append_block_to_buffer,
    collect_text_from_query,
    safe_query,
    with_turn_budget,
)

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
    provider: str = "anthropic",
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

    max_turns = 30
    system_prompt = INGESTOR_SYSTEM
    if provider == "openai":
        system_prompt += CODEX_SANDBOX_ADDENDUM
    backend = get_backend(provider)
    options = SDKOptions(
        system_prompt=with_turn_budget(system_prompt, max_turns, tools),
        allowed_tools=tools,
        max_turns=max_turns,
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
    last_sdk_error: Exception | None = None
    session_id: str | None = None

    for attempt in range(MAX_ATTEMPTS):
        effective_prompt = prompt + correction_hint

        try:
            async for msg in safe_query(prompt=effective_prompt, options=options, backend=backend):
                if msg.type == "result":
                    session_id = msg.session_id
                    usage = msg.usage
                    collect_text_from_query.last_usage = usage  # type: ignore[attr-defined]
                elif msg.type == "assistant":
                    for block in msg.content_blocks:
                        if message_buffer is not None:
                            append_block_to_buffer(block, message_buffer)
                        elif hasattr(block, "text") and not hasattr(block, "name"):
                            print(f"  [ingestor] {block.text[:200]}")
        except Exception as e:
            last_sdk_error = e
            if attempt == MAX_ATTEMPTS - 1:
                raise
            logger.warning(f"Ingestor attempt {attempt + 1}: SDK error ({e}), retrying")
            continue

        # Verify something was produced in data_dir
        data_files = list(data_dir.iterdir())
        output_files = [f for f in data_files if f.name != "ingest.py"]
        if not output_files:
            if attempt == MAX_ATTEMPTS - 1:
                err_msg = f"Ingestor agent did not produce any data files in {data_dir}"
                if last_sdk_error:
                    err_msg += f" (prior SDK error: {last_sdk_error})"
                raise FileNotFoundError(err_msg)
            correction_hint = (
                "\n\n<validation_error>\n"
                f"No data files were produced in {data_dir}. "
                "You must write at least one canonical data file to the data directory.\n"
                "</validation_error>"
            )
            logger.warning(f"Ingestor attempt {attempt + 1}: no data files, retrying")
            continue

        # Validate domain_config.json if a config_path was requested
        if config_path is not None:
            config_error: str | None = None

            if not config_path.exists():
                config_error = (
                    f"domain_config.json was not written to {config_path}. "
                    "You must create this file."
                )
            else:
                try:
                    raw_config = json.loads(config_path.read_text())
                    # Strict validation: reject dict data_paths at write time
                    if isinstance(raw_config.get("data_paths"), dict):
                        raise ValueError(
                            "data_paths must be a JSON list "
                            f'(e.g. ["data/file.csv"]), got dict: '
                            f"{raw_config['data_paths']}"
                        )
                    DomainConfig.model_validate(raw_config)
                except (
                    ValidationError,
                    json.JSONDecodeError,
                    TypeError,
                    ValueError,
                ) as e:
                    config_error = str(e)

            if config_error is not None:
                if attempt == MAX_ATTEMPTS - 1:
                    raise RuntimeError(
                        f"Ingestor config validation failed after "
                        f"{MAX_ATTEMPTS} attempts: {config_error}"
                    )
                correction_hint = (
                    f"\n\n<validation_error>\n"
                    f"The domain_config.json at {config_path} is invalid:\n"
                    f"{config_error}\n\n"
                    f"data_paths MUST be a JSON list (e.g. "
                    f'["data/file.csv"]), not a dict. Please fix the file.\n'
                    f"</validation_error>"
                )
                # Resume same session so the agent has context
                if session_id:
                    logger.warning(
                        f"Ingestor attempt {attempt + 1}: invalid "
                        f"domain_config.json, resuming session to fix"
                    )
                    clarification_max_turns = 10
                    options = SDKOptions(
                        system_prompt=with_turn_budget(
                            INGESTOR_SYSTEM, clarification_max_turns, tools
                        ),
                        allowed_tools=tools,
                        max_turns=clarification_max_turns,
                        permission_mode="acceptEdits",
                        cwd=output_dir,
                        model=model,
                        resume=session_id,
                        extra_args={"setting-sources": ""},
                    )
                else:
                    logger.warning(
                        f"Ingestor attempt {attempt + 1}: invalid "
                        f"domain_config.json, retrying (no session to resume)"
                    )
                continue

        return data_dir

    return data_dir  # unreachable
