"""Ingestor agent: Phase 0 data canonicalization.

Uses the SDK backend for a persistent session (may need multi-round human Q&A).
Tools: Bash (data inspection, conversion), Read/Write, Glob, Grep.
When interactive: also AskUserQuestion.
Produces: canonical dataset in {output_dir}/data/.
"""

import json
import logging
from pathlib import Path

from auto_core.notebook import NOTEBOOK_FILENAME
from auto_core.retry import QueryResult, agent_retry_loop
from auto_core.retry import ValidationError as RetryValidationError
from auto_core.sdk_backend import CODEX_SANDBOX_ADDENDUM, SDKOptions, get_backend
from auto_core.sdk_utils import (
    append_block_to_buffer,
    collect_text_from_query,
    prepare_turn_budget,
    resolve_prompt_provider,
    safe_query,
)
from pydantic import ValidationError

from auto_scientist.config import DomainConfig
from auto_scientist.prompts.ingestor import INGESTOR_USER, build_ingestor_system

logger = logging.getLogger(__name__)


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
    # `AskUserQuestion` is a Claude Code CLI built-in; the Codex backend
    # exposes no equivalent. If interactive mode was requested on a
    # non-Claude backend, fall back to autonomous with a clear warning
    # rather than silently advertising a tool the model cannot call.
    effective_interactive = interactive and provider == "anthropic"
    if interactive and not effective_interactive:
        logger.warning(
            "Ingestor --interactive is only supported with provider='anthropic' "
            "(Claude Code supplies AskUserQuestion). Falling back to autonomous "
            "mode for provider=%r.",
            provider,
        )
    if effective_interactive:
        tools.append("AskUserQuestion")

    mode = "interactive" if effective_interactive else "autonomous"

    max_turns = 30
    prompt_provider = resolve_prompt_provider(provider)
    system_prompt = build_ingestor_system(prompt_provider)
    if provider == "openai":
        system_prompt += CODEX_SANDBOX_ADDENDUM
    budget = prepare_turn_budget(system_prompt, max_turns, tools, provider=provider)
    backend = get_backend(provider)
    options = SDKOptions(
        system_prompt=budget.system_prompt,
        allowed_tools=budget.allowed_tools,
        max_turns=budget.max_turns,
        permission_mode="acceptEdits",
        cwd=output_dir,
        model=model,
        extra_args={},
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

    # Mutable state for the resume closure: on resume, reduce max_turns.
    current_options = [options]

    async def _query(prompt_text: str, resume_session_id: str | None) -> QueryResult:
        if resume_session_id is not None:
            clarification_max_turns = 10
            retry_budget = prepare_turn_budget(
                system_prompt, clarification_max_turns, tools, provider=provider
            )
            current_options[0] = SDKOptions(
                system_prompt=retry_budget.system_prompt,
                allowed_tools=retry_budget.allowed_tools,
                max_turns=retry_budget.max_turns,
                permission_mode="acceptEdits",
                cwd=output_dir,
                model=model,
                resume=resume_session_id,
                extra_args={},
            )
        else:
            current_options[0] = options
        sid: str | None = None
        async for msg in safe_query(
            prompt=prompt_text, options=current_options[0], backend=backend
        ):
            if msg.type == "result":
                sid = msg.session_id
                usage = msg.usage
                collect_text_from_query.last_usage = usage  # type: ignore[attr-defined]
            elif msg.type == "assistant":
                for block in msg.content_blocks:
                    if message_buffer is not None:
                        append_block_to_buffer(block, message_buffer)
                    elif hasattr(block, "text") and not hasattr(block, "name"):
                        print(f"  [ingestor] {block.text[:200]}")
        return QueryResult(raw_output="", session_id=sid, usage={})

    def _validate(result: QueryResult) -> Path:
        data_files = list(data_dir.iterdir())
        output_files = [f for f in data_files if f.name != "ingest.py"]
        if not output_files:
            raise RetryValidationError(
                "<validation_error>\n"
                f"No data files were produced in {data_dir}. "
                "You must write at least one canonical data file to the data directory.\n"
                "</validation_error>"
            )

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
                raise RetryValidationError(
                    f"<validation_error>\n"
                    f"The domain_config.json at {config_path} is invalid:\n"
                    f"{config_error}\n\n"
                    f"data_paths MUST be a JSON list (e.g. "
                    f'["data/file.csv"]), not a dict. Please fix the file.\n'
                    f"</validation_error>"
                )

        return data_dir

    def _on_exhausted(result: QueryResult | None, error: Exception) -> Path:
        if isinstance(error, RetryValidationError) and "No data files" in str(error):
            raise FileNotFoundError(f"Ingestor agent did not produce any data files in {data_dir}")
        if isinstance(error, RetryValidationError) and "domain_config.json" in str(error):
            raise RuntimeError(f"Ingestor config validation failed after 3 attempts: {error}")
        raise error

    return await agent_retry_loop(
        query_fn=_query,
        validate_fn=_validate,
        prompt=prompt,
        agent_name="Ingestor",
        on_exhausted=_on_exhausted,
    )
