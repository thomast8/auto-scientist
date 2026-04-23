"""Intake agent: LLM-driven PR canonicalization.

Given a natural-language review prompt pointing at some code - a GitHub PR
URL, `owner/repo#N`, a bare branch name, or "the current branch" - the
Intake agent parses the pointer, locates (or clones) the repo, resolves
base/head refs, computes the diff, snapshots the touched files, and writes
a populated `ReviewConfig`. The downstream agents (Surveyor, Hunter,
Prober, Findings) read the canonicalized workspace and never touch git
themselves.

Signature matches the shared `auto_core.Orchestrator._run_ingestion`
dispatch contract.
"""

from __future__ import annotations

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
    safe_query,
)
from pydantic import ValidationError

from auto_reviewer.config import ReviewConfig
from auto_reviewer.prompts.intake import INTAKE_USER, build_intake_system

logger = logging.getLogger(__name__)


async def run_intake(
    raw_data_path: Path,
    output_dir: Path,
    goal: str,
    interactive: bool = False,
    config_path: Path | None = None,
    model: str | None = None,
    message_buffer: list[str] | None = None,
    provider: str = "anthropic",
) -> Path:
    """Parse the review prompt and canonicalize a PR into a review workspace.

    Args:
        raw_data_path: Best-effort starting directory (typically the user's
            cwd). The agent uses this as the first candidate when locating
            the repository described by `goal`.
        output_dir: Review workspace root. Canonical artifacts land under
            `{output_dir}/data/`.
        goal: The user's natural-language review prompt. May contain a PR
            URL, owner/repo#N, a branch name, or a reference to "my
            current branch".
        interactive: If True, the agent has `AskUserQuestion` available to
            clarify ambiguous pointers (e.g. which repo to clone).
        config_path: Where the agent must write the refined ReviewConfig.
            The orchestrator reads this after intake to populate
            `self.config`.

    Returns:
        Path to the canonical data directory (`output_dir/data/`).
    """
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    notebook_path = output_dir / NOTEBOOK_FILENAME

    tools = ["Bash", "Read", "Write", "Glob", "Grep"]
    if interactive:
        tools.append("AskUserQuestion")

    mode = "interactive" if interactive else "autonomous"

    max_turns = 30
    prompt_provider = "gpt" if provider == "openai" else "claude"
    system_prompt = build_intake_system(prompt_provider)
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

    prompt = INTAKE_USER.format(
        prompt=goal,
        cwd=str(raw_data_path.resolve()),
        data_dir=str(data_dir),
        notebook_path=str(notebook_path),
        config_path=config_path_str,
        mode=mode,
    )

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
                        print(f"  [intake] {block.text[:200]}")
        return QueryResult(raw_output="", session_id=sid, usage={})

    def _validate(result: QueryResult) -> Path:
        diff_path = data_dir / "diff.patch"
        if not diff_path.exists() or diff_path.stat().st_size == 0:
            raise RetryValidationError(
                "<validation_error>\n"
                f"No diff.patch was produced at {diff_path} (or it is empty). "
                "Run `gh pr diff <ref>` or `git diff <base>...<head>` and write "
                "the output to that path.\n"
                "</validation_error>"
            )

        metadata_path = data_dir / "pr_metadata.json"
        if not metadata_path.exists():
            raise RetryValidationError(
                "<validation_error>\n"
                f"No pr_metadata.json was produced at {metadata_path}. "
                "Write the flattened PR metadata (title, body, author, url, "
                "baseRefName, headRefName) to that path.\n"
                "</validation_error>"
            )
        try:
            metadata = json.loads(metadata_path.read_text())
        except json.JSONDecodeError as e:
            raise RetryValidationError(
                "<validation_error>\n"
                f"pr_metadata.json at {metadata_path} is not valid JSON: {e}\n"
                "</validation_error>"
            ) from e
        missing = [k for k in ("title", "baseRefName", "headRefName") if not metadata.get(k)]
        if missing:
            raise RetryValidationError(
                "<validation_error>\n"
                f"pr_metadata.json is missing required keys: {missing}. "
                "At minimum it must carry title, baseRefName, and headRefName.\n"
                "</validation_error>"
            )

        touched_dir = data_dir / "touched_files"
        if not touched_dir.exists() or not any(touched_dir.iterdir()):
            raise RetryValidationError(
                "<validation_error>\n"
                f"touched_files/ is missing or empty at {touched_dir}. "
                "Snapshot each changed file at the head ref (use "
                "`git show <head>:<path>`) and write it with `/` replaced by "
                "`__` in the filename. Deleted files get a `.DELETED` "
                "tombstone.\n"
                "</validation_error>"
            )

        if config_path is not None:
            if not config_path.exists():
                raise RetryValidationError(
                    "<validation_error>\n"
                    f"No ReviewConfig was written to {config_path}. "
                    "You must create this file.\n"
                    "</validation_error>"
                )
            try:
                raw_config = json.loads(config_path.read_text())
                cfg = ReviewConfig.model_validate(raw_config)
            except (ValidationError, json.JSONDecodeError) as e:
                raise RetryValidationError(
                    "<validation_error>\n"
                    f"ReviewConfig at {config_path} is invalid: {e}\n"
                    "Required fields: name, repo_path, pr_ref, base_ref, "
                    "head_ref, and run_command must contain the literal "
                    '"{script_path}" placeholder.\n'
                    "</validation_error>"
                ) from e
            if not cfg.repo_path or not Path(cfg.repo_path).exists():
                raise RetryValidationError(
                    "<validation_error>\n"
                    f"ReviewConfig.repo_path ({cfg.repo_path!r}) is not set "
                    "or does not exist. It must be an absolute path to the "
                    "local clone of the repository under review.\n"
                    "</validation_error>"
                )
            if not cfg.pr_ref:
                raise RetryValidationError(
                    "<validation_error>\n"
                    "ReviewConfig.pr_ref is missing. Set it to the PR "
                    "identifier you resolved (URL, owner/repo#N, or branch "
                    "name).\n"
                    "</validation_error>"
                )
            if "{{script_path}}" in cfg.run_command:
                raise RetryValidationError(
                    "<validation_error>\n"
                    f"ReviewConfig.run_command has doubled braces: "
                    f"{cfg.run_command!r}. Use single braces: "
                    '"uv run pytest -x -s {script_path}". The Prober '
                    "substitutes `{script_path}` at runtime.\n"
                    "</validation_error>"
                )

        return data_dir

    def _on_exhausted(result: QueryResult | None, error: Exception) -> Path:
        if isinstance(error, RetryValidationError):
            msg = str(error)
            if "diff.patch" in msg:
                raise FileNotFoundError(
                    f"Intake did not produce a non-empty diff.patch in {data_dir}"
                ) from error
            if "ReviewConfig" in msg:
                raise RuntimeError(
                    f"Intake config validation failed after 3 attempts: {error}"
                ) from error
        raise error

    result: Path = await agent_retry_loop(
        query_fn=_query,
        validate_fn=_validate,
        prompt=prompt,
        agent_name="Intake",
        on_exhausted=_on_exhausted,
    )
    _reconcile_touched_files(result / "touched_files")
    return result


def _reconcile_touched_files(touched_dir: Path) -> None:
    """Drop spurious `.DELETED` tombstones written alongside a real snapshot.

    The LLM sometimes writes both a head-ref snapshot and a tombstone for
    every changed path (treating the two cases as conjunctive instead of
    exclusive). A tombstone is only meaningful when there is no snapshot
    for the same base name; otherwise it misleads the Surveyor.
    """
    if not touched_dir.exists():
        return
    snapshots = {p.name for p in touched_dir.iterdir() if not p.name.endswith(".DELETED")}
    for path in touched_dir.iterdir():
        if not path.name.endswith(".DELETED"):
            continue
        base = path.name.removesuffix(".DELETED")
        if base in snapshots:
            path.unlink()
            logger.debug("Removed spurious tombstone for %s", base)
