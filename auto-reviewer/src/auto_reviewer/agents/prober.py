"""Prober agent: implements and runs the Hunter's reproduction recipe.

Uses query() (fresh session, reads/writes files via tools).
Tools: Read, Write, Edit, Bash, Glob, Grep.
Input (via prompt): Hunter's BugPlan JSON + previous probe script + run config.
Output: probe script + run_result.json at {version_dir}/.
max_turns: 50
Safety hooks: block writes outside the review workspace, block mutation of
the target repo's source.
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
    resolve_prompt_provider,
)

from auto_reviewer.prompts.prober import (
    PROBER_USER,
    build_prober_system,
)

logger = logging.getLogger(__name__)

_STDERR_TRUNCATE = 3000

# Path-hack patterns the Prober must NOT write in probes. The contract is
# that probes run from `run_cwd` (target repo) via the `run_command`
# template; the target's native import resolution applies, so any manual
# sys.path / PYTHONPATH manipulation is a bypass of that contract and a
# red flag for "Prober ran the probe from the wrong directory."
_FORBIDDEN_PATH_HACKS: tuple[str, ...] = (
    "sys.path.insert",
    "sys.path.append",
    "PYTHONPATH=",
)


def _scan_probes_for_path_hacks(probes_dir: Path) -> tuple[Path, str] | None:
    """Return (probe_file, offending_pattern) if any probe has a forbidden
    import-path hack; else None.

    Only scans regular files under `probes_dir`; skips binaries and files
    we can't decode as text.
    """
    if not probes_dir.exists() or not probes_dir.is_dir():
        return None
    for probe_file in sorted(probes_dir.iterdir()):
        if not probe_file.is_file():
            continue
        try:
            text = probe_file.read_text()
        except (OSError, UnicodeDecodeError):
            continue
        for pattern in _FORBIDDEN_PATH_HACKS:
            if pattern in text:
                return probe_file, pattern
    return None


def _check_runtime_success(version_dir: Path) -> tuple[bool, str]:
    """Check whether the Prober's probe script ran successfully.

    Reads run_result.json first, falls back to exitcode.txt/stderr.txt.
    Returns (True, "") on success or (False, error_description) on failure.
    Timeouts are treated as success for retry purposes (they need Hunter
    rethinking, not a Prober retry).
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
    """Implement the Hunter's plan as a runnable probe script.

    Args:
        plan: Structured BugPlan dict from the Hunter.
        previous_script: Path to the previous iteration's probe script.
        output_dir: Base review-iteration directory.
        version: Version string for the new probe run (e.g., 'v01').
        domain_knowledge: Repo-level context.
        data_path: Absolute path to the target repo's root.
        data_files_listing: Pre-computed listing of files in data_path directory.

    Returns:
        Path to the newly created probe script.
    """
    version_dir = output_dir / version
    version_dir.mkdir(parents=True, exist_ok=True)
    new_script_path = version_dir / "run_result.json"

    # Codex seatbelt sandbox: uv panics (SCDynamicStore access denied).
    # Rewrite `uv run ...` to a python3 invocation that handles Intake's
    # `uv run pytest -x -s {script_path}` / `uv run python {script_path}`
    # shapes, not just the auto-scientist default `uv run {script_path}`.
    if provider == "openai":
        from auto_core.sdk_backend import rewrite_uv_run_for_codex

        run_command = rewrite_uv_run_for_codex(run_command)

    prompt_provider = resolve_prompt_provider(provider)
    system_prompt = build_prober_system(prompt_provider).format(
        data_path=data_path or "(not specified)",
        run_timeout_minutes=run_timeout_minutes,
        run_command=run_command,
    )
    if provider == "openai":
        system_prompt += CODEX_SANDBOX_ADDENDUM

    user_prompt = PROBER_USER.format(
        workspace_path=str(output_dir),
        version_dir=str(version_dir),
        plan_path=str(version_dir / "plan.json"),
        config_path=str(output_dir / "domain_config.json"),
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
        probe_path_hack = _scan_probes_for_path_hacks(version_dir / "probes")
        if probe_path_hack is not None:
            probe_file, offending = probe_path_hack
            raise ValidationError(
                "<validation_error>\n"
                f"Probe file {probe_file} contains a forbidden import-path "
                f"hack: {offending!r}. The contract is to run the probe from "
                "`run_cwd` (the target repo, as set in domain_config.json) "
                "using `run_command` with `{script_path}` substituted. When "
                "the probe runs from the target's own directory, the target's "
                "native import / module resolution applies and no manual path "
                "manipulation is needed. Remove the hack, ensure your Bash "
                "invocation does `cd $run_cwd` before `run_command`, and "
                "re-run.\n"
                "</validation_error>"
            )

        if not new_script_path.exists():
            raise ValidationError(
                "<validation_error>\n"
                f"You did not write run_result.json at {new_script_path}. "
                "After running the probe, write run_result.json to that exact path.\n"
                "</validation_error>"
            )

        runtime_ok, runtime_error = _check_runtime_success(new_script_path.parent)
        if not runtime_ok:
            raise ValidationError(
                "<runtime_error>\n"
                f"The probe reported failure in {new_script_path}:\n{runtime_error}\n"
                "Review the probe output, fix the issue, re-run the probe, "
                "and rewrite run_result.json.\n"
                "</runtime_error>"
            )

        return new_script_path

    def _on_exhausted(result: QueryResult | None, error: Exception) -> Path:
        if result is None:
            raise error
        if new_script_path.exists():
            return new_script_path
        raise FileNotFoundError(f"Prober agent did not write run_result.json at {new_script_path}")

    result: Path = await agent_retry_loop(
        query_fn=_query,
        validate_fn=_validate,
        prompt=user_prompt,
        agent_name="Prober",
        on_exhausted=_on_exhausted,
    )
    return result
