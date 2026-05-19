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
from dataclasses import replace
from pathlib import Path
from typing import Any

from auto_core.retry import QueryResult, ValidationError, agent_retry_loop
from auto_core.sdk_backend import SDKOptions, codex_sandbox_addendum, get_backend
from auto_core.sdk_utils import (
    append_block_to_buffer,
    collect_text_from_query,
    prepare_turn_budget,
    resolve_prompt_provider,
)

from auto_scientist.prompts.coder import (
    CODER_HAS_PREVIOUS,
    CODER_NO_PREVIOUS,
    CODER_USER,
    build_coder_system,
)

logger = logging.getLogger(__name__)

_STDERR_TRUNCATE = 3000


def _check_runtime_success(version_dir: Path) -> tuple[bool, str]:
    """Check whether the coder's experiment script ran successfully.

    Reads run_result.json first, falls back to exitcode.txt/stderr.txt.
    Returns (True, "") on success or (False, error_description) on failure.
    Timeouts are treated as success for retry purposes (they need Scientist
    rethinking, not a coder retry).
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


def _validate_deps(script_path: Path) -> tuple[bool, str]:
    """Check that every third-party import is covered by PEP 723 deps."""
    from auto_core.ensure_deps import validate_deps

    return validate_deps(script_path)


def _plan_constraint_text(plan: dict[str, Any], domain_knowledge: str) -> str:
    """Return normalized text used to detect hard implementation constraints."""
    return "\n".join(
        [
            json.dumps(plan, ensure_ascii=False, sort_keys=True),
            domain_knowledge,
        ]
    ).lower()


def _forbids_third_party(text: str) -> bool:
    phrases = (
        "standard-library only",
        "standard library only",
        "only the python standard library",
        "python standard library only",
        "stdlib only",
        "pure standard-library",
        "pure standard library",
        "no third-party",
        "no third party",
        "without third-party",
        "without third party",
        "no external dependencies",
        "no external packages",
        "without external dependencies",
        "without external packages",
    )
    return any(phrase in text for phrase in phrases)


def _forbids_plots(text: str) -> bool:
    phrases = (
        "no plots",
        "no plot",
        "without plots",
        "without plot",
        "do not plot",
        "don't plot",
        "no figures",
        "no figure",
        "without figures",
        "without figure",
        "do not generate plots",
        "do not save plots",
        "no png",
        "no pngs",
    )
    return any(phrase in text for phrase in phrases)


def _validate_plan_constraints(
    script_path: Path,
    plan: dict[str, Any],
    domain_knowledge: str = "",
) -> tuple[bool, str]:
    """Reject scripts that violate explicit plan constraints."""
    constraint_text = _plan_constraint_text(plan, domain_knowledge)
    source = script_path.read_text(encoding="utf-8")

    if _forbids_third_party(constraint_text):
        from auto_core.ensure_deps import extract_imports, extract_pep723_dep_strings

        imports = sorted(extract_imports(source))
        deps = sorted(extract_pep723_dep_strings(source))
        if imports or deps:
            details = []
            if deps:
                details.append(f"declared dependencies: {', '.join(deps)}")
            if imports:
                details.append(f"third-party imports: {', '.join(imports)}")
            return (
                False,
                "The plan explicitly forbids third-party packages, but the script "
                + "; ".join(details)
                + ". Rewrite it with only Python standard-library modules and an empty "
                "PEP 723 dependencies list.",
            )

    if _forbids_plots(constraint_text):
        plot_markers = (
            "matplotlib",
            "seaborn",
            "plotly",
            "bokeh",
            "altair",
            ".savefig(",
            "savefig(",
            ".png",
            ".pdf",
            ".svg",
        )
        found = sorted(marker for marker in plot_markers if marker in source.lower())
        if found:
            return (
                False,
                "The plan explicitly forbids plots or figures, but the script contains "
                f"plotting markers: {', '.join(found)}. Rewrite it without plot "
                "generation or plotting dependencies.",
            )

    return True, ""


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
    provider: str = "openai",
    network_access: bool = False,
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
    # Rewrite `uv run ...` to a python3 invocation; the ensure_deps prefix
    # added by the orchestrator is already Codex-aware (python3 <copy>).
    if provider == "openai":
        from auto_core.sdk_backend import rewrite_uv_run_for_codex

        run_command = rewrite_uv_run_for_codex(run_command)

    prompt_provider = resolve_prompt_provider(provider)
    system_prompt = build_coder_system(prompt_provider).format(
        data_path=data_path or "(not specified)",
        run_timeout_minutes=run_timeout_minutes,
        run_command=run_command,
    )
    if provider == "openai":
        system_prompt += codex_sandbox_addendum(network_access=network_access)

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
        network_access=network_access,
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

        # Validate: third-party imports must be declared in PEP 723 deps
        deps_ok, deps_error = _validate_deps(new_script_path)
        if not deps_ok:
            raise ValidationError(f"<validation_error>\n{deps_error}\n</validation_error>")

        constraints_ok, constraints_error = _validate_plan_constraints(
            new_script_path,
            plan,
            domain_knowledge,
        )
        if not constraints_ok:
            raise ValidationError(f"<validation_error>\n{constraints_error}\n</validation_error>")

        # Validate: script ran successfully at runtime
        runtime_ok, runtime_error = _check_runtime_success(new_script_path.parent)
        if not runtime_ok:
            raise ValidationError(
                "<runtime_error>\n"
                f"The script at {new_script_path} failed at runtime:\n{runtime_error}\n"
                "The script already exists on disk. Read it, fix the bug, and re-run it.\n"
                "</runtime_error>"
            )

        return new_script_path

    def _on_exhausted(result: QueryResult | None, error: Exception) -> Path:
        if result is None:
            raise error
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
