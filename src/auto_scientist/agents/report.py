"""Report agent: Phase 3 final summary generation.

Generates a comprehensive report covering the best approach, the journey from
first to final version, key insights, and recommendations for future work.

Returns the report content as a string; the orchestrator handles file writing.
"""

import logging
from pathlib import Path

from auto_scientist.prompts.report import REPORT_SYSTEM, REPORT_USER
from auto_scientist.sdk_backend import SDKOptions, get_backend
from auto_scientist.sdk_utils import (
    append_block_to_buffer,
    collect_text_from_query,
    prepare_turn_budget,
    safe_query,
    validate_report_structure,
)
from auto_scientist.state import ExperimentState

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 2

# Minimum report length to consider valid (characters)
MIN_REPORT_LENGTH = 100


async def run_report(
    state: ExperimentState,
    notebook_path: Path,
    output_dir: Path,
    model: str | None = None,
    message_buffer: list[str] | None = None,
    provider: str = "anthropic",
) -> str:
    """Generate the final experiment report.

    Args:
        state: Final experiment state with all version history.
        notebook_path: Path to the lab notebook.
        output_dir: Directory containing experiment artifacts (for reading).
        model: Optional model override.
        message_buffer: Optional buffer for streaming messages.

    Returns:
        Report content as a markdown string.
    """
    notebook_content = notebook_path.read_text() if notebook_path.exists() else "(no notebook)"

    user_prompt = REPORT_USER.format(
        domain=state.domain,
        goal=state.goal,
        total_iterations=state.iteration,
        best_version=state.versions[-1].version if state.versions else "none",
        notebook_content=notebook_content,
    )

    max_turns = 10
    allowed_tools = ["Read", "Glob"]
    budget = prepare_turn_budget(REPORT_SYSTEM, max_turns, allowed_tools, provider=provider)
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

    correction_hint = ""
    session_id: str | None = None
    full_text = ""

    for attempt in range(MAX_ATTEMPTS):
        effective_prompt = user_prompt + correction_hint

        report_parts: list[str] = []

        try:
            async for message in safe_query(
                prompt=effective_prompt, options=options, backend=backend
            ):
                if message.type == "assistant":
                    for block in message.content_blocks:
                        if message_buffer is not None:
                            append_block_to_buffer(block, message_buffer)
                        if hasattr(block, "text") and not hasattr(block, "name"):
                            report_parts.append(block.text)
                elif message.type == "result":
                    session_id = message.session_id
                    usage = message.usage
                    collect_text_from_query.last_usage = usage  # type: ignore[attr-defined]
        except Exception as e:
            if attempt == MAX_ATTEMPTS - 1:
                raise
            logger.warning(f"Report attempt {attempt + 1}: SDK error ({e}), retrying")
            continue

        full_text = "\n".join(report_parts)

        # Strip any conversational preamble before the first markdown heading.
        heading_idx = full_text.find("\n# ")
        if heading_idx != -1:
            full_text = full_text[heading_idx + 1 :]

        full_text = full_text.strip()

        # Validate: report should be non-empty and substantial
        if len(full_text) < MIN_REPORT_LENGTH:
            if attempt < MAX_ATTEMPTS - 1:
                correction_hint = (
                    "\n\n<validation_error>\n"
                    "Your previous output was too short or empty. "
                    "Please generate a comprehensive markdown report with headings, "
                    "covering the experiment journey, key findings, and recommendations.\n"
                    "</validation_error>"
                )
                logger.warning(f"Report attempt {attempt + 1}: output too short, retrying")
                continue
            break

        # Structural validation: check for required sections
        structure_issues = validate_report_structure(full_text)
        if not structure_issues:
            return full_text

        if attempt < MAX_ATTEMPTS - 1:
            issues_list = "\n".join(f"- {issue}" for issue in structure_issues)
            correction_hint = (
                f"\n\n<validation_error>\n"
                f"Your report is missing required sections or has structural issues:\n"
                f"{issues_list}\n\n"
                f"Please regenerate the report with all 10 required sections.\n"
                f"</validation_error>"
            )
            # Resume same session for targeted correction
            if session_id:
                logger.warning(
                    f"Report attempt {attempt + 1}: structural issues "
                    f"({len(structure_issues)} found), resuming session to fix"
                )
                retry_max_turns = 10
                retry_allowed_tools = ["Read", "Glob"]
                retry_budget = prepare_turn_budget(
                    REPORT_SYSTEM, retry_max_turns, retry_allowed_tools, provider=provider
                )
                options = SDKOptions(
                    system_prompt=retry_budget.system_prompt,
                    allowed_tools=retry_budget.allowed_tools,
                    max_turns=retry_budget.max_turns,
                    permission_mode="acceptEdits",
                    cwd=output_dir,
                    model=model,
                    resume=session_id,
                    extra_args={},
                )
            else:
                logger.warning(
                    f"Report attempt {attempt + 1}: structural issues "
                    f"({len(structure_issues)} found), retrying (no session to resume)"
                )
            continue

    # Return whatever we have, with a visible warning if incomplete
    if not full_text:
        raise RuntimeError(f"Report generation produced no output after {MAX_ATTEMPTS} attempts")

    remaining = validate_report_structure(full_text)
    if remaining:
        logger.warning(
            f"Returning incomplete report after {MAX_ATTEMPTS} attempts. "
            f"Remaining issues: {remaining}"
        )
        warning_header = (
            "> **WARNING: This report is incomplete.** "
            f"Missing sections: {', '.join(remaining)}\n\n"
        )
        full_text = warning_header + full_text

    return full_text
