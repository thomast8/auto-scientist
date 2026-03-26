"""Report agent: Phase 3 final summary generation.

Generates a comprehensive report covering the best model, the journey from
first to final version, key insights, and recommendations for future work.

Returns the report content as a string; the orchestrator handles file writing.
"""

import logging
from pathlib import Path

from claude_code_sdk import AssistantMessage, ClaudeCodeOptions, ResultMessage, TextBlock

from auto_scientist.prompts.report import REPORT_SYSTEM, REPORT_USER
from auto_scientist.sdk_utils import append_block_to_buffer, safe_query
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

    options = ClaudeCodeOptions(
        system_prompt=REPORT_SYSTEM,
        allowed_tools=["Read", "Glob"],
        max_turns=10,
        permission_mode="acceptEdits",
        cwd=output_dir,
        model=model,
        extra_args={"setting-sources": ""},
    )

    correction_hint = ""
    for attempt in range(MAX_ATTEMPTS):
        effective_prompt = user_prompt + correction_hint

        report_parts: list[str] = []

        try:
            async for message in safe_query(prompt=effective_prompt, options=options):
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if message_buffer is not None:
                            append_block_to_buffer(block, message_buffer)
                        if isinstance(block, TextBlock):
                            report_parts.append(block.text)
                elif isinstance(message, ResultMessage):
                    pass  # Agent is done
        except Exception as e:
            if attempt == MAX_ATTEMPTS - 1:
                raise
            logger.warning(f"Report attempt {attempt + 1}: SDK error ({e}), retrying")
            continue

        full_text = "\n".join(report_parts)

        # Strip any conversational preamble before the first markdown heading.
        heading_idx = full_text.find("\n# ")
        if heading_idx != -1:
            full_text = full_text[heading_idx + 1:]

        full_text = full_text.strip()

        # Validate: report should be non-empty and substantial
        if len(full_text) >= MIN_REPORT_LENGTH:
            return full_text

        if attempt < MAX_ATTEMPTS - 1:
            correction_hint = (
                "\n\n<validation_error>\n"
                "Your previous output was too short or empty. "
                "Please generate a comprehensive markdown report with headings, "
                "covering the experiment journey, key findings, and recommendations.\n"
                "</validation_error>"
            )
            logger.warning(f"Report attempt {attempt + 1}: output too short, retrying")

    # Return whatever we have, even if short
    return full_text
