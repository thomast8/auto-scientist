"""Report agent: Phase 3 final summary generation.

Generates a comprehensive report covering the best model, the journey from
first to final version, key insights, and recommendations for future work.

Returns the report content as a string; the orchestrator handles file writing.
"""

from pathlib import Path

from claude_code_sdk import AssistantMessage, ClaudeCodeOptions, ResultMessage, TextBlock

from auto_scientist.prompts.report import REPORT_SYSTEM, REPORT_USER
from auto_scientist.sdk_utils import append_block_to_buffer, safe_query
from auto_scientist.state import ExperimentState


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
        best_version=state.best_version or "none",
        best_score=state.best_score,
        notebook_content=notebook_content,
    )

    options = ClaudeCodeOptions(
        system_prompt=REPORT_SYSTEM,
        allowed_tools=["Read", "Glob"],
        max_turns=10,
        permission_mode="acceptEdits",
        cwd=output_dir,
        model=model,
    )

    report_parts: list[str] = []

    async for message in safe_query(prompt=user_prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if message_buffer is not None:
                    append_block_to_buffer(block, message_buffer)
                if isinstance(block, TextBlock):
                    report_parts.append(block.text)
        elif isinstance(message, ResultMessage):
            pass  # Agent is done

    return "\n".join(report_parts)
