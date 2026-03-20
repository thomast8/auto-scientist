"""Report agent: Phase 3 final summary generation.

Generates a comprehensive report covering the best model, the journey from
first to final version, key insights, and recommendations for future work.
"""

from pathlib import Path

from claude_code_sdk import AssistantMessage, ClaudeCodeOptions, ResultMessage, TextBlock

from auto_scientist.sdk_utils import safe_query

from auto_scientist.prompts.report import REPORT_SYSTEM, REPORT_USER
from auto_scientist.state import ExperimentState


async def run_report(
    state: ExperimentState,
    notebook_path: Path,
    output_dir: Path,
    model: str | None = None,
    message_buffer: list[str] | None = None,
) -> Path:
    """Generate the final experiment report.

    Args:
        state: Final experiment state with all version history.
        notebook_path: Path to the lab notebook.
        output_dir: Directory to write the report.

    Returns:
        Path to the generated report file.
    """
    notebook_content = notebook_path.read_text() if notebook_path.exists() else "(no notebook)"
    report_path = output_dir / "report.md"

    user_prompt = REPORT_USER.format(
        domain=state.domain,
        goal=state.goal,
        total_iterations=state.iteration,
        best_version=state.best_version or "none",
        best_score=state.best_score,
        notebook_content=notebook_content,
        report_path=str(report_path),
    )

    options = ClaudeCodeOptions(
        system_prompt=REPORT_SYSTEM,
        allowed_tools=["Read", "Write", "Glob"],
        max_turns=10,
        permission_mode="acceptEdits",
        cwd=output_dir,
        model=model,
    )

    async for message in safe_query(prompt=user_prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock) and message_buffer is not None:
                    message_buffer.append(block.text)
        elif isinstance(message, ResultMessage):
            pass  # Agent is done

    if not report_path.exists():
        raise FileNotFoundError(
            f"Report agent did not create the expected report at {report_path}"
        )

    return report_path
