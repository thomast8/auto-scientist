"""Report agent: Phase 3 final summary generation.

Generates a comprehensive report covering the best model, the journey from
first to final version, key insights, and recommendations for future work.
"""

from pathlib import Path

from auto_scientist.state import ExperimentState


async def run_report(
    state: ExperimentState,
    notebook_path: Path,
    output_dir: Path,
) -> Path:
    """Generate the final experiment report.

    Args:
        state: Final experiment state with all version history.
        notebook_path: Path to the lab notebook.
        output_dir: Directory to write the report.

    Returns:
        Path to the generated report file.
    """
    # TODO: Implement with claude-code-sdk query()
    # 1. Read lab notebook + all results files
    # 2. Identify best model and key turning points
    # 3. Generate report: journey, insights, best model, recommendations
    # 4. Write report to output_dir/report.md
    raise NotImplementedError("Report agent not yet implemented")
