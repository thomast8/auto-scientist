"""Analyst agent: structured analysis of experiment results + plots.

Uses query() (fresh session each iteration, bounded context).
Tools: Read (results file + plot PNGs), Glob (find output files).
Input: results text + plot images + lab notebook.
Output: structured JSON with success_score, failures, metrics, recommendations.
max_turns: 5
"""

from pathlib import Path
from typing import Any


async def run_analyst(
    results_path: Path,
    plot_paths: list[Path],
    notebook_path: Path,
    domain_knowledge: str = "",
) -> dict[str, Any]:
    """Analyze experiment results and produce structured assessment.

    Args:
        results_path: Path to the results text file.
        plot_paths: Paths to output plot PNGs (read as images).
        notebook_path: Path to the lab notebook.
        domain_knowledge: Domain-specific context injected into the prompt.

    Returns:
        Structured dict with keys:
            success_score: int (0-100)
            failures: list[str]
            key_metrics: dict[str, float]
            what_worked: list[str]
            what_didnt_work: list[str]
            stagnation_detected: bool
            paradigm_shift_recommended: bool
            should_stop: bool
            stop_reason: str | None
            recommended_changes: list[str]
    """
    # TODO: Implement with claude-code-sdk query()
    raise NotImplementedError("Analyst agent not yet implemented")
