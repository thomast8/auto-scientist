"""Compressed history builder for the Critic's context window."""

from auto_scientist.state import ExperimentState, VersionEntry


def build_compressed_history(state: ExperimentState) -> str:
    """Build a compact summary of all iterations for the Critic.

    Produces ~1 line per iteration extracted from the state's version entries.
    This keeps the Critic's context manageable even after many iterations.

    Args:
        state: Current experiment state with version history.

    Returns:
        Multi-line string with one summary line per version.
    """
    if not state.versions:
        return "No iterations completed yet."

    lines = [f"Experiment history ({len(state.versions)} versions):"]
    lines.append("")

    for entry in state.versions:
        status_icon = _status_icon(entry)
        score_str = f"score={entry.score}" if entry.score > 0 else "no score"
        hypothesis = entry.hypothesis[:80] + "..." if len(entry.hypothesis) > 80 else entry.hypothesis
        lines.append(f"  {status_icon} {entry.version} (iter {entry.iteration}): {score_str} - {hypothesis}")

    if state.dead_ends:
        lines.append("")
        lines.append(f"Dead ends: {', '.join(state.dead_ends)}")

    if state.best_version:
        lines.append("")
        lines.append(f"Best so far: {state.best_version} (score {state.best_score})")

    return "\n".join(lines)


def _status_icon(entry: VersionEntry) -> str:
    """Map version status to a compact icon."""
    return {
        "completed": "[ok]",
        "failed": "[FAIL]",
        "crashed": "[CRASH]",
        "running": "[...]",
        "pending": "[--]",
    }.get(entry.status, "[?]")
