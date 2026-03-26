"""Shared constants, helpers, and preferences for the auto-scientist UI."""

import json
from pathlib import Path

from rich.console import Console

_PREFS_PATH = Path.home() / ".config" / "auto-scientist" / "preferences.json"


def _load_prefs() -> dict:
    """Load user preferences from disk."""
    try:
        return json.loads(_PREFS_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _save_prefs(prefs: dict) -> None:
    """Save user preferences to disk (atomic write)."""
    _PREFS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = _PREFS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(prefs, indent=2))
    tmp.replace(_PREFS_PATH)


# Module-level console for one-time prints (startup banner in headless mode, etc.)
console = Console()

# Agent style palette
AGENT_STYLES = {
    "Analyst": "green",
    "Scientist": "cyan",
    "Coder": "magenta1",
    "Ingestor": "bright_red",
    "Report": "blue",
    "Critic": "yellow",
    "Debate": "yellow",
    "Results": "dim",
}

# Short descriptions shown immediately when an agent panel opens (before first summary)
AGENT_DESCRIPTIONS: dict[str, str] = {
    "Ingestor": "Preparing and canonicalizing raw data for experiment scripts...",
    "Analyst": "Analyzing experiment results and producing quantitative assessments...",
    "Scientist": "Formulating hypotheses and planning the next experiment...",
    "Critic": "Challenging the plan through critical debate...",
    "Revision": "Revising the experiment plan based on critique feedback...",
    "Coder": "Implementing the experiment plan as a Python script and running it...",
    "Report": "Generating a comprehensive summary report of all findings...",
}

# Maps orchestrator phase names to colors (matches the active agent)
PHASE_STYLES = {
    "INGESTION": "bright_red",
    "ANALYZE": "green",
    "PLAN": "cyan",
    "DEBATE": "yellow",
    "REVISE": "cyan",
    "IMPLEMENT": "magenta1",
    "REPORT": "blue",
}


def _score_style(score: int) -> str:
    """Return a Rich style string for a 0-100 score."""
    if score >= 70:
        return "green"
    if score >= 40:
        return "yellow"
    return "red"


def _format_elapsed(seconds: float) -> str:
    """Format seconds into a human-readable duration like '4m 32s'."""
    m, s = divmod(int(seconds), 60)
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"
