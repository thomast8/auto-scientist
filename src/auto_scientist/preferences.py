"""User preference helpers shared by Textual apps."""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

PREFS_PATH = Path.home() / ".config" / "auto-scientist" / "preferences.json"


def load_preferences() -> dict[str, object]:
    """Load user preferences from disk."""
    try:
        raw = json.loads(PREFS_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return raw if isinstance(raw, dict) else {}


def save_preferences(prefs: dict[str, object]) -> None:
    """Save user preferences to disk with an atomic replace."""
    try:
        PREFS_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = PREFS_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(prefs, indent=2))
        tmp.replace(PREFS_PATH)
    except OSError as e:
        logger.debug(f"Could not save preferences: {e}")


def load_theme() -> str | None:
    """Return the saved theme name if it exists."""
    theme_name = load_preferences().get("theme")
    return theme_name if isinstance(theme_name, str) else None


def save_theme(theme_name: str) -> None:
    """Persist the selected theme name."""
    prefs = load_preferences()
    prefs["theme"] = theme_name
    save_preferences(prefs)
