"""User preference helpers shared by Textual apps."""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

PREFS_PATH = Path.home() / ".config" / "auto-scientist" / "preferences.json"


def load_preferences() -> dict[str, object]:
    """Load user preferences from disk."""
    try:
        raw = json.loads(PREFS_PATH.read_text())
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as e:
        logger.warning(f"Preferences file is corrupt ({PREFS_PATH}), using defaults: {e}")
        return {}
    except OSError as e:
        logger.warning(f"Could not read preferences ({PREFS_PATH}): {e}")
        return {}
    return raw if isinstance(raw, dict) else {}


def save_preferences(prefs: dict[str, object]) -> None:
    """Save user preferences to disk with an atomic replace."""
    try:
        PREFS_PATH.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=PREFS_PATH.parent, suffix=".tmp", prefix="prefs-")
        try:
            with open(fd, "w") as fh:
                json.dump(prefs, fh, indent=2)
            Path(tmp_name).replace(PREFS_PATH)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise
    except OSError as e:
        logger.warning(f"Could not save preferences to {PREFS_PATH}: {e}")


def load_theme() -> str | None:
    """Return the saved theme name if it exists."""
    theme_name = load_preferences().get("theme")
    return theme_name if isinstance(theme_name, str) else None


def save_theme(theme_name: str) -> None:
    """Persist the selected theme name."""
    prefs = load_preferences()
    prefs["theme"] = theme_name
    save_preferences(prefs)
