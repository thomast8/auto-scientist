"""User preference helpers shared by Textual apps."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
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


def system_is_dark() -> bool | None:
    """Detect whether macOS is in dark mode.

    Returns True (dark), False (light), or None (unable to detect, e.g. Linux/Windows).
    """
    if sys.platform != "darwin":
        return None
    try:
        result = subprocess.run(
            ["defaults", "read", "-g", "AppleInterfaceStyle"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        return result.stdout.strip().lower() == "dark"
    except Exception:
        return None


# Bidirectional mapping of dark <-> light theme pairs.
# Themes without a counterpart (e.g. dracula) are left as-is.
_THEME_PAIRS: dict[str, str] = {}
for _dark, _light in [
    ("textual-dark", "textual-light"),
    ("solarized-dark", "solarized-light"),
    ("atom-one-dark", "atom-one-light"),
    ("catppuccin-mocha", "catppuccin-latte"),
    ("rose-pine", "rose-pine-dawn"),
    ("rose-pine-moon", "rose-pine-dawn"),
]:
    _THEME_PAIRS[_dark] = _light
    _THEME_PAIRS[_light] = _dark


def default_theme() -> str:
    """Return a Textual theme matching the OS appearance.

    If we can detect the system appearance (macOS only) and the saved theme
    has a dark/light counterpart, returns the variant that matches. On
    platforms where detection isn't available, the saved theme is returned
    unchanged. Falls back to textual-dark when nothing is saved.
    """
    dark = system_is_dark()
    saved = load_theme()
    if saved is not None:
        if dark is None:
            return saved  # can't detect system appearance, respect saved choice
        # Import here to avoid circular / heavy imports at module level
        from textual.theme import BUILTIN_THEMES

        theme_obj = BUILTIN_THEMES.get(saved)
        if theme_obj is not None and theme_obj.dark == dark:
            return saved  # already matches system
        # Try to flip to the counterpart
        counterpart = _THEME_PAIRS.get(saved)
        if counterpart is not None:
            return counterpart
        # No counterpart, keep the saved theme as-is
        return saved
    if dark is False:
        return "textual-light"
    return "textual-dark"
