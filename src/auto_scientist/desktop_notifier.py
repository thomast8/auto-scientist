"""Fire-and-forget macOS desktop notifications for long-running pipelines.

Mirrors the user's Claude Code ``Stop`` hook (``~/.claude/hooks/notify-stop.sh``)
which shells out to the ``alerter`` command. This module wraps the same tool
so an auto-scientist run can ping its operator when an agent finishes, when
an iteration wraps, or when the whole run terminates, without requiring the
TUI to stay focused.

Design:
- Opt-in via a ``level`` string (``off``/``run``/``iteration``/``agent``).
  Each level includes the coarser ones.
- Silent no-op when ``alerter`` is not on PATH (e.g. Linux, CI, minimal macOS).
- Subprocess is fully detached (``start_new_session=True`` + DEVNULL redirects)
  so notifications can never block or stall the orchestrator worker thread.
- All notifications share one ``--group auto-scientist`` so successive ones
  replace each other in macOS Notification Center instead of stacking.
- Each notification carries the run name in the subtitle and a rich message
  body with iteration context and per-agent stats where available.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


_MAX_MESSAGE_CHARS = 500
_GROUP = "auto-scientist"
_DEFAULT_ICON = Path(__file__).parent / "assets" / "notifier_icon.png"


class DesktopNotifier:
    """Emit desktop notifications at configurable granularity.

    The notifier keeps track of the *current iteration label* so that
    per-agent notifications can be stamped with their enclosing iteration
    without callers having to re-pass context on every call.
    """

    LEVELS: dict[str, int] = {"off": 0, "run": 1, "iteration": 2, "agent": 3}

    def __init__(
        self,
        level: str = "off",
        run_name: str = "",
        icon_path: Path | None = None,
    ) -> None:
        self._level = self.LEVELS.get(level, 0)
        self._run_name = run_name
        self._iteration_label: str = ""
        # Don't probe the filesystem at all when notifications are disabled.
        self._alerter: str | None = shutil.which("alerter") if self._level > 0 else None
        if self._level > 0 and self._alerter is None:
            logger.info(
                "Desktop notifications requested (level=%s) but `alerter` "
                "is not installed; notifications disabled.",
                level,
            )

        # Resolve icon: caller-supplied wins, otherwise fall back to the
        # bundled default if it exists, otherwise omit the flag entirely.
        resolved_icon: Path | None = None
        if icon_path is not None and icon_path.exists():
            resolved_icon = icon_path
        elif _DEFAULT_ICON.exists():
            resolved_icon = _DEFAULT_ICON
        self._icon_path: str | None = str(resolved_icon) if resolved_icon else None

    # -- Public API -----------------------------------------------------------

    def set_iteration(self, label: str) -> None:
        """Update the current iteration label (e.g. ``"Iteration 3/20"``).

        Called by PipelineLive at the start of each iteration so subsequent
        per-agent notifications can include the iteration context.
        """
        self._iteration_label = label

    def agent_done(
        self,
        name: str,
        elapsed: str,
        summary: str,
        *,
        num_turns: int | None = None,
        total_tokens: int | None = None,
    ) -> None:
        """Fire when a single agent finishes its run.

        Title carries the "what" (agent name), subtitle the "where"
        (run + iteration), message the "why" (elapsed + summary).
        The ``num_turns`` and ``total_tokens`` args are accepted for
        future use but intentionally omitted from the body to keep
        notifications scannable.
        """
        del num_turns, total_tokens  # intentionally unused, kept for API stability
        if self._level < self.LEVELS["agent"]:
            return

        body = f"{elapsed}\n{summary}" if summary else elapsed
        self._fire(
            title=f"{name} done",
            subtitle=self._compose_subtitle(),
            message=body,
        )

    def iteration_done(self, label: str, summary: str) -> None:
        """Fire when an iteration (or phase, e.g. Ingestion/Report) wraps."""
        if self._level < self.LEVELS["iteration"]:
            return

        body = summary if summary else label
        self._fire(
            title=label,
            subtitle=self._run_name,
            message=body,
        )

    def run_complete(self, status: str, summary: str) -> None:
        """Fire once the whole orchestrator run terminates."""
        if self._level < self.LEVELS["run"]:
            return
        self._fire(
            title=f"auto-scientist {status}",
            subtitle=self._run_name,
            message=summary,
            sound=True,
        )

    # -- Internals ------------------------------------------------------------

    def _compose_subtitle(self) -> str:
        """Run name + optional iteration label, separated by a middle dot."""
        if self._iteration_label:
            return f"{self._run_name} · {self._iteration_label}"
        return self._run_name

    def _fire(
        self,
        title: str,
        message: str,
        *,
        subtitle: str | None = None,
        sound: bool = False,
    ) -> None:
        if self._alerter is None:
            return

        clean_message = (message or "").replace("\r", " ")[:_MAX_MESSAGE_CHARS]
        effective_subtitle = subtitle if subtitle is not None else self._run_name

        args: list[str] = [
            self._alerter,
            "--title",
            title,
            "--subtitle",
            effective_subtitle,
            "--message",
            clean_message,
            "--group",
            _GROUP,
            # `--actions` flips alerter from banner mode (auto-retract after
            # ~5s) to alert mode (sticky until dismissed). Combined with
            # `--group` this means at most one notification is on screen at
            # a time: the newest one replaces its predecessor.
            "--actions",
            "OK",
            "--close-label",
            "Dismiss",
            "--ignore-dnd",
        ]
        if self._icon_path:
            args += ["--app-icon", self._icon_path]
        if sound:
            args += ["--sound", "default"]

        try:
            subprocess.Popen(
                args,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError:
            logger.debug("alerter launch failed", exc_info=True)
