"""Process-scoped cost ceiling for Claude CLI invocations.

Reads ``AUTO_SCIENTIST_MAX_RUN_USD`` at first access. Every ``ResultMessage``
from the SDK carries ``total_cost_usd``; the backend feeds those into
:func:`record_cost`. If the cumulative spend crosses the ceiling the next
``record_cost`` call raises :class:`RunBudgetExceededError` and the orchestrator
cancels the run.

Added after the 2026-04-23 zombie-CLI incident so a runaway loop bounded only
by ``max_turns`` cannot burn unbounded dollars.
"""

from __future__ import annotations

import os
import threading

_ENV_VAR = "AUTO_SCIENTIST_MAX_RUN_USD"
_DEFAULT_LIMIT_USD = 25.0


class RunBudgetExceededError(RuntimeError):
    """Raised when cumulative Claude CLI spend exceeds the configured ceiling."""


class _Accumulator:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._total_usd = 0.0
        self._limit_usd: float | None = None

    def _resolve_limit(self) -> float:
        if self._limit_usd is not None:
            return self._limit_usd
        raw = os.environ.get(_ENV_VAR)
        if raw is None:
            self._limit_usd = _DEFAULT_LIMIT_USD
        else:
            try:
                self._limit_usd = float(raw)
            except ValueError as exc:
                raise RuntimeError(f"{_ENV_VAR}={raw!r} is not a valid float") from exc
        return self._limit_usd

    def record(self, usd: float) -> None:
        if usd <= 0:
            return
        limit = self._resolve_limit()
        with self._lock:
            self._total_usd += usd
            total = self._total_usd
        if total > limit:
            raise RunBudgetExceededError(
                f"Claude CLI spend ${total:.4f} exceeded {_ENV_VAR}=${limit:.2f}. "
                f"Set {_ENV_VAR} to raise the ceiling, or investigate the runaway loop."
            )

    def reset(self, limit_usd: float | None = None) -> None:
        with self._lock:
            self._total_usd = 0.0
            self._limit_usd = limit_usd

    @property
    def total_usd(self) -> float:
        with self._lock:
            return self._total_usd


_accumulator = _Accumulator()


def record_cost(usd: float | None) -> None:
    """Feed one ``ResultMessage.total_cost_usd`` into the accumulator.

    ``None`` and non-positive values are silently ignored so callers can pass
    ``getattr(msg, "total_cost_usd", None)`` without extra branching.
    """
    if usd is None:
        return
    _accumulator.record(float(usd))


def reset_budget(limit_usd: float | None = None) -> None:
    """Reset the running total. Use between test cases or CLI invocations."""
    _accumulator.reset(limit_usd)


def total_usd() -> float:
    """Return cumulative cost recorded in this process."""
    return _accumulator.total_usd
