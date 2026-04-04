"""Shared retry/resume infrastructure for agent loops.

All agents follow the same pattern: query the LLM, validate the output,
retry with a correction hint on failure. This module extracts that pattern
into a reusable ``agent_retry_loop`` function.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ValidationError(Exception):
    """Raised by a validator when agent output is wrong but fixable.

    The ``correction_hint`` is appended to the prompt on the next attempt.
    """

    def __init__(self, correction_hint: str) -> None:
        self.correction_hint = correction_hint
        super().__init__(correction_hint)


@dataclass
class QueryResult:
    """Output from a single agent query attempt."""

    raw_output: str
    session_id: str | None
    usage: dict[str, Any]


async def agent_retry_loop(
    query_fn: Callable[[str, str | None], Awaitable[QueryResult]],
    validate_fn: Callable[[QueryResult], T],
    prompt: str,
    *,
    max_attempts: int = 3,
    agent_name: str,
    retryable_errors: tuple[type[Exception], ...] = (Exception,),
    on_exhausted: Callable[[QueryResult | None, Exception], T] | None = None,
) -> T:
    """Run an agent query with automatic retry, validation, and session resume.

    Parameters
    ----------
    query_fn:
        Async callable that runs the LLM query. Receives ``(effective_prompt,
        resume_session_id)`` and returns a ``QueryResult``. When
        ``resume_session_id`` is not None, the closure should rebuild SDK
        options with ``resume=session_id``.
    validate_fn:
        Validates the query result. Returns the validated output on success.
        Raises ``ValidationError`` on fixable errors.
    prompt:
        The original user prompt. Correction hints are appended by the wrapper.
    max_attempts:
        Maximum number of attempts (default 3).
    agent_name:
        Name for log messages.
    retryable_errors:
        Tuple of exception types to catch and retry on SDK/transport failures.
        Non-matching exceptions propagate immediately. Default ``(Exception,)``.
    on_exhausted:
        Called when all attempts fail. Receives ``(last_result, last_error)``.
        ``last_result`` is None if the last failure was an SDK error. If None,
        re-raises the last exception.

    Returns
    -------
    The validated output from ``validate_fn``, or the fallback from
    ``on_exhausted``.
    """
    session_id: str | None = None
    correction_hint = ""

    for attempt in range(max_attempts):
        effective_prompt = prompt + correction_hint

        try:
            result = await query_fn(effective_prompt, session_id)
        except retryable_errors as e:
            if attempt == max_attempts - 1:
                if on_exhausted:
                    return on_exhausted(None, e)
                raise
            logger.warning(f"{agent_name} attempt {attempt + 1}: SDK error ({e}), retrying")
            # Clear session (can't resume a failed connection) but keep
            # any correction hint from a prior validation failure so the
            # next attempt still benefits from the repair prompt.
            session_id = None
            continue

        try:
            return validate_fn(result)
        except ValidationError as e:
            if attempt == max_attempts - 1:
                if on_exhausted:
                    return on_exhausted(result, e)
                raise
            correction_hint = f"\n\n{e.correction_hint}"
            session_id = result.session_id
            logger.warning(f"{agent_name} attempt {attempt + 1}: validation failed, retrying")

    raise RuntimeError(f"{agent_name}: exhausted retries")  # unreachable safety net
