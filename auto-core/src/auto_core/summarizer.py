"""Agent output summarizer with periodic polling support.

Provides LLM-generated summaries of agent output during and after each
pipeline step. Used by the orchestrator when --summary-model is set.
"""

import asyncio
import contextlib
import logging
from collections.abc import Callable, Coroutine
from typing import Any

from openai import AsyncOpenAI, AuthenticationError, BadRequestError

logger = logging.getLogger(__name__)

# Track persistent failures so we log them once instead of spamming.
# Using a set (mutable container) avoids the need for `global` statements.
_persistent_failure_logged: set[bool] = set()

PROGRESS_PREFIX = (
    "Reply with ONE sentence, max 15 words. "
    "Use present participle (-ing) voice: 'Inspecting...', 'Writing...', 'Computing...'. "
    "Never refer to the agent in third person. Write as if you are the agent. "
    "Use plain Unicode for math symbols (R², σ, ε, ≈, Δ), not LaTeX notation."
)
FINAL_PREFIX = (
    "Reply with 2-3 sentences, max 40 words total. "
    "Use past tense. First sentence: the main outcome. "
    "Remaining sentences: key metrics, notable findings, or comparisons to prior iterations. "
    "Never refer to the agent in third person (no 'they', 'the agent', 'it'). "
    "Write as if you are the agent: 'Loaded 200 rows...', 'Found high variance in y...'. "
    "Use plain Unicode for math symbols (R², σ, ε, ≈, Δ), not LaTeX notation."
)

SUMMARY_PROMPTS: dict[str, str] = {}
"""Per-agent summary instructions. Populated by `auto_core.roles.install(...)`
at app bootstrap - each app (auto-scientist, auto-reviewer) ships its own
prompt set via its RoleRegistry. Left empty here so the core has no opinion
about what the agents are called."""


async def _query_summary(
    model: str,
    instructions: str,
    input_text: str,
    *,
    max_tokens: int = 60,
) -> str:
    """Call the OpenAI Responses API for a short summary."""
    client = AsyncOpenAI()
    response = await client.responses.create(
        model=model,
        instructions=instructions,
        input=input_text,
        max_output_tokens=max_tokens,
    )
    return response.output_text or ""


async def summarize_agent_output(
    agent_name: str,
    output: str | None,
    model: str,
    *,
    progress: bool = False,
) -> str:
    """Generate a 1-2 sentence summary of agent output.

    Args:
        agent_name: Agent type key (must match SUMMARY_PROMPTS).
        output: The agent's accumulated text output.
        model: OpenAI model to use for summarization.
        progress: If True, use progress prefix; otherwise use final prefix.

    Returns:
        Summary string, or "" on failure.
    """
    if not output:
        return ""

    try:
        fallback = "Summarize the following agent output in 1-2 sentences."
        instruction = SUMMARY_PROMPTS.get(agent_name)
        if instruction is None:
            for key in SUMMARY_PROMPTS:
                if agent_name.startswith(key):
                    instruction = SUMMARY_PROMPTS[key]
                    break
        if instruction is None:
            instruction = fallback
        prefix = PROGRESS_PREFIX if progress else FINAL_PREFIX
        instructions = f"{instruction}\n\n{prefix}"
        max_tokens = 60 if progress else 150
        return await _query_summary(
            model,
            instructions,
            f"Agent output:\n{output}",
            max_tokens=max_tokens,
        )
    except (AuthenticationError, BadRequestError, ImportError, TypeError) as e:
        if True not in _persistent_failure_logged:
            logger.error(f"Summarizer permanently broken ({type(e).__name__}): {e}")
            _persistent_failure_logged.add(True)
        return ""
    except Exception as e:
        logger.warning(f"Transient error summarizing {agent_name}: {e}")
        return ""


ITERATION_RECAP_PREFIX = (
    "Reply with 2-3 sentences, max 50 words total. "
    "Use past tense. Combine the agent summaries into a cohesive iteration recap. "
    "Focus on the main outcome and key findings. "
    "Write in first person plural: 'We explored...', 'We found...'."
)


async def summarize_iteration(
    agent_summaries: list[tuple[str, str]],
    model: str,
) -> str:
    """Generate a combined recap from all agents' final summaries.

    Args:
        agent_summaries: List of (agent_name, done_summary) pairs.
        model: OpenAI model to use for summarization.

    Returns:
        Combined summary string, or "" on failure.
    """
    if not agent_summaries:
        return ""

    lines = [f"{name}: {summary}" for name, summary in agent_summaries if summary]
    if not lines:
        return ""

    combined = "\n".join(lines)
    try:
        return await _query_summary(
            model,
            "Summarize this iteration's agent results into a cohesive recap."
            f"\n\n{ITERATION_RECAP_PREFIX}",
            f"Agent summaries:\n{combined}",
            max_tokens=100,
        )
    except (AuthenticationError, BadRequestError, ImportError, TypeError) as e:
        if True not in _persistent_failure_logged:
            logger.error(f"Summarizer permanently broken ({type(e).__name__}): {e}")
            _persistent_failure_logged.add(True)
        return ""
    except Exception as e:
        logger.warning(f"Transient error generating iteration recap: {e}")
        return ""


async def summarize_results(
    results_text: str,
    model: str,
) -> str:
    """Summarize experiment results.txt content.

    Args:
        results_text: Content of results.txt.
        model: OpenAI model to use for summarization.

    Returns:
        Summary string, or "" on failure.
    """
    if not results_text:
        return ""

    try:
        instruction = SUMMARY_PROMPTS["Results"]
        instructions = f"{instruction}\n\n{FINAL_PREFIX}"
        return await _query_summary(
            model,
            instructions,
            f"Results:\n{results_text}",
            max_tokens=150,
        )
    except (AuthenticationError, BadRequestError, ImportError, TypeError) as e:
        if True not in _persistent_failure_logged:
            logger.error(f"Summarizer permanently broken ({type(e).__name__}): {e}")
            _persistent_failure_logged.add(True)
        return ""
    except Exception as e:
        logger.warning(f"Transient error summarizing results: {e}")
        return ""


async def run_with_summaries[T](
    coro_fn: Callable[..., Coroutine[Any, Any, T]],
    agent_name: str,
    summary_model: str,
    message_buffer: list[str],
    interval: int | float = 15,
    label_prefix: str = "",
    summary_collector: list[tuple[str, str, str]] | None = None,
) -> T:
    """Run an agent coroutine with periodic live summaries.

    Launches a concurrent polling task that every `interval` seconds:
    1. Snapshots new entries in message_buffer since last poll
    2. Sends them to the summary model with a progress prompt
    3. Prints the result via print_summary

    When the coroutine completes, prints a final summary of all output.

    Args:
        coro_fn: Callable that accepts message_buffer and returns a coroutine.
        agent_name: Agent name for prompt selection and display.
        summary_model: OpenAI model for summarization.
        message_buffer: Shared list that the agent appends text to.
        interval: Seconds between periodic polls.
        label_prefix: Prefix for time/done labels (e.g. "Debate: openai:gpt-4o | ").

    Returns:
        The result of the agent coroutine.
    """
    elapsed = 0.0
    max_interval = interval * 16  # Cap backoff at 16x base interval
    progress_summaries: list[str] = []

    async def poll_loop():
        nonlocal elapsed
        current_interval = interval
        last_seen_len = 0
        waiting_emitted = False

        while True:
            await asyncio.sleep(current_interval)
            elapsed += current_interval

            buf_len = len(message_buffer)
            if buf_len == 0:
                # No content yet.  The panel's animated description dots
                # and elapsed-time footer already show the agent is alive,
                # so we only log here for diagnostics.
                if not waiting_emitted and elapsed >= interval * 2:
                    waiting_emitted = True
                    logger.debug(
                        f"{agent_name} [{label_prefix}{int(elapsed)}s]: "
                        "buffer still empty, waiting for model response"
                    )
                continue

            if buf_len == last_seen_len:
                # Buffer unchanged, back off
                current_interval = min(current_interval * 2, max_interval)
                continue

            # New content arrived, reset backoff
            last_seen_len = buf_len
            current_interval = interval

            tail = "\n".join(message_buffer[-10:])
            label = f"{label_prefix}{int(elapsed)}s"
            try:
                summary = await summarize_agent_output(
                    agent_name,
                    tail,
                    summary_model,
                    progress=True,
                )
                if summary:
                    progress_summaries.append(summary)
                    if summary_collector is not None:
                        summary_collector.append((agent_name, summary, label))
                    else:
                        logger.info(f"{agent_name} [{label}]: {summary}")
            except Exception as e:
                logger.warning(f"Periodic poll error for {agent_name}: {e}")

    poll_task = asyncio.create_task(poll_loop())
    try:
        result = await coro_fn(message_buffer)
    finally:
        poll_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await poll_task

        # Final summary: use the agent's actual output trace (tail) so the
        # recap reflects real results, not a lossy summary-of-summaries.
        if message_buffer:
            try:
                tail = "\n".join(message_buffer[-20:])
                summary = await summarize_agent_output(
                    agent_name,
                    tail,
                    summary_model,
                    progress=False,
                )
                if summary:
                    done_label = f"{label_prefix}done"
                    if summary_collector is not None:
                        summary_collector.append((agent_name, summary, done_label))
                    else:
                        logger.info(f"{agent_name} [{done_label}]: {summary}")
            except Exception as e:
                logger.warning(f"Final summary error for {agent_name}: {e}")

    return result
