"""Agent output summarizer with periodic polling support.

Provides LLM-generated summaries of agent output during and after each
pipeline step. Used by the orchestrator when --summary-model is set.
"""

import asyncio
import contextlib
import logging
from collections.abc import Callable, Coroutine
from typing import Any, TypeVar

from openai import AsyncOpenAI

from auto_scientist.console import print_summary

T = TypeVar("T")

logger = logging.getLogger(__name__)

PROGRESS_PREFIX = (
    "The agent is currently working. Based on the output so far, "
    "summarize what the agent is currently doing in 1-2 sentences."
)
FINAL_PREFIX = (
    "The agent has finished. Based on the full output, "
    "summarize what the agent accomplished in 1-2 sentences."
)

SUMMARY_PROMPTS: dict[str, str] = {
    "Ingestor": (
        "You are summarizing an Ingestor agent's output. "
        "Focus on: what files are being processed and what transformations are being applied?"
    ),
    "Analyst": (
        "You are summarizing an Analyst agent's output. "
        "Focus on: what findings have been observed? What metrics stand out? "
        "Any improvements or regressions?"
    ),
    "Scientist": (
        "You are summarizing a Scientist agent's output. "
        "Focus on: what hypothesis and strategy are being formulated? "
        "What is the expected impact?"
    ),
    "Scientist Revision": (
        "You are summarizing a Scientist Revision agent's output. "
        "Focus on: what is changing from the original plan? "
        "What revisions were adopted from the debate?"
    ),
    "Debate": (
        "You are summarizing a Debate phase's output. "
        "Focus on: what is being challenged and what positions are forming? "
        "What changed as a result?"
    ),
    "Coder": (
        "You are summarizing a Coder agent's output. "
        "Focus on: what code is being written and what approach is being taken? "
        "Describe the script structure, not line-by-line details."
    ),
    "Results": (
        "You are summarizing experiment results. "
        "Focus on: what did the experiment produce? Key numeric outcomes, "
        "whether the hypothesis was supported, comparison to previous iteration."
    ),
    "Report": (
        "You are summarizing a Report agent's output. "
        "Focus on: what key findings and results are being documented?"
    ),
}


async def _query_summary(model: str, instructions: str, input_text: str) -> str:
    """Call the OpenAI Responses API for a short summary."""
    client = AsyncOpenAI()
    response = await client.responses.create(
        model=model,
        instructions=instructions,
        input=input_text,
        max_output_tokens=150,
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
        instruction = SUMMARY_PROMPTS.get(agent_name, fallback)
        prefix = PROGRESS_PREFIX if progress else FINAL_PREFIX
        instructions = f"{instruction}\n\n{prefix}"
        return await _query_summary(model, instructions, f"Agent output:\n{output}")
    except Exception as e:
        print(f"  SUMMARY: error summarizing {agent_name}: {e}")
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
        return await _query_summary(model, instructions, f"Results:\n{results_text}")
    except Exception as e:
        print(f"  SUMMARY: error summarizing results: {e}")
        return ""


async def run_with_summaries(
    coro_fn: Callable[..., Coroutine[Any, Any, T]],
    agent_name: str,
    summary_model: str,
    message_buffer: list[str],
    interval: int | float = 15,
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

    Returns:
        The result of the agent coroutine.
    """
    last_summarized_index = 0
    elapsed = 0

    async def poll_loop():
        nonlocal last_summarized_index, elapsed
        while True:
            await asyncio.sleep(interval)
            elapsed += interval

            # Only summarize new content
            if len(message_buffer) <= last_summarized_index:
                continue

            new_content = "\n".join(message_buffer[last_summarized_index:])
            last_summarized_index = len(message_buffer)

            try:
                summary = await summarize_agent_output(
                    agent_name, new_content, summary_model, progress=True,
                )
                if summary:
                    label = f"{int(elapsed)}s"
                    print_summary(agent_name, summary, label=label)
            except Exception as e:
                print(f"  SUMMARY: periodic poll error for {agent_name}: {e}")

    poll_task = asyncio.create_task(poll_loop())
    try:
        result = await coro_fn(message_buffer)
    finally:
        poll_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await poll_task

        # Final summary of all output
        if message_buffer:
            try:
                full_output = "\n".join(message_buffer)
                summary = await summarize_agent_output(
                    agent_name, full_output, summary_model, progress=False,
                )
                if summary:
                    print_summary(agent_name, summary, label="done")
            except Exception as e:
                print(f"  SUMMARY: final summary error for {agent_name}: {e}")

    return result
