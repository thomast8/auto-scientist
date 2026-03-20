"""Agent output summarizer with periodic polling support.

Provides LLM-generated summaries of agent output during and after each
pipeline step. Used by the orchestrator when --summary-model is set.
"""

import logging

from auto_scientist.models.openai_client import query_openai

logger = logging.getLogger(__name__)

PROGRESS_PREFIX = "The agent is currently working. Based on the output so far, summarize what the agent is currently doing in 1-2 sentences."
FINAL_PREFIX = "The agent has finished. Based on the full output, summarize what the agent accomplished in 1-2 sentences."

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
        instruction = SUMMARY_PROMPTS.get(agent_name, "Summarize the following agent output in 1-2 sentences.")
        prefix = PROGRESS_PREFIX if progress else FINAL_PREFIX
        prompt = f"{instruction}\n\n{prefix}\n\nAgent output:\n{output}"
        return await query_openai(model, prompt, max_tokens=150)
    except Exception:
        logger.debug(f"SUMMARY: error summarizing {agent_name} output")
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
        prompt = f"{instruction}\n\n{FINAL_PREFIX}\n\nResults:\n{results_text}"
        return await query_openai(model, prompt, max_tokens=150)
    except Exception:
        logger.debug("SUMMARY: error summarizing results")
        return ""
