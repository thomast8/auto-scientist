"""Critic: multi-model critique dispatcher with Scientist debate loop.

Plain API call (OpenAI/Google/Anthropic SDK), no agent tools needed.
Input: scientist's plan + lab notebook + domain knowledge.
Output: critique with challenges, alternative hypotheses, suggestions,
plus the full debate transcript.

Neither the Critic nor the Scientist (during debate) sees the analysis JSON
or Python code. The plan already incorporates the analysis; passing both is
redundant. Both sides get symmetric context (equal footing).

Round 2+ critics are stateless: they receive the scientist's defense as
additional context but not their own prior critique. This avoids anchoring
bias and lets the critic form a fresh assessment each round.
"""

import asyncio
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from claude_code_sdk import ClaudeCodeOptions

from auto_scientist.agent_result import AgentResult
from auto_scientist.images import ImageData, encode_images_from_paths
from auto_scientist.model_config import AgentModelConfig
from auto_scientist.models.anthropic_client import query_anthropic
from auto_scientist.models.google_client import query_google
from auto_scientist.models.openai_client import query_openai
from auto_scientist.prompts.critic import (
    CRITIC_SYSTEM,
    CRITIC_USER,
    SCIENTIST_DEBATE_SYSTEM,
    SCIENTIST_DEBATE_USER,
)
from auto_scientist.sdk_utils import collect_text_from_query

logger = logging.getLogger(__name__)

MAX_RETRIES = 1  # 1 retry = 2 total attempts for empty responses
MIN_RESPONSE_LENGTH = 50  # minimum chars for a substantive response


async def _query_critic(
    config: AgentModelConfig,
    prompt: str,
    *,
    images: list[ImageData] | None = None,
) -> AgentResult:
    """Dispatch a prompt to the appropriate provider with web search enabled."""
    if config.provider == "openai":
        return await query_openai(
            config.model, prompt,
            web_search=True, reasoning=config.reasoning,
            images=images or [],
        )
    elif config.provider == "google":
        return await query_google(
            config.model, prompt,
            web_search=True, reasoning=config.reasoning,
            images=images or [],
        )
    elif config.provider == "anthropic":
        return await query_anthropic(
            config.model, prompt,
            web_search=True, reasoning=config.reasoning,
            images=images or [],
        )
    else:
        raise ValueError(f"Unknown critic provider: {config.provider!r}")


async def _query_with_retry(
    query_fn: Callable[..., Any],
    *args: Any,
    label: str = "",
    **kwargs: Any,
) -> AgentResult:
    """Call a query function, retrying if the response is empty, too short, or errors."""
    result = AgentResult(text="")
    for attempt in range(MAX_RETRIES + 1):
        try:
            result = await query_fn(*args, **kwargs)
        except Exception as e:
            if attempt < MAX_RETRIES:
                logger.warning(f"{label} SDK error ({e}), retrying (attempt {attempt + 1})")
                continue
            raise
        if result.text and len(result.text.strip()) >= MIN_RESPONSE_LENGTH:
            return result
        if attempt < MAX_RETRIES:
            reason = "empty" if not result.text or not result.text.strip() else "too short"
            logger.warning(f"{label} returned {reason} response, retrying (attempt {attempt + 1})")
    return result  # return whatever we got


async def run_single_critic_debate(
    config: AgentModelConfig,
    plan: dict[str, Any],
    notebook_content: str,
    domain_knowledge: str = "",
    success_criteria: str = "",
    max_rounds: int = 2,
    scientist_config: AgentModelConfig | None = None,
    message_buffer: list[str] | None = None,
    plot_paths: list[Path] | None = None,
    images: list[ImageData] | None = None,
    has_plots: bool = False,
) -> dict[str, Any]:
    """Run a multi-round debate between one critic and the scientist.

    Returns a dict with keys 'model', 'critique', 'transcript',
    'input_tokens', and 'output_tokens'.
    """
    if scientist_config is None:
        scientist_config = AgentModelConfig(model="claude-sonnet-4-6")

    label = f"{config.provider}:{config.model}"
    transcript: list[dict[str, str]] = []
    total_in = 0
    total_out = 0

    # Round 1: initial critique (no defense context)
    critic_prompt = _build_critic_prompt(
        plan, notebook_content, domain_knowledge,
        success_criteria=success_criteria,
        has_plots=has_plots,
    )
    critique_result = await _query_with_retry(
        _query_critic, config, critic_prompt, images=images or [],
        label=f"Critic ({label}) round 1",
    )
    total_in += critique_result.input_tokens
    total_out += critique_result.output_tokens
    critique_text = critique_result.text
    transcript.append({"role": "critic", "content": critique_text})
    if message_buffer is not None:
        message_buffer.append(f"[Critic] {critique_text}")

    # Rounds 2+: scientist responds, then critic critiques again (stateless)
    for round_num in range(1, max_rounds):
        scientist_user_prompt = _build_scientist_debate_user_prompt(
            plan=plan,
            notebook_content=notebook_content,
            domain_knowledge=domain_knowledge,
            success_criteria=success_criteria,
            critique=critique_text,
            has_plots=has_plots,
            plot_paths=plot_paths,
        )
        scientist_tools = ["WebSearch"]
        if has_plots:
            scientist_tools.append("Read")
        options = ClaudeCodeOptions(
            system_prompt=SCIENTIST_DEBATE_SYSTEM,
            allowed_tools=scientist_tools,
            max_turns=10,
            model=scientist_config.model,
            extra_args={"setting-sources": ""},
        )
        scientist_response = await collect_text_from_query(
            scientist_user_prompt, options, message_buffer,
            agent_name=f"Scientist debate round {round_num}",
        )
        transcript.append({"role": "scientist", "content": scientist_response})
        if message_buffer is not None:
            message_buffer.append(f"[Scientist] {scientist_response}")

        # Stateless: critic gets the defense but not their own prior critique
        critic_prompt = _build_critic_prompt(
            plan, notebook_content, domain_knowledge,
            success_criteria=success_criteria,
            scientist_defense=scientist_response,
            has_plots=has_plots,
        )
        critique_result = await _query_with_retry(
            _query_critic, config, critic_prompt, images=images or [],
            label=f"Critic ({label}) round {round_num + 1}",
        )
        total_in += critique_result.input_tokens
        total_out += critique_result.output_tokens
        critique_text = critique_result.text
        transcript.append({"role": "critic", "content": critique_text})
        if message_buffer is not None:
            message_buffer.append(f"[Critic] {critique_text}")

    return {
        "model": label,
        "critique": critique_text,
        "transcript": transcript,
        "input_tokens": total_in,
        "output_tokens": total_out,
    }


async def run_debate(
    critic_configs: list[AgentModelConfig],
    plan: dict[str, Any],
    notebook_content: str,
    domain_knowledge: str = "",
    success_criteria: str = "",
    max_rounds: int = 2,
    scientist_config: AgentModelConfig | None = None,
    message_buffer: list[str] | None = None,
    message_buffers: dict[str, list[str]] | None = None,
    plot_paths: list[Path] | None = None,
) -> list[dict[str, Any]]:
    """Run parallel multi-round critic-scientist debates for each critic model.

    Each critic runs its own independent debate with the scientist concurrently
    via asyncio.gather. Per-critic buffers can be provided via message_buffers.

    Args:
        critic_configs: List of AgentModelConfig for each critic.
        plan: Scientist's plan dict (hypothesis, strategy, changes).
        notebook_content: Current lab notebook content.
        domain_knowledge: Domain-specific context.
        success_criteria: Formatted top-level success criteria.
        max_rounds: Number of critique rounds (1 = no debate, 2 = default).
        scientist_config: Config for the Scientist's debate responses.
        message_buffer: Legacy single shared buffer (used if message_buffers not provided).
        message_buffers: Per-critic buffers keyed by "provider:model" label.
        plot_paths: Paths to plot images to forward to critics.

    Returns:
        List of dicts with keys 'model', 'critique', 'transcript',
        'input_tokens', and 'output_tokens'.
    """
    if not critic_configs:
        return []

    if scientist_config is None:
        scientist_config = AgentModelConfig(model="claude-sonnet-4-6")

    # Encode plot images once for all critics
    images: list[ImageData] = []
    if plot_paths:
        images = encode_images_from_paths(plot_paths)
    has_plots = bool(images)
    logger.info(f"Debate: {len(images)} plot image(s) will be forwarded to critics")

    tasks = []
    for config in critic_configs:
        label = f"{config.provider}:{config.model}"
        # Resolve buffer: per-critic dict > shared legacy buffer > None
        if message_buffers is not None:
            buf = message_buffers.setdefault(label, [])
        else:
            buf = message_buffer

        tasks.append(
            run_single_critic_debate(
                config=config,
                plan=plan,
                notebook_content=notebook_content,
                domain_knowledge=domain_knowledge,
                success_criteria=success_criteria,
                max_rounds=max_rounds,
                scientist_config=scientist_config,
                message_buffer=buf,
                plot_paths=plot_paths,
                images=images,
                has_plots=has_plots,
            )
        )

    results = await asyncio.gather(*tasks)
    return list(results)


def _build_critic_prompt(
    plan: dict[str, Any],
    notebook_content: str,
    domain_knowledge: str,
    success_criteria: str = "",
    scientist_defense: str = "",
    has_plots: bool = False,
) -> str:
    """Build the prompt sent to critic models.

    For round 1, scientist_defense is empty.
    For round 2+, it contains the scientist's response to the previous critique.
    The critic is not told they are "refining" anything (stateless design).
    """
    defense_section = ""
    if scientist_defense:
        defense_section = (
            f"<scientist_defense>{scientist_defense}</scientist_defense>"
        )

    plots_section = _plots_section_text() if has_plots else ""

    user = CRITIC_USER.format(
        domain_knowledge=domain_knowledge or "(none provided)",
        success_criteria=success_criteria or "(none defined yet)",
        notebook_content=notebook_content or "(empty)",
        plan_json=json.dumps(plan, indent=2),
        scientist_defense=defense_section,
        plots_section=plots_section,
    )
    return f"{CRITIC_SYSTEM}\n\n{user}"


def _build_scientist_debate_user_prompt(
    plan: dict[str, Any],
    notebook_content: str,
    domain_knowledge: str,
    success_criteria: str = "",
    critique: str = "",
    has_plots: bool = False,
    plot_paths: list[Path] | None = None,
) -> str:
    """Build the user prompt for the Scientist responding to a critique during debate.

    When plots are available, includes file paths so the SDK agent can read them
    with the Read tool (vision).
    """
    plots_section = ""
    if has_plots:
        plots_section = _plots_section_text()
        if plot_paths:
            path_list = "\n".join(f"- {p}" for p in plot_paths)
            plots_section += (
                f"\n<plot_files>\n"
                f"Use the Read tool to examine each plot file:\n"
                f"{path_list}\n"
                f"</plot_files>"
            )

    return SCIENTIST_DEBATE_USER.format(
        domain_knowledge=domain_knowledge or "(none provided)",
        success_criteria=success_criteria or "(none defined yet)",
        notebook_content=notebook_content or "(empty)",
        plan_json=json.dumps(plan, indent=2),
        critique=critique,
        plots_section=plots_section,
    )


def _plots_section_text() -> str:
    """Return the standard plots-attached section for debate prompts."""
    return (
        "<plots_attached>\n"
        "Experimental plots from the latest iteration are attached as images.\n"
        "Examine them for trends, patterns, anomalies, and numeric values.\n"
        "Reference specific plots when they support or contradict the plan.\n"
        "</plots_attached>"
    )
