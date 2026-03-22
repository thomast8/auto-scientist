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

import json
import logging
from collections.abc import Callable
from typing import Any

from claude_code_sdk import ClaudeCodeOptions

from auto_scientist.console import stream_separator
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
    on_token: Callable[[str], None] | None = None,
) -> str:
    """Dispatch a prompt to the appropriate provider with web search enabled."""
    if config.provider == "openai":
        return await query_openai(
            config.model, prompt,
            web_search=True, reasoning=config.reasoning, on_token=on_token,
        )
    elif config.provider == "google":
        return await query_google(
            config.model, prompt,
            web_search=True, reasoning=config.reasoning, on_token=on_token,
        )
    elif config.provider == "anthropic":
        return await query_anthropic(
            config.model, prompt,
            web_search=True, reasoning=config.reasoning, on_token=on_token,
        )
    else:
        raise ValueError(f"Unknown critic provider: {config.provider!r}")


async def _query_with_retry(
    query_fn: Callable[..., Any],
    *args: Any,
    label: str = "",
    **kwargs: Any,
) -> str:
    """Call a query function, retrying if the response is empty, too short, or errors."""
    result = ""
    for attempt in range(MAX_RETRIES + 1):
        try:
            result = await query_fn(*args, **kwargs)
        except Exception as e:
            if attempt < MAX_RETRIES:
                logger.warning(f"{label} SDK error ({e}), retrying (attempt {attempt + 1})")
                continue
            raise
        if result and len(result.strip()) >= MIN_RESPONSE_LENGTH:
            return result
        if attempt < MAX_RETRIES:
            reason = "empty" if not result or not result.strip() else "too short"
            logger.warning(f"{label} returned {reason} response, retrying (attempt {attempt + 1})")
    return result  # return whatever we got


async def run_debate(
    critic_configs: list[AgentModelConfig],
    plan: dict[str, Any],
    notebook_content: str,
    domain_knowledge: str = "",
    success_criteria: str = "",
    max_rounds: int = 2,
    scientist_config: AgentModelConfig | None = None,
    on_token_factory: Callable[[str], Callable[[str], None]] | None = None,
    message_buffer: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Run a multi-round critic-scientist debate for each critic model.

    Round 1: critic produces initial critique.
    Rounds 2+: scientist responds, then critic critiques again (stateless,
    with the scientist's defense as additional context).

    Both Critic and Scientist receive symmetric context: the plan, notebook,
    domain knowledge, and success criteria. Neither sees the analysis JSON
    or experiment scripts.

    Args:
        critic_configs: List of AgentModelConfig for each critic.
        plan: Scientist's plan dict (hypothesis, strategy, changes).
        notebook_content: Current lab notebook content.
        domain_knowledge: Domain-specific context.
        success_criteria: Formatted top-level success criteria.
        max_rounds: Number of critique rounds (1 = no debate, 2 = default).
        scientist_config: Config for the Scientist's debate responses.

    Returns:
        List of dicts with keys 'model', 'critique', and 'transcript'.
        transcript is a list of {"role": "critic"|"scientist", "content": str}.
    """
    if not critic_configs:
        return []

    # Default scientist config if not provided
    if scientist_config is None:
        scientist_config = AgentModelConfig(model="claude-sonnet-4-6")

    critiques = []
    for config in critic_configs:
        label = f"{config.provider}:{config.model}"
        transcript: list[dict[str, str]] = []

        # Round 1: initial critique (no defense context)
        critic_prompt = _build_critic_prompt(
            plan, notebook_content, domain_knowledge,
            success_criteria=success_criteria,
        )
        on_token = on_token_factory(f"Critic ({label}) round 1") if on_token_factory else None
        critique_text = await _query_with_retry(
            _query_critic, config, critic_prompt, on_token=on_token,
            label=f"Critic ({label}) round 1",
        )
        if on_token:
            stream_separator()
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
            )
            options = ClaudeCodeOptions(
                system_prompt=SCIENTIST_DEBATE_SYSTEM,
                allowed_tools=["WebSearch"],
                max_turns=10,
                model=scientist_config.model,
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
            )
            on_token = (
                on_token_factory(f"Critic ({label}) round {round_num + 1}")
                if on_token_factory else None
            )
            critique_text = await _query_with_retry(
                _query_critic, config, critic_prompt, on_token=on_token,
                label=f"Critic ({label}) round {round_num + 1}",
            )
            if on_token:
                stream_separator()
            transcript.append({"role": "critic", "content": critique_text})
            if message_buffer is not None:
                message_buffer.append(f"[Critic] {critique_text}")

        critiques.append({
            "model": label,
            "critique": critique_text,
            "transcript": transcript,
        })

    return critiques


def _build_critic_prompt(
    plan: dict[str, Any],
    notebook_content: str,
    domain_knowledge: str,
    success_criteria: str = "",
    scientist_defense: str = "",
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

    user = CRITIC_USER.format(
        domain_knowledge=domain_knowledge or "(none provided)",
        success_criteria=success_criteria or "(none defined yet)",
        notebook_content=notebook_content or "(empty)",
        plan_json=json.dumps(plan, indent=2),
        scientist_defense=defense_section,
    )
    return f"{CRITIC_SYSTEM}\n\n{user}"


def _build_scientist_debate_user_prompt(
    plan: dict[str, Any],
    notebook_content: str,
    domain_knowledge: str,
    success_criteria: str = "",
    critique: str = "",
) -> str:
    """Build the user prompt for the Scientist responding to a critique during debate."""
    return SCIENTIST_DEBATE_USER.format(
        domain_knowledge=domain_knowledge or "(none provided)",
        success_criteria=success_criteria or "(none defined yet)",
        notebook_content=notebook_content or "(empty)",
        plan_json=json.dumps(plan, indent=2),
        critique=critique,
    )
