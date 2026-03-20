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
from typing import Any

from auto_scientist.models.anthropic_client import query_anthropic
from auto_scientist.models.google_client import query_google
from auto_scientist.models.openai_client import query_openai
from auto_scientist.prompts.critic import (
    CRITIC_SYSTEM,
    CRITIC_USER,
    SCIENTIST_DEBATE_SYSTEM,
    SCIENTIST_DEBATE_USER,
)


def parse_critic_spec(spec: str) -> tuple[str, str]:
    """Parse a critic spec like 'openai:gpt-4o' into (provider, model).

    Args:
        spec: Critic specification in 'provider:model' format.

    Returns:
        Tuple of (provider, model_name).
    """
    parts = spec.split(":", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid critic spec: {spec!r}. Expected 'provider:model'.")
    return parts[0], parts[1]


async def _query_critic(provider: str, model: str, prompt: str) -> str:
    """Dispatch a prompt to the appropriate provider with web search enabled."""
    if provider == "openai":
        return await query_openai(model, prompt, web_search=True)
    elif provider == "google":
        return await query_google(model, prompt, web_search=True)
    elif provider == "anthropic":
        return await query_anthropic(model, prompt, web_search=True)
    else:
        raise ValueError(f"Unknown critic provider: {provider!r}")


async def run_debate(
    critic_specs: list[str],
    plan: dict[str, Any],
    notebook_content: str,
    domain_knowledge: str = "",
    max_rounds: int = 2,
    scientist_model: str = "claude-sonnet-4-6",
) -> list[dict[str, Any]]:
    """Run a multi-round critic-scientist debate for each critic model.

    Round 1: critic produces initial critique.
    Rounds 2+: scientist responds, then critic critiques again (stateless,
    with the scientist's defense as additional context).

    Both Critic and Scientist receive symmetric context: the plan, notebook,
    and domain knowledge. Neither sees the analysis JSON or experiment scripts.

    Args:
        critic_specs: List of critic specs (e.g., ['openai:gpt-4o']).
        plan: Scientist's plan dict (hypothesis, strategy, changes).
        notebook_content: Current lab notebook content.
        domain_knowledge: Domain-specific context.
        max_rounds: Number of critique rounds (1 = no debate, 2 = default).
        scientist_model: Anthropic model used for the Scientist's debate responses.

    Returns:
        List of dicts with keys 'model', 'critique', and 'transcript'.
        transcript is a list of {"role": "critic"|"scientist", "content": str}.
    """
    if not critic_specs:
        return []

    critiques = []
    for spec in critic_specs:
        provider, model = parse_critic_spec(spec)
        transcript: list[dict[str, str]] = []

        # Round 1: initial critique (no defense context)
        critic_prompt = _build_critic_prompt(
            plan, notebook_content, domain_knowledge
        )
        critique_text = await _query_critic(provider, model, critic_prompt)
        transcript.append({"role": "critic", "content": critique_text})

        # Rounds 2+: scientist responds, then critic critiques again (stateless)
        for _ in range(1, max_rounds):
            scientist_response = await query_anthropic(
                scientist_model,
                _build_scientist_debate_prompt(
                    plan=plan,
                    notebook_content=notebook_content,
                    domain_knowledge=domain_knowledge,
                    critique=critique_text,
                ),
                web_search=True,
            )
            transcript.append({"role": "scientist", "content": scientist_response})

            # Stateless: critic gets the defense but not their own prior critique
            critic_prompt = _build_critic_prompt(
                plan, notebook_content, domain_knowledge,
                scientist_defense=scientist_response,
            )
            critique_text = await _query_critic(provider, model, critic_prompt)
            transcript.append({"role": "critic", "content": critique_text})

        critiques.append({
            "model": spec,
            "critique": critique_text,
            "transcript": transcript,
        })

    return critiques


async def run_critic(
    critic_specs: list[str],
    plan: dict[str, Any],
    notebook_content: str,
    domain_knowledge: str = "",
) -> list[dict[str, Any]]:
    """Single-pass critique (backward-compatible wrapper).

    Equivalent to run_debate with max_rounds=1.
    """
    return await run_debate(
        critic_specs=critic_specs,
        plan=plan,
        notebook_content=notebook_content,
        domain_knowledge=domain_knowledge,
        max_rounds=1,
    )


def _build_critic_prompt(
    plan: dict[str, Any],
    notebook_content: str,
    domain_knowledge: str,
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
        notebook_content=notebook_content or "(empty)",
        plan_json=json.dumps(plan, indent=2),
        scientist_defense=defense_section,
    )
    return f"{CRITIC_SYSTEM}\n\n{user}"


def _build_scientist_debate_prompt(
    plan: dict[str, Any],
    notebook_content: str,
    domain_knowledge: str,
    critique: str,
) -> str:
    """Build the prompt for the Scientist responding to a critique during debate."""
    user = SCIENTIST_DEBATE_USER.format(
        domain_knowledge=domain_knowledge or "(none provided)",
        notebook_content=notebook_content or "(empty)",
        plan_json=json.dumps(plan, indent=2),
        critique=critique,
    )
    return f"{SCIENTIST_DEBATE_SYSTEM}\n\n{user}"
