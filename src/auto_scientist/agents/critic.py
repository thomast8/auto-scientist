"""Critic: multi-model critique dispatcher with Scientist debate loop.

Plain API call (OpenAI/Google/Anthropic SDK), no agent tools needed.
Input: scientist's plan + lab notebook + domain knowledge.
Output: critique with challenges, alternative hypotheses, suggestions,
plus the full debate transcript.

Neither the Critic nor the Scientist (during debate) sees the analysis JSON
or Python code. The plan already incorporates the analysis; passing both is
redundant. Both sides get symmetric context (equal footing).

When debate rounds > 1, the Scientist responds to each critique before the
critic refines. After the debate, the Scientist revision step produces a
revised plan (handled by the orchestrator, not this module).
"""

import json
from typing import Any

from auto_scientist.models.anthropic_client import query_anthropic
from auto_scientist.models.google_client import query_google
from auto_scientist.models.openai_client import query_openai


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
    Rounds 2+: scientist responds, then critic refines.
    Returns the final critique per model plus the full debate transcript.

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

        # Round 1: initial critique
        critic_prompt = _build_critic_prompt(
            plan, notebook_content, domain_knowledge
        )
        critique_text = await _query_critic(provider, model, critic_prompt)
        transcript.append({"role": "critic", "content": critique_text})

        # Rounds 2+: scientist responds, critic refines
        for _ in range(1, max_rounds):
            scientist_response = await query_anthropic(
                scientist_model,
                _build_scientist_response_prompt(
                    plan=plan,
                    notebook_content=notebook_content,
                    domain_knowledge=domain_knowledge,
                    critique=critique_text,
                ),
                web_search=True,
            )
            transcript.append({"role": "scientist", "content": scientist_response})
            refinement_prompt = _build_critic_refinement_prompt(
                plan=plan,
                notebook_content=notebook_content,
                domain_knowledge=domain_knowledge,
                critique=critique_text,
                defense=scientist_response,
            )
            critique_text = await _query_critic(provider, model, refinement_prompt)
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
) -> str:
    """Build the prompt sent to critic models."""
    parts = [
        "You are a scientific critic reviewing an autonomous experiment.",
        "Your role is to challenge the scientist's plan, propose alternative",
        "hypotheses, and identify blind spots. You have web search available",
        "to verify claims, look up papers, and check methods.",
        "",
        "## Domain Knowledge",
        domain_knowledge or "(none provided)",
        "",
        "## Lab Notebook",
        notebook_content or "(empty)",
        "",
        "## Scientist's Plan",
        json.dumps(plan, indent=2),
        "",
        "## Your Task",
        "Provide a critique of the scientist's plan covering:",
        "1. Challenges to the proposed hypothesis and strategy",
        "2. Alternative hypotheses the scientist hasn't considered",
        "3. Specific concerns about the planned changes",
        "4. Whether a different strategy type is needed"
        " (incremental/structural/exploratory)",
        "5. Whether the success criteria are well-chosen tests of the hypothesis"
        " (too lenient? redundant? missing obvious failure modes?)",
        "6. Any concerns about the expected impact or feasibility",
        "",
        "Use web search to verify scientific claims, look up relevant papers,",
        "and check whether proposed methods are sound.",
    ]
    return "\n".join(parts)


def _build_scientist_response_prompt(
    plan: dict[str, Any],
    notebook_content: str,
    domain_knowledge: str,
    critique: str,
) -> str:
    """Build the prompt for the Scientist responding to a critique during debate."""
    parts = [
        "You are the scientist who formulated this plan. A critic has reviewed it.",
        "Respond to their critique:",
        "- Defend choices that are well-motivated (explain your reasoning).",
        "- Acknowledge valid points and suggest how to address them.",
        "- Clarify any misunderstandings the critic may have about your plan.",
        "Be concise and substantive. Focus on the most important points.",
        "You have web search available to back up your claims with references.",
        "",
        "## Domain Knowledge",
        domain_knowledge or "(none provided)",
        "",
        "## Lab Notebook",
        notebook_content or "(empty)",
        "",
        "## Your Plan",
        json.dumps(plan, indent=2),
        "",
        "## Critic's Feedback",
        critique,
        "",
        "## Your Response",
        "Address each major point from the critic. Be honest about weaknesses",
        "but defend the reasoning behind your plan where it is sound.",
    ]
    return "\n".join(parts)


def _build_critic_refinement_prompt(
    plan: dict[str, Any],
    notebook_content: str,
    domain_knowledge: str,
    critique: str,
    defense: str,
) -> str:
    """Build the prompt for a critic to refine after seeing the scientist's response."""
    parts = [
        "You are a scientific critic reviewing an autonomous experiment.",
        "You previously critiqued the scientist's plan. The scientist has responded.",
        "Now refine your critique in light of their response:",
        "- Drop points that the scientist adequately addressed.",
        "- Sharpen points where their response was weak or evasive.",
        "- Add new observations prompted by their response.",
        "- Produce a final, actionable critique.",
        "You have web search available to verify any new claims.",
        "",
        "## Domain Knowledge",
        domain_knowledge or "(none provided)",
        "",
        "## Lab Notebook",
        notebook_content or "(empty)",
        "",
        "## Scientist's Plan",
        json.dumps(plan, indent=2),
        "",
        "## Your Original Critique",
        critique,
        "",
        "## Scientist's Response",
        defense,
        "",
        "## Your Task",
        "Produce a refined, final critique. This should be self-contained.",
        "Cover the same categories as before:",
        "1. Challenges to the proposed hypothesis and strategy",
        "2. Alternative hypotheses",
        "3. Specific concerns about the planned changes",
        "4. Whether a different strategy type is needed",
        "5. Whether the success criteria are appropriate",
        "6. Concerns about feasibility or expected impact",
    ]
    return "\n".join(parts)
