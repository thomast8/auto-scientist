"""Critic: multi-model critique dispatcher with optional debate loop.

Plain API call (OpenAI/Google/Anthropic SDK), no agent tools needed.
Input: analysis JSON + lab notebook + compressed history.
Output: free-text critique with challenges, alternative hypotheses, suggestions.

When debate rounds > 1, a lightweight defender (Claude via API) responds to
each critique before the critic refines. The Scientist only receives the
final refined critique from each model.
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
    """Dispatch a prompt to the appropriate provider."""
    if provider == "openai":
        return await query_openai(model, prompt)
    elif provider == "google":
        return await query_google(model, prompt)
    elif provider == "anthropic":
        return await query_anthropic(model, prompt)
    else:
        raise ValueError(f"Unknown critic provider: {provider!r}")


async def run_debate(
    critic_specs: list[str],
    analysis: dict[str, Any],
    compressed_history: str,
    notebook_content: str,
    domain_knowledge: str = "",
    script_content: str = "",
    max_rounds: int = 2,
    defender_model: str = "claude-sonnet-4-6",
) -> list[dict[str, str]]:
    """Run a multi-round critic-scientist debate for each critic model.

    Round 1: critic produces initial critique (same as single-pass).
    Rounds 2+: defender responds, then critic refines.
    Only the final critique per model is returned.

    Args:
        critic_specs: List of critic specs (e.g., ['openai:gpt-4o']).
        analysis: Structured analysis JSON from the Analyst.
        compressed_history: Compact history summary for context.
        notebook_content: Current lab notebook content.
        domain_knowledge: Domain-specific context.
        script_content: Current experiment script (for defender context).
        max_rounds: Number of critique rounds (1 = no debate, 2 = default).
        defender_model: Anthropic model used for the defender.

    Returns:
        List of dicts with keys 'model' and 'critique'.
    """
    if not critic_specs:
        return []

    critiques = []
    for spec in critic_specs:
        provider, model = parse_critic_spec(spec)

        # Round 1: initial critique
        critic_prompt = _build_critic_prompt(
            analysis, compressed_history, notebook_content, domain_knowledge
        )
        critique_text = await _query_critic(provider, model, critic_prompt)

        # Rounds 2+: defender responds, critic refines
        for _ in range(1, max_rounds):
            defense = await query_anthropic(
                defender_model,
                _build_defender_prompt(
                    analysis=analysis,
                    notebook_content=notebook_content,
                    compressed_history=compressed_history,
                    script_content=script_content,
                    critique=critique_text,
                ),
            )
            refinement_prompt = _build_critic_refinement_prompt(
                analysis=analysis,
                compressed_history=compressed_history,
                notebook_content=notebook_content,
                domain_knowledge=domain_knowledge,
                critique=critique_text,
                defense=defense,
            )
            critique_text = await _query_critic(provider, model, refinement_prompt)

        critiques.append({"model": spec, "critique": critique_text})

    return critiques


async def run_critic(
    critic_specs: list[str],
    analysis: dict[str, Any],
    compressed_history: str,
    notebook_content: str,
    domain_knowledge: str = "",
) -> list[dict[str, str]]:
    """Single-pass critique (backward-compatible wrapper).

    Equivalent to run_debate with max_rounds=1.
    """
    return await run_debate(
        critic_specs=critic_specs,
        analysis=analysis,
        compressed_history=compressed_history,
        notebook_content=notebook_content,
        domain_knowledge=domain_knowledge,
        max_rounds=1,
    )


def _build_critic_prompt(
    analysis: dict[str, Any],
    compressed_history: str,
    notebook_content: str,
    domain_knowledge: str,
) -> str:
    """Build the prompt sent to critic models."""
    parts = [
        "You are a scientific critic reviewing an autonomous modelling experiment.",
        "Your role is to challenge assumptions, propose alternative hypotheses,",
        "and identify blind spots the scientist may have missed.",
        "",
        "## Domain Knowledge",
        domain_knowledge or "(none provided)",
        "",
        "## Experiment History",
        compressed_history,
        "",
        "## Lab Notebook",
        notebook_content,
        "",
        "## Latest Analysis",
        json.dumps(analysis, indent=2),
        "",
        "## Your Task",
        "Provide a critique covering:",
        "1. Challenges to the current approach",
        "2. Alternative hypotheses the scientist hasn't considered",
        "3. Specific suggestions for the next iteration",
        "4. Whether a paradigm shift is needed (and what it might look like)",
        "5. Any concerns about overfitting, identifiability, or model complexity",
    ]
    return "\n".join(parts)


def _build_defender_prompt(
    analysis: dict[str, Any],
    notebook_content: str,
    compressed_history: str,
    script_content: str,
    critique: str,
) -> str:
    """Build the prompt for the scientist-defender responding to a critique."""
    parts = [
        "You are the scientist who designed and implemented the current experiment.",
        "A critic has reviewed your work. Respond to their critique:",
        "- Defend choices that are well-motivated (explain your reasoning).",
        "- Acknowledge valid points and suggest how to address them.",
        "- Clarify any misunderstandings the critic may have about your approach.",
        "Be concise and substantive. Focus on the most important points.",
        "",
        "## Experiment History",
        compressed_history,
        "",
        "## Lab Notebook",
        notebook_content,
        "",
        "## Latest Analysis",
        json.dumps(analysis, indent=2),
        "",
        "## Current Script",
        script_content or "(not available)",
        "",
        "## Critic's Feedback",
        critique,
        "",
        "## Your Response",
        "Address each major point from the critic. Be honest about weaknesses",
        "but defend choices that have sound reasoning behind them.",
    ]
    return "\n".join(parts)


def _build_critic_refinement_prompt(
    analysis: dict[str, Any],
    compressed_history: str,
    notebook_content: str,
    domain_knowledge: str,
    critique: str,
    defense: str,
) -> str:
    """Build the prompt for a critic to refine after seeing the defense."""
    parts = [
        "You are a scientific critic reviewing an autonomous modelling experiment.",
        "You previously provided a critique. The scientist has responded.",
        "Now refine your critique in light of their defense:",
        "- Drop points that the scientist adequately addressed.",
        "- Sharpen points where their defense was weak or evasive.",
        "- Add new observations prompted by their response.",
        "- Produce a final, actionable critique.",
        "",
        "## Domain Knowledge",
        domain_knowledge or "(none provided)",
        "",
        "## Experiment History",
        compressed_history,
        "",
        "## Lab Notebook",
        notebook_content,
        "",
        "## Latest Analysis",
        json.dumps(analysis, indent=2),
        "",
        "## Your Original Critique",
        critique,
        "",
        "## Scientist's Defense",
        defense,
        "",
        "## Your Task",
        "Produce a refined, final critique. This should be self-contained",
        "(the scientist will only see this version, not the debate history).",
        "Cover the same categories as before:",
        "1. Challenges to the current approach",
        "2. Alternative hypotheses",
        "3. Specific suggestions for the next iteration",
        "4. Whether a paradigm shift is needed",
        "5. Concerns about overfitting, identifiability, or model complexity",
    ]
    return "\n".join(parts)
