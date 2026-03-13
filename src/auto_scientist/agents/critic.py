"""Critic: multi-model critique dispatcher with optional debate loop.

Plain API call (OpenAI/Google/Anthropic SDK), no agent tools needed.
Input: scientist's plan + lab notebook + compressed history + domain knowledge.
Output: free-text critique with challenges, alternative hypotheses, suggestions.

Neither the Critic nor the Defender sees the analysis JSON or Python code.
The plan already incorporates the analysis; passing both is redundant.
Both sides get symmetric context (equal footing).

When debate rounds > 1, a lightweight defender (Claude via API) responds to
each critique before the critic refines. The Coder only receives the
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
    compressed_history: str,
    notebook_content: str,
    domain_knowledge: str = "",
    max_rounds: int = 2,
    defender_model: str = "claude-sonnet-4-6",
) -> list[dict[str, str]]:
    """Run a multi-round critic-scientist debate for each critic model.

    Round 1: critic produces initial critique (same as single-pass).
    Rounds 2+: defender responds, then critic refines.
    Only the final critique per model is returned.

    Both Critic and Defender receive symmetric context: the plan, notebook,
    compressed history, and domain knowledge. Neither sees the analysis JSON
    or experiment scripts.

    Args:
        critic_specs: List of critic specs (e.g., ['openai:gpt-4o']).
        plan: Scientist's plan dict (hypothesis, strategy, changes).
        compressed_history: Compact history summary for context.
        notebook_content: Current lab notebook content.
        domain_knowledge: Domain-specific context.
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
            plan, compressed_history, notebook_content, domain_knowledge
        )
        critique_text = await _query_critic(provider, model, critic_prompt)

        # Rounds 2+: defender responds, critic refines
        for _ in range(1, max_rounds):
            defense = await query_anthropic(
                defender_model,
                _build_defender_prompt(
                    plan=plan,
                    notebook_content=notebook_content,
                    compressed_history=compressed_history,
                    domain_knowledge=domain_knowledge,
                    critique=critique_text,
                ),
                web_search=True,
            )
            refinement_prompt = _build_critic_refinement_prompt(
                plan=plan,
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
    plan: dict[str, Any],
    compressed_history: str,
    notebook_content: str,
    domain_knowledge: str = "",
) -> list[dict[str, str]]:
    """Single-pass critique (backward-compatible wrapper).

    Equivalent to run_debate with max_rounds=1.
    """
    return await run_debate(
        critic_specs=critic_specs,
        plan=plan,
        compressed_history=compressed_history,
        notebook_content=notebook_content,
        domain_knowledge=domain_knowledge,
        max_rounds=1,
    )


def _build_critic_prompt(
    plan: dict[str, Any],
    compressed_history: str,
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
        "## Experiment History",
        compressed_history or "(no history yet)",
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
        "5. Any concerns about the expected impact or feasibility",
        "",
        "Use web search to verify scientific claims, look up relevant papers,",
        "and check whether proposed methods are sound.",
    ]
    return "\n".join(parts)


def _build_defender_prompt(
    plan: dict[str, Any],
    notebook_content: str,
    compressed_history: str,
    domain_knowledge: str,
    critique: str,
) -> str:
    """Build the prompt for the scientist-defender responding to a critique."""
    parts = [
        "You are the scientist who formulated the plan for the next experiment.",
        "A critic has reviewed your plan. Respond to their critique:",
        "- Defend choices that are well-motivated (explain your reasoning).",
        "- Acknowledge valid points and suggest how to address them.",
        "- Clarify any misunderstandings the critic may have about your plan.",
        "Be concise and substantive. Focus on the most important points.",
        "You have web search available to back up your claims with references.",
        "",
        "## Domain Knowledge",
        domain_knowledge or "(none provided)",
        "",
        "## Experiment History",
        compressed_history or "(no history yet)",
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
    compressed_history: str,
    notebook_content: str,
    domain_knowledge: str,
    critique: str,
    defense: str,
) -> str:
    """Build the prompt for a critic to refine after seeing the defense."""
    parts = [
        "You are a scientific critic reviewing an autonomous experiment.",
        "You previously critiqued the scientist's plan. The scientist has responded.",
        "Now refine your critique in light of their defense:",
        "- Drop points that the scientist adequately addressed.",
        "- Sharpen points where their defense was weak or evasive.",
        "- Add new observations prompted by their response.",
        "- Produce a final, actionable critique.",
        "You have web search available to verify any new claims.",
        "",
        "## Domain Knowledge",
        domain_knowledge or "(none provided)",
        "",
        "## Experiment History",
        compressed_history or "(no history yet)",
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
        "## Scientist's Defense",
        defense,
        "",
        "## Your Task",
        "Produce a refined, final critique. This should be self-contained",
        "(the coder will only see this version, not the debate history).",
        "Cover the same categories as before:",
        "1. Challenges to the proposed hypothesis and strategy",
        "2. Alternative hypotheses",
        "3. Specific concerns about the planned changes",
        "4. Whether a different strategy type is needed",
        "5. Concerns about feasibility or expected impact",
    ]
    return "\n".join(parts)
