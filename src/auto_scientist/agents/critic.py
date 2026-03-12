"""Critic: multi-model critique dispatcher.

Plain API call (OpenAI/Google/Anthropic SDK), no agent tools needed.
Input: analysis JSON + lab notebook + compressed history.
Output: free-text critique with challenges, alternative hypotheses, suggestions.
"""

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


async def run_critic(
    critic_specs: list[str],
    analysis: dict[str, Any],
    compressed_history: str,
    notebook_content: str,
    domain_knowledge: str = "",
) -> list[dict[str, str]]:
    """Send analysis to one or more critic models for debate.

    Args:
        critic_specs: List of critic specs (e.g., ['openai:gpt-4o', 'google:gemini-2.5-pro']).
        analysis: Structured analysis JSON from the Analyst.
        compressed_history: Compact history summary for context.
        notebook_content: Current lab notebook content.
        domain_knowledge: Domain-specific context.

    Returns:
        List of dicts with keys 'model' and 'critique'.
    """
    if not critic_specs:
        return []

    critiques = []
    for spec in critic_specs:
        provider, model = parse_critic_spec(spec)
        prompt = _build_critic_prompt(analysis, compressed_history, notebook_content, domain_knowledge)

        if provider == "openai":
            response = await query_openai(model, prompt)
        elif provider == "google":
            response = await query_google(model, prompt)
        elif provider == "anthropic":
            response = await query_anthropic(model, prompt)
        else:
            raise ValueError(f"Unknown critic provider: {provider!r}")

        critiques.append({"model": spec, "critique": response})

    return critiques


def _build_critic_prompt(
    analysis: dict[str, Any],
    compressed_history: str,
    notebook_content: str,
    domain_knowledge: str,
) -> str:
    """Build the prompt sent to critic models."""
    import json

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
