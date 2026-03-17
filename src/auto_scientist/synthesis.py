"""Periodic investigation synthesis.

Plain Anthropic API call (no tools, no agent). Runs every N iterations
before the Analyst, condensing the full notebook into a concise narrative.
The synthesis replaces raw notebook content in downstream prompts for that
iteration; the raw notebook file stays on disk untouched.
"""

from auto_scientist.models.anthropic_client import query_anthropic

SYNTHESIS_PROMPT = """\
You are a scientific editor. Your task is to condense a lab notebook into a
concise investigation narrative that captures the essential context.

## Lab Notebook (Full)
{notebook_content}

## Domain Knowledge
{domain_knowledge}

## Your Task
Produce a condensed narrative (target: 30-50% of the original notebook length)
that preserves:
1. The overall goal and what has been tried
2. Key hypotheses and whether they panned out
3. Major structural changes and their outcomes
4. Dead ends and why they were abandoned
5. The current state: what works, what doesn't, what's next

Write in a direct, factual style. Use section headers for clarity.
Do NOT include raw metric tables or full parameter lists; summarize trends.
This narrative will replace the full notebook in agent prompts, so it must
contain enough context for a scientist to plan the next iteration.
"""


async def run_synthesis(
    notebook_content: str,
    domain_knowledge: str = "",
    model: str = "claude-sonnet-4-6",
) -> str:
    """Condense the lab notebook into a concise narrative.

    Args:
        notebook_content: Full text of the lab notebook.
        domain_knowledge: Domain-specific context.
        model: Anthropic model to use.

    Returns:
        Condensed narrative string.
    """
    prompt = SYNTHESIS_PROMPT.format(
        notebook_content=notebook_content,
        domain_knowledge=domain_knowledge or "(none provided)",
    )
    return await query_anthropic(model, prompt)
