"""Compare current vs proposed critic personas on real debate inputs.

Loads actual plan, notebook, analysis, and prediction history from the
alien_minerals_first run (v03 - mid-investigation, stuck on Cryolux/Dravite)
and runs both persona sets against the same inputs using all 3 critic models
(OpenAI, Google, Anthropic).

Each persona is run once per model, so the comparison is:
  Current:  3 personas x 3 models = 9 calls
  Proposed: 3 personas x 3 models = 9 calls
  Total: 18 LLM API calls

Outputs a structured comparison: concern counts, category distribution,
overlap analysis, and example concerns from each persona.

Usage:
    uv run python scripts/compare_personas.py [--iteration v03] [--dry-run]
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# Add src to path (must happen before auto_scientist imports)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from claude_code_sdk import AssistantMessage, ClaudeCodeOptions, TextBlock  # noqa: E402

from auto_scientist.agents.critic import run_single_critic_debate  # noqa: E402
from auto_scientist.agents.debate_models import (  # noqa: E402
    CRITIC_OUTPUT_SCHEMA,
    Concern,
    CriticOutput,
    DebateResult,
    DebateRound,
)
from auto_scientist.agents.scientist import _format_predictions_for_prompt  # noqa: E402
from auto_scientist.model_config import AgentModelConfig  # noqa: E402
from auto_scientist.prompts.critic import CRITIC_SYSTEM_BASE, CRITIC_USER  # noqa: E402
from auto_scientist.prompts.critic import PERSONAS as CURRENT_PERSONAS  # noqa: E402
from auto_scientist.sdk_utils import (  # noqa: E402
    OutputValidationError,
    safe_query,
    validate_json_output,
)
from auto_scientist.state import PredictionRecord  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# All 3 critic models
# ---------------------------------------------------------------------------

CRITIC_MODELS: list[AgentModelConfig] = [
    AgentModelConfig(provider="openai", model="gpt-5.4"),
    AgentModelConfig(provider="google", model="gemini-3.1-pro-preview"),
    AgentModelConfig(provider="anthropic", model="claude-opus-4-6"),
]

# ---------------------------------------------------------------------------
# Proposed personas (the new design)
# ---------------------------------------------------------------------------

PROPOSED_PERSONAS: list[dict[str, str]] = [
    {
        "name": "Methodologist",
        "system_text": (
            "<persona>\n"
            "You are the Methodologist. Your single core question is:\n"
            "**Is this experiment valid?**\n"
            "\n"
            "Your lane covers: evaluation design, statistical rigor, data\n"
            "leakage, confounders, measurement error, sample size adequacy,\n"
            "data quality, label noise, and annotator modeling.\n"
            "\n"
            "Example concerns in your lane:\n"
            "- 'Cross-validation re-optimizes parameters inside a structure\n"
            "   chosen on the full dataset. The structure itself should be\n"
            "   validated, not just the parameters within it.'\n"
            "- 'With only N samples and K candidate models, the effective\n"
            "   sample-per-parameter ratio is too low for reliable selection.'\n"
            "- 'The correction factor assumes a constant offset, but without\n"
            "   a reference measurement, this could create artificial signal.'\n"
            "- 'The evaluation uses the same data for feature selection and\n"
            "   performance estimation, creating optimistic bias.'\n"
            "\n"
            "NOT in your lane (belongs to other personas):\n"
            "- 'The plan keeps adding complexity without progress'\n"
            "   -> Trajectory Critic (arc-level pattern)\n"
            "- 'The plan claims X=5.2 but the analysis says X=3.1'\n"
            "   -> Evidence Auditor (factual consistency)\n"
            "- 'If condition Z holds, the hypothesis fails'\n"
            "   -> Falsification Expert (failure scenario)\n"
            "- 'The goal has drifted from discovery to optimization'\n"
            "   -> Trajectory Critic (goal drift)\n"
            "\n"
            "Use web search to check for established statistical methods\n"
            "relevant to the experimental design.\n"
            "</persona>"
        ),
    },
    {
        "name": "Trajectory Critic",
        "system_text": (
            "<persona>\n"
            "You are the Trajectory Critic. Your single core question is:\n"
            "**Is this line of investigation working?**\n"
            "\n"
            "Your lane covers: progress across iterations (metric trends),\n"
            "circling vs converging, sub-problems stuck too long, sunk cost\n"
            "bias, goal drift across the arc, diminishing returns, and\n"
            "strategy-level complexity bloat.\n"
            "\n"
            "Example concerns in your lane:\n"
            "- 'The key metric has been below the target for three\n"
            "   iterations. Each attempt tweaks parameters but the\n"
            "   fundamental approach has not changed. Circling.'\n"
            "- 'The goal asks for causal discovery, but the last two\n"
            "   iterations optimized predictive accuracy. Goal drift.'\n"
            "- 'The aggregate metric improved, but entirely from one\n"
            "   sub-problem. Another sub-problem actually regressed.\n"
            "   The aggregate masks a broken component.'\n"
            "- 'Three iterations of parameter tuning yielded minimal\n"
            "   total improvement. Diminishing returns suggest a\n"
            "   structural pivot is needed, not more tuning.'\n"
            "\n"
            "NOT in your lane (belongs to other personas):\n"
            "- 'The method is statistically unsound for this sample size'\n"
            "   -> Methodologist (statistical validity)\n"
            "- 'The feature search has a multiple-testing problem'\n"
            "   -> Methodologist (statistical rigor)\n"
            "- 'If those specimens are measurement errors, the rule\n"
            "   fits noise' -> Falsification Expert (failure scenario)\n"
            "- 'The plan says X but the data shows not-X'\n"
            "   -> Evidence Auditor (factual consistency)\n"
            "\n"
            "You evaluate the arc, not the step. Read the notebook and\n"
            "prediction history first. Use web search to check for\n"
            "established solutions to problems the investigation is\n"
            "reinventing.\n"
            "</persona>"
        ),
        "instructions": (
            "<instructions>\n"
            "1. Read the lab notebook and prediction history to understand\n"
            "   the full investigation arc. What has been tried? What worked?\n"
            "   What failed? What patterns emerge across iterations?\n"
            "\n"
            "2. Evaluate whether the investigation is making genuine progress\n"
            "   toward the stated goal. Look for:\n"
            "   - Metric trends: improving, stagnating, or oscillating?\n"
            "   - Sub-problem health: aggregate improving while a sub-problem\n"
            "     is stuck or regressing?\n"
            "   - Strategy effectiveness: has the current approach class\n"
            "     (incremental/structural/exploratory) exhausted its potential?\n"
            "\n"
            "3. Check for circling: is the scientist re-attempting variations\n"
            "   of an approach that already failed? A new threshold on the\n"
            "   same feature set is not a new approach.\n"
            "\n"
            "4. Check for goal drift: has the scientist drifted toward a\n"
            "   proxy objective (matching a benchmark, optimizing a metric\n"
            "   that was never the goal)?\n"
            "\n"
            "5. Check for sunk cost: persisting because of prior investment\n"
            "   rather than evidence?\n"
            "\n"
            "6. If stuck, suggest what class of change is needed (structural\n"
            "   pivot, different sub-problem focus, fresh exploration)\n"
            "   without prescribing specific methods.\n"
            "\n"
            "IMPORTANT: Do not critique the plan's statistical methods.\n"
            "'CART is unstable on small samples' is the Methodologist's\n"
            "concern, not yours. Your concerns are about the investigation\n"
            "direction and progress, not the validity of individual methods.\n"
            "</instructions>"
        ),
    },
    {
        "name": "Falsification Expert",
        "system_text": (
            "<persona>\n"
            "You are the Falsification Expert. Your single core question is:\n"
            "**What would break this hypothesis?**\n"
            "\n"
            "Your lane covers: concrete data patterns or conditions that\n"
            "would falsify the hypothesis, untested assumptions the plan\n"
            "relies on, edge cases, and blind spots.\n"
            "\n"
            "Example concerns in your lane:\n"
            "- 'If the two distributions overlap by more than 30%, no\n"
            "   single threshold can separate them. What is the actual\n"
            "   overlap, and what is the fallback?'\n"
            "- 'The plan assumes those 3 outlier specimens are real data.\n"
            "   If they are measurement errors, the rule fits noise.'\n"
            "- 'The hypothesis depends on feature X cleanly separating\n"
            "   groups A and B. What if the threshold is off by 10%?\n"
            "   Is the separation robust or razor-thin?'\n"
            "- 'What if a specimen has the expected category label but\n"
            "   anomalous feature values? The plan has no fallback path.'\n"
            "\n"
            "NOT in your lane (belongs to other personas):\n"
            "- 'This method is known to be unstable on small samples'\n"
            "   -> Methodologist (method validity, not a failure scenario)\n"
            "- 'The investigation is drifting from the stated goal'\n"
            "   -> Trajectory Critic (goal drift)\n"
            "- 'The plan claims A but the data contradicts A'\n"
            "   -> Evidence Auditor (factual inconsistency)\n"
            "\n"
            "Every concern MUST take the form: 'If [specific condition],\n"
            "then [the hypothesis fails because].'\n"
            "'This method might not work' is NOT a falsification concern.\n"
            "'If X is true, the hypothesis fails because Y' IS.\n"
            "\n"
            "Use web search to check for published failure modes of\n"
            "similar approaches.\n"
            "</persona>"
        ),
    },
    {
        "name": "Evidence Auditor",
        "system_text": (
            "<persona>\n"
            "You are the Evidence Auditor. Your single core question is:\n"
            "**Does this plan match what the data says?**\n"
            "\n"
            "Your lane covers: cross-referencing the plan's empirical claims\n"
            "against the analysis metrics, checking that proposed threshold\n"
            "directions are consistent with per-class statistics, verifying\n"
            "that rule assignments match the data, and flagging when the plan\n"
            "ignores anomalous findings from the analysis or prediction\n"
            "history.\n"
            "\n"
            "Example concerns in your lane:\n"
            "- 'The plan states feature > threshold routes to class A, but\n"
            "   the analysis shows class A has the lowest mean for that\n"
            "   feature. The direction is reversed.'\n"
            "- 'The plan proposes a threshold of 443 to separate groups,\n"
            "   but the analysis reports corrected means of 417 and 455\n"
            "   after the calibration offset. The threshold should use\n"
            "   corrected values, not raw ones.'\n"
            "- 'Prediction 2.3 was refuted, but the plan proceeds as if\n"
            "   it was confirmed, building on an assumption that was\n"
            "   already disproven.'\n"
            "- 'The analysis flags a significant anomaly (17 specimens\n"
            "   misrouted), but the plan does not mention or address it.'\n"
            "\n"
            "Boundary fence - you do NOT:\n"
            "- Evaluate statistical methodology or evaluation design\n"
            "- Evaluate the investigation trajectory or arc\n"
            "- Construct hypothetical failure scenarios\n"
            "- Judge whether the strategy is good or bad\n"
            "\n"
            "You are a fact-checker, not a strategist. Read the analysis\n"
            "data carefully, then read every empirical claim in the plan,\n"
            "and verify each one against the numbers. If the plan says X\n"
            "and the data says not-X, that is your concern. Be specific:\n"
            "quote the plan's claim, quote the contradicting data point,\n"
            "and explain the discrepancy.\n"
            "</persona>"
        ),
    },
]


# ---------------------------------------------------------------------------
# Load inputs from a real run
# ---------------------------------------------------------------------------


def load_debate_inputs(
    run_dir: Path,
    iteration: str,
) -> dict[str, Any]:
    """Load all inputs needed for a debate from an existing run."""
    debate_path = run_dir / iteration / "debate.json"
    with open(debate_path) as f:
        debate_data = json.load(f)
    plan = debate_data["original_plan"]

    with open(run_dir / "state.json") as f:
        state = json.load(f)

    with open(run_dir / iteration / "analysis.json") as f:
        analysis = json.load(f)

    notebook_path = run_dir / "lab_notebook.xml"
    notebook_content = notebook_path.read_text() if notebook_path.exists() else ""

    pred_records = [PredictionRecord(**p) for p in state.get("prediction_history", [])]
    prediction_history = _format_predictions_for_prompt(pred_records)

    return {
        "plan": plan,
        "goal": state.get("goal", ""),
        "domain_knowledge": state.get("domain_knowledge", ""),
        "notebook_content": notebook_content,
        "analysis_json": json.dumps(analysis, indent=2),
        "prediction_history": prediction_history,
    }


# ---------------------------------------------------------------------------
# Run one persona on one model
# ---------------------------------------------------------------------------


async def _query_via_sdk(
    model: str,
    system_prompt: str,
    user_prompt: str,
) -> str:
    """Query an Anthropic model via Claude Code SDK (no API rate limits)."""
    options = ClaudeCodeOptions(
        system_prompt=system_prompt,
        model=model,
        max_turns=1,
        allowed_tools=[],
        permission_mode="bypassPermissions",
    )
    text_parts: list[str] = []
    async for msg in safe_query(prompt=user_prompt, options=options):
        if isinstance(msg, AssistantMessage):
            for block in msg.content:
                if isinstance(block, TextBlock):
                    text_parts.append(block.text)
    return "".join(text_parts)


async def run_single_sdk(
    persona: dict[str, str],
    config: AgentModelConfig,
    inputs: dict[str, Any],
    label: str,
) -> DebateResult:
    """Run a single persona via Claude Code SDK (for Anthropic models)."""
    model_label = f"{config.provider}:{config.model}"
    persona_name = persona["name"]

    print(f"  [{label}] {persona_name} x {model_label} (SDK)...")
    t0 = time.time()

    persona_text = persona.get("system_text", "")
    system = CRITIC_SYSTEM_BASE.format(
        persona_text=persona_text,
        critic_output_schema=json.dumps(CRITIC_OUTPUT_SCHEMA, indent=2),
    )

    defense_section = ""
    user = CRITIC_USER.format(
        goal=inputs["goal"] or "(no goal specified)",
        domain_knowledge=inputs["domain_knowledge"] or "(none provided)",
        notebook_content=inputs["notebook_content"] or "(empty)",
        analysis_json=inputs["analysis_json"] or "(no analysis yet)",
        prediction_history=inputs["prediction_history"] or "(no prediction history yet)",
        plan_json=json.dumps(inputs["plan"], indent=2),
        scientist_defense=defense_section,
    )

    raw_text = await _query_via_sdk(config.model, system, user)

    # Parse the structured output
    try:
        validated = validate_json_output(raw_text, CriticOutput, "Critic")
        critic_output = CriticOutput(**validated)
    except OutputValidationError:
        critic_output = CriticOutput(
            concerns=[
                Concern(
                    claim=f"[PARSE ERROR] {raw_text[:500]}",
                    severity="low",
                    confidence="low",
                    category="other",
                )
            ],
            alternative_hypotheses=[],
            overall_assessment=raw_text[:500],
        )

    elapsed = time.time() - t0
    n_concerns = len(critic_output.concerns)
    print(
        f"  [{label}] {persona_name} x {model_label} (SDK) done in {elapsed:.1f}s "
        f"({n_concerns} concerns)"
    )

    return DebateResult(
        persona=persona_name,
        critic_model=model_label,
        rounds=[DebateRound(critic_output=critic_output)],
        raw_transcript=[{"role": "critic", "content": raw_text}],
    )


async def run_single(
    persona: dict[str, str],
    config: AgentModelConfig,
    inputs: dict[str, Any],
    label: str,
) -> DebateResult:
    """Run a single persona on a single model.

    Uses Claude Code SDK for Anthropic (avoids API rate limits),
    direct API for OpenAI/Google.
    """
    if config.provider == "anthropic":
        return await run_single_sdk(persona, config, inputs, label)

    model_label = f"{config.provider}:{config.model}"
    persona_name = persona["name"]

    print(f"  [{label}] {persona_name} x {model_label}...")
    t0 = time.time()

    result = await run_single_critic_debate(
        config=config,
        plan=inputs["plan"],
        notebook_content=inputs["notebook_content"],
        domain_knowledge=inputs["domain_knowledge"],
        max_rounds=1,
        persona=persona,
        analysis_json=inputs["analysis_json"],
        prediction_history=inputs["prediction_history"],
        goal=inputs["goal"],
    )

    elapsed = time.time() - t0
    n_concerns = sum(len(r.critic_output.concerns) for r in result.rounds)
    print(
        f"  [{label}] {persona_name} x {model_label} done in {elapsed:.1f}s "
        f"({n_concerns} concerns, "
        f"{result.input_tokens} in / {result.output_tokens} out)"
    )
    return result


async def run_persona_set(
    personas: list[dict[str, str]],
    models: list[AgentModelConfig],
    label: str,
    inputs: dict[str, Any],
) -> list[DebateResult]:
    """Run every persona on every model.

    All personas x models run fully concurrently within each model batch.
    """
    results: list[DebateResult] = []
    for config in models:
        model_label = f"{config.provider}:{config.model}"
        print(f"  [{label}] Model batch: {model_label}")
        tasks = [run_single(persona, config, inputs, label) for persona in personas]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in batch_results:
            if isinstance(r, BaseException):
                logger.error(f"[{label}] {model_label} failed: {r}")
                print(f"  [{label}] FAILED: {r}")
            else:
                results.append(r)
        await asyncio.sleep(2)
    return results


# ---------------------------------------------------------------------------
# Analysis and comparison
# ---------------------------------------------------------------------------


def analyze_results(
    label: str,
    results: list[DebateResult],
) -> dict[str, Any]:
    """Extract structured metrics from debate results."""
    all_concerns: list[dict[str, Any]] = []
    by_persona: dict[str, list[Concern]] = {}
    by_model: dict[str, int] = {}

    for result in results:
        persona = result.persona
        model = result.critic_model
        by_persona.setdefault(persona, [])
        by_model.setdefault(model, 0)
        for rnd in result.rounds:
            for concern in rnd.critic_output.concerns:
                by_persona[persona].append(concern)
                by_model[model] += 1
                all_concerns.append(
                    {
                        "persona": persona,
                        "model": model,
                        "claim": concern.claim,
                        "severity": concern.severity,
                        "confidence": concern.confidence,
                        "category": concern.category,
                    }
                )

    # Category distribution
    categories: dict[str, int] = {}
    for c in all_concerns:
        categories[c["category"]] = categories.get(c["category"], 0) + 1

    # Severity distribution
    severities: dict[str, int] = {}
    for c in all_concerns:
        severities[c["severity"]] = severities.get(c["severity"], 0) + 1

    # Per-persona stats (aggregated across models)
    persona_stats = {}
    for persona, persona_concerns in by_persona.items():
        cats: dict[str, int] = {}
        for pc in persona_concerns:
            cats[pc.category] = cats.get(pc.category, 0) + 1
        persona_stats[persona] = {
            "count": len(persona_concerns),
            "high_severity": sum(1 for pc in persona_concerns if pc.severity == "high"),
            "categories": cats,
        }

    # Per-persona x model breakdown
    persona_model_stats: dict[str, dict[str, int]] = {}
    for c in all_concerns:
        key = c["persona"]
        persona_model_stats.setdefault(key, {})
        persona_model_stats[key][c["model"]] = persona_model_stats[key].get(c["model"], 0) + 1

    return {
        "label": label,
        "total_concerns": len(all_concerns),
        "categories": categories,
        "severities": severities,
        "by_model": by_model,
        "persona_stats": persona_stats,
        "persona_model_stats": persona_model_stats,
        "concerns": all_concerns,
    }


def compute_overlap(concerns: list[dict[str, Any]]) -> dict[str, float]:
    """Compute pairwise overlap between personas using keyword similarity.

    For each concern from persona A, find the most similar concern from
    persona B (by word-level Jaccard). Average these best-match scores.
    Higher = more overlap between the two personas.
    """
    from collections import defaultdict

    by_persona: dict[str, list[str]] = defaultdict(list)
    for c in concerns:
        by_persona[c["persona"]].append(c["claim"].lower())

    personas = sorted(by_persona.keys())
    overlaps: dict[str, float] = {}

    for i, p1 in enumerate(personas):
        for p2 in personas[i + 1 :]:
            pair_key = f"{p1} vs {p2}"
            max_similarities = []
            for claim1 in by_persona[p1]:
                words1 = set(claim1.split())
                best_sim = 0.0
                for claim2 in by_persona[p2]:
                    words2 = set(claim2.split())
                    if words1 | words2:
                        jaccard = len(words1 & words2) / len(words1 | words2)
                        best_sim = max(best_sim, jaccard)
                max_similarities.append(best_sim)
            avg_overlap = sum(max_similarities) / len(max_similarities) if max_similarities else 0
            overlaps[pair_key] = round(avg_overlap, 3)

    return overlaps


def print_comparison(current: dict, proposed: dict) -> None:
    """Print a formatted comparison of current vs proposed results."""
    print("\n" + "=" * 70)
    print("PERSONA COMPARISON RESULTS")
    print("=" * 70)

    # Summary table
    print(f"\n{'Metric':<30} {'Current':>15} {'Proposed':>15}")
    print("-" * 60)
    print(
        f"{'Total concerns':<30} {current['total_concerns']:>15} {proposed['total_concerns']:>15}"
    )
    for sev in ["high", "medium", "low"]:
        c_count = current["severities"].get(sev, 0)
        p_count = proposed["severities"].get(sev, 0)
        print(f"{'  ' + sev + ' severity':<30} {c_count:>15} {p_count:>15}")

    # Per-model concern counts
    print(f"\n{'Concerns by model:'}")
    all_models = sorted(set(list(current["by_model"].keys()) + list(proposed["by_model"].keys())))
    for model in all_models:
        c_count = current["by_model"].get(model, 0)
        p_count = proposed["by_model"].get(model, 0)
        print(f"  {model:<35} {c_count:>10} {p_count:>10}")

    # Category distribution
    print(f"\n{'Category distribution:'}")
    all_cats = sorted(set(list(current["categories"].keys()) + list(proposed["categories"].keys())))
    for cat in all_cats:
        c_count = current["categories"].get(cat, 0)
        p_count = proposed["categories"].get(cat, 0)
        print(f"  {cat:<25} {c_count:>10} {p_count:>10}")

    # Per-persona breakdown
    for dataset in [current, proposed]:
        print(f"\n--- {dataset['label']} ---")
        for persona, stats in dataset["persona_stats"].items():
            print(f"  {persona}: {stats['count']} concerns ({stats['high_severity']} high)")
            print(f"    Categories: {stats['categories']}")
            model_breakdown = dataset["persona_model_stats"].get(persona, {})
            for model, count in sorted(model_breakdown.items()):
                print(f"      {model}: {count}")

    # Overlap analysis
    print(f"\n{'Overlap analysis (avg Jaccard similarity):'}")
    c_overlap = compute_overlap(current["concerns"])
    p_overlap = compute_overlap(proposed["concerns"])
    print("  Current personas:")
    for pair, sim in sorted(c_overlap.items()):
        bar = "#" * int(sim * 40)
        print(f"    {pair:<45} {sim:.3f} {bar}")
    print("  Proposed personas:")
    for pair, sim in sorted(p_overlap.items()):
        bar = "#" * int(sim * 40)
        print(f"    {pair:<45} {sim:.3f} {bar}")

    # Average overlap
    c_avg = sum(c_overlap.values()) / len(c_overlap) if c_overlap else 0
    p_avg = sum(p_overlap.values()) / len(p_overlap) if p_overlap else 0
    print(f"\n  Average overlap: Current={c_avg:.3f}, Proposed={p_avg:.3f}")
    if p_avg < c_avg:
        reduction = (1 - p_avg / c_avg) * 100 if c_avg > 0 else 0
        print(f"  Overlap reduced by {reduction:.0f}%")

    # Sample concerns from each persona
    for dataset in [current, proposed]:
        print(f"\n--- {dataset['label']}: Sample concerns ---")
        by_persona: dict[str, list[dict]] = {}
        for c in dataset["concerns"]:
            by_persona.setdefault(c["persona"], []).append(c)
        for persona, concerns in by_persona.items():
            print(f"\n  {persona} ({len(concerns)} concerns across 3 models):")
            # Show one from each model if possible
            shown_models: set[str] = set()
            for c in concerns:
                if c["model"] not in shown_models and len(shown_models) < 3:
                    shown_models.add(c["model"])
                    claim_short = c["claim"][:120]
                    model_short = c["model"].split(":")[-1][:20]
                    sev_conf = f"{c['severity']}/{c['confidence']}"
                    print(f"    [{model_short}] [{sev_conf}] {c['category']}: {claim_short}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    parser = argparse.ArgumentParser(description="Compare current vs proposed critic personas")
    parser.add_argument(
        "--iteration",
        default="v03",
        help="Which iteration to use as test input (default: v03)",
    )
    parser.add_argument(
        "--run-dir",
        default="experiments/runs/alien_minerals_first",
        help="Path to the experiment run directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print inputs and personas without calling LLMs",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    iteration = args.iteration

    print(f"Loading debate inputs from {run_dir}/{iteration}...")
    inputs = load_debate_inputs(run_dir, iteration)

    print(f"  Goal: {inputs['goal'][:80]}...")
    print(f"  Plan hypothesis: {inputs['plan']['hypothesis'][:80]}...")
    print(f"  Analysis size: {len(inputs['analysis_json'])} chars")
    print(f"  Notebook size: {len(inputs['notebook_content'])} chars")
    print(f"  Prediction history size: {len(inputs['prediction_history'])} chars")

    models = CRITIC_MODELS
    print(f"\n  Models: {[f'{c.provider}:{c.model}' for c in models]}")
    n_cur = len(CURRENT_PERSONAS) * len(models)
    n_prop = len(PROPOSED_PERSONAS) * len(models)
    print(f"  Total API calls: {n_cur + n_prop} ({n_cur} current + {n_prop} proposed)")

    if args.dry_run:
        print("\n--- DRY RUN: Current personas ---")
        for p in CURRENT_PERSONAS:
            print(f"  {p['name']}")
            print(f"    system_text: {len(p['system_text'])} chars")
            for m in models:
                print(f"    x {m.provider}:{m.model}")

        print("\n--- DRY RUN: Proposed personas ---")
        for p in PROPOSED_PERSONAS:
            print(f"  {p['name']}")
            print(f"    system_text: {len(p['system_text'])} chars")
            if "instructions" in p:
                print(f"    instructions: {len(p['instructions'])} chars")
            for m in models:
                print(f"    x {m.provider}:{m.model}")
        return

    # Run both persona sets (all personas x all models)
    n_current = len(CURRENT_PERSONAS) * len(models)
    n_proposed = len(PROPOSED_PERSONAS) * len(models)
    print(f"\n--- Running CURRENT personas ({n_current} calls) ---")
    t0 = time.time()
    current_results = await run_persona_set(
        CURRENT_PERSONAS,
        models,
        "CURRENT",
        inputs,
    )
    current_elapsed = time.time() - t0
    print(f"  CURRENT total: {current_elapsed:.1f}s")

    print(f"\n--- Running PROPOSED personas ({n_proposed} calls) ---")
    t0 = time.time()
    proposed_results = await run_persona_set(
        PROPOSED_PERSONAS,
        models,
        "PROPOSED",
        inputs,
    )
    proposed_elapsed = time.time() - t0
    print(f"  PROPOSED total: {proposed_elapsed:.1f}s")

    # Analyze and compare
    current_analysis = analyze_results("CURRENT", current_results)
    proposed_analysis = analyze_results("PROPOSED", proposed_results)

    print_comparison(current_analysis, proposed_analysis)

    # Save raw results
    output_path = Path("experiments") / "persona_comparison.json"
    output_data = {
        "iteration": iteration,
        "run_dir": str(run_dir),
        "models": [f"{c.provider}:{c.model}" for c in models],
        "current": current_analysis,
        "proposed": proposed_analysis,
        "overlap": {
            "current": compute_overlap(current_analysis["concerns"]),
            "proposed": compute_overlap(proposed_analysis["concerns"]),
        },
    }
    output_path.write_text(json.dumps(output_data, indent=2, default=str))
    print(f"\nRaw results saved to {output_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(main())
