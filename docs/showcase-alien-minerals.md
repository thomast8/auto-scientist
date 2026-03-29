# Case Study: Classifying Alien Minerals

**An autonomous scientific investigation, start to finish, in under an hour.**

An LLM-driven system was given 500 specimens of six unknown mineral types, nine physical measurements per specimen, some mislabeled samples, and a simple directive: *figure out the classification rules*. No human touched the data, wrote any analysis code, or chose any model. Fifty-eight minutes later, the system delivered a five-feature decision hierarchy that correctly classifies every reliably-labeled specimen in the dataset.

This document walks through what happened, why it matters, and what it reveals about autonomous scientific reasoning.

---

## The Problem

A dataset of 500 alien mineral specimens, each belonging to one of six types: Aetherite, Borealis, Cryolux, Dravite, Erythian, and Fenrite. The specimens are described by nine continuous measurements (conductivity, magnetic susceptibility, fluorescence wavelength, color spectrum peak, refractive index, hardness, density, thermal expansion, solubility), one categorical property (crystal symmetry: cubic/hexagonal/orthorhombic), and collection metadata.

Three complications:
1. **Label noise.** Three geologists independently labeled every specimen. 25 of 500 disagreed; the majority vote was used as ground truth.
2. **Calibration artifacts.** 136 specimens carry instrument notes flagging potential measurement issues - fluorescence detector offsets, humidity effects, probe recalibrations.
3. **The goal is interpretability.** Not a black-box classifier. The system must express its findings as human-readable decision rules with explicit thresholds on named features.

```
auto-scientist run -c domains/alien_minerals/experiment.yaml
```

## The Agents

Auto-Scientist uses a multi-agent loop where each agent has a distinct role and strict information boundaries:

| Agent | Role | Sees | Does Not See |
|-------|------|------|--------------|
| **Ingestor** | Canonicalize raw data | Raw files | Nothing else yet |
| **Analyst** | Observe results, no recommendations | Results, plots, raw data | Code |
| **Scientist** | Plan experiments, define hypotheses | Analysis JSON, notebook | Code |
| **Critics** | Challenge the plan from multiple angles | Plan, analysis, predictions | Code |
| **Coder** | Implement and run experiments | Plan, previous script | Debate transcript |

The Scientist never sees code. The Coder never sees the debate. The Analyst is not allowed to recommend, only observe. These boundaries prevent the kind of confirmation bias that emerges when the same entity plans, implements, and evaluates.

For this run, the Scientist was Claude Opus 4.6. The Critics were GPT-5.4 and Claude Sonnet 4.6. Everything else ran on Sonnet 4.6.

## The Investigation

### Iteration 0: Exploration (10 min)

The system ingested two source files (a SQLite database and a CSV of field notes), denormalized them into a flat analysis table, and ran its first experiment: per-class feature distributions, ANOVA F-statistics, correlation analysis, calibration bias testing, and a baseline depth-4 decision tree.

**What it found:**

Crystal symmetry nearly perfectly partitions the six minerals into three groups: orthorhombic specimens are Fenrite (96.7%), hexagonal specimens are Aetherite or Borealis (96.7%), and cubic specimens are Erythian, Cryolux, or Dravite (95-99%). A baseline decision tree achieved macro F1 = 0.9152 on 5x5 repeated stratified cross-validation.

Two puzzles emerged immediately. First, `color_spectrum_peak` had the highest univariate ANOVA F of any feature (F = 278.87), yet the decision tree assigned it near-zero importance (0.012). Second, `hardness` showed statistically significant calibration bias (0.632 std shift, p = 0.009) between flagged and unflagged specimens.

**What it predicted (and got wrong):**

The Scientist made four testable predictions. Three were refuted: only one feature exceeded F > 50 (not two as predicted), calibration bias existed in hardness (the prediction said it wouldn't), and disagreement specimens were spread across 12 class pairs (not concentrated in 2-3). The baseline tree prediction was confirmed, massively exceeding the F1 >= 0.65 threshold at 0.9152.

Three out of four predictions wrong on the first try. This is actually fine. The predictions weren't guesses - they were hypotheses with pre-specified diagnostics and branching plans. Each refutation redirected the investigation productively.

### Iteration 1: The Dead End (23 min)

This is where it gets interesting. The Scientist looked at the crystal symmetry signal (96.7% purity per group) and concluded the classification must follow a hierarchical structure: split by symmetry first, then use one or two continuous features within each group.

The plan:
- Orthorhombic -> Fenrite
- Hexagonal -> split Aetherite from Borealis by conductivity
- Cubic -> split Erythian by magnetic susceptibility, then Cryolux from Dravite by fluorescence

Before the Coder touched any code, four critics challenged this plan from different angles:

> **Methodologist:** "The hierarchical structure was preselected using full v00 data, so the 5x5 CV likely leaked structural choices, making macro F1 optimistic."
>
> **Trajectory Critic (GPT-5.4):** "Aiming only to match macro F1 ~0.9152 could justify exception-heavy rule patches. Residual Cryolux/Dravite errors are likely noise-limited rather than separator-limited."
>
> **Falsification Expert:** "Cryolux vs Dravite within cubic+high magnetic_susceptibility could be impossible to separate, because top-feature gaps are only ~0.2 each."

The Scientist accepted some critiques (expanded calibration testing, constrained the Cryolux/Dravite search to 5 candidates) and rejected others (nested CV was overkill for a discovery iteration). The revised plan went to the Coder.

**Result: macro F1 = 0.6037.** A catastrophic 31-point regression.

Cryolux F1 collapsed to 0.134. Dravite hit 0.038. Within the cubic+high-magnetic-susceptibility subgroup, all five candidate features for separating Cryolux from Dravite had ANOVA F = 0.00. The best leave-one-out accuracy was 1.8%, worse than random.

The imposed hierarchy didn't just underperform - it destroyed the classification for two entire mineral types.

### Iteration 2: The Pivot (15 min)

The Analyst reported the regression without recommendations (as designed). The Scientist proposed a hybrid approach: extract rules from the v00 tree and graft them onto the hierarchy. Then the critics intervened again:

> **Trajectory Critic:** "This is sunk-cost structural circling. v00 already achieved interpretable depth-4 rules at macro F1 ~0.915 with Cryolux/Dravite F1 ~0.86-0.88. Two iterations have detoured into a failed hybrid."

This critique landed. The Scientist's revised plan opened with a remarkable admission:

> *"The v00 depth-4 decision tree already constitutes the interpretable rule set requested by the investigation goal; the remaining work is to extract its rules, verify their stability across resamples, confirm they are not driven by calibration artifacts, and document them - not to construct a new classifier."*

The system abandoned the hand-crafted hierarchy entirely. Instead, it fit depth-4 trees inside each of the 25 CV folds and analyzed which splits were stable across fold compositions.

**Result: macro F1 = 0.9276.** Full recovery, with additional insights:

- The top two splits (crystal symmetry orthorhombic, then hexagonal) appeared in 25/25 and 20/25 folds respectively - rock solid.
- Deeper splits (conductivity, fluorescence wavelength) were directionally consistent but threshold-unstable across folds.
- On the full-data fit, all 21 misclassified specimens came from the 25 geologist-disagreement set. Zero unanimous specimens were misclassified.

### Iteration 3: The Stop (3 min)

The Scientist evaluated the results against the investigation goal and decided to stop:

> *"Core question answered: interpretable classification rules for all 6 mineral types are identified and validated. Macro F1 = 0.9276 with all 6 classes above F1 = 0.87. 100% of errors are disagreement specimens - the noise floor is reached."*

The system generated a final report and shut down.

## What It Discovered

### The Final Rules

```
IF crystal_symmetry = orthorhombic:
    -> Fenrite

ELSE IF crystal_symmetry = hexagonal:
    IF conductivity > ~4.50:
        -> Aetherite
    ELSE:
        -> Borealis

ELSE (cubic):
    IF magnetic_susceptibility <= ~0.30:
        -> Erythian
    ELSE:
        IF fluorescence_wavelength <= ~450 nm:
            -> Cryolux
        ELSE:
            -> Dravite
```

Five features. Four decision levels. A physical interpretation that reads like a mineralogy textbook: crystal structure first, then ion mobility, then magnetism, then optical emission. The system noted this hierarchy "suggests that the six minerals differ fundamentally in their electronic structure at successive scales."

### Key Scientific Insights

**The color spectrum peak paradox.** The feature with the highest univariate separability (ANOVA F = 278.87) turned out to be completely useless. Within each crystal symmetry group, its discriminative power dropped below F = 3. It co-varies with crystal symmetry - specifically with Fenrite, which is 96.7% orthorhombic and has a distinctive orange-red peak at 555 nm. Once you condition on symmetry, it carries no additional information. The system caught this and proved it formally.

**Annotation noise is the binding constraint.** Every misclassification in the final model involves a specimen where geologists disagreed. The 475 unanimous specimens are classified perfectly. Improving the rules further is impossible without better labels for those 25 contested specimens. The system identified this ceiling and used it as a stopping criterion.

**Imposed structure fails where greedy optimization succeeds.** The v01 hierarchy assumed that crystal symmetry cleanly partitions classes and that summary-statistic thresholds would work within each partition. The greedy tree makes no such assumptions - it finds the split sequence that maximizes information gain at each step, discovering that magnetic susceptibility must come before fluorescence (to isolate Erythian first) in a way that the imposed hierarchy missed entirely.

## Framework Behavior

### What Worked

**Debate changed the outcome.** The Trajectory Critic's intervention at v02 - calling out the sunk-cost fallacy and pointing back to v00 - was the turning point. Without it, the Scientist was heading toward another iteration of patching the failed hierarchy. The debate didn't just catch errors; it redirected the scientific strategy.

**Information boundaries prevented confirmation bias.** The Analyst reported the v01 regression as pure observation: "Cryolux and Dravite collapsed. Here are the numbers." No spin, no suggestion to abandon the approach, no recommendation. The Scientist had to process the failure and decide what to do about it independently.

**Prediction tracking created accountability.** Every hypothesis came with pre-registered predictions, diagnostics, and branching plans. When three of four predictions were refuted in v00, the pre-specified "if refuted" plans kicked in automatically. This prevented the system from ignoring inconvenient results.

**The stopping decision was well-calibrated.** After reaching the annotation noise floor (0/475 unanimous errors), the system stopped instead of chasing marginal improvements. It explicitly listed open questions (threshold instability, the fluorescence LOO anomaly) as future work rather than trying to solve everything.

### What Could Be Better

**The hierarchy trap.** Both this run and a separate earlier run of the same problem hit the same failure mode: the Scientist sees near-perfect crystal symmetry partitioning and concludes a hand-crafted hierarchy must work, then wastes an iteration proving it doesn't. The Scientist prompt could benefit from a guard: "always validate the baseline model's structure before designing an alternative."

**Threshold instability at deeper levels.** The conductivity threshold varies from ~4.5 to ~6.0 across CV folds (CV = 24%), and the fluorescence threshold spans ~380-570 nm (CV = 10%). The rules are directionally right but the precise cutpoints are noisy. The system documented this honestly but didn't try to fix it (e.g., with ensemble thresholds or SIRUS-style rule stabilization). This would be natural next work.

## By the Numbers

| Metric | Value |
|--------|-------|
| Total wall time | 58 minutes |
| Iterations | 3 (+ ingestion + report) |
| Input tokens | 2.6M |
| Output tokens | 158K |
| Experiment scripts written | 3 |
| Testable predictions made | 14 |
| Predictions confirmed | 6 (43%) |
| Predictions refuted | 8 (57%) |
| Final macro F1 (5x5 CV) | 0.9276, 95% CI [0.916, 0.939] |
| Unanimous specimens misclassified | 0 / 475 |
| Features in final rules | 5 of 12 |

### Per-Iteration Breakdown

| Phase | Time | Input Tokens | Output Tokens |
|-------|------|-------------|---------------|
| Ingestion | 2.6 min | 295K | 9K |
| Iteration 0 (exploration) | 9.9 min | 550K | 26K |
| Iteration 1 (dead end) | 22.6 min | 879K | 59K |
| Iteration 2 (recovery) | 15.2 min | 696K | 44K |
| Iteration 3 (stop decision) | 2.8 min | 77K | 9K |
| Report generation | 4.6 min | 103K | 11K |

### Model Configuration

| Agent | Model | Provider |
|-------|-------|----------|
| Scientist | Claude Opus 4.6 | Anthropic |
| Analyst | Claude Sonnet 4.6 | Anthropic |
| Coder | Claude Sonnet 4.6 | Anthropic |
| Critic 1 | GPT-5.4 | OpenAI |
| Critic 2 | Claude Sonnet 4.6 | Anthropic |
| Summarizer | GPT-5.4-nano | OpenAI |

## Per-Class Results

| Class | Specimens | CV F1 | Std |
|-------|-----------|-------|-----|
| Aetherite | 90 | 0.957 | 0.026 |
| Borealis | 86 | 0.941 | 0.038 |
| Cryolux | 60 | 0.880 | 0.076 |
| Dravite | 58 | 0.903 | 0.065 |
| Erythian | 84 | 0.938 | 0.042 |
| Fenrite | 122 | 0.946 | 0.037 |

Cryolux has the lowest F1 and highest variance, reflecting the fluorescence threshold instability at the deepest level of the decision tree. Cryolux and Dravite together account for 11 of the 25 contested-label specimens, compounding the difficulty.

## Reproducing This Run

```bash
# Install
git clone https://github.com/thomast8/auto-scientist.git
cd auto-scientist
uv sync

# Set API keys
export ANTHROPIC_API_KEY="..."
export OPENAI_API_KEY="..."

# Run (takes ~1 hour with default preset)
auto-scientist run -c domains/alien_minerals/experiment.yaml
```

The investigation is non-deterministic (LLM sampling, CV fold randomness), so exact numbers will vary. The structural findings - crystal symmetry as root, the color spectrum peak paradox, the annotation noise ceiling - should reproduce consistently.

## What This Demonstrates

This isn't a cherry-picked success story. The system hit a dead end, wasted an iteration, and recovered. It made wrong predictions more often than right ones. The final result (F1 = 0.93) is good but not perfect, and the system is honest about why: annotation noise caps performance, and deeper thresholds are unstable.

What's interesting is the *process*. A multi-agent system with strict information boundaries, adversarial debate, and prediction tracking explored a moderately complex classification problem and arrived at a scientifically defensible answer in under an hour. It resolved puzzles (the color spectrum paradox), identified ceilings (the annotation noise floor), made a course correction when its theory failed (the hierarchy pivot), and stopped when continuing would produce diminishing returns.

The run captures what autonomous scientific investigation looks like when it works: not a straight line from question to answer, but a series of hypotheses, failures, critiques, and revisions that converge on something real.
