# Case Study: Alloy Composition-Property Relationships

**An autonomous investigation of what drives hardness in Fe-Cr-Ni-Mo-V alloys, completed in 80 minutes.**

An LLM-driven system was given 300 historical alloy records, 100 batch measurements from two labs (one with corrupt data), a literature review, and a goal: *discover the relationships between alloy composition and material properties, find optimal compositions*. No human selected features, chose models, or cleaned data. Eighty minutes later, the system delivered a validated Random Forest model (nested CV R² = 0.80, external validation R² = 0.70), characterized nonlinear element-hardness relationships via ALE plots, proved the corrosion resistance column was synthetic, and produced a Pareto frontier of cost-optimized compositions - all while navigating bad data, contradicting the literature, and correcting its own methodology three times.

---

## The Problem

An Fe-based alloy system with five compositional variables (Fe, Cr, Ni, Mo, V weight percentages) and three target properties (Vickers hardness, corrosion resistance, cost per kg). The data comes from three sources:

1. **Historical database** (300 records): composition, hardness (-2168 to +4044 HV, including 13 physically impossible negative values), corrosion resistance (-83 to +185, dimensionless), cost ($131-$573/kg), and sparse QC notes.
2. **Batch results** (100 records from 5 batches, 2 labs): Lab Alpha uses Vickers hardness and salt spray corrosion. Lab Beta uses Rockwell C hardness and electrochemical corrosion, with readings up to 362 HRC (physical maximum is ~70).
3. **Element spot prices**: Fe $1/kg, Cr $3.50, Ni $8, Mo $12, V $15.

Four complications:
1. **Corrupt data.** 13 negative hardness values, 14 negative corrosion values, and Lab Beta's Rockwell C readings are off by 5x.
2. **Incompatible measurement scales.** Two labs, two hardness methods, two corrosion methods, no validated conversion.
3. **A hidden trap.** The corrosion resistance column turns out not to be a physical measurement at all.
4. **Cost confusion.** The cost_per_kg column ranges $131-573/kg, but the weighted-average element spot price is only $1.3-5.7/kg (per weight fraction). The 100x gap is unexplained.

```
auto-scientist run -c domains/alloy_design/experiment.yaml
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

The Scientist never sees code. The Coder never sees the debate. The Analyst cannot recommend, only observe. These boundaries prevent confirmation bias: the entity that plans is not the one that evaluates, and the one that implements has no stake in the hypothesis.

For this run, the Scientist was Claude Opus 4.6. The Critics were GPT-5.4-mini and Claude Sonnet 4.6. Everything else ran on Sonnet 4.6.

## The Investigation

### Iteration 0: Exploration (10 min)

The system ingested a SQLite database, five batch JSON files, a YAML pricing table, and a literature review. It canonicalized everything into flat CSVs, immediately flagged Lab Beta as unusable (HRC values up to 362), and ran its first experiment: CLR-corrected correlations, anomaly characterization, interaction screening, and Lab Alpha compatibility testing.

**The first surprise: Chromium doesn't drive corrosion.**

Metallurgy textbooks say chromium is *the* primary driver of corrosion resistance in iron alloys via passive film formation. The CLR-corrected Spearman correlation between Cr and corrosion_resistance was -0.023. Essentially zero. Nickel dominated instead (r_s = +0.613).

This finding survived sensitivity analysis across three pseudocount values, with and without anomalous rows. It was stable and deeply puzzling.

**Other findings:**
- The 13 negative hardness values clustered at extreme compositions: 11/13 had (Ni > 10% or V > 5%) and Fe < 65%, with Cohen d = 3.40 for Ni. Extrapolation artifacts, safe to exclude.
- Cost_per_kg was NOT deterministic from element prices (mean residual $242/kg). It encodes processing cost.
- Neither Cr x Mo nor Cr x Ni interactions were significant (delta-R² = 0.0007 for Cr x Mo).

Before any code ran, two critics had already improved the plan:

> **Methodologist (GPT-5.4-mini):** "CLR-transformed pairwise correlations are unstable in singular, subcompositionally incoherent space. Fixed zero-replacement is a bias source."

> **Falsification Expert (Sonnet 4.6):** "The 10% Cr split likely mismatches the ~12% passive-film threshold. And if corrosion_resistance is formula-based, the whole Cr investigation is moot."

The Falsification Expert's suspicion about synthetic corrosion would prove prescient one iteration later.

### Iteration 1: The Corrosion Revelation (16 min)

The Scientist hypothesized that Cr's near-zero correlation hid a threshold effect at ~11% Cr, the known passive-film boundary. If the dataset spans both sides, a linear correlation would average across two regimes and appear near zero.

The critics pushed hard before any code ran:

> **Trajectory Critic (GPT-5.4-mini):** "You're circling back to a Cr explanation despite a near-zero signal and a failed Cr x Mo interaction. This is sunk-cost reasoning. Do a one-shot falsification and pivot if it fails."

> **Methodologist (Sonnet 4.6):** "Don't pool Lab Alpha corrosion with historical data. Salt spray and an unknown method have no validated scale alignment."

Both critiques landed. The Scientist accepted a one-shot Cr threshold test and restricted Lab Alpha pooling to hardness only.

**Results:**

The Cr threshold hypothesis was definitively killed: permutation p = 0.954, delta-R² = 0.000. There is no threshold.

But the real bombshell was the corrosion provenance test. An OLS regression of corrosion_resistance on the five raw weight percentages achieved R² = 0.84, with suggestively simple coefficients (Ni: +2.76, V: -5.42, Mo: -4.35). **Corrosion resistance is a synthetic score computed from composition, not an independent physical measurement.** The Cr paradox dissolves: Cr has a small coefficient (+0.70) in the formula, not because it doesn't affect real corrosion, but because whoever designed the scoring formula didn't weight it heavily.

This finding collapsed half the investigation. Modeling corrosion became pointless. Optimizing the "hardness-corrosion tradeoff" became optimizing hardness against a formula.

Meanwhile, hardness modeling was struggling. CLR main-effects OLS achieved CV R² = 0.297 with std = 0.50 (failing even the modest 0.30 threshold), and LASSO with quadratic terms improved to R² = 0.46 but with std = 0.47. Five quadratic terms survived regularization, confirming genuine nonlinearity, but the model was unstable.

### Iteration 2: Diagnosing Instability (24 min)

Three iterations of transform and regularization fixes (CLR, LASSO, ILR + Elastic Net) had treated collinearity as the root cause of hardness model instability. The debate exposed a different diagnosis:

> **Critics (post-debate revision):** "LASSO reduced CV std by only 6%. ILR and CLR produce equivalent fitted values. The real culprit is likely alloy-family heterogeneity across the broad composition space, not collinearity."

The Scientist abandoned the linear modeling track entirely and pivoted to Random Forest with nested cross-validation, using K-means clustering to test the heterogeneity hypothesis.

**Results:**

Clustering revealed two populations - a large conventional-alloy cluster (n=234, mean Fe=81%) with excellent in-cluster R² = 0.90, and a small highly-alloyed cluster (n=53, mean Fe=64%) with R² = 0.54. But within-cluster instability (std = 0.55 for the small cluster) exceeded the threshold, refuting the simple heterogeneity explanation. The instability comes from unmeasured process variables in extreme compositions, not from distinct alloy families.

The Random Forest achieved nested CV R² = 0.80, versus Elastic Net at R² = 0.47 - a gap of 0.33 confirming substantial nonlinearity beyond quadratic terms. Mo dominated with 56% permutation importance, Cr at 21%, Fe at 20%. Ni and V contributed less than 3% combined.

### Iteration 3: Characterizing Functional Forms (20 min)

The debate forced one more critical methodological fix:

> **Critics:** "PDPs on closed compositional data generate impossible compositions by marginalizing out correlated elements. Use ALE plots instead, which respect local data distribution."

The Scientist also elevated external validation (Lab Alpha) to a gate: no optimization recommendations without confirming the model generalizes.

**Results:**

The Mo ALE profile revealed a strongly nonlinear positive relationship:
- **0-6% Mo**: steep increase at 101 HV per percent
- **6-11% Mo**: transition zone
- **12%+ Mo**: saturation at 7 HV per percent (but only 19 observations here)

Cr showed a non-monotonic profile peaking at 49 HV/% in the 8-13% range. Fe was uniformly negative, steepest at low Fe (-42 HV/% below 66%). The 2D Mo x Cr ALE confirmed the interaction is negligible (range: -24 to +25 HV, versus Mo's main-effect span of 1,556 HV).

External validation on 54 Lab Alpha samples passed: R² = 0.70, MAE = 87 HV, mean residual = +42 HV. The model generalizes.

The optimization scan over 8,000 feasible Fe > 70% compositions produced a Pareto frontier. The highest-hardness composition: Fe 70.5%, Cr 15.5%, Mo 12.8%, V 0.9%, Ni 0.3%, with predicted hardness 2,619 HV (90% PI: 2,315-2,947 HV).

### Iteration 4: The Stop (3 min)

The Scientist evaluated the evidence and stopped:

> *"Core question answered: functional forms identified, model validated externally, optimization frontier produced. No further iterations required."*

## What It Discovered

### Key Scientific Findings

**1. Molybdenum is the primary hardness driver, with diminishing returns above 12%.** Each percent of Mo below 6% adds ~101 HV. Above 12%, only ~7 HV. This saturation means pushing Mo above 13% costs $12/kg per percent for minimal benefit - a critical insight for alloy design.

**2. Chromium does not drive corrosion resistance in this dataset - because corrosion resistance isn't real.** The corrosion_resistance column is a synthetic score (R² = 0.84 from raw compositions, near-integer coefficients). The Cr paradox that motivated an entire iteration of investigation dissolves once you realize the variable was computed, not measured.

**3. Cr's hardness contribution is real but non-monotonic.** The ALE profile peaks at 49 HV/% in the 8-13% range, suggesting carbide precipitation or solid-solution strengthening that saturates at high Cr.

**4. No element interactions matter.** Despite repeated literature citations of synergistic Cr+Mo effects, the 2D ALE spans only +/-25 HV across the joint space, compared to Mo's 1,556 HV main-effect range.

**5. High hardness and low cost are mutually exclusive.** High hardness requires Mo (expensive at $12/kg) and Cr (moderate at $3.50/kg); cheap alloys are high-Fe with low alloying. The optimization confirmed no composition achieves both high hardness and minimal alloying cost simultaneously.

**6. The cost_per_kg variable encodes processing, not materials.** True material cost from spot prices is only a few dollars per kg; cost_per_kg ranges $131-573/kg. The ~100x gap represents processing specification that can't be decomposed without metadata.

### Optimal Compositions (Fe > 70%, maximize hardness/cost ratio)

| Fe% | Cr% | Mo% | Predicted HV | 90% PI | Cost index |
|-----|-----|-----|-------------|--------|------------|
| 72.2 | 15.4 | 11.1 | 2,467 | 2,164-2,795 | 274 |
| 70.5 | 15.5 | 12.8 | 2,619 | 2,315-2,947 | 294 |
| 73.7 | 13.8 | 11.7 | 2,380 | 2,077-2,708 | 269 |

Cost index is `sum(weight_percent * element_price)`, a relative ranking metric. The true material cost per kg is ~100x lower (e.g., $2.94/kg for the top composition). All top compositions share the same recipe: maximize Mo (11-13%), keep Cr in the 13-16% sweet spot, minimize Ni and V.

## Framework Behavior

### What Worked

**Debate caught methodology errors three times.** (1) Critics forced the switch from raw correlations to CLR transforms for compositional data. (2) The Trajectory Critic called out sunk-cost reasoning on the Cr threshold hypothesis, limiting it to a one-shot falsification. (3) Critics caught that PDPs on compositional data generate impossible compositions, forcing the switch to ALE plots. Each fix changed the conclusions.

**The synthetic corrosion discovery.** The Falsification Expert raised suspicion in iteration 0. The provenance test in iteration 1 confirmed it. This finding collapsed half the investigation - but that's good. The system stopped trying to model a formula and redirected all effort to hardness, where the real science was.

**The instability diagnosis pivoted from collinearity to heterogeneity to noise.** Three iterations of increasingly sophisticated linear models (CLR, LASSO, ILR + Elastic Net) each failed to stabilize hardness predictions. Instead of trying a fourth transform, the system diagnosed the actual problem: unmeasured process variables in extreme compositions create irreducible noise. The Random Forest captured the nonlinear signal that linear models couldn't, and the system documented the noise ceiling honestly.

**Prediction tracking kept the investigation honest.** 17 testable predictions across 4 iterations, with 47% confirmed and 47% refuted. Every refutation redirected the investigation: the Cr threshold refutation killed the passive-film hypothesis, the clustering refutation shifted from heterogeneity to noise, the Pareto refutation proved hardness and corrosion can't be jointly optimized.

### What Could Be Better

**Two iterations focused on linear approaches before the RF pivot.** The v01 LASSO result (5 quadratic terms surviving, +0.17 R² improvement) was already strong evidence for nonlinearity, but the system spent v01 on linear fixes and didn't try Random Forest until v02. The RF ran alongside Elastic Net as a comparison in v02, but could have been tried one iteration sooner.

**The high-alloy regime (Fe < 70%) remains poorly understood.** Residual std of 392 HV versus 78 HV for conventional alloys. The system correctly identified this as a limitation and restricted optimization to Fe > 70%, but didn't attempt to collect or request process metadata that might explain the variance.

## By the Numbers

| Metric | Value |
|--------|-------|
| Total wall time | 80 minutes |
| Iterations | 4 productive + stop decision (+ ingestion + report) |
| Input tokens | 3.9M |
| Output tokens | 226K |
| Experiment scripts written | 4 |
| Testable predictions made | 17 |
| Predictions confirmed | 8 (47%) |
| Predictions refuted | 8 (47%) |
| Predictions inconclusive | 1 (6%) |
| Final nested CV R² | 0.7957 |
| External validation R² | 0.7015 |
| Mo permutation importance | 56% |

### Per-Iteration Breakdown

| Phase | Time | Input Tokens | Output Tokens |
|-------|------|-------------|---------------|
| Ingestion | 2.6 min | 273K | 9K |
| Iteration 0 (exploration) | 9.7 min | 464K | 34K |
| Iteration 1 (corrosion reveal) | 16.0 min | 806K | 49K |
| Iteration 2 (instability diagnosis) | 24.0 min | 959K | 54K |
| Iteration 3 (ALE + validation) | 19.7 min | 1,140K | 57K |
| Iteration 4 (stop decision) | 2.6 min | 88K | 8K |
| Report generation | 4.9 min | 203K | 15K |

### Model Configuration

| Agent | Model | Provider | Reasoning |
|-------|-------|----------|-----------|
| Scientist | Claude Opus 4.6 | Anthropic | medium |
| Analyst | Claude Sonnet 4.6 | Anthropic | medium |
| Coder | Claude Sonnet 4.6 | Anthropic | medium |
| Critic 1 | GPT-5.4-mini | OpenAI | medium |
| Critic 2 | Claude Sonnet 4.6 | Anthropic | medium |
| Summarizer | GPT-5.4-nano | OpenAI | off |

## Reproducing This Run

```bash
# Install
git clone https://github.com/thomast8/auto-scientist.git
cd auto-scientist
uv sync

# Set API keys
export ANTHROPIC_API_KEY="..."
export OPENAI_API_KEY="..."

# Run (takes ~80 minutes with default preset)
auto-scientist run -c domains/alloy_design/experiment.yaml
```

The investigation is non-deterministic (LLM sampling, CV fold randomness), so exact numbers will vary. The structural findings - Mo as dominant nonlinear driver, synthetic corrosion, no element interactions, high-alloy instability - should reproduce consistently.

## What This Demonstrates

This run shows the framework handling a problem with multiple traps: corrupt data that needs exclusion (Lab Beta), a target variable that isn't what it claims to be (corrosion resistance), a literature-motivated hypothesis that the data doesn't support (Cr threshold), and a model class that takes three iterations to get right (linear to nonlinear).

The system fell into the Cr threshold trap, spent an iteration investigating it, and was pulled out by its own critics. It discovered the corrosion column was synthetic before anyone told it. It correctly diagnosed that hardness model instability came from noise in extreme compositions rather than collinearity or heterogeneity, after systematically ruling out both alternatives.

The critics were particularly effective here. They caught compositional data handling errors (CLR transforms, ALE over PDP), prevented methodological shortcuts (pooling incompatible corrosion measurements, reusing test sets for model selection), and redirected the scientific strategy when the Scientist was circling (sunk-cost Cr investigation). The resulting analysis is more rigorous than it would have been without adversarial review.

The final model isn't perfect - R² = 0.80 with a 5x uncertainty gap for extreme compositions - and the system says so explicitly. That honesty about limitations, combined with the documented chain of hypotheses, predictions, refutations, and course corrections, is what makes the output trustworthy.
