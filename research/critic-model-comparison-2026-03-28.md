# Critic Model Comparison: Gemini vs GPT vs Anthropic

**Date:** 2026-03-28
**Branch:** `alien_minerals-fixes`
**Purpose:** Determine whether Gemini 3.1-pro-preview is pulling its weight as a critic model compared to GPT-5.4, and evaluate Anthropic Sonnet 4.6 as a potential replacement.

## Background

The auto-scientist debate phase runs 3 critic personas (Methodologist, Novelty Skeptic, Feasibility Assessor) in parallel. Model assignment rotates across iterations. The default config uses 2 Gemini + 1 GPT, meaning Gemini handles 2/3 of all critic slots.

During the `alien_minerals_first` runs, we noticed Gemini's output was dramatically shorter than GPT's. This investigation quantifies the gap and tests whether it's a quality issue or just a verbosity difference.

## Part 1: Historical Analysis (33 debate entries across 3 runs)

Analyzed all debate.json files from:
- `alien_minerals_first` (v01-v04)
- `alien_minerals_first_001` (v01-v04)
- `alien_minerals_first_002` (v01-v03)

**Scripts:** `experiments/analyze_debates.py`, `experiments/sample_debates.py`

### Aggregate Stats

| Metric | Gemini (n=16) | GPT (n=17) | Ratio |
|---|---|---|---|
| Avg output tokens | 406 | 6,337 | 0.06x |
| Avg # concerns | 2.8 | 8.2 | 0.34x |
| Avg # high-severity | 0.8 | 4.0 | 0.20x |
| Avg claim length (chars) | 128 | 491 | 0.26x |
| Avg # alt hypotheses | 2.0 | 4.2 | 0.48x |
| Avg alt hyp length (chars) | 213 | 292 | 0.73x |
| Avg assessment length (chars) | 510 | 730 | 0.70x |
| Avg input tokens | 10,689 | 47,032 | 0.23x |

### Qualitative Findings from Sampling

- Gemini's `claim` fields often **restate plan steps** rather than explaining why they're problematic. Example: "Re-derive the density threshold on the correctly-filtered CD subset (n=84) via grid search maximizing balanced accuracy" (this is the plan's own language, not a critique).
- GPT provides **multi-sentence analytical claims** with reasoning and literature citations.
- Gemini **never cites literature** despite the prompt instructing web search usage.
- Gemini's **assessments are the strongest part** of its output, concise but on-point.
- Gemini consistently **under-flags severity** (0.8 high per debate vs GPT's 4.0).

## Part 2: Controlled Head-to-Head (same prompt, 3 models, 3 personas)

Replayed the exact critic prompt from `alien_minerals_first_002/v02` through all 3 models with all 3 personas (9 total API calls). All models used `reasoning: high` and `web_search: true`.

**Scripts:** `experiments/compare_models.py` (Gemini + GPT), `experiments/compare_anthropic.py` (Anthropic)

### Summary Table

| Persona | Model | Time | Out tokens | #Concerns | #High | Avg claim (chars) | #Alt hyps | Assessment (chars) |
|---|---|---|---|---|---|---|---|---|
| Methodologist | Gemini 3.1-pro-preview | 45s | 499 | 4 | 2 | 124 | 4 | 622 |
| Methodologist | Sonnet 4.6 | 78s | 2,893 | 10 | 3 | 532 | 5 | 1,193 |
| Methodologist | GPT-5.4 | 171s | 9,621 | 9 | 5 | 385 | 5 | 709 |
| Novelty Skeptic | Gemini 3.1-pro-preview | 49s | 476 | 4 | 2 | 106 | 3 | 576 |
| Novelty Skeptic | Sonnet 4.6 | 83s | 3,421 | 10 | 5 | 564 | 5 | 1,916 |
| Novelty Skeptic | GPT-5.4 | 179s | 9,126 | 7 | 4 | 563 | 4 | 817 |
| Feasibility Assessor | Gemini 3.1-pro-preview | 21s | 482 | 3 | 1 | 114 | 2 | 663 |
| Feasibility Assessor | Sonnet 4.6 | 84s | 3,341 | 9 | 3 | 637 | 5 | 1,404 |
| Feasibility Assessor | GPT-5.4 | 193s | 10,299 | 7 | 3 | 587 | 4 | 727 |

### Qualitative Assessment per Model

**Gemini 3.1-pro-preview** - Weakest by a clear margin
- 3-4x fewer concerns, 4-5x shorter claims
- Catches real issues (one-hot encoding depth waste, topology stability) but misses most of what the others find
- Zero web search / literature citations despite prompt instruction
- Fast (21-49s) but depth is sacrificed
- Roughly 30-40% of GPT's value per debate slot

**GPT-5.4** - Strong all-rounder
- Best at web search/citations (CORELS, OCT, Dawid-Skene papers consistently referenced)
- Good concern detail and breadth (7-9 concerns)
- Most output tokens, but much is citation overhead
- 3-4x slower than Anthropic (171-193s)

**Anthropic Sonnet 4.6** - Surprisingly strong, best efficiency
- Matches or exceeds GPT on concern count (9-10 concerns)
- Longest and most specific claims (532-637 chars avg)
- Catches unique angles: fluorescence correction arithmetic inconsistency (12nm expected vs 2.4nm observed), geographic confounding, collector bias, RIPPER recommendation, "are Cryolux/Dravite actually the same mineral type?" hypothesis
- Most detailed assessments (1,193-1,916 chars vs GPT's 709-817)
- Runs in ~80s vs GPT's ~180s at 1/3 the output tokens (more efficient)
- No literature citations (doesn't use web search), compensates with deeper evidence-base analysis
- No evidence of being "soft" on the scientist despite Claude being the scientist model

## Part 0: Response Schema Investigation

Before the head-to-head, we tested whether Gemini's `response_schema` API constraint (JSON mode) was causing the terse output. Sent the same prompt with and without `response_schema`.

**Script:** `experiments/test_gemini_schema.py`

| Config | Out tokens | Chars |
|---|---|---|
| With `response_schema` | 466 | 2,343 |
| Without `response_schema` | 550 | 2,587 |

**Conclusion:** `response_schema` is not the bottleneck. Gemini just produces concise output regardless. Both versions parsed as valid JSON.

## Conclusions

1. Gemini 3.1-pro-preview provides ~30-40% of GPT-5.4's critic value per debate slot. With the default 2-Gemini/1-GPT rotation, the debate phase is getting less total critique coverage than it could.
2. Anthropic Sonnet 4.6 is a strong candidate replacement: matches GPT on breadth, exceeds it on depth-per-concern and assessment detail, runs 2x faster, uses 3x fewer output tokens.
3. The `response_schema` API constraint is not the cause of Gemini's terseness.
4. The concern about Anthropic being "too aligned" with the scientist (also Claude) is not supported by the data; Sonnet's critiques are the most aggressive of the three.

## Reproduction

All scripts are in `experiments/`:
- `analyze_debates.py` - aggregate stats across all historical debate.json files
- `sample_debates.py` - print qualitative samples side by side
- `compare_models.py` - controlled head-to-head (Gemini + GPT, parallel)
- `compare_anthropic.py` - Anthropic-only run (same prompts)
- `test_gemini_schema.py` - response_schema A/B test

Source data: `experiments/runs/alien_minerals_first*/v*/debate.json`
