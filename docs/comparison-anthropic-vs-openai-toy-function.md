# Backend Comparison: Anthropic vs OpenAI on Toy Function Discovery

**Same problem, same prompts, same critics - very different outcomes.**

Two runs of the toy function domain using equivalent preset levels but different LLM backends. The Anthropic run used Claude Opus 4.6 as scientist and Claude Sonnet 4.6 for other agents. The OpenAI run used GPT-5.4 as scientist and GPT-5.4-mini for other agents. Both runs used the same critic panel (GPT-5.4-mini + Claude Sonnet 4.6) and identical agent prompts.

The ground truth (unknown to the system): `y = 0.3x² + 2.5·sin(1.5x) + noise`.

---

## Results at a Glance

| | Anthropic (run 035_001) | OpenAI (run 047_002_004) |
|---|---|---|
| **Scientist** | claude-opus-4-6 | gpt-5.4 |
| **Analyst / Coder** | claude-sonnet-4-6 | gpt-5.4-mini |
| **Iterations** | 5 | 12 |
| **Stop proposals** | 1 (maintained) | 5 (withdrawn 4 times, maintained on 5th) |
| **Final RMSE** | **0.440** | 1.190 |
| **Final R-squared** | **0.977** | not computed |
| **Residuals** | White noise (Ljung-Box p=0.277) | Massive structure (lag-1 ACF = 0.866) |
| **Discovered formula** | y = 0.3x-squared + 2.5 sin(1.5x) | y ~ -0.06 - 0.07x + 0.31x-squared + 2.46 sin((pi/2)x) |
| **Claimed confidence** | Exact recovery with zero free parameters | "Approximation class, not a uniquely recovered generator" |
| **Predictions** | 16 (7 confirmed, 8 refuted, 1 inconclusive) | 51 (19 confirmed, 11 refuted, 8 inconclusive, 13 pending) |
| **Coder retries** | 0 | 0 |

The Anthropic run found the exact generating function with correct coefficients, white-noise residuals, and R-squared = 0.977. The OpenAI run settled for an approximate formula with the wrong frequency, 3x worse RMSE, and massive residual structure, then concluded the problem was unsolvable.

---

## The Divergence Point: Iteration 2

Both runs had identified a quadratic-plus-sine structure by iteration 1. The critical divergence happened at iteration 2.

### Anthropic scientist (opus-4-6): went for the coefficients

The Anthropic scientist spotted a pattern in the fitted amplitude:

> "The fitted amplitude A=2.5466 matches 8/pi = 2.54648 to 4 decimal places. This is almost certainly the true amplitude."

It proposed a specific, numerical plan:
- Run profile likelihood over omega in [1.0, 2.0] at 201 grid points to discriminate omega = 3/2 from omega = 5/4
- Fix omega = 3/2 and fit a and A by ordinary least squares (now a linear problem)
- Test specific constant candidates (a = 0.3, A = 2.5 vs 8/pi) with bootstrap CIs

This produced the decisive result: omega = 1.500 with a gap of 0.906 RMSE over the nearest competitor, the quadratic coefficient a = 0.3 inside a tight bootstrap CI, and white-noise residuals confirming exact recovery.

### OpenAI scientist (gpt-5.4): declared the problem unsolvable

The OpenAI scientist proposed stopping:

> "Investigation complete: on the observed interval and noise level, the dataset does not support unique recovery of a single generating formula."

Empty changes list. Zero testable predictions. It concluded "non-identifiability" after only two iterations, before ever attempting frequency profiling or coefficient identification.

The stop debate (same critics) forced a withdrawal. But the damage was done: the scientist's epistemic framework was now "non-identifiability" rather than "let me find the exact coefficients." Every subsequent plan was filtered through that lens.

---

## How the Wrong Framing Cascaded

The OpenAI scientist's premature non-identifiability conclusion created a self-reinforcing trap:

1. **Wrong validation approach.** Treated blocked cross-validation as the sole decision criterion. Blocked CV is good for model *selection* but too noisy to discriminate between omega = 1.5 and omega = pi/2 at n=200.

2. **Wrong frequency.** Settled on omega = pi/2 (approximately 1.571) without testing it against the true 1.5. Never performed profile likelihood. The 5% frequency error is small but fatal for residual diagnostics.

3. **Unnecessary model complexity.** Kept fitting a 5-parameter model (constant + linear + quadratic + sine + cosine). Never simplified down to the 3-parameter ground truth. The extra constant (-0.062) and linear term (-0.067x) are fitting noise.

4. **Large residuals confirmed the wrong belief.** RMSE of 1.19 (vs 0.44 for the correct model) produced strong residual autocorrelation. The scientist interpreted this as evidence that "exact recovery isn't possible" rather than "I have the wrong frequency."

5. **Seven wasted iterations.** Iterations 5 through 11 circled the same territory: propose to stop, critics object, grudgingly continue with abstract methodology, get mediocre results, propose to stop again.

---

## Debate Dynamics: Same Critics, Different Outcomes

A natural question: both runs used the same critic models (GPT-5.4-mini + Claude Sonnet 4.6). Why did the debates play out so differently?

The answer is that the scientist determines the quality of the debate, not the critics. Critics react to what they're given.

### Anthropic debates: constructive

The Anthropic scientist brought specific, testable plans. Critics sharpened them:

- v00: Critics caught missing multiplicative candidates (growing amplitude envelope). Scientist added them. They were tested and rejected, closing a real alternative.
- v01: Critics caught the phase-omission bug in the grid search and demanded windowed FFT for spectral leakage. Both were incorporated.
- v02: Critics flagged the FFT bin-resolution artifact (omega = 1.25 was an alias, not the true frequency). This led directly to the profile likelihood approach.
- v03: Critics identified a tautological RMSE argument for A = 8/pi over A = 2.5 (delta was 17x smaller than the SE). Scientist revised to prefer the simpler rational under parsimony.
- v04: Critics unanimously voted to stop: the answer was in hand, and additional validation was redundant.

Each debate made the next iteration materially better. The critics were doing their job, and the scientist was listening.

### OpenAI debates: frustrated

The OpenAI scientist brought premature stop proposals and abstract methodology. Critics had no choice but to object:

- v02: Scientist proposed stopping. Critics identified 5 unresolved gaps. Scientist withdrew.
- v05: Scientist proposed stopping again. Critics forced withdrawal with 4 more gaps.
- v06: Critics pointed out the "canonical oscillatory base" was actually *worse* than the quadratic baseline on the scientist's own blocked-holdout criterion.
- v09: Scientist proposed stopping a third time. Critics found protocol inconsistencies, missing uncertainty analysis, and untested structural alternatives.
- v10, v11: Two more stop proposals, two more withdrawals.

The critics did exactly their job in both runs. But when given a weak plan, critics can only demand "try harder," which produces another round of vague methodology rather than a focused numerical test. The debate amplified the scientist's weakness instead of compensating for it.

---

## What This Tells Us

### It's a model issue, not a prompt issue

Both runs used identical prompts. The difference is in how the models *interpreted* those prompts:

**Pattern recognition.** Opus spotted A = 2.5466 approximately equals 8/pi to 4 decimals and immediately exploited it. GPT-5.4 never made that connection across 12 iterations.

**Tool selection.** Profile likelihood over omega is a standard technique for frequency identification. Opus reached for it at the right moment. GPT-5.4 defaulted to blocked cross-validation for everything, a more conservative choice that works for model selection but not coefficient identification.

**Epistemic calibration.** Opus committed to specific numerical hypotheses and tested them. GPT-5.4 defaulted to "this might not be solvable" at iteration 2 and never fully recovered from that framing.

### The debate structure amplifies scientist quality

A confident scientist with a good plan creates a positive feedback loop: critics refine the plan, the next iteration improves, critics refine again.

A tentative scientist proposing to stop creates a negative feedback loop: critics force continuation, the scientist produces a vague plan, results are mediocre, the scientist proposes to stop again.

This means the scientist role is disproportionately important to overall investigation quality. A weaker scientist isn't just proportionally worse; it gets trapped in debate cycles that waste iterations.

### Caveats

This is a single-problem comparison. GPT-5.4 might perform differently on problems where its more conservative, methodology-heavy approach is appropriate (genuinely non-identifiable systems, problems requiring careful validation design rather than coefficient hunting). It might also improve with prompt modifications that discourage early stopping or require coefficient identification before allowing non-identifiability claims.

But on a clean function-discovery task with a definite answer, the Anthropic backend with opus-4-6 as scientist was unambiguously the stronger configuration.

---

## Run Details

### Anthropic (035_001)

- **Model config:** Scientist = claude-opus-4-6; Analyst/Coder/Ingestor/Report = claude-sonnet-4-6; Critics = gpt-5.4-mini + claude-sonnet-4-6; Summarizer = gpt-5.4-mini
- **Iterations:** 5 (v00-v04 experiments, v05 stop + report)
- **Key breakthroughs:** v00 (model family selection, additive over multiplicative), v02 (profile likelihood resolved omega = 3/2), v03 (amplitude disambiguation, parsimony selects A = 2.5)
- **Final answer:** y = 0.3x-squared + 2.5 sin(1.5x) with noise sigma approximately 0.44

### OpenAI (047_002_004)

- **Model config:** Scientist = gpt-5.4; Analyst/Coder/Ingestor/Report = gpt-5.4-mini; Critics = gpt-5.4-mini + claude-sonnet-4-6; Summarizer = gpt-5.4-nano
- **Iterations:** 12 (v00-v11 experiments, v12 stop + report)
- **Stop proposals at:** iterations 2, 5, 9, 10, 11 (maintained at 12)
- **Final answer:** y approximately equals -0.062 - 0.067x + 0.306x-squared + 2.457 sin((pi/2)x), reported as "best compact approximation, not a uniquely identified generating function"
