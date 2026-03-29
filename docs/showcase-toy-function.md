# Case Study: Discovering a Hidden Function

**From 200 noisy data points to the exact generating formula in 39 minutes.**

A system was handed a CSV with two columns - x and y - and told: "discover the mathematical function that generated this dataset." No hints about the function family, no guidance on methodology, no human in the loop. Two iterations later, it returned the exact formula with all three constants correct to within 1.5%, proved no simpler or more complex form fit better, and stopped.

The answer: **y = 0.3x² + 2.5·sin(1.5x)**

This was correct. The ground truth, unknown to the system, was `y = 2.5*sin(1.5*x) + 0.3*x²` with Gaussian noise (sigma = 0.5).

---

## The Problem

200 data points sampled uniformly on x in [-5, 5]. The y values range from about -2.7 to 9.8, with visible curvature, a suspicious flat region near x = -3.2, and noise throughout. The system knows nothing about the generating process. It must figure out the functional form, the constants, and the noise characteristics, then justify why its answer is right and alternatives are wrong.

```
auto-scientist run -c domains/toy_function/experiment.yaml
```

This is the "easy" built-in domain - a warmup problem that tests whether the framework can do basic scientific reasoning: form hypotheses, fit models, compare them rigorously, and know when to stop.

## The Investigation

### Ingestion (1 min)

The Ingestor loaded `toy_function.csv`, confirmed it was clean (200 rows, no missing values, no duplicates, float64 throughout), and produced a canonical copy. Nothing exciting here, but the system correctly noted the evenly-spaced x values (linspace pattern), the negative-y region near x = -1 to 0, and the global minimum near x = -0.78.

### Iteration 0: What Kind of Function Is This? (9 min)

The Scientist designed a broad screening: fit polynomials of degree 1-5, fit two trigonometric composite families (x² + cos and x + sin), estimate the noise floor with LOESS, and compute numerical derivatives as diagnostics.

Before any code ran, two critics challenged the plan:

> **Methodologist (GPT-5.4):** "This mixes candidate search and evaluation on the same 200 rows. R²/AIC/BIC on the fitting data are not decisive for true form. Add a validation layer."
>
> **Falsification Expert:** "The 'secondary plateau' could be noise. The Savitzky-Golay derivative settings aren't validated. A scaled x² + cos(x) could mimic the flattening."

The Scientist accepted both: added an 80/20 train/test split, adopted multi-start nonlinear fitting, and downgraded derivatives from "structural discriminator" to "sensitivity diagnostic."

**Results:**

The screening was decisive. Polynomials topped out at degree-5 with test R² = 0.813 and BIC = 91.4. The x² + cos composite hit test R² = 0.959 and BIC = -257.3. That's a BIC gap of 349 - not close. The LOESS noise floor estimated sigma = 0.44 and a theoretical R² ceiling of 0.978, giving the system a target to aim for.

A clue emerged from the cos fit's phase parameter: d = -11.002, which is approximately -3.5pi. That means the cosine was acting as a phase-shifted sine. The fitted constants (a = 0.302, b = 2.537, c = 1.506) looked suspiciously close to round numbers.

**Predictions scorecard:** 1 confirmed, 2 confirmed, 1 refuted. The refutation was informative - polynomials were predicted to reach R² > 0.85 but only hit 0.61, confirming the function is fundamentally non-polynomial.

### Iteration 1: Nailing the Exact Formula (23 min)

The Scientist hypothesized the generating function was y = 0.3x² + 2.5·sin(1.5x) and designed a confirmation strategy: fit the sin form directly (eliminating the phase ambiguity), compare against extended models (second harmonic, linear term), and run PySR symbolic regression as an independent cross-check.

The debate phase was particularly sharp here. Five critics (the largest panel in the run) found real issues:

> **Methodologist:** "The LOESS noise ceiling is treated as authoritative, but it's just an estimate. The 0.019 R² gap could be sampling error on n=40."
>
> **Trajectory Critic (GPT-5.4):** "The plan has shifted from 'discover the exact function' to 'close the gap to the noise ceiling.' That's goal drift. Add symbolic regression for exact recovery."
>
> **Evidence Auditor (GPT-5.4):** "The 'second harmonic' comparison has a frequency-unit mismatch. The proposed 'top 5 by test R²' selection reuses the test set."

The Scientist incorporated three key fixes: training-BIC-only model selection (no more test-set leakage), Fisher's g-test for proper residual periodogram significance, and PySR as independent triangulation.

**Results:**

The sin model with 100 multi-start fits converged to a = 0.3029, b = 2.5367, c = 1.5064, d = -0.067. All within 1.5% of the round values (0.3, 2.5, 1.5, 0.0).

Then the clean-constant model (fixing a = 0.3, b = 2.5, c = 1.5, d = 0.0 exactly) achieved BIC = -278.4, beating the free-parameter fit (BIC = -262.3) by 16.1 units. Under BIC, this is decisive: the data actively prefer the simpler model with no free parameters. The extra degrees of freedom in the free fit add noise without explanatory power.

Neither extension was justified: the second harmonic improved BIC by only 4.2, and the linear term by only 4.8, both well below the ΔBIC > 10 threshold.

Residual diagnostics confirmed completeness:
- Fisher's g-test: p = 0.483 (no remaining periodicity)
- Runs test: p = 0.158 (no autocorrelation)
- Training RMSE: 0.419, consistent with the LOESS noise estimate of 0.443

**The one surprise:** PySR failed. Its best expression - `square(0.383 - ((x * -0.592) - cos(x))) - 0.911` - was qualitatively different and 347 BIC units worse. The system noted this honestly: "the combination of a quadratic and a pure sinusoidal lies in a region of expression space that SR's Pareto-front traversal does not reach efficiently." An interesting finding about SR's limitations, not a challenge to the analytical result.

### Iteration 2: The Stop (2 min)

The Scientist evaluated the evidence and stopped:

> *"The core question is answered with strong, converging evidence across multiple validation methods. Exact constants identified. Clean-constant model is superior. No residual structure. Extensions ruled out. Near noise ceiling."*

## What It Got Right

The system recovered all three constants of the generating function (0.3, 2.5, 1.5) to within 1.5% accuracy, proved the intercept is zero, and estimated the noise at sigma = 0.44 versus the true sigma = 0.5. It did this without knowing the function family, the number of terms, or the noise level in advance.

More importantly, it proved the answer was right through multiple independent lines of evidence:

1. **BIC model selection**: clean-constant model preferred over free-parameter fit by ΔBIC = 16.1
2. **Parameter confidence intervals**: all 95% CIs contain the clean values
3. **Residual whiteness**: Fisher's g-test (p = 0.48) and runs test (p = 0.16) show no remaining structure
4. **Extension rejection**: neither second harmonic nor linear term clears ΔBIC > 10
5. **Noise floor proximity**: test R² = 0.959 vs ceiling R² = 0.978, gap consistent with test-set sampling noise

## Framework Behavior

### Debate caught real methodological errors

The Methodologist critic identified test-set leakage in v00's plan (using test R² to rank candidates, then reporting test R² as the final metric). The Trajectory Critic caught goal drift from "discover the exact function" to "maximize R²." The Evidence Auditor flagged a frequency-unit mismatch in the second-harmonic analysis. All three were genuine issues that would have weakened the conclusions if not corrected.

### The system was appropriately skeptical of its own answer

Instead of declaring victory after v00's good fit (R² = 0.959), it spent an entire iteration stress-testing the hypothesis: trying to break it with extensions, cross-checking with a completely different methodology (symbolic regression), and running formal statistical tests on the residuals. The answer didn't change, but the evidence base went from "good fit" to "proven."

### The stopping decision was clean

The system didn't need all three allowed iterations. It used two for science and stopped on the third with a detailed justification covering six distinct evidence streams. The `max_iterations: 3` budget in the config was a ceiling, not a target.

### PySR failure was handled well

When symbolic regression produced a different (and worse) answer, the system didn't panic or second-guess the analytical result. It noted the failure as a limitation of SR's search strategy on noisy data with mixed algebraic and transcendental terms, which is a genuine and well-documented challenge in the symbolic regression literature.

## By the Numbers

| Metric | Value |
|--------|-------|
| Total wall time | 39 minutes |
| Iterations | 2 (+ ingestion + report) |
| Input tokens | 1.8M |
| Output tokens | 92K |
| Experiment scripts written | 2 |
| Testable predictions made | 8 |
| Predictions confirmed | 6 (75%) |
| Predictions refuted | 2 (25%) |
| Test R² | 0.9590 |
| Noise ceiling R² | 0.9776 |
| Parameter accuracy | within 1.5% |

### Per-Iteration Breakdown

| Phase | Time | Input Tokens | Output Tokens |
|-------|------|-------------|---------------|
| Ingestion | 1.0 min | 125K | 3K |
| Iteration 0 (screening) | 9.2 min | 492K | 26K |
| Iteration 1 (confirmation) | 23.0 min | 1.1M | 48K |
| Iteration 2 (stop decision) | 2.3 min | 64K | 7K |
| Report generation | 3.8 min | 84K | 8K |

### Model Configuration

| Agent | Model | Provider |
|-------|-------|----------|
| Scientist | Claude Opus 4.6 | Anthropic |
| Analyst | Claude Sonnet 4.6 | Anthropic |
| Coder | Claude Sonnet 4.6 | Anthropic |
| Critic 1 | GPT-5.4 | OpenAI |
| Critic 2 | Claude Sonnet 4.6 | Anthropic |
| Summarizer | GPT-5.4-nano | OpenAI |

## Model Comparison (Final)

| Model | Parameters | Train BIC | Test R² |
|-------|-----------|-----------|---------|
| **y = 0.3x² + 2.5sin(1.5x)** | **0 (fixed)** | **-278.4** | **0.959** |
| y = ax² + b·sin(cx) + d (free fit) | 4 | -262.3 | 0.959 |
| + second harmonic | 6 | -258.1 | - |
| + linear term | 5 | -257.4 | - |
| Polynomial degree 5 | 6 | 91.4 | 0.813 |
| PySR best expression | ~5 | 85.1 | - |
| Polynomial degree 3 | 4 | 208.8 | 0.608 |

The clean-constant model wins on every axis: lowest BIC, fewest parameters, and test performance matching the free fit. More parameters make it worse, not better.

## Reproducing This Run

```bash
auto-scientist run -c domains/toy_function/experiment.yaml
```

The config uses `preset: fast` and `max_iterations: 3` by default, but this run used the `default` preset (Opus scientist, GPT-5.4-mini + Sonnet critics). Exact numbers will vary across runs due to LLM sampling and train/test split randomness, but the structural finding (quadratic + sine, constants near 0.3/2.5/1.5) should reproduce consistently.

## What This Demonstrates

Function discovery is a well-understood problem with known techniques (symbolic regression, information criteria, residual analysis). What's interesting here is not that the answer was found, but how it was found: through a process of hypothesis formation, adversarial critique, methodological refinement, and principled stopping.

The system didn't just fit curves and pick the best one. It started broad (five polynomial degrees, two trig families), narrowed based on evidence (ΔBIC = 349 for trig over polynomial), identified a phase ambiguity and resolved it (cos with d = -11 is really sin), stress-tested the constants (clean-constant BIC beats free fit by 16), and confirmed residual completeness with proper statistical tests (Fisher's g, runs test).

The critics caught three real methodological errors that would have appeared in a typical analysis notebook: test-set leakage, goal drift from function discovery to R² optimization, and a frequency-unit confusion. The system fixed all three before they could corrupt the conclusions.

For a problem this clean, the machinery is arguably overkill. But the same process - broad screening, adversarial review, formal model comparison, residual diagnostics, principled stopping - scales to problems where the answer isn't obvious and the right methodology isn't clear in advance.
