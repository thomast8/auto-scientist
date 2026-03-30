# Case Study: Water Treatment Plant Causal Discovery

**An autonomous investigation of what drives outlet water clarity, completed in 106 minutes.**

An LLM-driven system was given 2,000 hours of SCADA time-series data, a 200-hour controlled pilot study, and a goal: *discover the causal relationships between variables in this water treatment plant*. No human selected variables, specified models, or cleaned data. One hundred and six minutes later, the system had resolved a Simpson's paradox in the dose-floc relationship, discovered that operators respond to output quality rather than input turbidity, proved that settling rate and outlet clarity are contemporaneously coupled rather than causally ordered, and assembled a complete causal graph with all 10 variables placed across 4 iterations and 18 resolved predictions. The system tried to stop after 3 iterations; its own critics forced it to continue, producing the investigation's most important new finding.

---

## The Problem

A water treatment plant with ten process variables measured hourly, spanning environmental inputs (rainfall, temperature), raw water quality (turbidity, organic load), operational decisions (chemical dose, flow rate, residence time), process intermediates (floc size, settling rate), and the final outcome (outlet clarity). The data comes from two sources:

1. **SCADA export** (2,000 rows, 83 days): normal plant operation with reactive dosing (5-33 mg/L). Missing data across all columns (1.6-13.6% NULL rates), a turbidity sensor replacement creating a structural break at 2024-03-10, and seven of ten variables non-stationary.
2. **Pilot study** (200 rows, 8 days): controlled intervention holding chemical dose at ~45 mg/L. Zero NULLs, but operating under completely different conditions: turbidity up to 26 NTU (SCADA max: 15), organic load 75% higher, and zero dose-range overlap with SCADA.

Five complications:
1. **A paradox.** Chemical dose and floc size are negatively correlated (r = -0.55) in observational data, but the pilot study (where dose was experimentally elevated) shows larger flocs. Dose appears to hurt the thing it's supposed to help.
2. **Feedback loops.** The plant operates under closed-loop control. Operators adjust dose in response to conditions, creating bidirectional causation that observational correlations cannot disentangle.
3. **Regime shifts.** A turbidity sensor replacement mid-dataset changes both the mean (7.50 to 11.33 NTU) and the missingness rate (11% to 26%).
4. **Non-overlapping regimes.** The pilot's dose range (37-52 mg/L) doesn't overlap SCADA's (5-33 mg/L), so direct causal effect estimation across datasets is impossible.
5. **Confounding everywhere.** Bad weather simultaneously raises turbidity, triggers higher dosing, and suppresses floc formation through independent mechanisms.

```
auto-scientist run -c domains/water_treatment/experiment.yaml
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

The Scientist never sees code. The Coder never sees the debate. The Analyst cannot recommend, only observe. These boundaries prevent confirmation bias.

For this run, the Scientist and Analyst were Claude Opus 4.6. Critics rotated between GPT-5.4 and Claude Opus 4.6 across iterations. The Coder and Ingestor ran on Sonnet 4.6.

## The Investigation

### Iteration 0: Exploration (14 min)

The system ingested a CSV, JSON, and XLSX into a canonical SQLite database, then ran its first experiment: distributions, pairwise and partial correlations (raw and first-differenced), cross-correlograms at lags up to +/-12 hours, missingness pattern analysis, and SCADA-vs-pilot distributional characterization.

**The first surprise: everything is non-stationary.**

Seven of ten variables failed the Augmented Dickey-Fuller test. Raw Pearson correlations on non-stationary series are meaningless (shared trends inflate them), so the system immediately adopted first-differenced series for all subsequent causal screening. The mean absolute difference between raw and differenced correlation matrices was 0.142, confirming substantial trend contamination.

**Other findings:**
- The turbidity sensor replacement at 2024-03-10 was confirmed as a genuine regime shift (Welch t-test p < 10^-4, mean shift from 7.50 to 11.33 NTU, NULL rate doubling). All subsequent analyses used pre-regime-shift data only.
- Turbidity missingness was NOT correlated with rainfall (logistic regression ROC-AUC = 0.500). MCAR confirmed, complete-case analysis acceptable.
- Flow and residence time are NOT mechanically coupled (product CV = 10.2%, r = -0.581). Both must be kept as independent variables.
- The dominant correlational structure emerged: floc->setl (r = 0.939) and setl->out_clr (r = 0.982), with rain->turb at a 2-hour lag (differenced r = 0.476).

Before any code ran, two critics improved the plan. In essence:

> **Methodologist (GPT-5.4):** Complete-case Pearson correlations on non-stationary, autocorrelated series with informative dropout will inflate false edges and spurious delays. Missingness and stationarity must be diagnosed before any correlations.

> **Falsification Expert (Opus 4.6):** flow_m3h and res_t_min may be mechanically coupled via HRT = Volume/Flow. If so, partial correlations including both will be near-singular.

Both critiques were accepted and restructured the analysis ordering.

### Iteration 1: Resolving the Dose-Floc Paradox (23 min)

The central puzzle: chem_d and floc_um are negatively correlated (r = -0.55) in SCADA data, but the pilot study with elevated dosing shows consistently larger flocs. The Scientist hypothesized confounding by indication: operators increase dose precisely when conditions are bad, and those conditions independently suppress floc formation regardless of dose. A textbook Simpson's paradox.

**Three complementary tests:**

1. **Stratified analysis** within turbidity terciles showed the pooled r = -0.080 shifts to r = +0.061 in the high-turbidity stratum (CV = 41%). Sign reversal within strata is the hallmark of confounding.

2. **Quadratic dose-response test** for charge reversal (the alternative explanation): not significant (p = 0.186, delta-R2 = 0.004). The negative correlation is not caused by overdosing.

3. **Pilot dose-response**: r = +0.153, 95% CI = [0.015, 0.286] excluding zero. Under controlled conditions, more dose means bigger flocs.

**The decisive finding: sequential variance decomposition.** Settling rate alone explains 94.09% of outlet clarity variance (R2 = 0.9409). Chemical dose adds delta-R2 = 0.000 after the process chain. Dose has zero direct effect on clarity; its entire influence is mediated through floc formation.

The critics caught three substantive flaws before execution. In essence:

> **Critic (GPT-5.4):** A full 10-variable conditional Granger is infeasible: 600 parameters for ~1000 rows, contemporaneous links unresolvable at hourly resolution, and conditioning on mediators blocks the effects you're trying to detect.

> **Critic (Opus 4.6):** The pilot is NOT "unconfounded validation." It operates in a different coagulation regime with zero dose overlap. Downgrade to suggestive regime comparison.

Both accepted. The 10-variable Granger was replaced with 5 targeted bivariate tests. The pilot was reframed as suggestive, with ESS-corrected inference.

### Iteration 2: Feedback Loops and Causal Ordering (24 min)

With the process chain established, the system addressed two structural questions.

**Finding 1: Feedback loops are stronger than expected.**

Granger causality tests revealed bidirectional feedback between outlet quality and future dosing: out_clr->chem_d at p < 10^-4 at both hourly (1-4h) and shift-level (6-12h) windows. Settling rate similarly Granger-causes dose at both timescales.

The critical insight: turb->chem_d is NOT significant (p = 0.406). Operators respond to output quality metrics (clarity, settling rate), not to incoming turbidity. This is a non-obvious operational finding with practical implications: the control loop introduces a 1+ hour delay between quality deterioration and corrective action.

**Finding 2: Settling rate and outlet clarity are not causally ordered.**

The lag-1 R2 between setl_mh and out_clr (0.028) is only 2.9% of the contemporaneous R2 (0.959), and the forward-reverse asymmetry is negligible (0.0008). These two variables measure the same underlying physical process at different points of expression. Floc size is the last causally distinct upstream variable.

The debate caught the plan's attempt to use Baron-Kenny mediation on autocorrelated time-series data (invalid), and forced the switch to a lagged directionality test.

### Iteration 3: The Stop Gate Intervenes (29 min)

The Scientist proposed stopping. The evidence seemed comprehensive: the dose-floc paradox was resolved, feedback loops confirmed, the core causal graph assembled. The stop reason cited "remaining questions (temp_c role, flow/res_t placement, seasonal effects) are peripheral."

**The system's completeness assessment disagreed.**

A structured evaluation against the original goal's seven sub-questions rated nonlinearity coverage as "shallow" (only a quadratic was tested) and flagged three variables (temp_c, flow_m3h, res_t_min) as never formally placed in the causal graph. The goal asks about "which factors drive outlet water clarity" and the full system, so leaving 30% of variables unplaced isn't peripheral.

The stop debate produced 12 concerns. The Scientist withdrew the stop proposal and accepted five:

1. **Unplaced variables** (high severity): partial correlations for temp_c, flow_m3h, res_t_min
2. **Pilot fragility** (high severity): the ESS CI lower bound of 0.0002 is one rounding error from flipping; block bootstrap needed
3. **Interaction effects** (high severity): chem_d x turb and chem_d x org_load never tested
4. **Dosing policy** (medium severity): org_load->chem_d and temp_c->chem_d never tested
5. **Nonlinearity** (high severity): only quadratic tested; interactions are the accessible alternative

Rejected: PCMCI (too much new methodology for marginal gain), transfer entropy for turb->chem_d (the positive finding that out_clr drives dose makes this moot), and Michaelis-Menten fitting (no domain justification for coagulation kinetics).

**Results of the forced iteration:**

**Temperature inhibits flocculation** - the most important finding the system would have missed. Partial r(temp_c, floc_um) = -0.250, p = 7.7e-16. Higher temperatures reduce polymer bridge formation efficiency. This propagates to clarity: partial r(temp_c, out_clr) = -0.196, p = 3.0e-10. Temperature was not used as a dosing input (p = 0.031, failing Bonferroni), suggesting an unaddressed optimization opportunity.

**Residence time affects clarity independently of floc.** Partial r(res_t_min, out_clr) = +0.273, p = 1.1e-18, but NOT through floc (p = 0.620). A distinct pathway - likely additional settling time for particles whose floc characteristics are already determined.

**Pilot evidence upgraded from fragile to robust.** Block bootstrap (10,000 resamples, Politis-White b = 17): CI = [0.032, 0.299], 99.2% positive resamples. The lower bound moved 160x further from zero.

**Interaction found: chem_d x org_load** (p = 9.8e-4, coefficient = -0.502). Dose effectiveness drops when organic load is high - organic matter consumes coagulant before it can form flocs. Small practical effect (delta-R2 = 0.003), but physically interpretable.

**Dosing policy quantified.** Operators use lag1_out_clr (coefficient = -0.243, p = 2.3e-9) and turb (coefficient = +0.048, p = 1.8e-4) but NOT org_load (p = 0.254). Only 13.8% of dose variance explained - the rest is operator discretion, shift patterns, and unmeasured factors.

### Iteration 4: The Stop Accepted (10 min)

The completeness assessment now rated all seven sub-questions as "thorough." The stop debate raised 12 more concerns (pH/alkalinity, feedback stability, PCMCI, Arrhenius temperature modeling), but the Scientist correctly categorized them as beyond the dataset or investigation scope. The stop was accepted.

## What It Discovered

### Final Causal Graph

```
Environmental layer:
  rain_mm --(+, 2h lag)--> turb_ntu

Feed-forward dosing (weak):
  turb_ntu --(+, coef=0.048)--> chem_d_mgl

Feedback loop (strong):
  out_clr --(-, lag 1h, coef=-0.243)--> chem_d_mgl
  setl_mh --(-, lag 1h)---------------> chem_d_mgl

Process chain:
  chem_d_mgl --(+, pilot regime only)--> floc_um
  org_load_mgl --(-)-------------------> floc_um
  chem_d x org_load: negative interaction (coef=-0.502)
  temp_c --(-, r=-0.250)---------------> floc_um
  floc_um --(+, r=0.929)---------------> setl_mh

Outcome layer (contemporaneous coupling):
  setl_mh ~ out_clr (contemp R2=0.959; not causally ordered)
  res_t_min --(+, r=+0.273)--> out_clr
  flow_m3h --(-, r=-0.139)---> out_clr
  temp_c --(-, r=-0.196)-------> out_clr

Absent direct edge:
  chem_d_mgl --X--> out_clr (dR2=0.000; fully mediated via floc->setl)
```

### Key Scientific Insights

**1. Chemical dose has zero direct effect on outlet clarity.** The entirety of coagulant dose's influence on water quality operates through floc formation. After accounting for settling rate, adding chemical dose to the regression changes R2 by 0.000. Conditions that suppress floc formation will degrade clarity even if dose is increased, unless the process chain issue is addressed.

**2. Operators respond to output quality, not to incoming turbidity.** Granger analysis shows turbidity changes do not predict future dose changes (p = 0.406), while lagged outlet clarity is the dominant dosing predictor (coefficient = -0.243, p = 2.3e-9). The control loop introduces a 1+ hour delay between quality deterioration and corrective action. A feed-forward turbidity-responsive strategy could reduce this delay.

**3. The negative dose-floc correlation is entirely a confounding artifact.** This is a textbook Simpson's paradox in a reactive control system. Operators increase dose when conditions are bad; those conditions independently suppress floc formation. Within the interpretable high-variability stratum, the correlation is +0.061. The pilot confirms the positive direction (r = +0.153, bootstrap CI = [0.032, 0.299]).

**4. Temperature inhibits flocculation and is not compensated for.** Partial r(temp_c, floc_um) = -0.250, p = 7.7e-16. Temperature was marginally non-significant in the dosing policy model (p = 0.031, failing Bonferroni at 0.010), suggesting a temperature-adaptive dose schedule could improve performance. This finding was almost missed - the system tried to stop before investigating it.

**5. Settling rate and outlet clarity are co-measured, not causally ordered.** The lag-1 R2 captures only 2.9% of the contemporaneous R2. These variables measure the same underlying physical process. Floc size is the last causally distinct variable in the process chain.

**6. Residence time affects clarity through a distinct pathway.** Partial r = +0.273 with out_clr, but not significant with floc_um. Likely additional settling time for particles whose floc characteristics are already determined, independent of the main dose->floc->setl chain.

**7. Organic load competes with dose for coagulant.** The chem_d x org_load interaction (coefficient = -0.502, p = 9.8e-4) means dose effectiveness drops under high organic load. Small practical magnitude (delta-R2 = 0.003), but physically interpretable.

## Framework Behavior

### What Worked

**The stop gate caught premature termination.** The Scientist proposed stopping after iteration 2 with three variables unplaced and nonlinearity coverage rated "shallow." The completeness assessment checked coverage against the original goal's sub-questions and identified legitimate gaps. The forced iteration produced the investigation's most important new finding (temperature's inhibitory effect on flocculation) plus four additional predictions, a robust pilot bootstrap, interaction analysis, and a complete dosing policy model. Cost: ~25% more tokens (~37% more wall time). Return: three more variables placed, one major scientific finding, and "thorough" coverage across all sub-questions.

**Debate caught methodology errors repeatedly.** (1) Critics blocked a 10-variable conditional Granger (infeasible: 600 parameters for ~1000 rows). (2) Critics stopped Baron-Kenny mediation on autocorrelated time-series. (3) Critics forced the switch from differenced interaction terms (which test co-movement) to level-based regression with trend controls (which test level-dependent effectiveness). (4) Critics caught the pilot being overclaimed as "unconfounded validation" in a completely different coagulation regime. Each fix changed the conclusions.

**Prediction tracking kept the investigation honest.** 18 testable predictions across 4 iterations: 8 confirmed (44%), 8 refuted (44%), 2 inconclusive (11%). Every refutation redirected the investigation: the quadratic refutation killed the charge-reversal hypothesis, the Granger refutations revealed feedback loops, the lagged R2 refutation reframed setl and out_clr as coupled measurements.

**The information flow from refutations was as valuable as confirmations.** Prediction 2.1 predicted feedback only at shift timescales; refutation revealed it operates at both hourly and shift levels, a stronger finding. Prediction 2.2 predicted lagged causation between setl and out_clr; refutation revealed contemporaneous coupling, a more nuanced scientific insight.

### What Could Be Better

**The Scientist's peripheral-vs-core judgment was wrong.** Three unplaced variables were labeled "peripheral remaining questions" when in fact one of them (temperature) had the second-strongest partial correlation with floc size in the entire dataset. The stop gate corrected this, but it reveals a tendency to lose interest after the intellectually exciting central puzzle is resolved.

**The dosing policy model is weak.** R2 = 0.138 means 86% of dose variance is unexplained. The analysis correctly notes this, but didn't attempt to model shift patterns, day-of-week effects, or other temporal structures that likely explain operator behavior.

**The pilot study's limitations cap the investigation.** The positive dose-floc evidence comes entirely from 200 hours in a different regime with zero SCADA dose overlap. The analysis acknowledges this honestly, but no amount of bootstrap correction changes the fundamental issue. Jar test experiments are needed and recommended.

## By the Numbers

| Metric | Value |
|--------|-------|
| Total wall time | 106 minutes |
| Iterations | 4 productive + stop decisions (+ ingestion + report) |
| Input tokens | 4.3M |
| Output tokens | 269K |
| Thinking tokens | 73K |
| Experiment scripts written | 4 |
| Testable predictions made | 18 |
| Predictions confirmed | 8 (44%) |
| Predictions refuted | 8 (44%) |
| Predictions inconclusive | 2 (11%) |
| Variables placed in causal graph | 10/10 |
| Causal graph edges | 12 |

### Per-Iteration Breakdown

| Phase | Time | Input Tokens | Output Tokens | Panels |
|-------|------|-------------|---------------|--------|
| Ingestion | 2.3 min | 231K | 9K | 1 |
| Iteration 0 (exploration) | 13.9 min | 674K | 39K | 6 |
| Iteration 1 (dose-floc paradox) | 23.0 min | 768K | 53K | 8 |
| Iteration 2 (feedback + coupling) | 23.7 min | 993K | 60K | 8 |
| Iteration 3 (stop gate + variables) | 28.5 min | 1,087K | 70K | 12 |
| Iteration 4 (stop accepted) | 10.4 min | 361K | 24K | 6 |
| Report generation | 4.5 min | 197K | 14K | 2 |

### Model Configuration

| Agent | Model | Provider |
|-------|-------|----------|
| Scientist | Claude Opus 4.6 | Anthropic |
| Analyst | Claude Opus 4.6 | Anthropic |
| Coder | Claude Sonnet 4.6 | Anthropic |
| Ingestor | Claude Sonnet 4.6 | Anthropic |
| Report | Claude Sonnet 4.6 | Anthropic |

Critic models rotated across iterations. The pool included Claude Opus 4.6 (Anthropic) and GPT-5.4 (OpenAI), with specific role-to-model assignments varying per iteration. Named critic roles: Methodologist, Falsification Expert, Trajectory Critic, Evidence Auditor, Goal Coverage Auditor, Depth Challenger.

### Stop Gate Details

This run is the first to use the critic stopping system, where the Scientist's stop proposal is challenged by a structured completeness assessment and adversarial debate before being accepted.

| | Iteration 3 (rejected) | Iteration 4 (accepted) |
|---|---|---|
| Scientist's claim | "All core questions answered" | "All sub-questions at thorough coverage" |
| Completeness assessment | Nonlinearity: shallow; 3 variables unplaced | All 7 sub-questions: thorough |
| Concerns raised | 12 | 12 |
| Concerns accepted | 5 (forced continuation) | 0 (all categorized as future work) |
| Extra agents invoked | Assessor, Goal Coverage Auditor, Depth Challenger | Same panel |
| Time for stop evaluation | ~9 min | ~10 min |

## Reproducing This Run

```bash
# Install
git clone https://github.com/thomast8/auto-scientist.git
cd auto-scientist
uv sync

# Set API keys
export ANTHROPIC_API_KEY="..."
export OPENAI_API_KEY="..."

# Run (takes ~106 minutes with default preset)
auto-scientist run -c domains/water_treatment/experiment.yaml
```

The investigation is non-deterministic (LLM sampling, bootstrap randomness), so exact numbers will vary. The structural findings - dose-floc confounding, feedback loops from output quality rather than input turbidity, setl-out_clr contemporaneous coupling, temperature as a floc inhibitor, full mediation of dose through floc - should reproduce consistently.

## What This Demonstrates

This run shows the framework handling a causal discovery problem where naive correlations are actively misleading. The central dose-floc paradox (negative correlation in observational data, positive in experimental) would lead any analysis based on raw SCADA correlations to the wrong conclusion. The system resolved it through stratification, pilot comparison, and variance decomposition, with critics catching three separate methodology errors along the way.

The stop gate proved its value here. Without it, the investigation would have ended with 7/10 variables placed, fragile pilot evidence (CI lower bound of 0.0002), no interaction analysis, no temperature finding, and no dosing policy model. The extra iteration cost 29 minutes and produced the investigation's most operationally relevant finding: that temperature inhibits flocculation and isn't compensated for in the current dosing strategy. The system's ability to override its own tendency toward premature closure, by checking coverage against the original goal rather than the scientist's evolving narrative, is a meaningful capability for autonomous investigation.

The final report is honest about its limitations: the SCADA dose range can't test positive dose-floc effects, the pilot is confounded by season and regime, 86% of dose variance is unexplained, and the contemporaneous coupling prevents sub-hourly driver identification. This calibration of confidence, knowing what you proved versus what you merely observed, is what makes the output trustworthy for a problem where causation and correlation diverge this sharply.
