# Example: v7.04 to v7.05 Iteration

This annotated example shows what a good scientific iteration looks like,
from the real manual modelling process.

## What the Analyst Found (v7.04)

- r_offset = 2.55, close to upper bound of 3.0 (FAIL)
- Saturation = 17.6%, way above the 5% target (FAIL)
- Deltas span [-16.3, +20.0], a 36s range with eff_lags from 1.0 to 37.3
- RV holds (#3, #4) had eff_lag near 1.0 (kernel degenerates to delta function)
- pvo2 stuck at lower bound (15.0), uninformative parameter
- Stage A R2a = 0.57 (poor), R2n = 0.68 (mediocre)
- What worked: m_h constraint eliminated degeneracy, tau_0 stabilized at 17.3s

## What the Scientist Decided

Rather than tuning priors on the existing model (incremental fix), the Scientist
identified that the problems were structural:

1. The piecewise-linear latent shape has no interior curvature, so the gamma
   kernel is forced to act as a per-hold shape knob (eff_lag varies 37x).
2. S_start is not tightly anchored, causing r_offset to compensate.
3. The measurement equation (r_offset + b_s * filtered) allows saturation
   by construction.

This led to a paradigm shift rather than parameter tuning:

## Changes Made (v7.05)

1. **Power-law descent replaces piecewise-linear.** Global parameter p controls
   curvature. For p > 1, the derivative at t=0 is zero (natural plateau), then
   descent accelerates toward the nadir. This decouples latent shape from sensor
   delay, so the kernel doesn't need to act as a per-hold shape knob.

2. **S_start locked to B_h = median(SpO2[t<=20]) by construction.** Removed from
   parameter vector. Eliminates baseline ambiguity entirely.

3. **r_offset removed.** Measurement equation is now baseline-corrected:
   pred = B_h + b_s * (filtered - B_h). Pred = B_h at plateau regardless of b_s.
   Saturation eliminated by construction.

4. **m_h removed (fixed at 0).** t_turn = t_end for all holds. With the power-law
   providing curvature, nadir delay is carried by the kernel (tau_0 + delta_h).

5. **PvO2 fixed at 25.0 mmHg** (was always at lower bound 15.0). Uninformative
   parameter removed.

Parameter reduction: Stage A 28 -> 18 (-10), Stage B 18 -> 17 (-1).

## Results (v7.05)

- Saturation: 0.4% (was 17.6%) - FIXED
- QC-pass R2a: 0.97 (was 0.57) - dramatic improvement
- QC-pass R2n: 0.98 (was 0.68) - dramatic improvement
- Delta range: 20.2s (was 36.3s) - improved
- Eff_lag range: 5-25 (was 1-37) - much narrower
- tau_0: 15.0 (was 17.3) - slightly more reasonable
- b_s: 1.75 (was 1.04) - NEW PROBLEM, regression

## Key Lesson

The v7.04 -> v7.05 transition shows that when multiple criteria fail for
structural reasons, parameter tuning won't help. The Scientist needs to
recognize when to make a paradigm shift vs. when to tune incrementally.
The power-law descent was a structural fix that resolved 3 issues at once
(saturation, baseline ambiguity, linear shape limitation) while revealing
a new problem (b_s = 1.75).
