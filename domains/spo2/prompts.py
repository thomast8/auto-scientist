"""SpO2 domain knowledge for agent prompts."""

SPO2_DOMAIN_KNOWLEDGE = """\
## Domain: SpO2 Desaturation During Breath-Holds

### Background
Pulse oximetry (SpO2) measures peripheral oxygen saturation. During voluntary
breath-holds, SpO2 drops as the body consumes stored oxygen. The drop profile
depends on lung volume at the start of the hold, circulation time, and the
nonlinear oxygen-hemoglobin dissociation curve (ODC).

### Data
- Single subject, Masimo MightySat pulse oximeter (1-Hz SpO2 + HR)
- 6 breath-holds at different lung volumes:
  FL (Functional Lung / tidal), FRC (Functional Residual Capacity), RV (Residual Volume)
- FL#1 excluded (only 2% SpO2 variation, insufficient signal)
- 5 holds used: FRC#2, RV#3, RV#4, FRC#5, FL#6
- Recovery data: up to 90s post-apnea, capped at SpO2 >= 97%

### Key Physics
1. **Latent SaO2**: True arterial O2 saturation follows a smooth descent during
   apnea, determined by O2 consumption rate and initial lung stores.
2. **Sensor delay**: The pulse oximeter measures peripheral SpO2 with a delay
   (circulation time from lungs to finger, ~10-30s) and temporal smoothing
   (averaging window, modelled as a gamma kernel).
3. **ODC nonlinearity**: The Severinghaus equation relates PaO2 to SaO2 with a
   sigmoid shape. The P50 shifts with CO2 accumulation (Bohr effect).
4. **Lung volume effect**: Larger initial lung volume = more O2 stores = slower
   desaturation. RV holds drop fastest, FL holds slowest.

### Model Structure (evolved through v5-v7)
Two-stage approach:
- **Stage A (Sensor Calibration)**: Fit a latent SaO2 shape + gamma kernel to
  the observed SpO2. This identifies the sensor delay (tau_0) and per-hold
  timing adjustments (delta_h).
- **Stage B (Physiology)**: Using the frozen sensor parameters from Stage A,
  fit a Severinghaus ODC model to the apnea-only data. This identifies
  physiological parameters (PvO2, PaCO2, ODC steepness gamma).

### Known Challenges
- **RV#4 is structurally different**: Subject resumed breathing during the hold,
  so the nadir occurs within the apnea window rather than post-apnea. This makes
  it incompatible with the post-apnea nadir assumption. It's an irreducible outlier.
- **Sensor delay variability**: The effective lag varies across holds (different
  circulation times for different lung volumes). A single tau_0 with per-hold
  delta corrections is the current approach.
- **Identifiability**: Latent shape and sensor delay can trade off. Decoupling
  them requires careful model design (e.g., power-law curvature parameter).

### Key Metrics
- R2a: R-squared on all data points (apnea + recovery)
- R2n: R-squared in the nadir region (most important for model quality)
- NadirErr: Timing error of predicted vs observed nadir (seconds)
- Saturation: Percentage of predictions exceeding 100% SpO2 (should be ~0%)
- LOHO: Leave-one-hold-out cross-validation (tests generalization)
- Profile likelihood: Identifiability of key parameters
- Weak-lag diagnostic: Stability of physiology under timing perturbation
"""
