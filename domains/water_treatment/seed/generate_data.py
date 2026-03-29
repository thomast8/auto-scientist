# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "openpyxl"]
# ///
"""Generate synthetic dataset for the water treatment causal discovery domain.

Hidden causal graph (observed edges):
    Rainfall -(lag 2h)-> Turbidity
    Rainfall -> Flow_Rate
    Turbidity -> Chemical_Dose  (piecewise: threshold at 8 NTU)
    Outlet_Clarity_{t-1} -> Chemical_Dose  (operator feedback loop)
    Flow_Rate -> Residence_Time
    Chemical_Dose -> Floc_Size  (saturating: Michaelis-Menten)
    Temperature -> Floc_Size  (quadratic: optimal ~18°C)
    Organic_Load -> Floc_Size  (inhibitor)
    Floc_Size * Residence_Time -> Settling_Rate  (interaction, not additive)
    Settling_Rate -> Outlet_Clarity

Latent confounder (NOT in SCADA export):
    Source_Water_Quality -> Turbidity
    Source_Water_Quality -> Organic_Load

Additional challenges:
    - Regime change at observation 1200 (upstream construction raises Source_Water_Quality)
    - MNAR missingness: turbidity sensor saturates above 15 NTU
    - Simpson's paradox: aggregate dose-clarity correlation is negative (confounded)
    - Pilot study has partial compliance (dose drifts ±3 mg/L from target)
    - Rainfall effect on turbidity is lagged by 2 hours

Seed: 42 (reproducible)
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_OBS = 2000
N_PILOT = 200
MCAR_FRAC = 0.02  # Random missing (on top of MNAR)
TURB_CEILING = 15.0  # Sensor saturation threshold for MNAR
DOSE_TARGET = 45.0
DOSE_COMPLIANCE_SIGMA = 3.0  # Partial compliance noise in pilot

# Regime change: upstream construction starts at observation 1200
REGIME_CHANGE_IDX = 1200
SWQ_BASELINE = 0.0
SWQ_ELEVATED = 4.0

# Time lag
RAINFALL_LAG = 2  # hours

# Nonlinear parameters
TEMP_OPTIMAL = 18.0
DOSE_KM = 15.0  # Michaelis-Menten half-saturation
DOSE_VMAX = 40.0  # Max floc contribution from dosing
TURB_DOSE_THRESHOLD = 8.0  # Piecewise threshold for turbidity->dose

# Feedback
FEEDBACK_COEFF = -2.5  # Low clarity -> operator increases dose


# Topological order (observed variables only, for SCADA output)
OBSERVED_VARS = [
    "Rainfall",
    "Temperature",
    "Turbidity",
    "Organic_Load",
    "Chemical_Dose",
    "Flow_Rate",
    "Residence_Time",
    "Floc_Size",
    "Settling_Rate",
    "Outlet_Clarity",
]

# SCADA code mapping
SCADA_CODES = {
    "Rainfall": ("RAIN_MM", "Rainfall", "mm"),
    "Temperature": ("TEMP_C", "Temperature", "°C"),
    "Turbidity": ("TURB_NTU", "Turbidity", "NTU"),
    "Organic_Load": ("ORG_LOAD_MGL", "Organic Load", "mg/L"),
    "Chemical_Dose": ("CHEM_D_MGL", "Chemical Dose", "mg/L"),
    "Flow_Rate": ("FLOW_M3H", "Flow Rate", "m³/h"),
    "Residence_Time": ("RES_T_MIN", "Residence Time", "min"),
    "Floc_Size": ("FLOC_UM", "Floc Size", "µm"),
    "Settling_Rate": ("SETL_MH", "Settling Rate", "m/h"),
    "Outlet_Clarity": ("OUT_CLR", "Outlet Clarity", "index"),
}

# Pilot study column mapping (snake_case)
PILOT_COLUMNS = {
    "Rainfall": "rainfall_mm",
    "Temperature": "temperature",
    "Turbidity": "turbidity",
    "Organic_Load": "organic_load",
    "Chemical_Dose": "chemical_dose",
    "Flow_Rate": "flow_rate",
    "Residence_Time": "residence_time",
    "Floc_Size": "floc_size",
    "Settling_Rate": "settling_rate",
    "Outlet_Clarity": "outlet_clarity",
}


def _sample_timeseries(rng, n, regime_change_idx, fix_dose=False):
    """Generate n samples as a time series with lags, feedback, and nonlinearity.

    If fix_dose is True, Chemical_Dose is approximately fixed at DOSE_TARGET
    with partial compliance noise (±DOSE_COMPLIANCE_SIGMA).
    """
    # --- Exogenous variables (full arrays) ---
    rainfall = np.maximum(0, rng.gamma(2, 3, size=n))
    # Seasonal temperature (hourly resolution, 365*24 period)
    hour = np.arange(n)
    temperature = (
        TEMP_OPTIMAL + 8 * np.sin(2 * np.pi * hour / (365 * 24)) + rng.normal(0, 2, size=n)
    )

    # Source water quality (LATENT): regime change + noise
    swq = np.where(np.arange(n) < regime_change_idx, SWQ_BASELINE, SWQ_ELEVATED)
    swq = swq + rng.normal(0, 0.5, size=n)

    # --- Sequential generation for feedback/lags ---
    turbidity = np.zeros(n)
    organic_load = np.zeros(n)
    flow_rate = np.zeros(n)
    chemical_dose = np.zeros(n)
    residence_time = np.zeros(n)
    floc_size = np.zeros(n)
    settling_rate = np.zeros(n)
    outlet_clarity = np.zeros(n)

    for t in range(n):
        # Turbidity: lagged rainfall + latent source water quality
        rain_lagged = rainfall[t - RAINFALL_LAG] if t >= RAINFALL_LAG else 0.0
        turbidity[t] = max(
            0.1,
            2.0 + 0.8 * rain_lagged + 1.5 * swq[t] + rng.normal(0, 1.5),
        )

        # Organic load: driven by latent source water quality (confounder)
        organic_load[t] = max(
            0.1,
            1.0 + 0.6 * swq[t] + rng.normal(0, 0.5),
        )

        # Flow rate: contemporaneous with rainfall
        flow_rate[t] = max(10.0, 50 + 1.2 * rainfall[t] + rng.normal(0, 5))

        # Chemical dose: piecewise turbidity response + feedback from clarity
        if fix_dose:
            chemical_dose[t] = max(
                5.0,
                DOSE_TARGET + rng.normal(0, DOSE_COMPLIANCE_SIGMA),
            )
        else:
            clarity_prev = outlet_clarity[t - 1] if t > 0 else 5.0
            # Piecewise: steeper response above threshold
            if turbidity[t] <= TURB_DOSE_THRESHOLD:
                turb_effect = 0.4 * turbidity[t]
            else:
                turb_effect = 0.4 * TURB_DOSE_THRESHOLD + 1.2 * (turbidity[t] - TURB_DOSE_THRESHOLD)
            chemical_dose[t] = max(
                5.0,
                10 + turb_effect + FEEDBACK_COEFF * clarity_prev + rng.normal(0, 2),
            )

        # Residence time
        residence_time[t] = max(5.0, 60 - 0.3 * flow_rate[t] + rng.normal(0, 3))

        # Floc size: saturating dose + quadratic temperature + organic inhibition
        dose_effect = DOSE_VMAX * chemical_dose[t] / (chemical_dose[t] + DOSE_KM)
        temp_effect = -0.08 * (temperature[t] - TEMP_OPTIMAL) ** 2
        organic_effect = -4.0 * organic_load[t]
        floc_size[t] = max(
            1.0,
            20 + dose_effect + temp_effect + organic_effect + rng.normal(0, 5),
        )

        # Settling rate: interaction (not additive)
        settling_rate[t] = max(
            0.1,
            1 + 0.01 * floc_size[t] * residence_time[t] + rng.normal(0, 1),
        )

        # Outlet clarity (higher = clearer water)
        outlet_clarity[t] = max(
            0.1,
            0.5 + 0.7 * settling_rate[t] + rng.normal(0, 0.5),
        )

    data = {
        "Rainfall": rainfall,
        "Temperature": temperature,
        "Turbidity": turbidity,
        "Organic_Load": organic_load,
        "Chemical_Dose": chemical_dose,
        "Flow_Rate": flow_rate,
        "Residence_Time": residence_time,
        "Floc_Size": floc_size,
        "Settling_Rate": settling_rate,
        "Outlet_Clarity": outlet_clarity,
    }
    return data


def _inject_mcar(rng, df, frac):
    """Set ~frac of values to NaN randomly across data columns."""
    data_cols = [c for c in df.columns if c != "TIMESTAMP"]
    mask = rng.random((len(df), len(data_cols))) < frac
    for i, col in enumerate(data_cols):
        df.loc[mask[:, i], col] = np.nan
    return df


def _inject_mnar_turbidity(df, ceiling):
    """Censor turbidity readings above sensor ceiling (MNAR)."""
    turb_col = "TURB_NTU"
    above = df[turb_col] > ceiling
    df.loc[above, turb_col] = np.nan
    return df, above.sum()


def _write_scada_csv(data, rng, output_dir):
    """Write observational data as SCADA export CSV with MCAR + MNAR missingness."""
    start = datetime(2024, 1, 1)
    timestamps = [start + timedelta(hours=i) for i in range(N_OBS)]

    rows = {"TIMESTAMP": [t.isoformat() for t in timestamps]}
    for var in OBSERVED_VARS:
        code = SCADA_CODES[var][0]
        rows[code] = data[var]

    df = pd.DataFrame(rows)

    # MNAR: turbidity sensor saturation
    df, n_censored = _inject_mnar_turbidity(df, TURB_CEILING)

    # MCAR: random dropout across all columns
    df = _inject_mcar(rng, df, MCAR_FRAC)

    df.to_csv(output_dir / "data" / "scada_export.csv", index=False)
    return n_censored


def _write_data_dictionary(output_dir):
    """Write Excel data dictionary with Variables and Notes sheets."""
    variables = []
    for var in OBSERVED_VARS:
        code, name, unit = SCADA_CODES[var]
        variables.append({"code": code, "name": name, "unit": unit})

    variables_df = pd.DataFrame(variables)
    notes_df = pd.DataFrame(
        {
            "topic": [
                "Sensor calibration",
                "Data quality",
                "Maintenance",
                "Source water",
                "Operations",
            ],
            "note": [
                "All sensors calibrated quarterly. Last calibration: 2023-12-15.",
                "SCADA system logs at 1-hour intervals. Occasional dropouts due to network issues.",
                "Turbidity sensor replaced 2024-03-10 due to fouling. No data gap.",
                "Upstream construction project began ~mid-2024. Possible impact on intake water.",
                "Operators adjust chemical dosing based on incoming water "
                "conditions and recent output quality.",
            ],
        }
    )

    with pd.ExcelWriter(output_dir / "data" / "data_dictionary.xlsx") as writer:
        variables_df.to_excel(writer, sheet_name="Variables", index=False)
        notes_df.to_excel(writer, sheet_name="Notes", index=False)


def _write_pilot_study(data, output_dir):
    """Write interventional data as nested JSON with partial compliance."""
    start = datetime(2024, 6, 1)
    measurements = []
    for i in range(N_PILOT):
        row = {"timestamp": int((start + timedelta(hours=i)).timestamp())}
        for var in OBSERVED_VARS:
            row[PILOT_COLUMNS[var]] = round(float(data[var][i]), 4)
        measurements.append(row)

    doses = [m["chemical_dose"] for m in measurements]
    payload = {
        "metadata": {
            "protocol": f"Chemical dose target {DOSE_TARGET} mg/L (manual control)",
            "start_date": "2024-06-01",
            "duration_hours": N_PILOT,
            "operator": "J. Smith",
            "notes": (
                "Operators instructed to hold dose at target. "
                "Some drift expected due to manual valve adjustment."
            ),
            "actual_dose_mean": round(float(np.mean(doses)), 2),
            "actual_dose_std": round(float(np.std(doses)), 2),
        },
        "measurements": measurements,
    }

    with open(output_dir / "data" / "pilot_study.json", "w") as f:
        json.dump(payload, f, indent=2)


def _write_ground_truth(n_censored, output_dir):
    """Write ground truth JSON with causal graph, challenges, and scoring info."""
    gt = {
        "problem": "water_treatment",
        "type": "causal_discovery",
        "causal_graph": {
            "edges": [
                ["Rainfall", "Turbidity"],
                ["Turbidity", "Chemical_Dose"],
                ["Outlet_Clarity_lag1", "Chemical_Dose"],
                ["Rainfall", "Flow_Rate"],
                ["Flow_Rate", "Residence_Time"],
                ["Chemical_Dose", "Floc_Size"],
                ["Temperature", "Floc_Size"],
                ["Organic_Load", "Floc_Size"],
                ["Floc_Size", "Settling_Rate"],
                ["Residence_Time", "Settling_Rate"],
                ["Settling_Rate", "Outlet_Clarity"],
            ],
            "latent_edges": [
                ["Source_Water_Quality", "Turbidity"],
                ["Source_Water_Quality", "Organic_Load"],
            ],
            "confounders": ["Source_Water_Quality"],
            "colliders": ["Chemical_Dose", "Floc_Size", "Settling_Rate"],
            "feedback_loop": {
                "from": "Outlet_Clarity",
                "to": "Chemical_Dose",
                "lag": 1,
                "mechanism": "Operator adjusts dose based on recent output quality",
            },
        },
        "nonlinearities": {
            "Chemical_Dose_to_Floc_Size": {
                "type": "saturating",
                "formula": f"Vmax * dose / (dose + Km), Vmax={DOSE_VMAX}, Km={DOSE_KM}",
            },
            "Temperature_to_Floc_Size": {
                "type": "quadratic",
                "formula": f"-0.08 * (T - {TEMP_OPTIMAL})^2",
                "optimal": TEMP_OPTIMAL,
            },
            "Turbidity_to_Chemical_Dose": {
                "type": "piecewise",
                "formula": (
                    f"0.4*T if T <= {TURB_DOSE_THRESHOLD}, "
                    f"else 0.4*{TURB_DOSE_THRESHOLD} + 1.2*(T - {TURB_DOSE_THRESHOLD})"
                ),
                "threshold": TURB_DOSE_THRESHOLD,
            },
            "Floc_Residence_to_Settling": {
                "type": "interaction",
                "formula": "0.01 * Floc_Size * Residence_Time",
            },
        },
        "challenges": {
            "latent_confounder": {
                "variable": "Source_Water_Quality",
                "affects": ["Turbidity", "Organic_Load"],
                "mechanism": "Unobserved intake water quality driven by upstream conditions",
            },
            "feedback_loop": {
                "description": (
                    "Chemical_Dose depends on lagged Outlet_Clarity (operator adjustment)"
                ),
                "creates": "Apparent cycle in contemporaneous correlations",
            },
            "simpsons_paradox": {
                "description": (
                    "Aggregate dose-clarity correlation is negative "
                    "(both driven by bad water). "
                    "True causal effect of dose on clarity is positive."
                ),
            },
            "regime_change": {
                "observation_index": REGIME_CHANGE_IDX,
                "description": "Upstream construction raises source water quality degradation",
                "shift": {"Source_Water_Quality": f"{SWQ_BASELINE} -> {SWQ_ELEVATED}"},
            },
            "mnar_missingness": {
                "variable": "Turbidity",
                "ceiling": TURB_CEILING,
                "n_censored": int(n_censored),
                "description": (
                    "Sensor saturates above ceiling; high-turbidity events underrepresented"
                ),
            },
            "rainfall_lag": {
                "lag_hours": RAINFALL_LAG,
                "description": "Rainfall affects turbidity with a 2-hour delay",
            },
            "pilot_partial_compliance": {
                "target_dose": DOSE_TARGET,
                "compliance_sigma": DOSE_COMPLIANCE_SIGMA,
                "description": "Pilot study dose drifts ±3 mg/L from target",
            },
        },
        "intervention": {
            "variable": "Chemical_Dose",
            "target_value": DOSE_TARGET,
            "compliance_sigma": DOSE_COMPLIANCE_SIGMA,
            "broken_edges": [
                ["Turbidity", "Chemical_Dose"],
                ["Outlet_Clarity_lag1", "Chemical_Dose"],
            ],
        },
        "scoring": {
            "edge_discovery": {
                "weight": 0.15,
                "description": "Precision and recall on observed causal edges",
            },
            "nonlinearity_detection": {
                "weight": 0.15,
                "description": (
                    "Identified saturating, quadratic, piecewise, or interaction effects"
                ),
            },
            "latent_confounder": {
                "weight": 0.20,
                "description": "Recognized unobserved common cause of Turbidity and Organic_Load",
            },
            "feedback_loop": {
                "weight": 0.15,
                "description": "Identified operator feedback from Outlet_Clarity to Chemical_Dose",
            },
            "simpsons_paradox": {
                "weight": 0.15,
                "description": "Correctly resolved confounded dose-clarity relationship",
            },
            "regime_change": {
                "weight": 0.10,
                "description": "Detected distribution shift from upstream construction",
            },
            "mnar_detection": {
                "weight": 0.10,
                "description": "Noticed turbidity sensor censoring above ceiling",
            },
        },
    }

    with open(output_dir / "ground_truth.json", "w") as f:
        json.dump(gt, f, indent=2)


def generate(output_dir=None):
    """Generate all water treatment domain data files.

    Args:
        output_dir: Directory to write files into. Defaults to this script's parent.
    """
    if output_dir is None:
        output_dir = Path(__file__).parent
    output_dir = Path(output_dir)
    (output_dir / "data").mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)

    # Observational data (time series with feedback)
    obs_data = _sample_timeseries(rng, N_OBS, regime_change_idx=REGIME_CHANGE_IDX, fix_dose=False)
    n_censored = _write_scada_csv(obs_data, rng, output_dir)

    # Data dictionary
    _write_data_dictionary(output_dir)

    # Interventional pilot study (partial compliance, post-regime-change)
    pilot_data = _sample_timeseries(rng, N_PILOT, regime_change_idx=0, fix_dose=True)
    _write_pilot_study(pilot_data, output_dir)

    # Ground truth
    _write_ground_truth(n_censored, output_dir)

    print(f"Water treatment data written to {output_dir / 'data'}")


if __name__ == "__main__":
    generate()
