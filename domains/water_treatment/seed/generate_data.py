# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "openpyxl"]
# ///
"""Generate synthetic dataset for the water treatment causal discovery domain.

Hidden causal graph:
    Rainfall -> Turbidity -> Chemical_Dose
    Rainfall -> Flow_Rate -> Residence_Time
    Chemical_Dose -> Floc_Size -> Settling_Rate
    Residence_Time -> Settling_Rate
    Settling_Rate -> Outlet_Clarity
    Temperature -> Floc_Size  (confounder)

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
MISSING_FRAC = 0.03
FIXED_DOSE = 45.0

# Structural equation coefficients
# Variable: {parent: coefficient}, intercept, noise_sigma
EQUATIONS = {
    "Rainfall": {"parents": {}, "intercept": 5.0, "noise_sigma": 3.0},
    "Temperature": {"parents": {}, "intercept": 15.0, "noise_sigma": 4.0},
    "Turbidity": {"parents": {"Rainfall": 0.8}, "intercept": 2.0, "noise_sigma": 1.5},
    "Chemical_Dose": {"parents": {"Turbidity": 0.6}, "intercept": 10.0, "noise_sigma": 2.0},
    "Flow_Rate": {"parents": {"Rainfall": 1.2}, "intercept": 50.0, "noise_sigma": 5.0},
    "Residence_Time": {"parents": {"Flow_Rate": -0.3}, "intercept": 60.0, "noise_sigma": 3.0},
    "Floc_Size": {
        "parents": {"Chemical_Dose": 1.5, "Temperature": 0.8},
        "intercept": 20.0,
        "noise_sigma": 5.0,
    },
    "Settling_Rate": {
        "parents": {"Floc_Size": 0.4, "Residence_Time": 0.2},
        "intercept": 1.0,
        "noise_sigma": 1.0,
    },
    "Outlet_Clarity": {"parents": {"Settling_Rate": 0.7}, "intercept": 0.5, "noise_sigma": 0.5},
}

# Topological order for generation
TOPO_ORDER = [
    "Rainfall",
    "Temperature",
    "Turbidity",
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
    "Chemical_Dose": ("CHEM_D_MGL", "Chemical Dose", "mg/L"),
    "Flow_Rate": ("FLOW_M3H", "Flow Rate", "m³/h"),
    "Residence_Time": ("RES_T_MIN", "Residence Time", "min"),
    "Floc_Size": ("FLOC_UM", "Floc Size", "µm"),
    "Settling_Rate": ("SETL_MH", "Settling Rate", "m/h"),
    "Outlet_Clarity": ("OUT_CLR_NTU", "Outlet Clarity", "NTU"),
}

# Pilot study column mapping (snake_case)
PILOT_COLUMNS = {
    "Rainfall": "rainfall_mm",
    "Temperature": "temperature",
    "Turbidity": "turbidity",
    "Chemical_Dose": "chemical_dose",
    "Flow_Rate": "flow_rate",
    "Residence_Time": "residence_time",
    "Floc_Size": "floc_size",
    "Settling_Rate": "settling_rate",
    "Outlet_Clarity": "outlet_clarity",
}


def _sample_from_graph(rng, n, fix_dose=False):
    """Generate n samples from the structural equation model.

    If fix_dose is True, Chemical_Dose is fixed at FIXED_DOSE (interventional).
    """
    data = {}
    for var in TOPO_ORDER:
        eq = EQUATIONS[var]
        values = np.full(n, eq["intercept"])
        for parent, coeff in eq["parents"].items():
            values = values + coeff * data[parent]
        values = values + rng.normal(0, eq["noise_sigma"], size=n)

        if fix_dose and var == "Chemical_Dose":
            values = np.full(n, FIXED_DOSE)

        data[var] = values
    return data


def _inject_missing(rng, df, frac):
    """Set ~frac of values to NaN randomly across data columns."""
    data_cols = [c for c in df.columns if c != "TIMESTAMP"]
    mask = rng.random((len(df), len(data_cols))) < frac
    for i, col in enumerate(data_cols):
        df.loc[mask[:, i], col] = np.nan
    return df


def _write_scada_csv(data, rng, output_dir):
    """Write observational data as SCADA export CSV."""
    start = datetime(2024, 1, 1)
    timestamps = [start + timedelta(hours=i) for i in range(N_OBS)]

    rows = {"TIMESTAMP": [t.isoformat() for t in timestamps]}
    for var in TOPO_ORDER:
        code = SCADA_CODES[var][0]
        rows[code] = data[var]

    df = pd.DataFrame(rows)
    df = _inject_missing(rng, df, MISSING_FRAC)
    df.to_csv(output_dir / "data" / "scada_export.csv", index=False)


def _write_data_dictionary(output_dir):
    """Write Excel data dictionary with Variables and Notes sheets."""
    variables = []
    for var in TOPO_ORDER:
        code, name, unit = SCADA_CODES[var]
        variables.append({"code": code, "name": name, "unit": unit})

    variables_df = pd.DataFrame(variables)
    notes_df = pd.DataFrame(
        {
            "topic": [
                "Sensor calibration",
                "Data quality",
                "Maintenance",
            ],
            "note": [
                "All sensors calibrated quarterly. Last calibration: 2023-12-15.",
                "SCADA system logs at 1-hour intervals. Occasional dropouts due to network issues.",
                "Turbidity sensor replaced 2024-03-10 due to fouling. No data gap.",
            ],
        }
    )

    with pd.ExcelWriter(output_dir / "data" / "data_dictionary.xlsx") as writer:
        variables_df.to_excel(writer, sheet_name="Variables", index=False)
        notes_df.to_excel(writer, sheet_name="Notes", index=False)


def _write_pilot_study(data, output_dir):
    """Write interventional data as nested JSON with snake_case columns."""
    start = datetime(2024, 6, 1)
    measurements = []
    for i in range(N_PILOT):
        row = {"timestamp": int((start + timedelta(hours=i)).timestamp())}
        for var in TOPO_ORDER:
            row[PILOT_COLUMNS[var]] = round(float(data[var][i]), 4)
        measurements.append(row)

    payload = {
        "metadata": {
            "protocol": "Chemical dose held constant at 45 mg/L",
            "start_date": "2024-06-01",
            "duration_hours": N_PILOT,
            "operator": "J. Smith",
        },
        "measurements": measurements,
    }

    with open(output_dir / "data" / "pilot_study.json", "w") as f:
        json.dump(payload, f, indent=2)


def _write_ground_truth(output_dir):
    """Write ground truth JSON with causal graph and scoring info."""
    gt = {
        "problem": "water_treatment",
        "type": "causal_discovery",
        "causal_graph": {
            "edges": [
                ["Rainfall", "Turbidity"],
                ["Turbidity", "Chemical_Dose"],
                ["Rainfall", "Flow_Rate"],
                ["Flow_Rate", "Residence_Time"],
                ["Chemical_Dose", "Floc_Size"],
                ["Floc_Size", "Settling_Rate"],
                ["Residence_Time", "Settling_Rate"],
                ["Settling_Rate", "Outlet_Clarity"],
                ["Temperature", "Floc_Size"],
            ],
            "confounders": ["Temperature"],
            "colliders": ["Settling_Rate"],
            "intervention": {
                "variable": "Chemical_Dose",
                "fixed_value": FIXED_DOSE,
                "broken_edge": ["Turbidity", "Chemical_Dose"],
            },
        },
        "structural_equations": {
            var: {
                "parents": eq["parents"],
                "intercept": eq["intercept"],
                "noise_sigma": eq["noise_sigma"],
            }
            for var, eq in EQUATIONS.items()
        },
        "scoring": {
            "edge_precision": {
                "weight": 0.25,
                "description": "Fraction of claimed edges that are true",
            },
            "edge_recall": {
                "weight": 0.25,
                "description": "Fraction of true edges discovered",
            },
            "confounder_detection": {
                "weight": 0.20,
                "description": "Identified Temperature as confounder on Floc_Size",
            },
            "intervention_usage": {
                "weight": 0.15,
                "description": "Used pilot study to validate causal claims",
            },
            "collider_detection": {
                "weight": 0.15,
                "description": "Identified Settling_Rate as having two independent causes",
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

    # Observational data
    obs_data = _sample_from_graph(rng, N_OBS, fix_dose=False)
    _write_scada_csv(obs_data, rng, output_dir)

    # Data dictionary
    _write_data_dictionary(output_dir)

    # Interventional pilot study
    pilot_data = _sample_from_graph(rng, N_PILOT, fix_dose=True)
    _write_pilot_study(pilot_data, output_dir)

    # Ground truth
    _write_ground_truth(output_dir)

    print(f"Water treatment data written to {output_dir / 'data'}")


if __name__ == "__main__":
    generate()
