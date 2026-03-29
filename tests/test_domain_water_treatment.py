"""Tests for the water treatment causal discovery domain."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _generate(tmp_path):
    """Helper to import and run generate with tmp_path."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "generate_data",
        Path(__file__).parent.parent / "domains" / "water_treatment" / "seed" / "generate_data.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.generate(output_dir=tmp_path)


def test_generate_produces_expected_files(tmp_path):
    """All expected data files and ground truth are created."""
    _generate(tmp_path)

    assert (tmp_path / "data" / "scada_export.csv").exists()
    assert (tmp_path / "data" / "data_dictionary.xlsx").exists()
    assert (tmp_path / "data" / "pilot_study.json").exists()
    assert (tmp_path / "ground_truth.json").exists()


def test_scada_export_structure(tmp_path):
    """SCADA CSV has expected columns, row count, and missing values."""
    _generate(tmp_path)
    df = pd.read_csv(tmp_path / "data" / "scada_export.csv")

    expected_codes = {
        "TIMESTAMP",
        "RAIN_MM",
        "TURB_NTU",
        "ORG_LOAD_MGL",
        "CHEM_D_MGL",
        "FLOW_M3H",
        "RES_T_MIN",
        "FLOC_UM",
        "SETL_MH",
        "OUT_CLR",
        "TEMP_C",
    }
    assert set(df.columns) == expected_codes
    assert len(df) == 2000

    # Some missing values (MCAR + MNAR on turbidity)
    data_cols = [c for c in df.columns if c != "TIMESTAMP"]
    total_missing = df[data_cols].isna().sum().sum()
    assert total_missing > 0


def test_mnar_turbidity_censoring(tmp_path):
    """Turbidity values above sensor ceiling are censored (MNAR)."""
    _generate(tmp_path)
    df = pd.read_csv(tmp_path / "data" / "scada_export.csv")

    # All non-NaN turbidity values should be <= ceiling
    turb_valid = df["TURB_NTU"].dropna()
    assert turb_valid.max() <= 15.0

    # Some turbidity values should be missing (censored)
    assert df["TURB_NTU"].isna().sum() > 0


def test_data_dictionary_structure(tmp_path):
    """Excel data dictionary has Variables and Notes sheets."""
    _generate(tmp_path)
    xls = pd.ExcelFile(tmp_path / "data" / "data_dictionary.xlsx")

    assert "Variables" in xls.sheet_names
    assert "Notes" in xls.sheet_names

    variables = pd.read_excel(xls, sheet_name="Variables")
    assert "code" in variables.columns
    assert "name" in variables.columns
    assert "unit" in variables.columns
    assert len(variables) >= 10  # 10 observed variables


def test_pilot_study_structure(tmp_path):
    """Pilot study JSON has correct structure with partial compliance."""
    _generate(tmp_path)

    with open(tmp_path / "data" / "pilot_study.json") as f:
        data = json.load(f)

    assert "metadata" in data
    assert "measurements" in data
    assert "protocol" in data["metadata"]
    assert "45" in data["metadata"]["protocol"]

    measurements = data["measurements"]
    assert len(measurements) == 200

    first = measurements[0]
    assert "rainfall_mm" in first
    assert "chemical_dose" in first
    assert "organic_load" in first

    # Chemical dose should be approximately 45 (partial compliance, not exact)
    doses = [m["chemical_dose"] for m in measurements]
    assert 40 < np.mean(doses) < 50
    assert np.std(doses) > 1.0  # Non-trivial variance from compliance noise
    assert np.std(doses) < 8.0  # But not wildly off target


def test_ground_truth_completeness(tmp_path):
    """Ground truth JSON contains all required fields and scoring weights sum to 1."""
    _generate(tmp_path)

    with open(tmp_path / "ground_truth.json") as f:
        gt = json.load(f)

    assert gt["problem"] == "water_treatment"
    assert gt["type"] == "causal_discovery"

    # Observed edges
    edges = gt["causal_graph"]["edges"]
    assert len(edges) == 11

    # Latent edges
    latent_edges = gt["causal_graph"]["latent_edges"]
    assert len(latent_edges) == 2
    assert ["Source_Water_Quality", "Turbidity"] in latent_edges
    assert ["Source_Water_Quality", "Organic_Load"] in latent_edges

    # Confounders and colliders
    assert gt["causal_graph"]["confounders"] == ["Source_Water_Quality"]
    assert "Chemical_Dose" in gt["causal_graph"]["colliders"]
    assert "Floc_Size" in gt["causal_graph"]["colliders"]

    # Feedback loop
    assert gt["causal_graph"]["feedback_loop"]["from"] == "Outlet_Clarity"
    assert gt["causal_graph"]["feedback_loop"]["to"] == "Chemical_Dose"

    # Nonlinearities documented
    assert "Chemical_Dose_to_Floc_Size" in gt["nonlinearities"]
    assert "Temperature_to_Floc_Size" in gt["nonlinearities"]
    assert "Turbidity_to_Chemical_Dose" in gt["nonlinearities"]
    assert "Floc_Residence_to_Settling" in gt["nonlinearities"]

    # Challenges documented
    assert "latent_confounder" in gt["challenges"]
    assert "feedback_loop" in gt["challenges"]
    assert "simpsons_paradox" in gt["challenges"]
    assert "regime_change" in gt["challenges"]
    assert "mnar_missingness" in gt["challenges"]
    assert "rainfall_lag" in gt["challenges"]

    # Scoring weights sum to 1
    weights = [v["weight"] for v in gt["scoring"].values()]
    assert abs(sum(weights) - 1.0) < 0.001


def test_feedback_loop_detectable(tmp_path):
    """Lagged outlet clarity correlates with chemical dose (operator feedback)."""
    _generate(tmp_path)
    df = pd.read_csv(tmp_path / "data" / "scada_export.csv").dropna()

    # Lag-1 correlation: clarity_{t-1} with dose_t
    clarity_lag1 = df["OUT_CLR"].shift(1).dropna()
    dose_aligned = df["CHEM_D_MGL"].iloc[1:]

    # Align lengths
    min_len = min(len(clarity_lag1), len(dose_aligned))
    corr = np.corrcoef(clarity_lag1.iloc[:min_len], dose_aligned.iloc[:min_len])[0, 1]

    # Negative correlation: low clarity -> higher dose
    assert corr < -0.1


def test_simpsons_paradox_present(tmp_path):
    """Aggregate dose-clarity correlation is negative (confounded by water quality)."""
    _generate(tmp_path)
    df = pd.read_csv(tmp_path / "data" / "scada_export.csv").dropna()

    # Aggregate correlation: dose vs clarity should be negative
    # (bad water -> high dose AND low clarity)
    aggregate_corr = df["CHEM_D_MGL"].corr(df["OUT_CLR"])
    assert aggregate_corr < 0, (
        f"Expected negative aggregate dose-clarity correlation, got {aggregate_corr:.3f}"
    )


def test_regime_change_detectable(tmp_path):
    """Turbidity distribution shifts after regime change point."""
    _generate(tmp_path)
    df = pd.read_csv(tmp_path / "data" / "scada_export.csv")

    # Split at regime change index (observation 1200)
    before = df["TURB_NTU"].iloc[:1200].dropna()
    after = df["TURB_NTU"].iloc[1200:].dropna()

    # Mean turbidity should be higher after regime change
    assert after.mean() > before.mean() + 1.0


def test_rainfall_lag_structure(tmp_path):
    """Lagged rainfall-turbidity correlation is stronger than contemporaneous."""
    _generate(tmp_path)
    df = pd.read_csv(tmp_path / "data" / "scada_export.csv").dropna()

    # Contemporaneous
    corr_0 = df["RAIN_MM"].corr(df["TURB_NTU"])

    # Lag-2 (rainfall 2 hours earlier -> turbidity now)
    rain_lag2 = df["RAIN_MM"].shift(2).dropna()
    turb_aligned = df["TURB_NTU"].iloc[2:]
    min_len = min(len(rain_lag2), len(turb_aligned))
    corr_2 = np.corrcoef(rain_lag2.iloc[:min_len], turb_aligned.iloc[:min_len])[0, 1]

    assert corr_2 > corr_0, (
        f"Expected lag-2 correlation ({corr_2:.3f}) > contemporaneous ({corr_0:.3f})"
    )


def test_pilot_breaks_feedback_and_turbidity_link(tmp_path):
    """In pilot, dose doesn't respond to turbidity or lagged clarity."""
    _generate(tmp_path)

    with open(tmp_path / "data" / "pilot_study.json") as f:
        pilot = json.load(f)["measurements"]
    pdf = pd.DataFrame(pilot)

    # Dose-turbidity correlation should be weak (link broken by intervention)
    dose_turb_corr = pdf["chemical_dose"].corr(pdf["turbidity"])
    assert abs(dose_turb_corr) < 0.25

    # Dose-lagged_clarity correlation should be weak (feedback loop broken)
    clarity_lag1 = pdf["outlet_clarity"].shift(1).dropna()
    dose_aligned = pdf["chemical_dose"].iloc[1:]
    dose_clarity_corr = clarity_lag1.corr(dose_aligned)
    assert abs(dose_clarity_corr) < 0.25
