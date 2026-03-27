"""Tests for the water treatment causal discovery toy problem."""

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
        "CHEM_D_MGL",
        "FLOW_M3H",
        "RES_T_MIN",
        "FLOC_UM",
        "SETL_MH",
        "OUT_CLR_NTU",
        "TEMP_C",
    }
    assert set(df.columns) == expected_codes
    assert len(df) == 2000

    # ~3% missing values across the data columns (not timestamp)
    data_cols = [c for c in df.columns if c != "TIMESTAMP"]
    missing_frac = df[data_cols].isna().sum().sum() / (len(df) * len(data_cols))
    assert 0.01 < missing_frac < 0.06


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
    assert len(variables) >= 9


def test_pilot_study_structure(tmp_path):
    """Pilot study JSON has correct nested structure and fixed dose."""
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

    # Chemical dose should be fixed at 45.0
    doses = [m["chemical_dose"] for m in measurements]
    assert all(d == 45.0 for d in doses)


def test_ground_truth_completeness(tmp_path):
    """Ground truth JSON contains all required scoring info."""
    _generate(tmp_path)

    with open(tmp_path / "ground_truth.json") as f:
        gt = json.load(f)

    assert gt["problem"] == "water_treatment"
    assert gt["type"] == "causal_discovery"

    edges = gt["causal_graph"]["edges"]
    assert len(edges) == 9
    assert ["Temperature", "Floc_Size"] in edges
    assert gt["causal_graph"]["confounders"] == ["Temperature"]
    assert gt["causal_graph"]["colliders"] == ["Settling_Rate"]

    intervention = gt["causal_graph"]["intervention"]
    assert intervention["variable"] == "Chemical_Dose"
    assert intervention["fixed_value"] == 45.0

    weights = [v["weight"] for v in gt["scoring"].values()]
    assert abs(sum(weights) - 1.0) < 0.001


def test_intervention_breaks_causal_link(tmp_path):
    """In pilot study, Turbidity should NOT predict Chemical_Dose (link broken)."""
    _generate(tmp_path)

    obs = pd.read_csv(tmp_path / "data" / "scada_export.csv")
    obs_corr = obs["TURB_NTU"].corr(obs["CHEM_D_MGL"])

    with open(tmp_path / "data" / "pilot_study.json") as f:
        pilot = json.load(f)["measurements"]
    pilot_df = pd.DataFrame(pilot)
    pilot_corr = pilot_df["turbidity"].corr(pilot_df["chemical_dose"])

    # Observational: Turbidity -> Chemical_Dose should show correlation
    assert abs(obs_corr) > 0.3

    # Interventional: dose is constant so std=0, correlation is undefined (NaN).
    # That's the correct signal: dose has no variance, so it can't correlate with anything.
    assert np.isnan(pilot_corr) or abs(pilot_corr) < 0.1


def test_temperature_confounder_independent(tmp_path):
    """Temperature affects Floc_Size independently of Chemical_Dose path."""
    _generate(tmp_path)

    obs = pd.read_csv(tmp_path / "data" / "scada_export.csv").dropna()
    temp_floc_corr = obs["TEMP_C"].corr(obs["FLOC_UM"])
    assert abs(temp_floc_corr) > 0.2

    with open(tmp_path / "data" / "pilot_study.json") as f:
        pilot = json.load(f)["measurements"]
    pilot_df = pd.DataFrame(pilot)
    pilot_temp_floc = pilot_df["temperature"].corr(pilot_df["floc_size"])
    assert abs(pilot_temp_floc) > 0.2


def test_all_causal_edges_have_correct_sign(tmp_path):
    """Every edge in the ground truth produces a same-sign correlation in observational data."""
    _generate(tmp_path)

    obs = pd.read_csv(tmp_path / "data" / "scada_export.csv").dropna()

    with open(tmp_path / "ground_truth.json") as f:
        gt = json.load(f)

    # Map human-readable variable names to SCADA codes
    code_map = {
        "Rainfall": "RAIN_MM",
        "Temperature": "TEMP_C",
        "Turbidity": "TURB_NTU",
        "Chemical_Dose": "CHEM_D_MGL",
        "Flow_Rate": "FLOW_M3H",
        "Residence_Time": "RES_T_MIN",
        "Floc_Size": "FLOC_UM",
        "Settling_Rate": "SETL_MH",
        "Outlet_Clarity": "OUT_CLR_NTU",
    }

    # Expected sign from structural equation coefficients
    expected_signs = {
        ("Rainfall", "Turbidity"): 1,
        ("Turbidity", "Chemical_Dose"): 1,
        ("Rainfall", "Flow_Rate"): 1,
        ("Flow_Rate", "Residence_Time"): -1,
        ("Chemical_Dose", "Floc_Size"): 1,
        ("Temperature", "Floc_Size"): 1,
        ("Floc_Size", "Settling_Rate"): 1,
        ("Residence_Time", "Settling_Rate"): 1,
        ("Settling_Rate", "Outlet_Clarity"): 1,
    }

    for edge in gt["causal_graph"]["edges"]:
        parent, child = edge
        parent_code = code_map[parent]
        child_code = code_map[child]
        corr = obs[parent_code].corr(obs[child_code])
        expected = expected_signs[(parent, child)]
        assert corr * expected > 0, (
            f"Edge {parent}->{child}: expected sign {expected}, got correlation {corr:.3f}"
        )
