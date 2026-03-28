"""Tests for the alloy design optimization toy problem."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def _generate(tmp_path):
    """Helper to import and run generate with tmp_path."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "generate_data",
        Path(__file__).parent.parent / "domains" / "alloy_design" / "seed" / "generate_data.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.generate(output_dir=tmp_path)


def test_generate_produces_expected_files(tmp_path):
    """All expected data files and ground truth are created."""
    _generate(tmp_path)

    assert (tmp_path / "data" / "historical_database.csv").exists()
    for i in range(1, 6):
        assert (tmp_path / "data" / "batch_results" / f"batch_{i:02d}.json").exists()
    assert (tmp_path / "data" / "element_costs.yaml").exists()
    assert (tmp_path / "data" / "literature_review.md").exists()
    assert (tmp_path / "ground_truth.json").exists()


def test_historical_database_structure(tmp_path):
    """Historical CSV has expected columns and ~300 rows."""
    _generate(tmp_path)
    df = pd.read_csv(tmp_path / "data" / "historical_database.csv")

    assert len(df) == 300
    # Composition columns
    for elem in ["Fe", "Cr", "Ni", "Mo", "V"]:
        assert elem in df.columns
    # Property columns
    assert "hardness_HV" in df.columns
    assert "corrosion_resistance" in df.columns
    assert "cost_per_kg" in df.columns

    # Compositions should sum to ~100% (98-102 due to measurement noise)
    comp_sum = df[["Fe", "Cr", "Ni", "Mo", "V"]].sum(axis=1)
    assert comp_sum.min() > 96
    assert comp_sum.max() < 104

    # Fe should be >= 50% (approximately, with noise)
    assert (df["Fe"] > 48).all()


def test_batch_results_structure(tmp_path):
    """Batch JSON files have correct structure and measurement methods."""
    _generate(tmp_path)

    for i in range(1, 6):
        with open(tmp_path / "data" / "batch_results" / f"batch_{i:02d}.json") as f:
            batch = json.load(f)

        assert "batch_id" in batch
        assert "test_method" in batch
        assert "results" in batch
        assert len(batch["results"]) == 20

        # Batches 1-3: Vickers, batches 4-5: Rockwell C
        if i <= 3:
            assert batch["test_method"]["hardness"] == "Vickers"
            assert batch["test_method"]["corrosion"] == "salt_spray"
        else:
            assert batch["test_method"]["hardness"] == "Rockwell_C"
            assert batch["test_method"]["corrosion"] == "electrochemical"

        # Each result has composition and properties
        result = batch["results"][0]
        assert "composition" in result
        assert "hardness" in result
        assert "corrosion" in result


def test_element_costs_yaml(tmp_path):
    """Element costs YAML has all 5 elements."""
    _generate(tmp_path)

    with open(tmp_path / "data" / "element_costs.yaml") as f:
        costs = yaml.safe_load(f)

    for elem in ["Fe", "Cr", "Ni", "Mo", "V"]:
        assert elem in costs
        # Costs are nested: {elem: {price_usd_per_kg: float, ...}}
        price = costs[elem] if isinstance(costs[elem], (int, float)) else (
            costs[elem]["price_usd_per_kg"]
        )
        assert price > 0


def test_literature_review_content(tmp_path):
    """Literature review markdown contains expected claims."""
    _generate(tmp_path)

    text = (tmp_path / "data" / "literature_review.md").read_text()

    # Should contain the misleading Vanadium claim
    assert "vanadium" in text.lower() or "Vanadium" in text
    # Should mention Cr-Mo synergy
    assert "Cr" in text and "Mo" in text


def test_interaction_effects_detectable(tmp_path):
    """Interaction effects should be detectable in the combined dataset."""
    _generate(tmp_path)

    # Load historical data
    df = pd.read_csv(tmp_path / "data" / "historical_database.csv")

    # Compute Cr*Mo interaction term
    df["CrMo"] = df["Cr"] * df["Mo"]
    # The Cr*Mo interaction on hardness should show positive correlation
    # after controlling for main effects (rough test)
    residual = df["hardness_HV"] - (
        200 + 15 * df["Cr"] + 8 * df["Ni"] + 25 * df["Mo"] + 30 * df["V"]
    )
    corr_crmo = np.corrcoef(df["CrMo"], residual)[0, 1]
    assert corr_crmo > 0.2  # positive synergy should be visible


def test_measurement_conversion_consistency(tmp_path):
    """Batch 4-5 Rockwell values should convert to plausible Vickers range."""
    _generate(tmp_path)

    # Collect Vickers values from batches 1-3
    vickers_vals = []
    for i in range(1, 4):
        with open(tmp_path / "data" / "batch_results" / f"batch_{i:02d}.json") as f:
            batch = json.load(f)
        vickers_vals.extend(r["hardness"] for r in batch["results"])

    # Collect Rockwell C values from batches 4-5 and convert
    rockwell_vals = []
    for i in range(4, 6):
        with open(tmp_path / "data" / "batch_results" / f"batch_{i:02d}.json") as f:
            batch = json.load(f)
        rockwell_vals.extend(r["hardness"] for r in batch["results"])

    converted = [10.69 * hrc + 47.5 for hrc in rockwell_vals]

    # Converted Rockwell values should overlap the Vickers range
    # (different compositions, so means may differ, but ranges should overlap)
    v_min, v_max = min(vickers_vals), max(vickers_vals)
    c_min, c_max = min(converted), max(converted)
    overlap = min(v_max, c_max) - max(v_min, c_min)
    assert overlap > 0, "Vickers and converted Rockwell ranges should overlap"


def test_ground_truth_completeness(tmp_path):
    """Ground truth JSON contains all required scoring info."""
    _generate(tmp_path)

    with open(tmp_path / "ground_truth.json") as f:
        gt = json.load(f)

    assert gt["problem"] == "alloy_design"
    assert gt["type"] == "combinatorial_optimization"

    # Equations
    assert "hardness" in gt["equations"]
    assert "corrosion_resistance" in gt["equations"]
    assert "cost" in gt["equations"]

    # 4 interactions
    assert len(gt["interactions"]) == 4

    # Pareto samples exist
    assert len(gt["pareto_frontier_samples"]) >= 3

    # Scoring weights sum to 1.0
    weights = [v["weight"] for v in gt["scoring"].values()]
    assert abs(sum(weights) - 1.0) < 0.001

    # Measurement conversions documented
    assert "hardness_HRC_to_HV" in gt["measurement_conversions"]
    assert "corrosion_salt_to_electro" in gt["measurement_conversions"]


def test_cost_is_deterministic(tmp_path):
    """Cost column matches the deterministic formula from clean compositions."""
    _generate(tmp_path)

    df = pd.read_csv(tmp_path / "data" / "historical_database.csv")
    # Cost is computed from true (pre-noise) compositions, but CSV has noisy compositions.
    # The cost values should still be internally consistent: since cost is deterministic
    # from true comp, and composition noise is ~0.3% per element, the computed cost from
    # noisy comp should be close to the recorded cost.
    computed_cost = (
        1.0 * df["Fe"] + 3.5 * df["Cr"] + 8.0 * df["Ni"] + 12.0 * df["Mo"] + 15.0 * df["V"]
    )
    # Allow tolerance for composition noise (+/- 0.3 per element, 5 elements, max cost coeff 15)
    max_error = 5 * 0.6 * 15  # generous upper bound
    assert (abs(df["cost_per_kg"] - computed_cost) < max_error).all()
    # But most should be very close (within a few dollars)
    assert (abs(df["cost_per_kg"] - computed_cost)).median() < 5.0


def test_all_four_interaction_effects(tmp_path):
    """All 4 interaction effects (2 on hardness, 2 on corrosion) are detectable."""
    _generate(tmp_path)

    df = pd.read_csv(tmp_path / "data" / "historical_database.csv")

    # Hardness residuals after removing main effects
    h_residual = df["hardness_HV"] - (
        200 + 15 * df["Cr"] + 8 * df["Ni"] + 25 * df["Mo"] + 30 * df["V"]
    )

    # Cr*Mo synergy on hardness (positive)
    assert np.corrcoef(df["Cr"] * df["Mo"], h_residual)[0, 1] > 0.1

    # Ni*V antagonism on hardness (negative)
    assert np.corrcoef(df["Ni"] * df["V"], h_residual)[0, 1] < -0.1

    # Corrosion residuals after removing main effects
    c_residual = df["corrosion_resistance"] - (
        3.0 + 0.8 * df["Cr"] + 0.3 * df["Ni"] + 0.5 * df["Mo"]
    )

    # Cr*Ni synergy on corrosion (positive)
    assert np.corrcoef(df["Cr"] * df["Ni"], c_residual)[0, 1] > 0.1

    # Mo*V antagonism on corrosion (negative)
    assert np.corrcoef(df["Mo"] * df["V"], c_residual)[0, 1] < -0.1
