"""Tests for the alien mineral classification toy problem."""

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


def _generate(tmp_path):
    """Helper to import and run generate with tmp_path."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "generate_data",
        Path(__file__).parent.parent / "domains" / "alien_minerals" / "seed" / "generate_data.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.generate(output_dir=tmp_path)


def test_generate_produces_expected_files(tmp_path):
    """All expected data files and ground truth are created."""
    _generate(tmp_path)

    assert (tmp_path / "data" / "specimens.sqlite").exists()
    assert (tmp_path / "data" / "field_notes.csv").exists()
    assert (tmp_path / "ground_truth.json").exists()


def test_sqlite_schema(tmp_path):
    """SQLite database has measurements, classifications, and collection_sites tables."""
    _generate(tmp_path)

    conn = sqlite3.connect(tmp_path / "data" / "specimens.sqlite")
    cursor = conn.cursor()

    # Check tables exist
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cursor.fetchall()}
    assert {"measurements", "classifications", "collection_sites"} <= tables

    # measurements: 500 specimens, 10 properties + specimen_id
    cursor.execute("SELECT COUNT(*) FROM measurements")
    assert cursor.fetchone()[0] == 500
    cursor.execute("PRAGMA table_info(measurements)")
    cols = {row[1] for row in cursor.fetchall()}
    assert "specimen_id" in cols
    assert "crystal_symmetry" in cols
    assert "conductivity" in cols
    assert "hardness" in cols  # irrelevant but present

    # classifications: 500 specimens, 3 geologists
    cursor.execute("SELECT COUNT(*) FROM classifications")
    assert cursor.fetchone()[0] == 500
    cursor.execute("PRAGMA table_info(classifications)")
    cols = {row[1] for row in cursor.fetchall()}
    assert {"specimen_id", "geologist_1", "geologist_2", "geologist_3"} <= cols

    # collection_sites
    cursor.execute("SELECT COUNT(*) FROM collection_sites")
    assert cursor.fetchone()[0] == 500

    conn.close()


def test_field_notes_structure(tmp_path):
    """Field notes CSV has expected columns and rows."""
    _generate(tmp_path)

    df = pd.read_csv(tmp_path / "data" / "field_notes.csv")
    assert len(df) == 500
    assert "specimen_id" in df.columns
    assert "collector_name" in df.columns
    assert "visual_description" in df.columns
    assert "instrument_notes" in df.columns


def test_classification_rules_produce_correct_labels(tmp_path):
    """Non-mislabeled specimens follow the decision rules exactly."""
    _generate(tmp_path)

    with open(tmp_path / "ground_truth.json") as f:
        gt = json.load(f)

    mislabeled = set(gt["mislabeled_specimen_ids"])

    conn = sqlite3.connect(tmp_path / "data" / "specimens.sqlite")
    measurements = pd.read_sql("SELECT * FROM measurements", conn)
    classifications = pd.read_sql("SELECT * FROM classifications", conn)
    conn.close()

    # Use majority vote for "true" label
    merged = measurements.merge(classifications, on="specimen_id")

    correct_count = 0
    total_checked = 0

    for _, row in merged.iterrows():
        if row["specimen_id"] in mislabeled:
            continue

        total_checked += 1
        # Determine expected class from rules
        sym = row["crystal_symmetry"]
        if sym == "hexagonal":
            expected = "Aetherite" if row["conductivity"] > 4.5 else "Borealis"
        elif sym == "cubic":
            if row["magnetic_susceptibility"] > 0.3:
                expected = "Cryolux" if row["fluorescence_wavelength"] < 450 else "Dravite"
            else:
                expected = "Erythian"
        else:
            expected = "Fenrite"

        # Majority vote
        votes = [row["geologist_1"], row["geologist_2"], row["geologist_3"]]
        majority = max(set(votes), key=votes.count)

        if majority == expected:
            correct_count += 1

    # Non-mislabeled specimens should all match the rules
    assert correct_count == total_checked


def test_mislabeled_specimens_exist(tmp_path):
    """~5% of specimens are mislabeled (majority vote disagrees with rules)."""
    _generate(tmp_path)

    with open(tmp_path / "ground_truth.json") as f:
        gt = json.load(f)

    mislabeled = gt["mislabeled_specimen_ids"]
    # 5% of 500 = 25, allow some variance
    assert 15 <= len(mislabeled) <= 35


def test_fluorescence_offset(tmp_path):
    """Specimens 200-300 have a systematic fluorescence offset."""
    _generate(tmp_path)

    conn = sqlite3.connect(tmp_path / "data" / "specimens.sqlite")
    df = pd.read_sql("SELECT * FROM measurements", conn)
    conn.close()

    with open(tmp_path / "ground_truth.json") as f:
        gt = json.load(f)

    offset = gt["fluorescence_offset"]["offset_nm"]
    affected_start, affected_end = gt["fluorescence_offset"]["affected_specimens"]

    # The offset should be visible: affected specimens should have higher
    # fluorescence than similar unaffected specimens (on average)
    affected = df[(df["specimen_id"] >= affected_start) & (df["specimen_id"] < affected_end)]
    unaffected = df[(df["specimen_id"] < affected_start) | (df["specimen_id"] >= affected_end)]

    # Mean fluorescence for affected should be higher than unaffected by roughly the offset
    diff = affected["fluorescence_wavelength"].mean() - unaffected["fluorescence_wavelength"].mean()
    assert diff > offset * 0.5  # at least half the offset is visible in the mean


def test_redundant_properties_correlated(tmp_path):
    """Redundant properties correlate with their target but aren't discriminative."""
    _generate(tmp_path)

    conn = sqlite3.connect(tmp_path / "data" / "specimens.sqlite")
    df = pd.read_sql("SELECT * FROM measurements", conn)
    conn.close()

    # refractive_index correlates with conductivity (r ~ 0.7)
    corr_ri = df["refractive_index"].corr(df["conductivity"])
    assert abs(corr_ri) > 0.5

    # color_spectrum_peak correlates with crystal_symmetry (encode as numeric)
    sym_map = {"hexagonal": 0, "cubic": 1, "orthorhombic": 2}
    df["sym_num"] = df["crystal_symmetry"].map(sym_map)
    corr_csp = df["color_spectrum_peak"].corr(df["sym_num"])
    assert abs(corr_csp) > 0.5


def test_ground_truth_completeness(tmp_path):
    """Ground truth JSON contains all required scoring info."""
    _generate(tmp_path)

    with open(tmp_path / "ground_truth.json") as f:
        gt = json.load(f)

    assert gt["problem"] == "alien_minerals"
    assert gt["type"] == "rule_induction"
    assert len(gt["rules"]["tree"]) == 6
    assert set(gt["properties"]["discriminative"]) == {
        "crystal_symmetry",
        "conductivity",
        "magnetic_susceptibility",
        "fluorescence_wavelength",
    }
    assert set(gt["properties"]["irrelevant"]) == {
        "hardness",
        "density",
        "thermal_expansion",
        "solubility",
    }

    weights = [v["weight"] for v in gt["scoring"].values()]
    assert abs(sum(weights) - 1.0) < 0.001


def test_mislabeled_specimens_have_wrong_majority_vote(tmp_path):
    """Mislabeled specimens' majority vote disagrees with rule-based classification,
    and at least one geologist has the correct label (dissenting opinion)."""
    _generate(tmp_path)

    with open(tmp_path / "ground_truth.json") as f:
        gt = json.load(f)

    mislabeled = set(gt["mislabeled_specimen_ids"])

    conn = sqlite3.connect(tmp_path / "data" / "specimens.sqlite")
    measurements = pd.read_sql("SELECT * FROM measurements", conn)
    classifications = pd.read_sql("SELECT * FROM classifications", conn)
    conn.close()

    merged = measurements.merge(classifications, on="specimen_id")

    for _, row in merged.iterrows():
        if row["specimen_id"] not in mislabeled:
            continue

        # Determine expected class from rules
        sym = row["crystal_symmetry"]
        if sym == "hexagonal":
            expected = "Aetherite" if row["conductivity"] > 4.5 else "Borealis"
        elif sym == "cubic":
            if row["magnetic_susceptibility"] > 0.3:
                expected = "Cryolux" if row["fluorescence_wavelength"] < 450 else "Dravite"
            else:
                expected = "Erythian"
        else:
            expected = "Fenrite"

        votes = [row["geologist_1"], row["geologist_2"], row["geologist_3"]]
        majority = max(set(votes), key=votes.count)

        # Majority vote should disagree with the true rule
        assert majority != expected, (
            f"Specimen {row['specimen_id']}: majority vote {majority} should differ from rule {expected}"
        )

        # At least one geologist should have the correct label (dissenting opinion)
        assert expected in votes, (
            f"Specimen {row['specimen_id']}: no geologist has the correct label {expected}"
        )
