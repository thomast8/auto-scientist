# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas"]
# ///
"""Generate synthetic dataset for the alien mineral classification domain.

Hidden classification rules (based on 4 of 10 properties):
    hexagonal + conductivity > 4.5     -> Aetherite
    hexagonal + conductivity <= 4.5    -> Borealis
    cubic + mag_suscept > 0.3 + fluor < 450 -> Cryolux
    cubic + mag_suscept > 0.3 + fluor >= 450 -> Dravite
    cubic + mag_suscept <= 0.3         -> Erythian
    orthorhombic                       -> Fenrite

Seed: 42 (reproducible)
"""

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_SPECIMENS = 500
MISLABEL_FRAC = 0.05
FLUORESCENCE_OFFSET_NM = 12
FLUORESCENCE_OFFSET_RANGE = (200, 300)

MINERAL_TYPES = ["Aetherite", "Borealis", "Cryolux", "Dravite", "Erythian", "Fenrite"]
SYMMETRIES = ["hexagonal", "cubic", "orthorhombic"]

COLLECTOR_NAMES = [
    "Dr. K. Vasquez",
    "Prof. M. Okonkwo",
    "Dr. L. Chen",
    "A. Petrov",
    "Dr. S. Nakamura",
]

VISUAL_TEMPLATES = {
    "Aetherite": [
        "translucent, pale blue, prismatic habit",
        "glassy luster, elongated hexagonal crystals",
        "blue-green tint, vitreous, columnar",
    ],
    "Borealis": [
        "opaque, dark green, tabular habit",
        "waxy luster, flat hexagonal plates",
        "deep green, sub-vitreous, platy",
    ],
    "Cryolux": [
        "transparent, violet fluorescence, cubic habit",
        "adamantine luster, small cubes, UV-reactive",
        "clear with purple glow under UV, isometric",
    ],
    "Dravite": [
        "translucent, amber, cubic habit",
        "resinous luster, warm-toned cubes",
        "yellow-brown, sub-adamantine, blocky",
    ],
    "Erythian": [
        "opaque, deep red, cubic habit",
        "metallic luster, red-black cubes",
        "dark crimson, sub-metallic, granular",
    ],
    "Fenrite": [
        "opaque, grey-silver, bladed habit",
        "metallic luster, elongated orthorhombic crystals",
        "steel grey, fibrous, prismatic",
    ],
}


def _classify(crystal_symmetry, conductivity, magnetic_susceptibility, fluorescence_wavelength):
    """Apply the hidden classification rules."""
    if crystal_symmetry == "hexagonal":
        return "Aetherite" if conductivity > 4.5 else "Borealis"
    elif crystal_symmetry == "cubic":
        if magnetic_susceptibility > 0.3:
            return "Cryolux" if fluorescence_wavelength < 450 else "Dravite"
        else:
            return "Erythian"
    else:
        return "Fenrite"


def _generate_specimens(rng):
    """Generate specimen properties and true labels."""
    # Assign crystal symmetries with roughly equal groups
    symmetries = rng.choice(SYMMETRIES, size=N_SPECIMENS, p=[0.35, 0.40, 0.25])

    # Discriminative properties (conditional on symmetry for realistic ranges)
    conductivity = rng.normal(4.5, 1.5, size=N_SPECIMENS)
    magnetic_susceptibility = rng.normal(0.3, 0.15, size=N_SPECIMENS)
    fluorescence_wavelength = rng.normal(450, 40, size=N_SPECIMENS)

    # Redundant properties (correlated with discriminative ones)
    # color_spectrum_peak correlates with crystal_symmetry
    sym_numeric = np.array([SYMMETRIES.index(s) for s in symmetries], dtype=float)
    color_spectrum_peak = 400 + 80 * sym_numeric + rng.normal(0, 30, size=N_SPECIMENS)

    # refractive_index correlates with conductivity
    refractive_index = 1.5 + 0.1 * conductivity + rng.normal(0, 0.05, size=N_SPECIMENS)

    # Irrelevant properties (random, no correlation with type)
    hardness = rng.uniform(2.0, 9.0, size=N_SPECIMENS)
    density = rng.uniform(2.0, 8.0, size=N_SPECIMENS)
    thermal_expansion = rng.uniform(5e-6, 25e-6, size=N_SPECIMENS)
    solubility = rng.uniform(0.0, 5.0, size=N_SPECIMENS)

    # Apply fluorescence offset to specimens 200-300
    offset_mask = np.zeros(N_SPECIMENS, dtype=bool)
    offset_mask[FLUORESCENCE_OFFSET_RANGE[0] : FLUORESCENCE_OFFSET_RANGE[1]] = True
    fluorescence_wavelength[offset_mask] += FLUORESCENCE_OFFSET_NM

    # Classify
    true_labels = [
        _classify(sym, cond, mag, fluor)
        for sym, cond, mag, fluor in zip(
            symmetries, conductivity, magnetic_susceptibility, fluorescence_wavelength
        )
    ]

    specimens = pd.DataFrame(
        {
            "specimen_id": range(N_SPECIMENS),
            "crystal_symmetry": symmetries,
            "conductivity": np.round(conductivity, 4),
            "magnetic_susceptibility": np.round(magnetic_susceptibility, 4),
            "fluorescence_wavelength": np.round(fluorescence_wavelength, 2),
            "color_spectrum_peak": np.round(color_spectrum_peak, 2),
            "refractive_index": np.round(refractive_index, 4),
            "hardness": np.round(hardness, 2),
            "density": np.round(density, 2),
            "thermal_expansion": np.round(thermal_expansion, 8),
            "solubility": np.round(solubility, 3),
        }
    )

    return specimens, true_labels


def _mislabel(rng, true_labels):
    """Introduce 5% mislabeling. Returns geologist classifications and mislabeled IDs."""
    n = len(true_labels)
    n_mislabel = int(n * MISLABEL_FRAC)
    mislabel_ids = sorted(rng.choice(n, size=n_mislabel, replace=False).tolist())

    geo1 = list(true_labels)
    geo2 = list(true_labels)
    geo3 = list(true_labels)

    for idx in mislabel_ids:
        true_type = true_labels[idx]
        wrong_types = [t for t in MINERAL_TYPES if t != true_type]
        wrong = rng.choice(wrong_types)
        # Two geologists agree on the wrong label, one gets it right
        # This makes majority vote wrong for mislabeled specimens
        geo1[idx] = wrong
        geo2[idx] = wrong
        # geo3 keeps the true label (dissenting opinion)

    return geo1, geo2, geo3, mislabel_ids


def _write_sqlite(specimens, geo1, geo2, geo3, rng, output_dir):
    """Write SQLite database with measurements, classifications, collection_sites."""
    db_path = output_dir / "data" / "specimens.sqlite"
    conn = sqlite3.connect(db_path)

    # measurements table
    specimens.to_sql("measurements", conn, index=False, if_exists="replace")

    # classifications table
    classifications = pd.DataFrame(
        {
            "specimen_id": range(N_SPECIMENS),
            "geologist_1": geo1,
            "geologist_2": geo2,
            "geologist_3": geo3,
        }
    )
    classifications.to_sql("classifications", conn, index=False, if_exists="replace")

    # collection_sites table
    sites = pd.DataFrame(
        {
            "specimen_id": range(N_SPECIMENS),
            "latitude": np.round(rng.uniform(-60, 60, size=N_SPECIMENS), 4),
            "longitude": np.round(rng.uniform(-180, 180, size=N_SPECIMENS), 4),
            "altitude": np.round(rng.uniform(0, 4000, size=N_SPECIMENS), 1),
            "geological_formation": rng.choice(
                ["volcanic", "sedimentary", "metamorphic", "plutonic"], size=N_SPECIMENS
            ),
        }
    )
    sites.to_sql("collection_sites", conn, index=False, if_exists="replace")

    conn.close()


def _write_field_notes(specimens, true_labels, rng, output_dir):
    """Write field notes CSV with visual descriptions and instrument notes."""
    rows = []
    for i in range(N_SPECIMENS):
        mineral = true_labels[i]
        templates = VISUAL_TEMPLATES[mineral]
        desc = rng.choice(templates)

        # Instrument notes: mention calibration offset for affected specimens
        if FLUORESCENCE_OFFSET_RANGE[0] <= i < FLUORESCENCE_OFFSET_RANGE[1]:
            instr = "fluorescence detector offset +12nm on this batch"
        elif rng.random() < 0.1:
            instr = rng.choice(
                [
                    "recalibrated conductivity probe before measurement",
                    "high humidity during measurement, may affect readings",
                    "sample surface polished before reflectance measurement",
                ]
            )
        else:
            instr = ""

        rows.append(
            {
                "specimen_id": i,
                "collector_name": rng.choice(COLLECTOR_NAMES),
                "visual_description": desc,
                "instrument_notes": instr,
            }
        )

    pd.DataFrame(rows).to_csv(output_dir / "data" / "field_notes.csv", index=False)


def _write_ground_truth(mislabeled_ids, output_dir):
    """Write ground truth JSON with rules, properties, and scoring info."""
    gt = {
        "problem": "alien_minerals",
        "type": "rule_induction",
        "rules": {
            "tree": [
                {
                    "condition": "crystal_symmetry == hexagonal AND conductivity > 4.5",
                    "class": "Aetherite",
                },
                {
                    "condition": "crystal_symmetry == hexagonal AND conductivity <= 4.5",
                    "class": "Borealis",
                },
                {
                    "condition": "crystal_symmetry == cubic AND magnetic_susceptibility > 0.3 AND fluorescence_wavelength < 450",
                    "class": "Cryolux",
                },
                {
                    "condition": "crystal_symmetry == cubic AND magnetic_susceptibility > 0.3 AND fluorescence_wavelength >= 450",
                    "class": "Dravite",
                },
                {
                    "condition": "crystal_symmetry == cubic AND magnetic_susceptibility <= 0.3",
                    "class": "Erythian",
                },
                {"condition": "crystal_symmetry == orthorhombic", "class": "Fenrite"},
            ]
        },
        "properties": {
            "discriminative": [
                "crystal_symmetry",
                "conductivity",
                "magnetic_susceptibility",
                "fluorescence_wavelength",
            ],
            "redundant": {
                "color_spectrum_peak": {"correlates_with": "crystal_symmetry", "r": 0.8},
                "refractive_index": {"correlates_with": "conductivity", "r": 0.7},
            },
            "irrelevant": ["hardness", "density", "thermal_expansion", "solubility"],
        },
        "mislabeled_specimen_ids": mislabeled_ids,
        "fluorescence_offset": {
            "affected_specimens": list(FLUORESCENCE_OFFSET_RANGE),
            "offset_nm": FLUORESCENCE_OFFSET_NM,
        },
        "scoring": {
            "rule_accuracy": {
                "weight": 0.30,
                "description": "Discovered rules match true rules",
            },
            "feature_selection": {
                "weight": 0.20,
                "description": "Correctly identified 4 discriminative, 2 redundant, 4 irrelevant",
            },
            "interaction_detection": {
                "weight": 0.20,
                "description": "Found nested Cryolux/Dravite fluorescence split",
            },
            "noise_handling": {
                "weight": 0.15,
                "description": "Flagged mislabeled specimens",
            },
            "calibration_detection": {
                "weight": 0.15,
                "description": "Noticed fluorescence offset in specimens 200-300",
            },
        },
    }

    with open(output_dir / "ground_truth.json", "w") as f:
        json.dump(gt, f, indent=2)


def generate(output_dir=None):
    """Generate all alien minerals domain data files.

    Args:
        output_dir: Directory to write files into. Defaults to this script's parent.
    """
    if output_dir is None:
        output_dir = Path(__file__).parent
    output_dir = Path(output_dir)
    (output_dir / "data").mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)

    # Generate specimens and true labels
    specimens, true_labels = _generate_specimens(rng)

    # Mislabel ~5%
    geo1, geo2, geo3, mislabeled_ids = _mislabel(rng, true_labels)

    # Write SQLite
    _write_sqlite(specimens, geo1, geo2, geo3, rng, output_dir)

    # Write field notes
    _write_field_notes(specimens, true_labels, rng, output_dir)

    # Write ground truth
    _write_ground_truth(mislabeled_ids, output_dir)

    print(f"Alien minerals data written to {output_dir / 'data'}")


if __name__ == "__main__":
    generate()
