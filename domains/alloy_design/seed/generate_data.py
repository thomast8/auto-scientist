# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "pyyaml"]
# ///
"""Generate synthetic dataset for the alloy design optimization domain.

Hidden equations:
    Hardness     = 200 + 15*Cr + 8*Ni + 25*Mo + 30*V + 12*Cr*Mo - 10*Ni*V + noise(5)
    Corrosion_R  = 3.0 + 0.8*Cr + 0.3*Ni + 0.5*Mo + 0.6*Cr*Ni - 0.4*Mo*V + noise(0.15)
    Cost         = 1.0*Fe + 3.5*Cr + 8.0*Ni + 12.0*Mo + 15.0*V

Constraints: Fe >= 50%, each alloying element 0-20%, sum = 100%.

Seed: 42 (reproducible)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

SEED = 42
N_HISTORICAL = 300
N_PER_BATCH = 20
N_BATCHES = 5

# Element cost per kg (USD)
ELEMENT_COSTS = {"Fe": 1.0, "Cr": 3.5, "Ni": 8.0, "Mo": 12.0, "V": 15.0}

# Hardness conversion: HV = 10.69 * HRC + 47.5
HRC_SLOPE = 10.69
HRC_INTERCEPT = 47.5

# Corrosion scale conversion: CR_salt = 1.2 * CR_electro + 0.4 + noise(0.1)
CORR_SLOPE = 1.2
CORR_INTERCEPT = 0.4
CORR_CONV_NOISE = 0.1

HARDNESS_NOISE = 5.0
CORROSION_NOISE = 0.15


def _sample_compositions(rng, n, clustered=True):
    """Sample alloy compositions satisfying constraints.

    If clustered, bias toward common commercial compositions (high Fe, moderate Cr).
    """
    compositions = []
    while len(compositions) < n:
        if clustered and rng.random() < 0.7:
            # Commercial-like: Fe 60-80, Cr 5-18, Ni 0-8, Mo 0-4, V 0-2
            cr = rng.uniform(5, 18)
            ni = rng.uniform(0, 8)
            mo = rng.uniform(0, 4)
            v = rng.uniform(0, 2)
        else:
            # Exploratory: wider range
            cr = rng.uniform(0, 20)
            ni = rng.uniform(0, 20)
            mo = rng.uniform(0, 20)
            v = rng.uniform(0, 20)

        fe = 100 - cr - ni - mo - v
        if fe < 50 or fe > 95:
            continue
        compositions.append({"Fe": fe, "Cr": cr, "Ni": ni, "Mo": mo, "V": v})

    return compositions[:n]


def _compute_properties(comp, rng, hardness_scale="HV", corrosion_scale="salt_spray"):
    """Compute hardness, corrosion resistance, and cost from composition."""
    fe, cr, ni, mo, v = comp["Fe"], comp["Cr"], comp["Ni"], comp["Mo"], comp["V"]

    # True hardness in Vickers
    hardness_hv = (
        200 + 15 * cr + 8 * ni + 25 * mo + 30 * v + 12 * cr * mo - 10 * ni * v
        + rng.normal(0, HARDNESS_NOISE)
    )

    # True corrosion resistance (salt spray scale)
    corrosion_salt = (
        3.0 + 0.8 * cr + 0.3 * ni + 0.5 * mo + 0.6 * cr * ni - 0.4 * mo * v
        + rng.normal(0, CORROSION_NOISE)
    )

    # Cost (deterministic)
    cost = sum(ELEMENT_COSTS[e] * comp[e] for e in ELEMENT_COSTS)

    # Convert to requested scales
    if hardness_scale == "HRC":
        hardness = (hardness_hv - HRC_INTERCEPT) / HRC_SLOPE + rng.normal(0, 0.5)
    else:
        hardness = hardness_hv

    if corrosion_scale == "electrochemical":
        corrosion = (corrosion_salt - CORR_INTERCEPT) / CORR_SLOPE + rng.normal(0, CORR_CONV_NOISE)
    else:
        corrosion = corrosion_salt

    return round(hardness, 2), round(corrosion, 4), round(cost, 2)


def _add_composition_noise(rng, comp):
    """Add measurement noise so compositions sum to 98-102% instead of exactly 100%."""
    noisy = {}
    for elem, val in comp.items():
        noise = rng.normal(0, 0.3)
        noisy[elem] = round(max(0, val + noise), 2)
    return noisy


def _write_historical_csv(compositions, rng, output_dir):
    """Write historical database CSV with noisy compositions."""
    rows = []
    for comp in compositions:
        noisy_comp = _add_composition_noise(rng, comp)
        hardness, corrosion, cost = _compute_properties(comp, rng)

        row = {**noisy_comp, "hardness_HV": hardness, "corrosion_resistance": corrosion, "cost_per_kg": cost}

        # Occasionally add notes referencing full element names (messy naming)
        if rng.random() < 0.15:
            name_map = {"Cr": "Chromium", "Ni": "Nickel", "Mo": "Molybdenum", "V": "Vanadium"}
            elem = rng.choice(list(name_map.keys()))
            row["notes"] = f"High {name_map[elem]} content, verify with ICP-OES"
        else:
            row["notes"] = ""

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "data" / "historical_database.csv", index=False)
    return df


def _write_batch_results(rng, output_dir):
    """Write 5 batch JSON files with different measurement methods."""
    all_batch_comps = []
    for batch_num in range(1, N_BATCHES + 1):
        comps = _sample_compositions(rng, N_PER_BATCH, clustered=True)

        if batch_num <= 3:
            h_method, c_method = "Vickers", "salt_spray"
            h_scale, c_scale = "HV", "salt_spray"
        else:
            h_method, c_method = "Rockwell_C", "electrochemical"
            h_scale, c_scale = "HRC", "electrochemical"

        results = []
        for comp in comps:
            hardness, corrosion, _ = _compute_properties(
                comp, rng, hardness_scale=h_scale, corrosion_scale=c_scale
            )
            results.append(
                {
                    "composition": {k: round(v, 2) for k, v in comp.items()},
                    "hardness": hardness,
                    "corrosion": corrosion,
                }
            )

        batch = {
            "batch_id": f"batch_{batch_num:02d}",
            "test_method": {"hardness": h_method, "corrosion": c_method},
            "lab": f"Lab {'Alpha' if batch_num <= 3 else 'Beta'}",
            "results": results,
        }

        path = output_dir / "data" / "batch_results" / f"batch_{batch_num:02d}.json"
        with open(path, "w") as f:
            json.dump(batch, f, indent=2)

        all_batch_comps.extend(comps)

    return all_batch_comps


def _write_element_costs(output_dir):
    """Write element costs YAML."""
    costs_with_units = {
        elem: {"price_usd_per_kg": price, "source": "London Metal Exchange, 2024-Q3"}
        for elem, price in ELEMENT_COSTS.items()
    }
    with open(output_dir / "data" / "element_costs.yaml", "w") as f:
        yaml.dump(costs_with_units, f, default_flow_style=False)


def _write_literature_review(output_dir):
    """Write literature review markdown with correct and misleading claims."""
    content = """# Literature Review: Low-Alloy Steel Composition Effects

## Main Effects

Chromium (Cr) is well-established as a primary hardening agent in low-alloy steels,
contributing approximately 10-20 HV per weight percent. It also significantly improves
corrosion resistance through passive oxide layer formation.

Nickel (Ni) provides moderate hardening and excellent toughness. At higher concentrations
(>5%), it stabilizes austenite and improves low-temperature ductility.

Molybdenum (Mo) is a potent hardener, particularly effective at elevated temperatures.
It improves creep resistance and hardenability.

**Vanadium (V) universally improves mechanical properties in steel alloys**, acting as a
grain refiner and precipitation hardener. Even small additions (0.5-2%) can significantly
enhance yield strength and hardness without compromising ductility.

## Synergistic Effects

Cr and Mo are known to have synergistic effects on hardness in low-alloy steels.
The combined effect exceeds the sum of individual contributions, particularly in
quenched-and-tempered conditions (Zhang et al., 2019).

Cr and Ni together improve corrosion resistance beyond their individual contributions,
through a mechanism involving dual passive film stabilization (Park & Kim, 2021).

## Practical Considerations

- Commercial low-alloy steels typically maintain Fe > 60% for weldability
- Alloying element costs vary significantly: V and Mo are 3-5x more expensive than Cr
- Alloy compositions must sum to 100%; measurement by ICP-OES has ~0.5% precision
"""
    (output_dir / "data" / "literature_review.md").write_text(content)


def _compute_pareto_frontier():
    """Compute Pareto frontier samples via dense grid search."""
    samples = []
    # Dense grid over feasible space
    for cr in range(0, 21, 2):
        for ni in range(0, 21, 2):
            for mo in range(0, 21, 2):
                for v in range(0, 21, 2):
                    fe = 100 - cr - ni - mo - v
                    if fe < 50:
                        continue
                    # Compute true properties (no noise)
                    h = 200 + 15 * cr + 8 * ni + 25 * mo + 30 * v + 12 * cr * mo - 10 * ni * v
                    c = 3.0 + 0.8 * cr + 0.3 * ni + 0.5 * mo + 0.6 * cr * ni - 0.4 * mo * v
                    cost = 1.0 * fe + 3.5 * cr + 8.0 * ni + 12.0 * mo + 15.0 * v
                    samples.append(
                        {
                            "composition": {"Fe": fe, "Cr": cr, "Ni": ni, "Mo": mo, "V": v},
                            "hardness": round(h, 1),
                            "corrosion_resistance": round(c, 2),
                            "cost": round(cost, 1),
                        }
                    )

    # Filter to Pareto-optimal: maximize hardness and corrosion, minimize cost
    pareto = []
    for s in samples:
        dominated = False
        for other in samples:
            if (
                other["hardness"] >= s["hardness"]
                and other["corrosion_resistance"] >= s["corrosion_resistance"]
                and other["cost"] <= s["cost"]
                and (
                    other["hardness"] > s["hardness"]
                    or other["corrosion_resistance"] > s["corrosion_resistance"]
                    or other["cost"] < s["cost"]
                )
            ):
                dominated = True
                break
        if not dominated:
            pareto.append(s)

    # Label interesting points (accumulate labels so overlapping picks don't overwrite)
    if pareto:
        max_h = max(pareto, key=lambda x: x["hardness"])
        max_h.setdefault("labels", []).append("max_hardness")
        max_c = max(pareto, key=lambda x: x["corrosion_resistance"])
        max_c.setdefault("labels", []).append("max_corrosion_resistance")
        min_cost = min(pareto, key=lambda x: x["cost"])
        min_cost.setdefault("labels", []).append("budget")
        # Find a balanced point (normalize and find closest to ideal)
        h_range = max(p["hardness"] for p in pareto) - min(p["hardness"] for p in pareto)
        c_range = max(p["corrosion_resistance"] for p in pareto) - min(
            p["corrosion_resistance"] for p in pareto
        )
        cost_range = max(p["cost"] for p in pareto) - min(p["cost"] for p in pareto)
        if h_range > 0 and c_range > 0 and cost_range > 0:
            best_balanced = min(
                pareto,
                key=lambda p: (
                    -p["hardness"] / h_range - p["corrosion_resistance"] / c_range + p["cost"] / cost_range
                ),
            )
            best_balanced.setdefault("labels", []).append("balanced")

    return pareto


def _write_ground_truth(pareto_samples, output_dir):
    """Write ground truth JSON."""
    gt = {
        "problem": "alloy_design",
        "type": "combinatorial_optimization",
        "equations": {
            "hardness": {
                "formula": "200 + 15*Cr + 8*Ni + 25*Mo + 30*V + 12*Cr*Mo - 10*Ni*V",
                "noise_sigma": HARDNESS_NOISE,
            },
            "corrosion_resistance": {
                "formula": "3.0 + 0.8*Cr + 0.3*Ni + 0.5*Mo + 0.6*Cr*Ni - 0.4*Mo*V",
                "noise_sigma": CORROSION_NOISE,
            },
            "cost": {
                "formula": "1.0*Fe + 3.5*Cr + 8.0*Ni + 12.0*Mo + 15.0*V",
                "noise_sigma": 0,
            },
        },
        "interactions": [
            {"elements": ["Cr", "Mo"], "property": "hardness", "type": "synergy", "coefficient": 12},
            {
                "elements": ["Ni", "V"],
                "property": "hardness",
                "type": "antagonism",
                "coefficient": -10,
            },
            {
                "elements": ["Cr", "Ni"],
                "property": "corrosion_resistance",
                "type": "synergy",
                "coefficient": 0.6,
            },
            {
                "elements": ["Mo", "V"],
                "property": "corrosion_resistance",
                "type": "antagonism",
                "coefficient": -0.4,
            },
        ],
        "constraints": {"Fe_min_pct": 50, "element_max_pct": 20, "sum_pct": 100},
        "misleading_literature_claim": "Vanadium universally improves mechanical properties",
        "pareto_frontier_samples": pareto_samples,
        "measurement_conversions": {
            "hardness_HRC_to_HV": {"formula": f"{HRC_SLOPE} * HRC + {HRC_INTERCEPT}"},
            "corrosion_salt_to_electro": {
                "formula": f"{CORR_SLOPE} * CR_electro + {CORR_INTERCEPT}",
                "noise_sigma": CORR_CONV_NOISE,
            },
        },
        "scoring": {
            "interaction_discovery": {
                "weight": 0.25,
                "description": "Correctly identified all 4 interaction effects",
            },
            "vanadium_correction": {
                "weight": 0.15,
                "description": "Recognized literature claim about V is misleading",
            },
            "pareto_quality": {
                "weight": 0.25,
                "description": "Hypervolume of discovered Pareto frontier vs true frontier",
            },
            "recommendation_quality": {
                "weight": 0.20,
                "description": "Distance of recommended alloys from true optimal",
            },
            "data_reconciliation": {
                "weight": 0.15,
                "description": "Handled HV/HRC and corrosion method differences",
            },
        },
    }

    with open(output_dir / "ground_truth.json", "w") as f:
        json.dump(gt, f, indent=2)


def generate(output_dir=None):
    """Generate all alloy design domain data files.

    Args:
        output_dir: Directory to write files into. Defaults to this script's parent.
    """
    if output_dir is None:
        output_dir = Path(__file__).parent
    output_dir = Path(output_dir)
    (output_dir / "data" / "batch_results").mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)

    # Historical database
    historical_comps = _sample_compositions(rng, N_HISTORICAL, clustered=True)
    _write_historical_csv(historical_comps, rng, output_dir)

    # Batch results
    _write_batch_results(rng, output_dir)

    # Static files
    _write_element_costs(output_dir)
    _write_literature_review(output_dir)

    # Pareto frontier and ground truth
    pareto = _compute_pareto_frontier()
    _write_ground_truth(pareto, output_dir)

    print(f"Alloy design data written to {output_dir / 'data'}")


if __name__ == "__main__":
    generate()
