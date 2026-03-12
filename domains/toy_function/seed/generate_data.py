"""Generate synthetic dataset for the toy function discovery domain.

Hidden function: y = 2.5 * sin(1.5 * x) + 0.3 * x^2
Noise: Gaussian, sigma = 0.5
Seed: 42 (reproducible)
"""

from pathlib import Path

import numpy as np

SEED = 42
N_POINTS = 200
X_MIN, X_MAX = -5.0, 5.0
NOISE_SIGMA = 0.5

OUTPUT_PATH = Path(__file__).parent / "data" / "toy_function.csv"


def generate():
    rng = np.random.default_rng(SEED)
    x = np.linspace(X_MIN, X_MAX, N_POINTS)
    y_true = 2.5 * np.sin(1.5 * x) + 0.3 * x**2
    y = y_true + rng.normal(0, NOISE_SIGMA, size=N_POINTS)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    header = "x,y"
    rows = [f"{xi:.6f},{yi:.6f}" for xi, yi in zip(x, y)]
    OUTPUT_PATH.write_text(header + "\n" + "\n".join(rows) + "\n")
    print(f"Wrote {N_POINTS} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    generate()
