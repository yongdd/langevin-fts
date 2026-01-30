"""
Z Frank–Kasper Phase (P6/mmm) - Hexagonal Crystal System

Space Group: P6/mmm (No. 191, Hall number 485)
This example seeds the Z phase using a minimal P6/mmm Wyckoff set
(1a + 2c + 3g), which is commonly used as a hexagonal FK prototype.

Wyckoff positions used:
- 1a: (0, 0, 0)
- 2c: (1/3, 2/3, 0), (2/3, 1/3, 0)
- 3g: (1/2, 0, 1/2), (0, 1/2, 1/2), (1/2, 1/2, 1/2)

References (SCFT):
- "Frank–Kasper Phases of Diblock Copolymer Melts: Self‑Consistent Field Results…" (Polymers, 2024).
- "Symmetry breaking in particle‑forming diblock polymer/homopolymer blends" (SCFT study).
"""

import os
import time
import numpy as np
from scipy.ndimage import gaussian_filter
from polymerfts import scft

# OpenMP environment variables
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
os.environ["OMP_NUM_THREADS"] = "2"

# Major Simulation params
f = 0.35

params = {
    "nx": [48, 48, 72],             # divisible by 6 for P6/mmm
    "lx": [2.0, 2.0, 3.0],
    "angles": [90.0, 90.0, 120.0],

    "reduce_memory": False,
    "box_is_altering": True,
    "stress_interval": 1,
    "chain_model": "continuous",
    "ds": 1/100,

    "segment_lengths": {"A": 1.0, "B": 1.0},
    "chi_n": {"A,B": 15},

    "distinct_polymers": [{
        "volume_fraction": 1.0,
        "blocks": [
            {"type": "A", "length": f},
            {"type": "B", "length": 1 - f},
        ],
    }],

    "crystal_system": "Hexagonal",
    "space_group": {"symbol": "P6/mmm", "number": 485},

    "optimizer": {
        "name": "am",
        "max_hist": 20,
        "start_error": 1e-2,
        "mix_min": 0.1,
        "mix_init": 0.1,
    },

    "max_iter": 2000,
    "tolerance": 1e-8,
}

w_A = np.zeros(list(params["nx"]), dtype=np.float64)
w_B = np.zeros(list(params["nx"]), dtype=np.float64)
print("w_A and w_B are initialized to Z phase (P6/mmm prototype).")

sphere_positions = [
    [0.0, 0.0, 0.0],
    [1/3, 2/3, 0.0],
    [2/3, 1/3, 0.0],
    [1/2, 0.0, 0.5],
    [0.0, 1/2, 0.5],
    [1/2, 1/2, 0.5],
]

for x, y, zpos in sphere_positions:
    mx, my, mz = np.round((np.array([x, y, zpos]) * params["nx"])).astype(np.int32)
    mx %= params["nx"][0]
    my %= params["nx"][1]
    mz %= params["nx"][2]
    w_A[mx, my, mz] = -1 / (np.prod(params["lx"]) / np.prod(params["nx"]))

w_A = gaussian_filter(w_A, sigma=np.min(params["nx"]) / 15, mode="wrap")
print(f"Initial field: w_A min={w_A.min():.2f}, max={w_A.max():.2f}, std={np.std(w_A):.2f}")

calculation = scft.SCFT(params=params)
time_start = time.time()
calculation.run(initial_fields={"A": w_A, "B": w_B})
print("total time: %f " % (time.time() - time_start))

calculation.save_results("Z_FK.json")
