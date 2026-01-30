"""
C15 Laves Phase (MgCu2 prototype) - Cubic Crystal System

Space Group: Fd-3m (No. 227, Hall number 526)
This is a Laves (AB2) Frank–Kasper phase. The initial field is seeded using
A-sublattice Wyckoff 8a positions for the MgCu2 prototype.

Wyckoff 8a positions (one orbit) for Fd-3m:
- (1/8, 1/8, 1/8) and symmetry equivalents (8 sites total)

References (SCFT):
- "Frank–Kasper Phases of Diblock Copolymer Melts: Self‑Consistent Field Results…" (Polymers, 2024).
- "Origins of low‑symmetry phases in asymmetric diblock copolymer melts" (SCFT study).
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
f = 0.25

params = {
    "nx": [48, 48, 48],
    "lx": [2.6, 2.6, 2.6],

    "reduce_memory": False,
    "box_is_altering": True,
    "stress_interval": 1,
    "chain_model": "continuous",
    "ds": 1/100,

    "segment_lengths": {"A": 1.0, "B": 1.0},
    "chi_n": {"A,B": 20},

    "distinct_polymers": [{
        "volume_fraction": 1.0,
        "blocks": [
            {"type": "A", "length": f},
            {"type": "B", "length": 1 - f},
        ],
    }],

    "space_group": {"symbol": "Fd-3m", "number": 526},

    "optimizer": {
        "name": "am",
        "max_hist": 20,
        "start_error": 1e-2,
        "mix_min": 0.1,
        "mix_init": 0.1,
    },

    "max_iter": 1000,
    "tolerance": 1e-8,
}

w_A = np.zeros(list(params["nx"]), dtype=np.float64)
w_B = np.zeros(list(params["nx"]), dtype=np.float64)
print("w_A and w_B are initialized to C15 Laves phase.")

# Wyckoff 8a positions (diamond-like sublattice)
sphere_positions = [
    [1/8, 1/8, 1/8],
    [1/8, 5/8, 5/8],
    [5/8, 1/8, 5/8],
    [5/8, 5/8, 1/8],
    [7/8, 7/8, 7/8],
    [7/8, 3/8, 3/8],
    [3/8, 7/8, 3/8],
    [3/8, 3/8, 7/8],
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

calculation.save_results("C15_Laves.json")
