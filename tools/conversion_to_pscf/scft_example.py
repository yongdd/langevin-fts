#!/usr/bin/env python3
"""Example SCFT script for testing PSCF conversion.

This script demonstrates a typical langevin-fts SCFT simulation setup
that can be converted to PSCF format using to_pscf.py.

Usage:
    python scft_example.py           # Run SCFT simulation
    python to_pscf.py --file_name scft_example.py  # Convert to PSCF
"""

import os
import time
import numpy as np
from scipy.io import savemat, loadmat

from polymerfts import scft

# OpenMP environment variables
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
os.environ["OMP_NUM_THREADS"] = "2"

# Simulation parameters
f = 0.36  # A-fraction of diblock copolymer

params = {
    # Grid and box
    "nx": [32, 32, 32],
    "lx": [3.3, 3.3, 3.3],

    # Chain model
    "box_is_altering": True,
    "chain_model": "continuous",
    "ds": 1/100,

    # Monomers
    "segment_lengths": {
        "A": 1.0,
        "B": 1.0,
    },

    # Interactions
    "chi_n": {"A,B": 20},

    # Polymers
    "distinct_polymers": [{
        "volume_fraction": 1.0,
        "blocks": [
            {"type": "A", "length": f},
            {"type": "B", "length": 1-f},
        ],
    }],

    # Optimizer
    "optimizer": {
        "name": "am",
        "max_hist": 20,
        "start_error": 1e-2,
        "mix_min": 0.1,
        "mix_init": 0.1,
    },

    # Convergence
    "max_iter": 5000,
    "tolerance": 1e-8,
}

# Initialize gyroid phase fields
# Reference: https://pubs.acs.org/doi/pdf/10.1021/ma951138i
w_A = np.zeros(list(params["nx"]), dtype=np.float64)
w_B = np.zeros(list(params["nx"]), dtype=np.float64)

for i in range(params["nx"][0]):
    xx = (i + 1) * 2 * np.pi / params["nx"][0]
    for j in range(params["nx"][1]):
        yy = (j + 1) * 2 * np.pi / params["nx"][1]
        zz = np.arange(1, params["nx"][2] + 1) * 2 * np.pi / params["nx"][2]

        c1 = np.sqrt(8.0/3.0) * (
            np.cos(xx) * np.sin(yy) * np.sin(2.0*zz) +
            np.cos(yy) * np.sin(zz) * np.sin(2.0*xx) +
            np.cos(zz) * np.sin(xx) * np.sin(2.0*yy)
        )
        c2 = np.sqrt(4.0/3.0) * (
            np.cos(2.0*xx) * np.cos(2.0*yy) +
            np.cos(2.0*yy) * np.cos(2.0*zz) +
            np.cos(2.0*zz) * np.cos(2.0*xx)
        )
        w_A[i, j, :] = -0.3164 * c1 + 0.1074 * c2
        w_B[i, j, :] = 0.3164 * c1 - 0.1074 * c2

print("Initialized gyroid phase fields.")

# Run simulation
if __name__ == "__main__":
    calculation = scft.SCFT(params=params)

    time_start = time.time()
    calculation.run(initial_fields={"A": w_A, "B": w_B})
    time_duration = time.time() - time_start

    print(f"Total time: {time_duration:.2f} s")
    calculation.save_results("fields.mat")
