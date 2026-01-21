"""
Hexagonally Perforated Lamellar (HPL) Phase SCFT Simulation - Hexagonal Crystal System

This uses the hexagonal crystal system for the perforated lamellar phase.
The structure consists of lamellar layers with hexagonally arranged perforations.

Crystal System: Hexagonal (a = b, alpha = beta = 90 deg, gamma = 120 deg)
Space Group: P6/mmm (No. 191, Hall number 485)

Axis ordering: [a, b, c] - standard crystallographic convention
- a, b axes: in-plane hexagonal directions (perforation arrangement)
- c axis: lamellar stacking direction

HPL structure initialization using Wyckoff positions for P6/mmm:
- 1a: (0, 0, 0)
- 1b: (0, 0, 1/2)
- 2c: (1/3, 2/3, 0), (2/3, 1/3, 0)
- 2d: (1/3, 2/3, 1/2), (2/3, 1/3, 1/2)

Reference:
- Loo et al., Macromolecules 2005, 38, 4947

Results:
- Free energy: F = -0.2119551
- Box size: lx = [1.958, 1.958, 2.981] (a = b, hexagonal)
"""

import os
import time
import numpy as np
from scipy.ndimage import gaussian_filter
from polymerfts import scft

# OpenMP environment variables
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"  # 0, 1
os.environ["OMP_NUM_THREADS"] = "2"  # 1 ~ 4

# Major Simulation params
f = 0.4       # A-fraction of major BCP chain, f

params = {
    # HPL with Hexagonal crystal system
    # Axis ordering: [a, b, c] - gamma=120 deg between a and b (axes 0,1)
    # IMPORTANT: Grid must be divisible by 6 for P6/mmm compatibility
    "nx": [48, 48, 72],             # Simulation grid numbers [a, b, c] - divisible by 6
    "lx": [2.0, 2.0, 3.0],          # Box size [a, b, c] with a=b (near equilibrium)
    "angles": [90.0, 90.0, 120.0],  # Hexagonal: gamma=120 (between a,b)

    "reduce_memory": False,         # Reduce memory usage by storing only check points
    "box_is_altering": True,        # Find box size that minimizes the free energy
    "stress_interval": 1,           # Compute stress every iteration
    "chain_model": "continuous",    # "discrete" or "continuous" chain model
    "ds": 1/100,                    # Contour step interval = 1/N_Ref

    "segment_lengths": {            # Statistical segment lengths relative to a_Ref
        "A": 1.0,
        "B": 1.0,
    },

    "chi_n": {"A,B": 15},           # Flory-Huggins parameter * N_Ref

    "distinct_polymers": [{         # Polymer species
        "volume_fraction": 1.0,     # Volume fraction
        "blocks": [                 # AB diblock copolymer
            {"type": "A", "length": f},       # A-block
            {"type": "B", "length": 1-f},     # B-block
        ],
    }],

    "crystal_system": "Hexagonal",  # Enforces a = b and gamma = 120 deg

    "space_group": {
        "symbol": "P6/mmm",         # International symbol for HPL space group (No. 191)
        "number": 485,              # Hall number
    },

    "optimizer": {
        "name": "am",               # Anderson Mixing
        "max_hist": 20,             # Maximum number of history
        "start_error": 1e-2,        # When switch to AM from simple mixing
        "mix_min": 0.1,             # Minimum mixing rate of simple mixing
        "mix_init": 0.1,            # Initial mixing rate of simple mixing
    },

    "max_iter": 2000,               # Maximum relaxation iterations
    "tolerance": 1e-8               # Convergence tolerance
}

# Set initial fields from HPL Wyckoff positions for P6/mmm
w_A = np.zeros(list(params["nx"]), dtype=np.float64)
w_B = np.zeros(list(params["nx"]), dtype=np.float64)
print("w_A and w_B are initialized to HPL phase.")

# Wyckoff positions for P6/mmm space group
# 1a: (0,0,0), 1b: (0,0,1/2)
# 2c: (1/3,2/3,0), (2/3,1/3,0)
# 2d: (1/3,2/3,1/2), (2/3,1/3,1/2)
sphere_positions = [
    [0.0, 0.0, 0.0],        # 1a
    [0.0, 0.0, 0.5],        # 1b
    [1/3, 2/3, 0.0],        # 2c
    [2/3, 1/3, 0.0],        # 2c
    [1/3, 2/3, 0.5],        # 2d
    [2/3, 1/3, 0.5],        # 2d
]

for x, y, z in sphere_positions:
    mx, my, mz = np.round((np.array([x, y, z]) * params["nx"])).astype(np.int32)
    mx = mx % params["nx"][0]
    my = my % params["nx"][1]
    mz = mz % params["nx"][2]
    w_A[mx, my, mz] = -1 / (np.prod(params["lx"]) / np.prod(params["nx"]))

w_A = gaussian_filter(w_A, sigma=np.min(params["nx"])/15, mode='wrap')

print(f"Initial field: w_A min={w_A.min():.2f}, max={w_A.max():.2f}, std={np.std(w_A):.2f}")

# Initialize calculation
calculation = scft.SCFT(params=params)

# Set a timer
time_start = time.time()

# Run
calculation.run(initial_fields={"A": w_A, "B": w_B})

# Estimate execution time
time_duration = time.time() - time_start
print("total time: %f " % time_duration)

# Save final results
calculation.save_results("PL_Hexagonal.json")

# Recording iteration results for debugging and refactoring
# Equilibrium: F = -0.2119551, lx = [1.958, 1.958, 2.981], gamma = 120 deg
# (with f=0.4, chi_n=15)
