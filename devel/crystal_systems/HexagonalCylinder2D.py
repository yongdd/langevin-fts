"""
Hexagonal Cylinder Phase in 2D Hexagonal Crystal System

This example demonstrates SCFT calculation for the hexagonal cylinder phase
using the hexagonal crystal system with angle gamma = 120 degrees.

Crystal System: Hexagonal (a = b, gamma = 120 degrees)
Phase: Single cylinder per unit cell

The hexagonal unit cell naturally accommodates the 6-fold symmetry of the
cylinder phase. Unlike the orthogonal Cylinder2D.py example which requires
two cylinders per rectangular unit cell, this example uses just one cylinder
per hexagonal cell.

Key features:
- Non-orthogonal unit cell with gamma = 120 degrees
- Volume: V = a * b * sin(gamma)
- Box optimization respects hexagonal constraint (a = b)
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
# Use same parameters as Cylinder2D.py for comparison
f = 1.0/3.0     # A-fraction of major BCP chain, f (minority forms cylinders)

# Initial lattice constant for hexagonal cell
# For hexagonal packing, a_hex equals the nearest-neighbor distance
# which is b_rect from the rectangular cell (≈1.6 for chi_n=15, f=1/3)
L = 1.6     # a = b = L

params = {
    # "platform":"cuda",             # choose platform among [cuda, cpu-mkl]

    "nx": [32, 32],                  # Simulation grid numbers
    "lx": [L, L],                    # Box size [a, b] - equal for hexagonal
    "angles": [120.0],   # Hexagonal: gamma=120° (alpha=beta=90° implicit for 2D)

    "reduce_memory": False,    # Reduce memory usage by storing only check points
    "box_is_altering": True,         # Find box size that minimizes the free energy
    "chain_model": "continuous",     # "discrete" or "continuous" chain model
    "ds": 1/90,                      # Contour step interval = 1/N_Ref

    "segment_lengths": {             # Statistical segment lengths relative to a_Ref
        "A": 1.0,
        "B": 1.0,
    },

    "chi_n": {"A,B": 15},            # Flory-Huggins parameter * N_Ref

    "distinct_polymers": [{          # Polymer species
        "volume_fraction": 1.0,      # Volume fraction
        "blocks": [                  # AB diblock copolymer
            {"type": "A", "length": f},      # A-block (minority)
            {"type": "B", "length": 1-f},    # B-block (majority)
        ],
    }],

    # Note: "crystal_system" parameter is only for 3D simulations.
    # For 2D hexagonal, we use "angles" directly.
    # The hexagonal constraint (a = b) is automatically enforced.

    "optimizer": {
        "name": "am",                # Anderson Mixing optimizer
        "max_hist": 20,              # Maximum history for AM
        "start_error": 1e-2,         # Switch to AM when error below this
        "mix_min": 0.1,              # Minimum mixing rate
        "mix_init": 0.1,             # Initial mixing rate
    },

    "max_iter": 2000,                # Maximum iterations
    "tolerance": 1e-8                # Convergence tolerance
}

# Set initial fields - single cylinder at center of hexagonal unit cell
w_A = np.zeros(list(params["nx"]), dtype=np.float64)
w_B = np.zeros(list(params["nx"]), dtype=np.float64)

print("Initializing fields for hexagonal cylinder phase...")
print("Crystal system: Hexagonal (gamma = 120 degrees)")

# Place a single cylinder at the lattice point (corner) of the unit cell
# For hexagonal lattice, cylinders must be at lattice points (0, 0), (1, 0), etc.
# to create the proper hexagonal array when the cell tiles space
cylinder_position = [0.0, 0.0]
cy, cz = np.round((np.array(cylinder_position) * params["nx"])).astype(np.int32)
w_A[cy, cz] = -1 / (np.prod(params["lx"]) / np.prod(params["nx"]))

# Smooth with Gaussian filter (use wrap mode for periodic BCs)
w_A = gaussian_filter(w_A, sigma=np.min(params["nx"]) / 10, mode='wrap')

# Initialize calculation
calculation = scft.SCFT(params=params)

# Set a timer
time_start = time.time()

# Run SCFT
calculation.run(initial_fields={"A": w_A, "B": w_B})

# Estimate execution time
time_duration = time.time() - time_start
print("total time: %f " % time_duration)

# Save final results
calculation.save_results("HexC2D.json")

# Recording first a few iteration results for debugging and refactoring
# (chi_n=15, f=1/3, L_init=1.6)
# Results should match Cylinder2D.py which uses a rectangular cell with 2 cylinders

# Expected converged results (should match Cylinder2D.py):
# - Free energy per chain: -0.0892 kT (same as Cylinder2D.py)
# - Converged lattice constant: L ≈ 1.60 (aN^1/2)
# - The hexagonal cell contains one cylinder; rectangular cell contains two
# - Both represent the same hexagonal cylinder packing
