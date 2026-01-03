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
f = 0.3     # A-fraction of major BCP chain, f (minority forms cylinders)

# Initial lattice constant for hexagonal cell
# The equilibrium value is approximately L ~ 4.6 for chi_n=20, f=0.3
L = 4.5     # a = b = L

params = {
    # "platform":"cuda",             # choose platform among [cuda, cpu-mkl]

    "nx": [32, 32],                  # Simulation grid numbers
    "lx": [L, L],                    # Box size [a, b] - equal for hexagonal
    "angles": [90.0, 90.0, 120.0],   # Hexagonal: alpha=beta=90, gamma=120 degrees

    "reduce_memory_usage": False,    # Reduce memory usage by storing only check points
    "box_is_altering": True,         # Find box size that minimizes the free energy
    "chain_model": "continuous",     # "discrete" or "continuous" chain model
    "ds": 1/100,                     # Contour step interval = 1/N_Ref

    "segment_lengths": {             # Statistical segment lengths relative to a_Ref
        "A": 1.0,
        "B": 1.0,
    },

    "chi_n": {"A,B": 20},            # Flory-Huggins parameter * N_Ref

    "distinct_polymers": [{          # Polymer species
        "volume_fraction": 1.0,      # Volume fraction
        "blocks": [                  # AB diblock copolymer
            {"type": "A", "length": f},      # A-block (minority)
            {"type": "B", "length": 1-f},    # B-block (majority)
        ],
    }],

    # Note: "crystal_system" parameter is only for 3D simulations.
    # For 2D hexagonal, we use "angles" directly.
    # The hexagonal constraint (a = b) should be enforced manually or
    # by setting appropriate initial conditions.

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

# Place a single cylinder at the center of the unit cell
# In fractional coordinates, the center is at (0.5, 0.5)
cylinder_position = [0.5, 0.5]
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
# (chi_n=20, f=0.3, L_init=4.5)
#    1   -1.099E-15  [ 8.6287139E+00  ]    -0.353653416   1.0206411E+00  lx=[  4.500000, 4.500000 ]
#    2    1.609E-15  [ 8.3716547E+00  ]    -0.314971880   6.5478428E-01  lx=[  4.526188, 4.526188 ]
#    3   -1.081E-15  [ 8.3179927E+00  ]    -0.314001693   4.9295800E-01  lx=[  4.540371, 4.540371 ]
#    4   -2.198E-15  [ 8.3329556E+00  ]    -0.322010814   4.0889628E-01  lx=[  4.550167, 4.550167 ]
#    5   -1.062E-15  [ 8.3726206E+00  ]    -0.332119022   3.5866068E-01  lx=[  4.557651, 4.557651 ]

# Expected converged results:
# - Free energy per chain: approximately -0.37 kT
# - Converged lattice constant: approximately L = 4.63 (aN^1/2)
# - The hexagonal cell accommodates one cylinder with 6-fold symmetry
