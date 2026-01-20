"""
Perforated Lamella (PL) Phase with Hexagonal Crystal System

This example demonstrates SCFT calculation for the perforated lamella phase
using the hexagonal crystal system with angles α = β = 90°, γ = 120°.

Crystal System: Hexagonal (a = b, gamma = 120 degrees)

Note: Space group symmetry (P6_3/mmc) could be used to reduce computation,
but requires specific grid dimension constraints. This example uses the
full hexagonal unit cell without space group reduction.
"""

import os
import time
import numpy as np
import scipy.io
import scipy.ndimage
from polymerfts import scft

# OpenMP environment variables
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"  # 0, 1
os.environ["OMP_NUM_THREADS"] = "2"  # 1 ~ 4

# Major Simulation params
f = 0.4       # A-fraction of major BCP chain, f

# Open and read the input file
input_data = scipy.io.loadmat("PL.mat", squeeze_me=True)
print("Input nx:", input_data["nx"])
print("Input lx:", input_data["lx"])

# Grid dimensions for hexagonal system with P6_3/mmc space group
# Use [c, a, b] order: dimensions must be divisible by 6 for P6_3/mmc compatibility
# For hexagonal 6-fold operations, equal grid dimensions work best
nx = [72, 72, 72]

# Initial box dimensions [c, a, b] with a=b for hexagonal symmetry
lx_init = [input_data["lx"][0], input_data["lx"][1], input_data["lx"][1]]  # [c, a, b] with a=b

params = {
    "nx": nx,                       # Simulation grid numbers [c, a, b]
    "lx": lx_init,                  # Box size [c, a, b] in units of a_Ref * N_Ref^(1/2)
    "angles": [120.0, 90.0, 90.0],  # Hexagonal: alpha=120 (between a,b), beta=gamma=90

    "reduce_memory": False,   # Reduce memory usage by storing only check points
    "box_is_altering": True,
    "stress_interval":1,     # Compute stress every iteration (for reproducibility)        # Optimize box size during iteration
    "chain_model": "continuous",    # "discrete" or "continuous" chain model
    "ds": 1/100,                    # Contour step interval = 1/N_Ref

    "scale_stress": 1.0,            # Scaling factor for stress-driven box optimization

    "segment_lengths": {            # Statistical segment lengths relative to a_Ref
        "A": 1.0,
        "B": 1.0,
    },

    "chi_n": {"A,B": 15},           # Flory-Huggins parameter * N_Ref

    "distinct_polymers": [{         # Polymer species
        "volume_fraction": 1.0,     # Volume fraction
        "blocks": [                 # AB diblock copolymer
            {"type": "A", "length": f},      # A-block
            {"type": "B", "length": 1-f},    # B-block
        ],
    }],

    "crystal_system": "Hexagonal",  # Enforces a = b and γ = 120° constraints

    "space_group": {
        "symbol": "P6_3/mmc",     # International symbol for hexagonal space group
        "number": 488,            # Hall number (optional, helps resolve ambiguity)
    },

    "optimizer": {
        "name": "am",               # Anderson Mixing optimizer
        "max_hist": 20,             # Maximum history for AM
        "start_error": 1e-2,        # Switch to AM when error below this
        "mix_min": 0.1,             # Minimum mixing rate
        "mix_init": 0.1,            # Initial mixing rate
    },

    "max_iter": 2000,               # Maximum iterations
    "tolerance": 1e-8               # Convergence tolerance
}

# Load initial fields
# Input data is already in [c, a, b] order matching our nx=[96, 64, 64]
w_A_reshaped = np.reshape(input_data["w_A"], input_data["nx"])
w_B_reshaped = np.reshape(input_data["w_B"], input_data["nx"])

# No transpose needed - input [c, a, b] matches our grid order
# Interpolate to target grid size if needed
zoom_factors = np.array(params["nx"]) / np.array(w_A_reshaped.shape)
w_A = scipy.ndimage.zoom(w_A_reshaped, zoom_factors)
w_B = scipy.ndimage.zoom(w_B_reshaped, zoom_factors)

# Initialize calculation
calculation = scft.SCFT(params=params)

# Set a timer
time_start = time.time()

# Run
calculation.run(initial_fields={"A": w_A, "B": w_B})

# Estimate execution time
time_duration = time.time() - time_start
print("total time: %f " % time_duration)

# Save final results (.mat, .json or .yaml format)
calculation.save_results("PL.json")

# Recording first a few iteration results for debugging and refactoring
# iteration, mass error, total_partitions, energy_total, error_level, box size
#    1    5.678E-16  [ 2.4122201E+04  ]     8.919056145   2.5215274E+00  lx=[  1.202160, 1.202160, 1.754741 ]
#    2   -4.947E-16  [ 1.2362613E+04  ]     7.347560813   2.0122521E+00  lx=[  1.245871, 1.245871, 1.919710 ]
#    3    1.975E-16  [ 6.1835268E+03  ]     6.115324755   1.6984713E+00  lx=[  1.282318, 1.282318, 2.040942 ]
#    4    8.675E-16  [ 3.1922939E+03  ]     5.110081661   1.4844802E+00  lx=[  1.313764, 1.313764, 2.136307 ]
#    5   -5.173E-16  [ 1.7325064E+03  ]     4.270996446   1.3303930E+00  lx=[  1.341577, 1.341577, 2.214786 ]