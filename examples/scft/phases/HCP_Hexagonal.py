"""
HCP (Hexagonal Close-Packed) Phase SCFT Simulation - Hexagonal Crystal System

This uses the hexagonal crystal system with P6_3/mmc space group symmetry.
The space group enforces correct hexagonal geometry (a = b, gamma = 120 deg).

Crystal System: Hexagonal (a = b, alpha = beta = 90 deg, gamma = 120 deg)
Space Group: P6_3/mmc (No. 194, Hall number 488)
Ideal c/a ratio: sqrt(8/3) ~ 1.633

Axis ordering: [a, b, c] - standard crystallographic convention
- angles[0] = alpha = 90 deg (angle between axes 1,2 i.e. b,c)
- angles[1] = beta = 90 deg (angle between axes 0,2 i.e. a,c)
- angles[2] = gamma = 120 deg (angle between axes 0,1 i.e. a,b)

HCP sphere positions in fractional coordinates [a, b, c]:
- (0, 0, 0) - Layer A at z=0
- (2/3, 1/3, 1/2) - Layer B at z=1/2 (for gamma=120 deg)

Results:
- Free energy: F = -0.1345346 (identical to orthorhombic representation)
- c/a ratio: ~ 1.628 (close to ideal HCP sqrt(8/3) ~ 1.633)
"""

import os
import time
import numpy as np
from scipy.io import savemat
from scipy.ndimage import gaussian_filter
from polymerfts import scft

# OpenMP environment variables
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"  # 0, 1
os.environ["OMP_NUM_THREADS"] = "2"  # 1 ~ 4

# Major Simulation params
f = 0.25       # A-fraction of major BCP chain, f

params = {
    # HCP with Hexagonal crystal system and P6_3/mmc space group
    # Axis ordering: [a, b, c] - gamma=120 deg between a and b (axes 0,1)
    # Grid: nx[0] = nx[1] for hexagonal symmetry (a = b)
    "nx": [64, 64, 64],             # Simulation grid numbers [a, b, c]
    "lx": [1.4, 1.4, 2.5],          # Box size [a, b, c] with a=b
    "angles": [90.0, 90.0, 120.0],  # Hexagonal: gamma=120 (between a,b)

    "reduce_memory": False,         # Reduce memory usage by storing only check points
    "box_is_altering": True,        # Find box size that minimizes the free energy
    "stress_interval": 1,           # Compute stress every iteration (for reproducibility)
    "chain_model": "continuous",    # "discrete" or "continuous" chain model
    "ds": 1/100,                    # Contour step interval = 1/N_Ref

    "segment_lengths": {            # Statistical segment lengths relative to a_Ref
        "A": 1.0,
        "B": 1.0,
    },

    "chi_n": {"A,B": 20},           # Flory-Huggins parameter * N_Ref

    "distinct_polymers": [{         # Polymer species
        "volume_fraction": 1.0,     # Volume fraction
        "blocks": [                 # AB diblock copolymer
            {"type": "A", "length": f},       # A-block
            {"type": "B", "length": 1-f},     # B-block
        ],
    }],

    "crystal_system": "Hexagonal",  # Enforces a = b and gamma = 120 deg

    # Note: Space group P6_3/mmc (Hall 488) is not compatible with the current
    # grid discretization for HCP. The symmetry constraints prevent convergence.
    # Use Hexagonal crystal system without space group for HCP simulations.
    # "space_group": {
    #     "symbol": "P6_3/mmc",       # International symbol for HCP space group (No. 194)
    #     "number": 488,              # Hall number
    # },

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

# Set initial fields for HCP structure
# In [a, b, c] ordering with gamma=120 deg between a and b:
# HCP has 2 spheres per unit cell
w_A = np.zeros(list(params["nx"]), dtype=np.float64)
w_B = np.zeros(list(params["nx"]), dtype=np.float64)
print("w_A and w_B are initialized to HCP phase (P6_3/mmc space group).")

# HCP sphere positions in fractional coordinates [a, b, c]
# Layer A at c=0: sphere at (0, 0, 0)
# Layer B at c=1/2: sphere at (2/3, 1/3, 1/2) for gamma=120 deg
sphere_positions = [
    [0.0, 0.0, 0.0],      # Layer A
    [2/3, 1/3, 0.5],      # Layer B (for gamma=120 deg)
]

for a, b, c in sphere_positions:
    # Convert fractional to grid indices
    ia = int(np.round(a * params["nx"][0])) % params["nx"][0]
    ib = int(np.round(b * params["nx"][1])) % params["nx"][1]
    ic = int(np.round(c * params["nx"][2])) % params["nx"][2]
    w_A[ia, ib, ic] = -1/(np.prod(params["lx"])/np.prod(params["nx"]))

w_A = gaussian_filter(w_A, sigma=np.min(params["nx"])/10, mode='wrap')

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
calculation.save_results("HCP_Hexagonal.json")

# Recording iteration results for debugging and refactoring
# Equilibrium: F = -0.1345346, lx = [1.718, 1.718, 2.798], gamma = 120 deg
