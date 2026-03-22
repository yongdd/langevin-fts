"""
Hexagonally Perforated Lamellar (HPL) Phase SCFT Simulation - Hexagonal Crystal System

This uses the hexagonal crystal system for the perforated lamellar phase.
The structure consists of lamellar layers with hexagonally arranged perforations,
where perforations in adjacent layers are staggered (ABAB stacking).

Crystal System: Hexagonal (a = b, alpha = beta = 90 deg, gamma = 120 deg)
Space Group: P6_3/mmc (No. 194, Hall number 488)

Axis ordering: [a, b, c] - standard crystallographic convention
- a, b axes: in-plane hexagonal directions (perforation arrangement)
- c axis: lamellar stacking direction

Initial fields loaded from PL.mat (converged PSCF solution).

Reference:
- Loo et al., Macromolecules 2005, 38, 4947

Results:
- Free energy: F = -0.2100188
- Box size: lx = [1.963, 1.963, 2.952] (a = b, hexagonal, c/a ~ 1.504)
"""

import os
import numpy as np
import scipy.io
from scipy.ndimage import zoom
from polymerfts import scft

# OpenMP environment variables
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"  # 0, 1
os.environ["OMP_NUM_THREADS"] = "2"  # 1 ~ 4

# Major Simulation params
f = 0.4       # A-fraction of major BCP chain, f

params = {
    # HPL with Hexagonal crystal system and P6_3/mmc space group
    # Axis ordering: [a, b, c] - gamma=120 deg between a and b (axes 0,1)
    # IMPORTANT: Grid must be divisible by 6 for P6_3/mmc compatibility
    "nx": [48, 48, 72],             # Simulation grid numbers [a, b, c] - divisible by 6
    "lx": [1.96, 1.96, 2.98],       # Box size [a, b, c] with a=b (near equilibrium)
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
        "symbol": "P6_3/mmc",       # International symbol for HPL space group (No. 194)
        "number": 488,              # Hall number
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

# Load initial fields from PL.mat (converged PSCF solution)
# The .mat file uses axis ordering [c, a, b], we need [a, b, c]
mat_data = scipy.io.loadmat(os.path.join(os.path.dirname(__file__), "PL.mat"))
mat_nx = mat_data["nx"].flatten()       # [96, 64, 64] in [c, a, b] order
mat_w_A = mat_data["w_A"].reshape(mat_nx)  # shape: (96, 64, 64)
mat_w_B = mat_data["w_B"].reshape(mat_nx)

# Transpose from [c, a, b] to [a, b, c] and interpolate to target grid
mat_w_A = np.transpose(mat_w_A, (1, 2, 0))  # (64, 64, 96)
mat_w_B = np.transpose(mat_w_B, (1, 2, 0))

# Rescale fields from source chi_n to target chi_n
# Source: chi_n=50 (w_A mean≈30, w_B mean≈20), Target: chi_n=15
chi_n_source = 50
chi_n_target = params["chi_n"]["A,B"]
scale = chi_n_target / chi_n_source

w_A_src = mat_w_A * scale
w_B_src = mat_w_B * scale

# Interpolate to target grid
zoom_factors = [params["nx"][i] / w_A_src.shape[i] for i in range(3)]
w_A = zoom(w_A_src, zoom_factors, mode='wrap', order=3)
w_B = zoom(w_B_src, zoom_factors, mode='wrap', order=3)

print(f"Initial field loaded from PL.mat (rescaled chi_n: {chi_n_source} -> {chi_n_target})")
print(f"  w_A: min={w_A.min():.2f}, max={w_A.max():.2f}, mean={w_A.mean():.2f}")
print(f"  w_B: min={w_B.min():.2f}, max={w_B.max():.2f}, mean={w_B.mean():.2f}")

# Initialize calculation
calculation = scft.SCFT(params=params)

# Set a timer

# Run
calculation.run(initial_fields={"A": w_A, "B": w_B})

# Estimate execution time

# Save final results
calculation.save_results("PL_Hexagonal.json")

# Recording iteration results for debugging and refactoring
# Equilibrium: F = -0.2100188, lx = [1.9625, 1.9625, 2.9515], gamma = 120 deg
# Space group: P6_3/mmc (No. 194, Hall 488) - staggered perforations (ABAB)
# (with f=0.4, chi_n=15)
