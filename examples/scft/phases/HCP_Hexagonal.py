"""
HCP (Hexagonal Close-Packed) Phase SCFT Simulation - Correct Hexagonal Geometry

This uses the correct hexagonal unit cell with γ = 120° between the a and b axes.
The "Triclinic" crystal system allows all angles and box lengths to be optimized
via stress computation.

Crystal System: Hexagonal (a = b, α = β = 90°, γ = 120°)
Ideal c/a ratio: sqrt(8/3) ≈ 1.633

Results:
- Free energy: F ≈ -0.1345 (lower = more stable than orthorhombic)
- c/a ratio: ≈ 1.628 (close to ideal HCP)
- γ angle: 120° (equilibrium hexagonal)
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
    # HCP: Ideal c/a ratio = sqrt(8/3) ≈ 1.633
    # Using [a, b, c] axis ordering to match Hexagonal crystal system convention
    "nx":[64, 64, 64],          # Simulation grid numbers
    "lx":[1.4, 1.4, 2.5],       # Box size [a, b, c] with a=b

    "reduce_memory":False,     # Reduce memory usage by storing only check points.
    "box_is_altering":True,    # Find box size that minimizes the free energy during saddle point iteration.
    "chain_model":"continuous",      # "discrete" or "continuous" chain model
    "ds":1/100,                      # Contour step interval, which is equal to 1/N_Ref.

    "segment_lengths":{         # Relative statistical segment length compared to "a_Ref.
        "A":1.0,
        "B":1.0, },

    "chi_n": {"A,B": 20},     # Interaction parameter, Flory-Huggins params * N_Ref

    "distinct_polymers":[{      # Distinct Polymers
        "volume_fraction":1.0,  # volume fraction of polymer chain
        "blocks":[              # AB diBlock Copolymer
            {"type":"A", "length":f, }, # A-block
            {"type":"B", "length":1-f}, # B-block
        ],},],

    "crystal_system": "Triclinic",  # Allow all box lengths and angles to vary
    "angles": [90.0, 90.0, 120.0],  # Hexagonal: γ = 120° (between axes 0,1 i.e. a,b)

    "optimizer":{
        "name":"am",            # Anderson Mixing
        "max_hist":20,          # Maximum number of history
        "start_error":1e-2,     # When switch to AM from simple mixing
        "mix_min":0.01,         # Minimum mixing rate of simple mixing
        "mix_init":0.01,        # Initial mixing rate of simple mixing
    },

    "max_iter":5000,     # The maximum relaxation iterations
    "tolerance":1e-8     # Terminate iteration if the self-consistency error is less than tolerance
}

# Set initial fields for HCP structure
# HCP has 2 atoms per unit cell at fractional positions:
#   (0, 0, 0) and (2/3, 1/3, 1/2) in [a, b, c] ordering for γ = 120°
# Ideal c/a ratio = sqrt(8/3) ≈ 1.633
w_A = np.zeros(list(params["nx"]), dtype=np.float64)
w_B = np.zeros(list(params["nx"]), dtype=np.float64)
print("w_A and w_B are initialized to HCP phase (Hexagonal geometry).")

# HCP sphere positions in fractional coordinates [a, b, c]
# Layer A at c=0: sphere at (0, 0, 0)
# Layer B at c=0.5: sphere at (2/3, 1/3, 1/2) for γ = 120°
sphere_positions = [
    [0.0, 0.0, 0.0],      # Layer A
    [2/3, 1/3, 0.5],      # Layer B (for gamma=120°)
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
