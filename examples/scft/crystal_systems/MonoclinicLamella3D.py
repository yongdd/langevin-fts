"""
Monoclinic crystal system example: AB diblock copolymer lamella phase
with angle optimization.

This demonstrates:
1. Non-orthogonal unit cell with β ≠ 90°
2. Box relaxation optimizing both lengths (a,b,c) and angle (β)
"""

import os
import time
import numpy as np
from polymerfts import scft

# OpenMP environment variables
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
os.environ["OMP_NUM_THREADS"] = "2"

# Major Simulation params
f = 0.5  # A-fraction of diblock chain

params = {
    # Grid and box settings
    "nx": [16, 16, 32],           # Simulation grid numbers
    "lx": [2.0, 2.0, 4.0],        # Box lengths
    "angles": [80.0, 90.0, 90.0],  # [α, β, γ] in degrees - Monoclinic has β ≠ 90°

    "crystal_system": "Monoclinic",  # Enable Monoclinic optimization (optimizes β angle)
    "box_is_altering": True,         # Optimize box size and angle

    "chain_model": "continuous",
    "ds": 1/50,

    "segment_lengths": {
        "A": 1.0,
        "B": 1.0,
    },

    "chi_n": {"A,B": 15},

    "distinct_polymers": [{
        "volume_fraction": 1.0,
        "blocks": [
            {"type": "A", "length": f},
            {"type": "B", "length": 1-f},
        ],
    }],

    "optimizer": {
        "name": "am",
        "max_hist": 20,
        "start_error": 1e-2,
        "mix_min": 0.02,
        "mix_init": 0.02,
    },

    "scale_stress": 0.1,  # Scale factor for stress-driven box optimization

    "max_iter": 200,
    "tolerance": 1e-7
}

# Set initial fields - lamellar structure along z
w_A = np.zeros(list(params["nx"]), dtype=np.float64)
w_B = np.zeros(list(params["nx"]), dtype=np.float64)
print("w_A and w_B are initialized to lamellar phase along z.")
for k in range(params["nx"][2]):
    zz = k * 2 * np.pi / params["nx"][2]
    w_A[:, :, k] = np.cos(zz) * 1.0
    w_B[:, :, k] = -np.cos(zz) * 1.0

# Initialize calculation
print("\n=== Monoclinic Crystal System SCFT ===")
print(f"Initial angles: α={params['angles'][0]}°, β={params['angles'][1]}°, γ={params['angles'][2]}°")
print(f"Initial box: lx={params['lx']}")
calculation = scft.SCFT(params=params)

# Set a timer
time_start = time.time()

# Run
calculation.run(initial_fields={"A": w_A, "B": w_B})

# Estimate execution time
time_duration = time.time() - time_start
print(f"\nTotal time: {time_duration:.2f} seconds")

# Print final box parameters
print(f"\nFinal box lengths: {calculation.cb.get_lx()}")
print(f"Final angles: α={calculation.cb.get_angles_degrees()[0]:.2f}°, β={calculation.cb.get_angles_degrees()[1]:.2f}°, γ={calculation.cb.get_angles_degrees()[2]:.2f}°")
print(f"Is orthogonal: {calculation.cb.is_orthogonal()}")
