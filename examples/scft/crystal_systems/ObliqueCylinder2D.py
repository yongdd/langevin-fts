"""
Oblique 2D crystal system example: AB diblock copolymer cylinder phase
with angle optimization.

This demonstrates:
1. Starting from a non-optimal angle (γ = 100°)
2. Box relaxation optimizing lengths (a, b) and angle (γ)
3. The system naturally finding the optimal hexagonal arrangement (γ → 120°)

For cylinder phase with hexagonal packing:
- Optimal γ = 120° gives equidistant nearest-neighbor cylinders
- Optimal a = b for hexagonal symmetry
"""

import os
import time
import numpy as np
from scipy.ndimage import gaussian_filter
from polymerfts import scft

# OpenMP environment variables
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
os.environ["OMP_NUM_THREADS"] = "2"

# Parameters
f = 1.0/3.0  # A-fraction - minority forms cylinders
chi_n = 18   # Flory-Huggins parameter

# Start with deliberately non-optimal angle
initial_gamma = 100.0  # Will optimize toward 120° for hexagonal packing

# Lattice constant
L = 1.8

params = {
    # Grid and box settings
    "nx": [32, 32],
    "lx": [L, L],
    "angles": [90.0, 90.0, initial_gamma],  # Only gamma matters for 2D

    "crystal_system": "Oblique2D",  # Enable 2D angle optimization
    "box_is_altering": True,        # Optimize box size and angle

    "chain_model": "continuous",
    "ds": 1/90,

    "segment_lengths": {
        "A": 1.0,
        "B": 1.0,
    },

    "chi_n": {"A,B": chi_n},

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
        "mix_min": 0.05,
        "mix_init": 0.05,
    },

    "scale_stress": 0.3,  # Scale factor for stress-driven box optimization

    "max_iter": 500,
    "tolerance": 1e-7
}

# Set initial fields - cylinder at origin
print("Initializing cylinder phase...")
nx, ny = params["nx"]
w_A = np.zeros([nx, ny], dtype=np.float64)

cylinder_radius = 0.2  # Fractional radius
gamma_rad = initial_gamma * np.pi / 180.0
cos_gamma = np.cos(gamma_rad)

for i in range(nx):
    for j in range(ny):
        x_frac = i / nx
        y_frac = j / ny

        # Apply periodic wrapping
        if x_frac > 0.5:
            x_frac -= 1.0
        if y_frac > 0.5:
            y_frac -= 1.0

        # Cartesian distance using metric
        r_sq = x_frac**2 + y_frac**2 + 2*cos_gamma*x_frac*y_frac
        r = np.sqrt(max(0, r_sq))

        # Cylinder profile
        interface_width = 0.05
        phi = 0.5 * (1.0 - np.tanh((r - cylinder_radius) / interface_width))
        w_A[i, j] = -phi

# Smooth and normalize
w_A = gaussian_filter(w_A, sigma=1.0, mode='wrap')
w_A = w_A / np.abs(w_A).max() * chi_n * 0.5
w_B = -w_A

# Initialize
print("\n" + "="*60)
print("Oblique 2D Crystal System SCFT - Cylinder Phase")
print("="*60)
print(f"\nStarting with NON-OPTIMAL angle γ = {initial_gamma:.1f}°")
print(f"Expected optimal angle: γ → 120° for hexagonal packing")
print(f"\nInitial box: lx = {params['lx']}")
print(f"Parameters: f = {f:.3f}, χN = {chi_n}")

calculation = scft.SCFT(params=params)

time_start = time.time()
calculation.run(initial_fields={"A": w_A, "B": w_B})
time_duration = time.time() - time_start

print(f"\nTotal time: {time_duration:.2f} seconds")

# Results
final_angles = calculation.cb.get_angles_degrees()
final_lx = calculation.cb.get_lx()

print("\n" + "="*60)
print("RESULTS - Angle Optimization")
print("="*60)
print(f"\nFinal box: lx = [{final_lx[0]:.4f}, {final_lx[1]:.4f}]")
print(f"\nAngle optimization:")
print(f"  γ: {initial_gamma:.1f}° → {final_angles[2]:.2f}° (target: 120° for hex)")

gamma_error = abs(final_angles[2] - 120.0)
print(f"\n  |γ - 120°| = {gamma_error:.2f}°")

if gamma_error < 5.0:
    print("\n  SUCCESS: Angle optimized toward hexagonal arrangement!")
else:
    print(f"\n  Note: May need more iterations or different initial conditions.")

print(f"\nFree energy: {calculation.free_energy:.6f}")
