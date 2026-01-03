"""
Triclinic crystal system example: AB diblock copolymer cylinder phase
with full angle optimization.

This demonstrates:
1. Starting from a non-optimal tilted unit cell
2. Box relaxation optimizing all lengths (a,b,c) and all angles (α,β,γ)
3. The system naturally finding the optimal hexagonal arrangement (γ → 120°)

For cylinder phase with hexagonal packing:
- Optimal γ = 120° in the plane perpendicular to cylinder axis
- Optimal α = β = 90° (cylinders perpendicular to the tilted plane)
"""

import os
import time
import numpy as np
from scipy.ndimage import gaussian_filter
from polymerfts import scft

# OpenMP environment variables
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
os.environ["OMP_NUM_THREADS"] = "2"

# Major Simulation params
f = 1.0/3.0  # A-fraction - minority forms cylinders

# Start with deliberately non-optimal angles to demonstrate optimization
initial_gamma = 100.0  # Will optimize toward 120° for hexagonal packing

# Estimate optimal lattice constant for cylinder phase
# For hexagonal cylinder at chi_n=18, L ~ 1.8 is reasonable
L = 1.8

params = {
    # Grid and box settings - xy plane will have hexagonal cylinder packing
    "nx": [24, 24, 8],           # Fewer points in z (cylinder axis direction)
    "lx": [L, L, L/2],           # Shorter in z direction
    "angles": [90.0, 90.0, initial_gamma],  # Start with non-optimal γ

    "crystal_system": "Triclinic",  # Enable full angle optimization
    "box_is_altering": True,        # Optimize box size and angles

    "chain_model": "continuous",
    "ds": 1/90,

    "segment_lengths": {
        "A": 1.0,
        "B": 1.0,
    },

    "chi_n": {"A,B": 18},  # High enough for well-defined cylinders

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

    "scale_stress": 0.2,  # Scale factor for stress-driven box optimization

    "max_iter": 500,
    "tolerance": 1e-7
}

# Set initial fields - cylinder at origin extending along z
w_A = np.zeros(list(params["nx"]), dtype=np.float64)

# Create a cylinder seed at the origin
print("Initializing cylinder phase...")
nx, ny, nz = params["nx"]
cylinder_radius = 0.2  # Fractional radius

for i in range(nx):
    for j in range(ny):
        # Position in fractional coordinates
        x_frac = i / nx
        y_frac = j / ny

        # Apply periodic wrapping
        if x_frac > 0.5:
            x_frac -= 1.0
        if y_frac > 0.5:
            y_frac -= 1.0

        # Convert to Cartesian distance using metric (for initial_gamma angle)
        gamma_rad = initial_gamma * np.pi / 180.0
        cos_gamma = np.cos(gamma_rad)
        r_sq = x_frac**2 + y_frac**2 + 2*cos_gamma*x_frac*y_frac
        r = np.sqrt(max(0, r_sq))

        # Cylinder profile along z
        interface_width = 0.05
        phi = 0.5 * (1.0 - np.tanh((r - cylinder_radius) / interface_width))
        w_A[i, j, :] = -phi  # Negative for A-attracting

# Smooth the initial condition
w_A = gaussian_filter(w_A, sigma=1.0, mode='wrap')

# Normalize to have proper amplitude
w_A = w_A / np.abs(w_A).max() * params["chi_n"]["A,B"] * 0.5

w_B = -w_A  # Complementary field for B

# Initialize calculation
print("\n" + "="*60)
print("Triclinic Crystal System SCFT - Cylinder Phase")
print("="*60)
print(f"\nStarting with NON-OPTIMAL angles to test optimization:")
print(f"  Initial α = {params['angles'][0]:.1f}° (expect → 90°)")
print(f"  Initial β = {params['angles'][1]:.1f}° (expect → 90°)")
print(f"  Initial γ = {params['angles'][2]:.1f}° (expect → 120° for hex)")
print(f"\nInitial box: lx = {params['lx']}")
print(f"Parameters: f = {f:.3f}, χN = {params['chi_n']['A,B']}")

calculation = scft.SCFT(params=params)

# Set a timer
time_start = time.time()

# Run
calculation.run(initial_fields={"A": w_A, "B": w_B})

# Estimate execution time
time_duration = time.time() - time_start
print(f"\nTotal time: {time_duration:.2f} seconds")

# Print final box parameters
final_angles = calculation.cb.get_angles_degrees()
final_lx = calculation.cb.get_lx()

print("\n" + "="*60)
print("RESULTS - Angle Optimization")
print("="*60)
print(f"\nFinal box lengths: [{final_lx[0]:.4f}, {final_lx[1]:.4f}, {final_lx[2]:.4f}]")
print(f"\nAngle optimization results:")
print(f"  α: {params['angles'][0]:.1f}° → {final_angles[0]:.2f}° (target: 90°)")
print(f"  β: {params['angles'][1]:.1f}° → {final_angles[1]:.2f}° (target: 90°)")
print(f"  γ: {params['angles'][2]:.1f}° → {final_angles[2]:.2f}° (target: 120° for hex)")

# Check if angles converged to expected values
alpha_error = abs(final_angles[0] - 90.0)
beta_error = abs(final_angles[1] - 90.0)
gamma_error = abs(final_angles[2] - 120.0)

print(f"\nAngle errors from optimal:")
print(f"  |α - 90°| = {alpha_error:.2f}°")
print(f"  |β - 90°| = {beta_error:.2f}°")
print(f"  |γ - 120°| = {gamma_error:.2f}°")

if alpha_error < 5.0 and beta_error < 5.0 and gamma_error < 10.0:
    print("\n✓ Angles successfully optimized toward hexagonal arrangement!")
else:
    print("\n⚠ Angles may need more iterations to fully converge.")

print(f"\nFree energy: {calculation.free_energy:.6f}")
print(f"Is orthogonal: {calculation.cb.is_orthogonal()}")
