"""
Angle Optimization Demonstration for Cylinder Phase

This script demonstrates that γ = 120° minimizes the free energy for the
hexagonal cylinder phase of an AB diblock copolymer.

We run SCFT calculations at different angles and show that:
1. Free energy has a minimum at γ = 120°
2. The stress tensor component σ_xy drives the angle toward this minimum
"""

import os
import sys
import io
import numpy as np
from scipy.ndimage import gaussian_filter
from polymerfts import scft

# OpenMP environment variables
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
os.environ["OMP_NUM_THREADS"] = "2"

# Parameters
f = 1.0/3.0  # A-fraction
chi_n = 18
L = 1.8
ds = 1/90

def run_scft_at_angle(gamma, max_iter=300, verbose=False):
    """Run SCFT at a specific angle and return free energy and stress."""

    params = {
        "nx": [32, 32],
        "lx": [L, L],
        "angles": [gamma],  # Single gamma for 2D
        "crystal_system": "Oblique2D",
        "box_is_altering": False,  # Fixed box for comparison
        "chain_model": "continuous",
        "ds": ds,
        "segment_lengths": {"A": 1.0, "B": 1.0},
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
            "mix_min": 0.1,
            "mix_init": 0.1,
        },
        "max_iter": max_iter,
        "tolerance": 1e-8
    }

    # Initialize cylinder at origin
    nx, ny = params["nx"]
    w_A = np.zeros([nx, ny], dtype=np.float64)
    gamma_rad = gamma * np.pi / 180.0
    cos_gamma = np.cos(gamma_rad)

    cylinder_radius = 0.2
    for i in range(nx):
        for j in range(ny):
            x_frac = i / nx
            y_frac = j / ny
            if x_frac > 0.5: x_frac -= 1.0
            if y_frac > 0.5: y_frac -= 1.0
            r_sq = x_frac**2 + y_frac**2 + 2*cos_gamma*x_frac*y_frac
            r = np.sqrt(max(0, r_sq))
            phi = 0.5 * (1.0 - np.tanh((r - cylinder_radius) / 0.05))
            w_A[i, j] = -phi

    w_A = gaussian_filter(w_A, sigma=1.0, mode='wrap')
    w_A = w_A / np.abs(w_A).max() * chi_n * 0.5
    w_B = -w_A

    # Suppress output during run
    if not verbose:
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

    calc = scft.SCFT(params=params)
    calc.run(initial_fields={"A": w_A, "B": w_B})

    if not verbose:
        sys.stdout = old_stdout

    # Get stress tensor
    calc.solver.compute_stress()
    stress = np.array(calc.solver.get_stress())

    return calc.free_energy, stress


print("="*70)
print("ANGLE OPTIMIZATION DEMONSTRATION")
print("Cylinder Phase of AB Diblock Copolymer")
print("="*70)
print(f"\nParameters: f = {f:.3f}, χN = {chi_n}, L = {L}")
print("\nScanning angles from 90° to 150° to find the free energy minimum...")
print("(γ = 120° corresponds to hexagonal packing)\n")

# Scan angles
angles = [90, 100, 110, 115, 118, 120, 122, 125, 130, 140, 150]
results = []

print(f"{'Angle (°)':<12} {'Free Energy':<15} {'σ_xy (stress)':<15} {'dF/dγ sign':<12}")
print("-" * 55)

for gamma in angles:
    F, stress = run_scft_at_angle(gamma)
    # σ_xy is stress[3] for 2D (index 3 in the 6-component stress tensor)
    sigma_xy = stress[3] if len(stress) > 3 else 0.0

    # The sign of σ_xy indicates the direction of angle gradient
    # Negative σ_xy → γ should increase; Positive σ_xy → γ should decrease
    gradient_sign = "← decrease" if sigma_xy > 0.001 else ("→ increase" if sigma_xy < -0.001 else "≈ 0 (min)")

    results.append((gamma, F, sigma_xy))
    print(f"{gamma:<12} {F:<15.6f} {sigma_xy:<15.6f} {gradient_sign:<12}")

# Find minimum
min_idx = np.argmin([r[1] for r in results])
optimal_angle = results[min_idx][0]
optimal_F = results[min_idx][1]

print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"\nFree energy minimum found at γ = {optimal_angle}°")
print(f"Minimum free energy: F = {optimal_F:.6f}")

# Compare with γ = 90° (orthogonal)
F_90 = next(r[1] for r in results if r[0] == 90)
F_improvement = F_90 - optimal_F
print(f"\nFree energy improvement from γ=90° to γ={optimal_angle}°:")
print(f"  ΔF = {F_improvement:.6f} kT per chain")

if abs(optimal_angle - 120) <= 5:
    print(f"\n✓ Optimal angle ({optimal_angle}°) is close to hexagonal (120°)")
    print("  This confirms that hexagonal packing minimizes the free energy.")
else:
    print(f"\n  Note: Optimal angle {optimal_angle}° differs from 120°")
    print("  This may be due to finite size effects or insufficient convergence.")

# Show the stress gradient behavior
print("\n" + "-"*70)
print("Stress Analysis:")
print("-"*70)
print("\nThe off-diagonal stress σ_xy acts as a 'force' on the angle:")
print("  • σ_xy < 0: System wants to increase γ (toward 120°)")
print("  • σ_xy > 0: System wants to decrease γ (toward 120°)")
print("  • σ_xy ≈ 0: System is at equilibrium angle")
print("\nObserved behavior:")
for gamma, F, sigma in results:
    direction = "↑" if sigma < -0.001 else ("↓" if sigma > 0.001 else "○")
    print(f"  γ = {gamma:3d}°: σ_xy = {sigma:+.4f} {direction}")
