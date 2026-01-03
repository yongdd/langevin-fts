"""
Demonstration of non-trivial optimal angle under constraint.

When the box dimensions are constrained to non-optimal values,
the optimal angle can deviate from both 90° and 120°.

This shows that the angle optimization genuinely minimizes free energy
given the constraints, rather than just converging to symmetric values.
"""

import os
import sys
import io
import numpy as np
from scipy.ndimage import gaussian_filter
from polymerfts import scft

os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
os.environ["OMP_NUM_THREADS"] = "2"

# Parameters
f = 1.0/3.0
chi_n = 18
ds = 1/90

def run_scft_at_angle(lx, ly, gamma, max_iter=300):
    """Run SCFT at specific box dimensions and angle."""
    params = {
        "nx": [32, 32],
        "lx": [lx, ly],
        "angles": [90.0, 90.0, gamma],
        "crystal_system": "Oblique2D",
        "box_is_altering": False,
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

    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    calc = scft.SCFT(params=params)
    calc.run(initial_fields={"A": w_A, "B": w_B})
    sys.stdout = old_stdout

    return calc.free_energy


print("="*70)
print("NON-TRIVIAL OPTIMAL ANGLE DEMONSTRATION")
print("="*70)

# Case 1: Symmetric box (a = b) - optimal should be ~120°
print("\n" + "-"*70)
print("Case 1: Symmetric box (a = b = 1.8)")
print("-"*70)
print("For symmetric box, optimal γ should be 120° (hexagonal)")

L = 1.8
angles_1 = list(range(100, 145, 5))
results_1 = []
for gamma in angles_1:
    F = run_scft_at_angle(L, L, gamma)
    results_1.append((gamma, F))
    print(f"  γ = {gamma:3d}°: F = {F:.6f}")

min_idx_1 = np.argmin([r[1] for r in results_1])
print(f"\n  Optimal angle: γ = {results_1[min_idx_1][0]}°")

# Case 2: Asymmetric box (a ≠ b) - optimal may deviate from 120°
print("\n" + "-"*70)
print("Case 2: Asymmetric box (a = 2.0, b = 1.5)")
print("-"*70)
print("For asymmetric box, optimal γ may deviate from 120°")

lx, ly = 2.0, 1.5
angles_2 = list(range(80, 145, 5))
results_2 = []
for gamma in angles_2:
    F = run_scft_at_angle(lx, ly, gamma)
    results_2.append((gamma, F))
    print(f"  γ = {gamma:3d}°: F = {F:.6f}")

min_idx_2 = np.argmin([r[1] for r in results_2])
print(f"\n  Optimal angle: γ = {results_2[min_idx_2][0]}°")

# Case 3: Another asymmetric case
print("\n" + "-"*70)
print("Case 3: Asymmetric box (a = 1.5, b = 2.2)")
print("-"*70)

lx, ly = 1.5, 2.2
angles_3 = list(range(80, 145, 5))
results_3 = []
for gamma in angles_3:
    F = run_scft_at_angle(lx, ly, gamma)
    results_3.append((gamma, F))
    print(f"  γ = {gamma:3d}°: F = {F:.6f}")

min_idx_3 = np.argmin([r[1] for r in results_3])
print(f"\n  Optimal angle: γ = {results_3[min_idx_3][0]}°")

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"\nCase 1 (a=b=1.8):     Optimal γ = {results_1[min_idx_1][0]}° (expected ~120°)")
print(f"Case 2 (a=2.0, b=1.5): Optimal γ = {results_2[min_idx_2][0]}°")
print(f"Case 3 (a=1.5, b=2.2): Optimal γ = {results_3[min_idx_3][0]}°")

if results_2[min_idx_2][0] != 120 or results_3[min_idx_3][0] != 120:
    print("\n✓ Non-trivial angles found for asymmetric boxes!")
    print("  This demonstrates that the optimal angle depends on box geometry,")
    print("  not just the phase symmetry.")
