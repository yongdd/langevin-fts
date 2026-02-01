"""
Non-Periodic Boundary Conditions: Mixed BC Propagator Example (2D)

This example demonstrates propagator computation with mixed boundary
conditions in 2D: reflecting BC in x-direction and absorbing BC in y-direction.

Physical interpretation:
- A polymer confined between two parallel reflecting walls (x-direction)
- With absorbing surfaces at top and bottom (y-direction)
- This models a thin film with impenetrable side walls and reactive top/bottom

This example uses the high-level PropagatorSolver API for simplicity.
"""

import os
import numpy as np

# OpenMP environment variables
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"

from polymerfts import PropagatorSolver


def main():
    # Simulation parameters
    NX, NY = 32, 24   # Grid points
    LX, LY = 4.0, 3.0 # Domain size
    ds = 0.01         # Contour step
    n_segments = 50   # Number of segments

    dx = LX / NX
    dy = LY / NY
    dV = dx * dy

    print("=" * 60)
    print("Mixed BC Propagator Computation Demo (2D)")
    print("=" * 60)
    print(f"Grid: {NX} x {NY}")
    print(f"Domain: {LX} x {LY}")
    print(f"Contour step: {ds}")
    print(f"Number of segments: {n_segments}")
    print()

    # Create solver with mixed BC
    # Format: [x_low, x_high, y_low, y_high]
    solver = PropagatorSolver(
        nx=[NX, NY], lx=[LX, LY],
        bc=["reflecting", "reflecting", "absorbing", "absorbing"],
        ds=ds,
        bond_lengths={"A": 1.0},
        chain_model="continuous",
        platform="cpu-fftw"  # Use CPU for 2D non-periodic BC
    )

    # Add a simple homopolymer
    solver.add_polymer(volume_fraction=1.0, blocks=[["A", 1.0, 0, 1]])

    # Display solver configuration
    print(solver.info)
    print(f"    x-direction: reflecting (walls)")
    print(f"    y-direction: absorbing (reactive surfaces)")
    print()

    # Zero potential field
    M = NX * NY
    w = np.zeros(M)
    solver.set_fields({"A": w})

    # Initial condition: Gaussian centered in domain
    q_init = np.zeros(M)
    for i in range(NX):
        x = (i + 0.5) * dx
        for j in range(NY):
            y = (j + 0.5) * dy
            idx = i * NY + j
            q_init[idx] = np.exp(-((x - LX/2)**2 + (y - LY/2)**2) / (2 * 0.5**2))

    initial_mass = np.sum(q_init) * dV

    print("Evolution with mixed BC:")
    print("-" * 60)
    print(f"{'Step':>6} {'Mass':>14} {'Mass Ratio':>14} {'x-symmetry':>14}")
    print("-" * 60)

    # Evolve propagator
    q = q_init.copy()
    for step in range(n_segments):
        q = solver.advance(q, "A")

        if (step + 1) % 10 == 0 or step == 0:
            current_mass = np.sum(q) * dV
            mass_ratio = current_mass / initial_mass

            # Check x-symmetry (should be preserved due to symmetric IC and reflecting BC)
            q_2d = q.reshape(NX, NY)
            sym_error = np.max(np.abs(q_2d - q_2d[::-1, :]))  # Flip in x

            print(f"{step+1:>6} {current_mass:>14.6e} {mass_ratio:>14.6f} {sym_error:>14.6e}")

    final_mass = np.sum(q) * dV
    mass_ratio = final_mass / initial_mass

    print("-" * 60)
    print(f"\nMixed BC behavior test:")
    print(f"  Initial mass: {initial_mass:.10e}")
    print(f"  Final mass:   {final_mass:.10e}")
    print(f"  Mass ratio:   {mass_ratio:.6f}")

    # The x-direction (reflecting) should preserve symmetry
    q_2d = q.reshape(NX, NY)
    x_sym_error = np.max(np.abs(q_2d - q_2d[::-1, :]))
    print(f"\n  X-symmetry error (reflecting BC): {x_sym_error:.10e}")

    if x_sym_error < 1e-10:
        print("  PASSED: X-symmetry preserved by reflecting BC")
    else:
        print("  Note: Some numerical asymmetry detected")

    if final_mass < initial_mass:
        print("  PASSED: Mass decreased due to absorbing BC in y-direction")
    else:
        print("  WARNING: Mass should decrease with absorbing BC")

    # Show profile along x at y=LY/2
    mid_j = NY // 2
    x_profile = q_2d[:, mid_j]

    print(f"\n  Profile along x at y={LY/2} (center):")
    print(f"  Max: {np.max(x_profile):.6e}")
    print(f"  Peak position: x = {(np.argmax(x_profile) + 0.5) * dx:.2f}")

    # Save results
    np.savez("mixed_bc_propagator_2d.npz",
             x=np.linspace(dx/2, LX-dx/2, NX),
             y=np.linspace(dy/2, LY-dy/2, NY),
             q_init=q_init.reshape(NX, NY),
             q_final=q.reshape(NX, NY))
    print(f"\nResults saved to mixed_bc_propagator_2d.npz")


if __name__ == "__main__":
    main()
