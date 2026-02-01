"""
Non-Periodic Boundary Conditions: Absorbing BC Propagator Example

This example demonstrates propagator computation with absorbing (Dirichlet)
boundary conditions using the pseudo-spectral method with DST.

Physical interpretation:
- Absorbing BC (Dirichlet): q = 0 at boundaries
- Polymer chains reaching the boundary are "absorbed"
- Mass decreases over time as chains diffuse to boundaries
- Uses Discrete Sine Transform (DST) instead of FFT

This simulates a homopolymer chain in a domain with absorbing walls,
such as a confined polymer near reactive surfaces.

This example uses the high-level PropagatorSolver API for simplicity.
"""

import os
import numpy as np

# OpenMP environment variables
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"

from polymerfts import PropagatorSolver


def main():
    # Simulation parameters
    N = 64            # Grid points
    L = 4.0           # Domain length
    ds = 0.01         # Contour step
    n_segments = 100  # Number of segments

    dx = L / N

    print("=" * 60)
    print("Absorbing BC Propagator Computation Demo")
    print("=" * 60)
    print(f"Grid points: {N}")
    print(f"Domain length: {L}")
    print(f"Contour step: {ds}")
    print(f"Number of segments: {n_segments}")
    print()

    # Create solver with absorbing BC
    solver = PropagatorSolver(
        nx=[N], lx=[L],
        bc=["absorbing", "absorbing"],
        ds=ds,
        bond_lengths={"A": 1.0},
        chain_model="continuous"
    )

    # Add a simple homopolymer
    solver.add_polymer(volume_fraction=1.0, blocks=[["A", 1.0, 0, 1]])

    # Display solver configuration
    print(solver.info)
    print()

    # Zero potential field
    w = np.zeros(N)
    solver.set_fields({"A": w})

    # Initial condition: Gaussian centered in domain
    q_init = np.zeros(N)
    for i in range(N):
        x = (i + 0.5) * dx
        q_init[i] = np.exp(-((x - L/2)**2) / (2 * 0.5**2))

    initial_mass = np.sum(q_init) * dx

    print("Evolution with absorbing BC:")
    print("-" * 40)
    print(f"{'Step':>6} {'Mass':>14} {'Mass Ratio':>14}")
    print("-" * 40)

    # Evolve propagator
    q = q_init.copy()
    for step in range(n_segments):
        q = solver.advance(q, "A")

        if (step + 1) % 20 == 0 or step == 0:
            current_mass = np.sum(q) * dx
            mass_ratio = current_mass / initial_mass
            print(f"{step+1:>6} {current_mass:>14.6e} {mass_ratio:>14.6f}")

    final_mass = np.sum(q) * dx
    mass_ratio = final_mass / initial_mass

    print("-" * 40)
    print(f"\nMass absorption test:")
    print(f"  Initial mass: {initial_mass:.10e}")
    print(f"  Final mass:   {final_mass:.10e}")
    print(f"  Mass ratio:   {mass_ratio:.6f}")

    if final_mass < initial_mass:
        print("  PASSED: Mass decreased due to absorbing BC")
    else:
        print("  WARNING: Mass should decrease with absorbing BC")

    # Compare with reflecting BC for reference
    print(f"\n--- Comparison with Reflecting BC ---")

    solver_reflect = PropagatorSolver(
        nx=[N], lx=[L],
        bc=["reflecting", "reflecting"],
        ds=ds,
        bond_lengths={"A": 1.0},
        chain_model="continuous"
    )
    solver_reflect.add_polymer(volume_fraction=1.0, blocks=[["A", 1.0, 0, 1]])
    solver_reflect.set_fields({"A": w})

    q_reflect = q_init.copy()
    for step in range(n_segments):
        q_reflect = solver_reflect.advance(q_reflect, "A")

    reflect_mass = np.sum(q_reflect) * dx

    print(f"  Absorbing BC final mass: {final_mass:.10e}")
    print(f"  Reflecting BC final mass: {reflect_mass:.10e}")
    print(f"  Absorbing/Reflecting ratio: {final_mass/reflect_mass:.6f}")

    # Save results
    np.savez("absorbing_bc_propagator.npz",
             x=np.linspace(dx/2, L-dx/2, N),
             q_init=q_init,
             q_final_absorbing=q,
             q_final_reflecting=q_reflect)
    print(f"\nResults saved to absorbing_bc_propagator.npz")


if __name__ == "__main__":
    main()
