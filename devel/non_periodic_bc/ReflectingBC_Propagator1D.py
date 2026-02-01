"""
Non-Periodic Boundary Conditions: Reflecting BC Propagator Example

This example demonstrates propagator computation with reflecting (Neumann)
boundary conditions using the pseudo-spectral method with DCT.

Physical interpretation:
- Reflecting BC (Neumann): Zero flux at boundaries
- The chain propagator satisfies dq/dn = 0 at boundaries
- Mass (integral of propagator) is conserved
- Uses Discrete Cosine Transform (DCT) instead of FFT

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
    n_segments = 100  # Number of segments (corresponds to N=100)

    dx = L / N

    print("=" * 60)
    print("Reflecting BC Propagator Computation Demo")
    print("=" * 60)
    print(f"Grid points: {N}")
    print(f"Domain length: {L}")
    print(f"Contour step: {ds}")
    print(f"Number of segments: {n_segments}")
    print()

    # Create solver with reflecting BC
    # BC format: [x_low, x_high] for 1D
    solver = PropagatorSolver(
        nx=[N], lx=[L],
        bc=["reflecting", "reflecting"],
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

    # Initial condition: Gaussian centered at L/4 (not symmetric about L/2)
    # This tests that reflecting BC properly confines the diffusion
    q_init = np.zeros(N)
    for i in range(N):
        x = (i + 0.5) * dx
        q_init[i] = np.exp(-((x - L/4)**2) / (2 * 0.3**2))

    initial_mass = np.sum(q_init) * dx

    print("Evolution with reflecting BC:")
    print("-" * 40)
    print(f"{'Step':>6} {'Mass':>14} {'Mass Error':>14}")
    print("-" * 40)

    # Evolve propagator
    q = q_init.copy()
    for step in range(n_segments):
        q = solver.advance(q, "A")

        if (step + 1) % 20 == 0 or step == 0:
            current_mass = np.sum(q) * dx
            mass_error = abs(current_mass - initial_mass) / initial_mass
            print(f"{step+1:>6} {current_mass:>14.6e} {mass_error:>14.6e}")

    final_mass = np.sum(q) * dx
    mass_error = abs(final_mass - initial_mass) / initial_mass

    print("-" * 40)
    print(f"\nMass conservation test:")
    print(f"  Initial mass: {initial_mass:.10e}")
    print(f"  Final mass:   {final_mass:.10e}")
    print(f"  Relative error: {mass_error:.10e}")

    if mass_error < 1e-6:
        print("  PASSED: Mass is conserved (error < 1e-6)")
    else:
        print("  WARNING: Mass conservation error is larger than expected")

    # Verify the propagator spread over time but stayed within boundaries
    print(f"\nPropagator evolution:")
    print(f"  Initial peak position: x = {L/4:.2f}")
    print(f"  Initial max value: {np.max(q_init):.6f}")
    print(f"  Final max value: {np.max(q):.6f}")
    print(f"  Propagator diffused but mass conserved by reflecting BC")

    # Save results for visualization
    np.savez("reflecting_bc_propagator.npz",
             x=np.linspace(dx/2, L-dx/2, N),
             q_init=q_init,
             q_final=q)
    print(f"\nResults saved to reflecting_bc_propagator.npz")


if __name__ == "__main__":
    main()
