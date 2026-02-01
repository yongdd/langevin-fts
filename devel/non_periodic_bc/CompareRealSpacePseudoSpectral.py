"""
Compare Real-Space and Pseudo-Spectral Methods for Non-Periodic BCs

This test compares propagator computation using two different numerical methods:
1. Pseudo-spectral method: DCT/DST transforms (high accuracy for smooth solutions)
2. Real-space method: Crank-Nicolson finite differences (more general BCs)

Both methods should converge to the same solution as grid resolution increases.
The pseudo-spectral method is typically more accurate for the same grid size,
while the real-space method is more flexible for complex boundary conditions.

Boundary conditions tested:
- Reflecting (Neumann): dq/dn = 0 at boundaries
- Absorbing (Dirichlet): q = 0 at boundaries
- Mixed: Reflecting in x, Absorbing in y (2D)

This example uses the high-level PropagatorSolver API for simplicity.
"""

import os
import numpy as np

# OpenMP environment variables
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from polymerfts import PropagatorSolver


def run_comparison_1d(bc_type, n_grids=[32, 64, 128]):
    """
    Compare real-space and pseudo-spectral methods in 1D.

    Args:
        bc_type: "reflecting" or "absorbing"
        n_grids: List of grid sizes to test convergence
    """
    L = 4.0           # Domain length
    ds = 0.01         # Contour step
    n_segments = 50   # Number of propagator steps

    print(f"\n{'='*70}")
    print(f"1D Comparison: {bc_type.upper()} BC")
    print(f"{'='*70}")
    print(f"Domain: [0, {L}], ds={ds}, segments={n_segments}")
    print(f"{'Grid':>8} {'PS Mass':>14} {'RS Mass':>14} {'Max Diff':>14} {'Rel Err':>14}")
    print("-" * 70)

    results = []

    for N in n_grids:
        dx = L / N

        # BC for both boundaries
        bc = [bc_type, bc_type]

        # Create pseudo-spectral solver
        solver_ps = PropagatorSolver(
            nx=[N], lx=[L], bc=bc, ds=ds,
            bond_lengths={"A": 1.0}, chain_model="continuous",
            numerical_method="rqm4", platform="cpu-fftw"
        )
        solver_ps.add_polymer(1.0, [["A", 1.0, 0, 1]])

        # Create real-space solver
        solver_rs = PropagatorSolver(
            nx=[N], lx=[L], bc=bc, ds=ds,
            bond_lengths={"A": 1.0}, chain_model="continuous",
            numerical_method="cn-adi2", platform="cpu-fftw"
        )
        solver_rs.add_polymer(1.0, [["A", 1.0, 0, 1]])

        # Initialize with zero potential
        w = np.zeros(N)
        solver_ps.set_fields({"A": w})
        solver_rs.set_fields({"A": w})

        # Gaussian initial condition centered in domain
        q_init = np.zeros(N)
        for i in range(N):
            x = (i + 0.5) * dx
            q_init[i] = np.exp(-((x - L/2)**2) / (2 * 0.4**2))

        # Evolve propagators
        q_ps = q_init.copy()
        q_rs = q_init.copy()

        for step in range(n_segments):
            q_ps = solver_ps.advance(q_ps, "A")
            q_rs = solver_rs.advance(q_rs, "A")

        # Compare results
        mass_ps = np.sum(q_ps) * dx
        mass_rs = np.sum(q_rs) * dx
        max_diff = np.max(np.abs(q_ps - q_rs))
        rel_err = max_diff / np.max(np.abs(q_ps)) if np.max(np.abs(q_ps)) > 0 else 0

        print(f"{N:>8} {mass_ps:>14.6e} {mass_rs:>14.6e} {max_diff:>14.6e} {rel_err:>14.6e}")
        results.append((N, mass_ps, mass_rs, max_diff, rel_err))

    # Check convergence (error should decrease with finer grid)
    if len(results) >= 2:
        first_err = results[0][4]
        last_err = results[-1][4]
        if last_err < first_err:
            print(f"\nCONVERGENCE: Error decreased from {first_err:.2e} to {last_err:.2e}")
        else:
            print(f"\nWARNING: Error did not decrease as expected")

    return results


def run_comparison_2d(bc_x, bc_y, n_grids=[(16, 12), (32, 24), (64, 48)]):
    """
    Compare real-space and pseudo-spectral methods in 2D.

    Args:
        bc_x: BC type for x-direction ("reflecting", "absorbing", "periodic")
        bc_y: BC type for y-direction
        n_grids: List of (nx, ny) grid sizes
    """
    LX, LY = 4.0, 3.0  # Domain size
    ds = 0.01          # Contour step
    n_segments = 30    # Number of propagator steps

    print(f"\n{'='*70}")
    print(f"2D Comparison: X={bc_x.upper()}, Y={bc_y.upper()}")
    print(f"{'='*70}")
    print(f"Domain: [0,{LX}] x [0,{LY}], ds={ds}, segments={n_segments}")
    print(f"{'Grid':>12} {'PS Mass':>14} {'RS Mass':>14} {'Max Diff':>14} {'Rel Err':>14}")
    print("-" * 70)

    results = []

    for (NX, NY) in n_grids:
        dx = LX / NX
        dy = LY / NY
        dV = dx * dy
        M = NX * NY

        # BC format: [x_low, x_high, y_low, y_high]
        bc = [bc_x, bc_x, bc_y, bc_y]

        # Create pseudo-spectral solver
        solver_ps = PropagatorSolver(
            nx=[NX, NY], lx=[LX, LY], bc=bc, ds=ds,
            bond_lengths={"A": 1.0}, chain_model="continuous",
            numerical_method="rqm4", platform="cpu-fftw"
        )
        solver_ps.add_polymer(1.0, [["A", 1.0, 0, 1]])

        # Create real-space solver
        solver_rs = PropagatorSolver(
            nx=[NX, NY], lx=[LX, LY], bc=bc, ds=ds,
            bond_lengths={"A": 1.0}, chain_model="continuous",
            numerical_method="cn-adi2", platform="cpu-fftw"
        )
        solver_rs.add_polymer(1.0, [["A", 1.0, 0, 1]])

        # Initialize with zero potential
        w = np.zeros(M)
        solver_ps.set_fields({"A": w})
        solver_rs.set_fields({"A": w})

        # Gaussian initial condition centered in domain
        q_init = np.zeros(M)
        for i in range(NX):
            x = (i + 0.5) * dx
            for j in range(NY):
                y = (j + 0.5) * dy
                idx = i * NY + j
                q_init[idx] = np.exp(-((x - LX/2)**2 + (y - LY/2)**2) / (2 * 0.4**2))

        # Evolve propagators
        q_ps = q_init.copy()
        q_rs = q_init.copy()

        for step in range(n_segments):
            q_ps = solver_ps.advance(q_ps, "A")
            q_rs = solver_rs.advance(q_rs, "A")

        # Compare results
        mass_ps = np.sum(q_ps) * dV
        mass_rs = np.sum(q_rs) * dV
        max_diff = np.max(np.abs(q_ps - q_rs))
        rel_err = max_diff / np.max(np.abs(q_ps)) if np.max(np.abs(q_ps)) > 0 else 0

        grid_str = f"{NX}x{NY}"
        print(f"{grid_str:>12} {mass_ps:>14.6e} {mass_rs:>14.6e} {max_diff:>14.6e} {rel_err:>14.6e}")
        results.append((NX, NY, mass_ps, mass_rs, max_diff, rel_err))

    return results


def run_comparison_3d(bc_type, n_grids=[(8, 8, 8), (16, 16, 16)]):
    """
    Compare real-space and pseudo-spectral methods in 3D.
    """
    LX, LY, LZ = 4.0, 4.0, 4.0  # Domain size
    ds = 0.01                    # Contour step
    n_segments = 20              # Number of propagator steps

    print(f"\n{'='*70}")
    print(f"3D Comparison: {bc_type.upper()} BC")
    print(f"{'='*70}")
    print(f"Domain: [0,{LX}]^3, ds={ds}, segments={n_segments}")
    print(f"{'Grid':>14} {'PS Mass':>14} {'RS Mass':>14} {'Max Diff':>14} {'Rel Err':>14}")
    print("-" * 72)

    results = []

    for (NX, NY, NZ) in n_grids:
        dx = LX / NX
        dy = LY / NY
        dz = LZ / NZ
        dV = dx * dy * dz
        M = NX * NY * NZ

        # BC format: [x_low, x_high, y_low, y_high, z_low, z_high]
        bc = [bc_type, bc_type, bc_type, bc_type, bc_type, bc_type]

        # Create pseudo-spectral solver
        solver_ps = PropagatorSolver(
            nx=[NX, NY, NZ], lx=[LX, LY, LZ], bc=bc, ds=ds,
            bond_lengths={"A": 1.0}, chain_model="continuous",
            numerical_method="rqm4", platform="cpu-fftw"
        )
        solver_ps.add_polymer(1.0, [["A", 1.0, 0, 1]])

        # Create real-space solver
        solver_rs = PropagatorSolver(
            nx=[NX, NY, NZ], lx=[LX, LY, LZ], bc=bc, ds=ds,
            bond_lengths={"A": 1.0}, chain_model="continuous",
            numerical_method="cn-adi2", platform="cpu-fftw"
        )
        solver_rs.add_polymer(1.0, [["A", 1.0, 0, 1]])

        # Initialize with zero potential
        w = np.zeros(M)
        solver_ps.set_fields({"A": w})
        solver_rs.set_fields({"A": w})

        # Gaussian initial condition centered in domain
        q_init = np.zeros(M)
        for i in range(NX):
            x = (i + 0.5) * dx
            for j in range(NY):
                y = (j + 0.5) * dy
                for k in range(NZ):
                    z = (k + 0.5) * dz
                    idx = (i * NY + j) * NZ + k
                    q_init[idx] = np.exp(-((x - LX/2)**2 + (y - LY/2)**2 + (z - LZ/2)**2) / (2 * 0.5**2))

        # Evolve propagators
        q_ps = q_init.copy()
        q_rs = q_init.copy()

        for step in range(n_segments):
            q_ps = solver_ps.advance(q_ps, "A")
            q_rs = solver_rs.advance(q_rs, "A")

        # Compare results
        mass_ps = np.sum(q_ps) * dV
        mass_rs = np.sum(q_rs) * dV
        max_diff = np.max(np.abs(q_ps - q_rs))
        rel_err = max_diff / np.max(np.abs(q_ps)) if np.max(np.abs(q_ps)) > 0 else 0

        grid_str = f"{NX}x{NY}x{NZ}"
        print(f"{grid_str:>14} {mass_ps:>14.6e} {mass_rs:>14.6e} {max_diff:>14.6e} {rel_err:>14.6e}")
        results.append((NX, NY, NZ, mass_ps, mass_rs, max_diff, rel_err))

    return results


def main():
    print("=" * 70)
    print("Real-Space vs Pseudo-Spectral Comparison for Non-Periodic BCs")
    print("=" * 70)
    print("\nPseudo-spectral: DCT (reflecting) / DST (absorbing)")
    print("Real-space: Crank-Nicolson finite differences (ADI)")
    print("\nNote: Both methods should converge to the same solution.")
    print("The pseudo-spectral method has higher accuracy for smooth solutions.")

    all_passed = True

    # 1D Tests
    print("\n" + "=" * 70)
    print("1D TESTS")
    print("=" * 70)

    results_1d_reflect = run_comparison_1d("reflecting")
    results_1d_absorb = run_comparison_1d("absorbing")

    # 2D Tests
    print("\n" + "=" * 70)
    print("2D TESTS")
    print("=" * 70)

    results_2d_reflect = run_comparison_2d("reflecting", "reflecting")
    results_2d_absorb = run_comparison_2d("absorbing", "absorbing")
    results_2d_mixed = run_comparison_2d("reflecting", "absorbing")

    # 3D Tests (smaller grids due to memory)
    print("\n" + "=" * 70)
    print("3D TESTS")
    print("=" * 70)

    results_3d_reflect = run_comparison_3d("reflecting")
    results_3d_absorb = run_comparison_3d("absorbing")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Check if finest grid results are within tolerance
    tol = 0.05  # 5% relative error tolerance for comparison

    tests = [
        ("1D Reflecting", results_1d_reflect[-1][4]),
        ("1D Absorbing", results_1d_absorb[-1][4]),
        ("2D Reflecting", results_2d_reflect[-1][5]),
        ("2D Absorbing", results_2d_absorb[-1][5]),
        ("2D Mixed", results_2d_mixed[-1][5]),
        ("3D Reflecting", results_3d_reflect[-1][6]),
        ("3D Absorbing", results_3d_absorb[-1][6]),
    ]

    print(f"\n{'Test':>20} {'Rel Error':>14} {'Status':>10}")
    print("-" * 50)

    for name, rel_err in tests:
        status = "PASS" if rel_err < tol else "FAIL"
        if status == "FAIL":
            all_passed = False
        print(f"{name:>20} {rel_err:>14.6e} {status:>10}")

    print("-" * 50)

    if all_passed:
        print("\nAll tests PASSED!")
        print("Real-space and pseudo-spectral methods agree within tolerance.")
    else:
        print("\nSome tests FAILED!")
        print("Check numerical parameters or implementation.")

    return all_passed


if __name__ == "__main__":
    main()
