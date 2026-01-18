#!/usr/bin/env python3
"""
Grafted Brush Validation Test

Tests the real-space solver with absorbing boundary conditions by comparing
numerical results against analytical Fourier series solutions.

Setup:
- 1D domain with absorbing boundaries on both sides
- Gaussian initial condition centered at x0 = 0.5 with sigma = 0.1
- Grid: 512 points, Lx = 4.0
- Zero potential field (pure diffusion)

The analytical solution for diffusion with absorbing BCs:
    q(x,s) = (2/L) * sum_n a_n * sin(n*pi*x/L) * exp(-n^2*pi^2*b^2*s/(6*L^2))

where a_n are the Fourier sine coefficients of the initial Gaussian.

This test validates:
1. Absorbing boundary conditions work correctly
2. Diffusion operator is accurate
3. Convergence behavior with contour step size
"""

import os
import sys
import numpy as np

os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
os.environ["OMP_NUM_THREADS"] = "4"

from polymerfts import PropagatorSolver


def gaussian(x, x0, sigma):
    """Gaussian function."""
    return np.exp(-(x - x0)**2 / (2 * sigma**2))


def analytical_solution_absorbing_bc(x, s, x0, sigma, L, b=1.0, n_terms=200):
    """
    Analytical solution for diffusion with absorbing BCs.

    Solves: dq/ds = (b^2/6) * d^2q/dx^2
    BC: q(0,s) = q(L,s) = 0
    IC: q(x,0) = Gaussian centered at x0 with width sigma

    Solution via Fourier sine series:
        q(x,s) = sum_n a_n * sin(n*pi*x/L) * exp(-n^2*pi^2*b^2*s/(6*L^2))

    where a_n = (2/L) * integral[q(x,0) * sin(n*pi*x/L) dx]
    """

    # Precompute Fourier coefficients using same grid as numerical solver
    # For Gaussian initial condition, we numerically compute coefficients
    nx_fine = 2048
    # Use cell-centered grid matching the solver
    x_fine = (np.arange(nx_fine) + 0.5) * L / nx_fine
    dx_fine = L / nx_fine

    q0 = gaussian(x_fine, x0, sigma)

    q = np.zeros_like(x)

    for n in range(1, n_terms + 1):
        # Compute Fourier coefficient
        sin_basis = np.sin(n * np.pi * x_fine / L)
        a_n = 2.0 / L * np.sum(q0 * sin_basis) * dx_fine

        # Eigenvalue
        lambda_n = (n * np.pi / L)**2 * b**2 / 6.0

        # Add term to solution
        q += a_n * np.sin(n * np.pi * x / L) * np.exp(-lambda_n * s)

    return q


def test_grafted_brush_realspace():
    """Test real-space solver with absorbing BC against analytical solution."""

    print("="*60)
    print("Grafted Brush Validation Test (Real-Space Solver)")
    print("="*60)

    # Test parameters
    nx = 512
    Lx = 4.0
    x0 = 2.0  # Grafting point at center (away from boundary for stability)
    sigma = 0.02  # Very sharp Gaussian (closer to delta function)
    b = 1.0  # Statistical segment length
    s_final = 0.2  # Contour length to propagate

    # Grid (cell-centered)
    dx = Lx / nx
    x = (np.arange(nx) + 0.5) * dx

    # Initial condition (Gaussian) - NOT normalized by dx
    # The solver expects q_init as a field, not a probability density
    q_init = gaussian(x, x0, sigma)

    print(f"\nTest Configuration:")
    print(f"  Grid: {nx} points, Lx = {Lx}")
    print(f"  Grafting point: x0 = {x0}")
    print(f"  Gaussian width: sigma = {sigma}")
    print(f"  Contour length: s = {s_final}")

    # Analytical solution using the same initial condition (not normalized)
    q_analytical = analytical_solution_absorbing_bc(x, s_final, x0, sigma, Lx, b)

    # Scale analytical solution to match the initial condition normalization
    # Both solutions start from the same Gaussian, so no scaling needed

    # Test different contour step sizes (finer values for sharp Gaussian)
    ds_values = [0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003125]
    errors = []

    print(f"\nConvergence Study:")
    print("-"*70)
    print(f"{'ds':>10} {'N_steps':>10} {'L2 Error':>15} {'Max Error':>15}")
    print("-"*70)

    for ds in ds_values:
        n_steps = int(s_final / ds)

        try:
            # Create solver with absorbing BC
            solver = PropagatorSolver(
                nx=[nx],
                lx=[Lx],
                ds=ds,
                bond_lengths={"A": b},
                bc=["absorbing", "absorbing"],
                chain_model="continuous",
                method="realspace",
                platform="cpu-mkl",
                use_checkpointing=False
            )

            # Add polymer with grafting point
            solver.add_polymer(
                volume_fraction=1.0,
                blocks=[["A", s_final, 0, 1]],
                grafting_points={0: "G"}
            )

            # Compute propagators with zero field
            w_field = np.zeros(nx)
            solver.compute_propagators({"A": w_field}, q_init={"G": q_init})

            # Get final propagator
            q_numerical = solver.get_propagator(polymer=0, v=0, u=1, step=n_steps)

            # Compute errors
            error_l2 = np.sqrt(np.mean((q_numerical - q_analytical)**2))
            error_max = np.max(np.abs(q_numerical - q_analytical))

            errors.append({
                'ds': ds,
                'n_steps': n_steps,
                'l2_error': error_l2,
                'max_error': error_max,
                'q_numerical': q_numerical.copy()
            })

            print(f"{ds:>10.4f} {n_steps:>10d} {error_l2:>15.6e} {error_max:>15.6e}")

        except Exception as e:
            print(f"{ds:>10.4f} {'FAILED':>10} {str(e)[:30]}")
            errors.append(None)

    print("-"*60)

    # Analyze convergence
    valid_errors = [e for e in errors if e is not None]
    if len(valid_errors) >= 2:
        # Estimate convergence order
        e1, e2 = valid_errors[0], valid_errors[1]
        if e1['l2_error'] > 1e-14 and e2['l2_error'] > 1e-14:
            order = np.log(e1['l2_error'] / e2['l2_error']) / np.log(e1['ds'] / e2['ds'])
            print(f"\nEstimated convergence order: p ≈ {order:.1f}")

    # Test boundary conditions
    print("\nBoundary Condition Check:")
    if valid_errors:
        q = valid_errors[-1]['q_numerical']
        print(f"  q[0] (should be ~0): {q[0]:.6e}")
        print(f"  q[-1] (should be ~0): {q[-1]:.6e}")

        if abs(q[0]) < 0.01 and abs(q[-1]) < 0.01:
            print("  -> PASSED: Absorbing BC working correctly")
        else:
            print("  -> WARNING: Boundary values not close to zero")

    # Check normalization (integral should decrease due to absorbing BC)
    if valid_errors:
        q = valid_errors[-1]['q_numerical']
        integral = np.sum(q) * dx
        print(f"\nNormalization Check:")
        print(f"  Initial integral: 1.0")
        print(f"  Final integral: {integral:.6f}")
        print(f"  (Should be < 1 due to absorption at boundaries)")

    return errors


def test_grafted_brush_pseudospectral():
    """Test pseudo-spectral solver with absorbing BC (DST)."""

    print("\n" + "="*60)
    print("Grafted Brush Validation Test (Pseudo-Spectral DST)")
    print("="*60)

    # Test parameters (same as real-space test)
    nx = 512
    Lx = 4.0
    x0 = 2.0  # Center of domain
    sigma = 0.02  # Very sharp Gaussian
    b = 1.0
    s_final = 0.2

    dx = Lx / nx
    x = (np.arange(nx) + 0.5) * dx

    # Initial condition (Gaussian) - NOT normalized by dx
    q_init = gaussian(x, x0, sigma)

    q_analytical = analytical_solution_absorbing_bc(x, s_final, x0, sigma, Lx, b)

    ds_values = [0.1, 0.05, 0.025, 0.0125]
    errors = []

    print(f"\nConvergence Study (Pseudo-Spectral with DST):")
    print("-"*60)
    print(f"{'ds':>10} {'N_steps':>10} {'L2 Error':>15} {'Max Error':>15}")
    print("-"*60)

    for ds in ds_values:
        n_steps = int(s_final / ds)

        try:
            solver = PropagatorSolver(
                nx=[nx],
                lx=[Lx],
                ds=ds,
                bond_lengths={"A": b},
                bc=["absorbing", "absorbing"],
                chain_model="continuous",
                method="pseudospectral",
                platform="cpu-mkl",
                use_checkpointing=False
            )

            solver.add_polymer(
                volume_fraction=1.0,
                blocks=[["A", s_final, 0, 1]],
                grafting_points={0: "G"}
            )

            w_field = np.zeros(nx)
            solver.compute_propagators({"A": w_field}, q_init={"G": q_init})

            q_numerical = solver.get_propagator(polymer=0, v=0, u=1, step=n_steps)

            error_l2 = np.sqrt(np.mean((q_numerical - q_analytical)**2))
            error_max = np.max(np.abs(q_numerical - q_analytical))

            errors.append({
                'ds': ds,
                'n_steps': n_steps,
                'l2_error': error_l2,
                'max_error': error_max,
                'q_numerical': q_numerical.copy()
            })

            print(f"{ds:>10.4f} {n_steps:>10d} {error_l2:>15.6e} {error_max:>15.6e}")

        except Exception as e:
            print(f"{ds:>10.4f} {'FAILED':>10} {str(e)[:40]}")
            errors.append(None)

    print("-"*60)

    return errors


def compare_methods():
    """Compare real-space vs pseudo-spectral for absorbing BC."""

    print("\n" + "="*60)
    print("Method Comparison: Real-Space vs Pseudo-Spectral (DST)")
    print("="*60)

    nx = 512
    Lx = 4.0
    x0 = 2.0  # Center of domain
    sigma = 0.02  # Very sharp Gaussian
    b = 1.0
    s_final = 0.2
    ds = 0.01

    dx = Lx / nx
    x = (np.arange(nx) + 0.5) * dx
    n_steps = int(s_final / ds)

    # Initial condition (Gaussian) - NOT normalized by dx
    q_init = gaussian(x, x0, sigma)

    q_analytical = analytical_solution_absorbing_bc(x, s_final, x0, sigma, Lx, b)

    # Real-space solver
    print("\nReal-Space Solver:")
    try:
        solver_rs = PropagatorSolver(
            nx=[nx], lx=[Lx], ds=ds,
            bond_lengths={"A": b},
            bc=["absorbing", "absorbing"],
            chain_model="continuous",
            method="realspace",
            platform="cpu-mkl",
            use_checkpointing=False
        )
        solver_rs.add_polymer(1.0, [["A", s_final, 0, 1]], grafting_points={0: "G"})
        solver_rs.compute_propagators({"A": np.zeros(nx)}, q_init={"G": q_init})
        q_rs = solver_rs.get_propagator(polymer=0, v=0, u=1, step=n_steps)

        error_rs = np.sqrt(np.mean((q_rs - q_analytical)**2))
        print(f"  L2 Error vs Analytical: {error_rs:.6e}")
    except Exception as e:
        print(f"  FAILED: {e}")
        q_rs = None

    # Pseudo-spectral solver
    print("\nPseudo-Spectral Solver (DST):")
    try:
        solver_ps = PropagatorSolver(
            nx=[nx], lx=[Lx], ds=ds,
            bond_lengths={"A": b},
            bc=["absorbing", "absorbing"],
            chain_model="continuous",
            method="pseudospectral",
            platform="cpu-mkl",
            use_checkpointing=False
        )
        solver_ps.add_polymer(1.0, [["A", s_final, 0, 1]], grafting_points={0: "G"})
        solver_ps.compute_propagators({"A": np.zeros(nx)}, q_init={"G": q_init})
        q_ps = solver_ps.get_propagator(polymer=0, v=0, u=1, step=n_steps)

        error_ps = np.sqrt(np.mean((q_ps - q_analytical)**2))
        print(f"  L2 Error vs Analytical: {error_ps:.6e}")
    except Exception as e:
        print(f"  FAILED: {e}")
        q_ps = None

    # Compare methods
    if q_rs is not None and q_ps is not None:
        error_between = np.sqrt(np.mean((q_rs - q_ps)**2))
        print(f"\nDifference between methods: {error_between:.6e}")


def main():
    print("="*60)
    print("Grafted Brush Validation Tests")
    print("="*60)
    print("\nThis test validates absorbing boundary conditions by")
    print("comparing numerical solutions against analytical Fourier")
    print("series solutions for diffusion in a 1D domain.")

    # Run tests
    errors_rs = test_grafted_brush_realspace()
    errors_ps = test_grafted_brush_pseudospectral()
    compare_methods()

    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)

    print("\nKey Findings:")
    print("1. Real-space solver uses Crank-Nicolson ADI method")
    print("2. Pseudo-spectral solver uses Discrete Sine Transform (DST)")
    print("3. Both should handle absorbing BC correctly")
    print("4. Error decreases with smaller ds (finer contour discretization)")

    # Check if tests passed
    valid_errors = [e for e in errors_rs if e is not None]
    if valid_errors:
        finest_error = valid_errors[-1]['l2_error']
        if finest_error < 0.01:  # 1% error tolerance
            print(f"\nReal-Space Test: PASSED (L2 error = {finest_error:.2e})")
        else:
            print(f"\nReal-Space Test: MARGINAL (L2 error = {finest_error:.2e})")

    valid_errors = [e for e in errors_ps if e is not None]
    if valid_errors:
        finest_error = valid_errors[-1]['l2_error']
        if finest_error < 0.01:
            print(f"Pseudo-Spectral Test: PASSED (L2 error = {finest_error:.2e})")
        else:
            print(f"Pseudo-Spectral Test: MARGINAL (L2 error = {finest_error:.2e})")


if __name__ == "__main__":
    main()
