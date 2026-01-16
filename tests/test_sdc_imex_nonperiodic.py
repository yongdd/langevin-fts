#!/usr/bin/env python3
"""
Test SDC IMEX mode with non-periodic boundary conditions.

This script verifies that IMEX mode works correctly for all boundary conditions:
- Periodic BC: Uses FFT-based diffusion solve
- Non-periodic BC: Uses PCG-based diffusion solve
- 1D: Uses tridiagonal solver
"""

import os
import sys
import time
import numpy as np
import io
from contextlib import redirect_stdout

os.environ["OMP_MAX_ACTIVE_LEVELS"] = "0"

from polymerfts import SCFT


def test_imex_with_boundary_conditions():
    """Test IMEX mode with various boundary conditions."""
    print("=" * 70)
    print("Testing SDC IMEX mode with various boundary conditions")
    print("=" * 70)

    results = []

    # Common parameters (ds will be adjusted for 1D)
    base_params = {
        "chain_model": "continuous",
        "ds": 1/16,  # For 2D/3D; 1D uses finer discretization
        "segment_lengths": {"A": 1.0, "B": 1.0},
        "chi_n": {"A,B": 15.0},
        "distinct_polymers": [{
            "volume_fraction": 1.0,
            "blocks": [
                {"type": "A", "length": 0.5},
                {"type": "B", "length": 0.5}
            ]
        }],
        "optimizer": {
            "name": "am",
            "max_hist": 20,
            "start_error": 1e-2,
            "mix_min": 0.1,
            "mix_init": 0.1,
        },
        "max_iter": 100,
        "tolerance": 1e-6,
        "verbose_level": 0,  # Must be 0 for consistent convergence
        "platform": "cuda",
        "numerical_method": "sdc-2",
        "box_is_altering": False,
    }

    # Test cases
    test_cases = [
        # (name, nx, lx, boundary_conditions)
        ("2D Periodic", [32, 32], [4.0, 4.0], ["periodic", "periodic", "periodic", "periodic"]),
        ("2D All Reflecting", [32, 32], [4.0, 4.0], ["reflecting", "reflecting", "reflecting", "reflecting"]),
        ("2D X:Absorbing Y:Reflecting", [32, 32], [4.0, 4.0], ["absorbing", "absorbing", "reflecting", "reflecting"]),
        ("1D Periodic", [64], [4.0], ["periodic", "periodic"]),
        ("1D Reflecting", [64], [4.0], ["reflecting", "reflecting"]),
        ("1D Absorbing", [64], [4.0], ["absorbing", "absorbing"]),
    ]

    for test_name, nx, lx, bc in test_cases:
        print(f"\nTest: {test_name}")
        print("-" * 50)

        params = base_params.copy()
        params["nx"] = nx
        params["lx"] = lx
        params["boundary_conditions"] = bc

        # Use finer discretization for 1D to reduce Strang splitting error
        if len(nx) == 1:
            params["ds"] = 1/128

        # Generate initial fields (keep a copy since SCFT may modify them)
        n_grid = int(np.prod(nx))
        np.random.seed(42)
        w_A = np.random.normal(0.0, 2.0, n_grid)
        w_B = -w_A

        # Test without IMEX (fully implicit)
        params_no_imex = params.copy()
        params_no_imex["sdc_imex_mode"] = False

        try:
            with redirect_stdout(io.StringIO()):
                scft_no_imex = SCFT(params_no_imex)
            start = time.perf_counter()
            with redirect_stdout(io.StringIO()):
                scft_no_imex.run({"A": w_A.copy(), "B": w_B.copy()})
            time_no_imex = time.perf_counter() - start
            H_no_imex = scft_no_imex.free_energy
            print(f"  Without IMEX: H={H_no_imex:.6f}, time={time_no_imex*1000:.1f}ms")
        except Exception as e:
            print(f"  Without IMEX: ERROR - {e}")
            results.append((test_name, "no_imex", False, str(e)))
            continue

        # Test with IMEX enabled
        params_imex = params.copy()
        params_imex["sdc_imex_mode"] = True

        try:
            with redirect_stdout(io.StringIO()):
                scft_imex = SCFT(params_imex)
            start = time.perf_counter()
            with redirect_stdout(io.StringIO()):
                scft_imex.run({"A": w_A.copy(), "B": w_B.copy()})
            time_imex = time.perf_counter() - start
            H_imex = scft_imex.free_energy
            print(f"  With IMEX:    H={H_imex:.6f}, time={time_imex*1000:.1f}ms")

            # Compare results - allow 5% relative difference
            rel_diff = abs(H_imex - H_no_imex) / max(abs(H_no_imex), 1e-10)
            if rel_diff < 0.05:
                print(f"  PASSED (rel_diff={rel_diff:.2e})")
                results.append((test_name, "both", True, rel_diff))
            else:
                print(f"  WARNING: Large difference (rel_diff={rel_diff:.2e})")
                results.append((test_name, "both", False, f"rel_diff={rel_diff:.2e}"))
        except Exception as e:
            print(f"  With IMEX: ERROR - {e}")
            results.append((test_name, "imex", False, str(e)))

    # Summary
    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)
    all_passed = True
    for test_name, mode, passed, info in results:
        status = "PASSED" if passed else "FAILED"
        print(f"  {test_name}: {status}")
        if not passed:
            print(f"    Error: {info}")
            all_passed = False

    return all_passed


if __name__ == "__main__":
    success = test_imex_with_boundary_conditions()
    sys.exit(0 if success else 1)
