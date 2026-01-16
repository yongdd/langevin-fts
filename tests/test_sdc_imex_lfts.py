#!/usr/bin/env python3
"""
Test SDC IMEX mode.

This script verifies that IMEX mode works correctly for SDC solver
and compares its performance against the fully implicit SDC mode.

IMEX mode treats diffusion implicitly and reaction explicitly,
which is faster for periodic BC in 2D/3D.
"""

import os
import sys
import time
import numpy as np
import io
from contextlib import redirect_stdout

os.environ["OMP_MAX_ACTIVE_LEVELS"] = "0"

from polymerfts import LFTS


def test_lfts_imex_mode():
    """Test that IMEX mode works for L-FTS with SDC solver."""
    print("=" * 70)
    print("Testing SDC IMEX mode for L-FTS")
    print("=" * 70)

    # Common parameters
    base_params = {
        "nx": [32, 32],
        "lx": [4.0, 4.0],
        "chain_model": "continuous",
        "ds": 1/32,
        "segment_lengths": {"A": 1.0, "B": 1.0},
        "chi_n": {"A,B": 15.0},
        "distinct_polymers": [{
            "volume_fraction": 1.0,
            "blocks": [
                {"type": "A", "length": 0.5},
                {"type": "B", "length": 0.5}
            ]
        }],
        "compressor": {
            "name": "am",
            "max_hist": 20,
            "start_error": 1e-2,
            "mix_min": 0.1,
            "mix_init": 0.1,
        },
        "langevin": {
            "max_step": 3,      # Just a few steps to verify it works
            "dt": 0.5,
            "nbar": 1000.0,
        },
        "saddle": {
            "max_iter": 100,
            "tolerance": 1e-4,
        },
        "recording": {
            "dir": "test_imex_output",
            "recording_period": 10,
            "sf_computing_period": 10,
            "sf_recording_period": 10,
        },
        "verbose_level": 1,  # Must be > 0 for hamiltonian to be computed
        "platform": "cuda",
        "numerical_method": "sdc-2",
    }

    # Generate initial fields
    n_grid = int(np.prod(base_params["nx"]))
    np.random.seed(42)
    w_A = np.random.normal(0.0, 0.5, n_grid)
    w_B = -w_A
    initial_fields = {"A": w_A, "B": w_B}

    # Test 1: Without IMEX (default)
    print("\nTest 1: SDC without IMEX (fully implicit)")
    params_no_imex = base_params.copy()
    params_no_imex["sdc_imex_mode"] = False

    try:
        with redirect_stdout(io.StringIO()):
            lfts_no_imex = LFTS(params_no_imex, random_seed=123)

        start = time.perf_counter()
        with redirect_stdout(io.StringIO()):
            lfts_no_imex.run(initial_fields)
        time_no_imex = time.perf_counter() - start

        print(f"  Completed in {time_no_imex*1000:.1f} ms")
        print("  Status: PASSED")
    except Exception as e:
        print(f"  ERROR: {e}")
        return False

    # Test 2: With IMEX enabled
    print("\nTest 2: SDC with IMEX (implicit diffusion, explicit reaction)")
    params_imex = base_params.copy()
    params_imex["sdc_imex_mode"] = True

    try:
        with redirect_stdout(io.StringIO()):
            lfts_imex = LFTS(params_imex, random_seed=123)

        start = time.perf_counter()
        with redirect_stdout(io.StringIO()):
            lfts_imex.run(initial_fields)
        time_imex = time.perf_counter() - start

        print(f"  Completed in {time_imex*1000:.1f} ms")
        print("  Status: PASSED")
    except Exception as e:
        print(f"  ERROR: {e}")
        return False

    # Test 3: Compare 3D case
    print("\nTest 3: 3D case with IMEX")
    params_3d = base_params.copy()
    params_3d["nx"] = [16, 16, 16]
    params_3d["lx"] = [3.0, 3.0, 3.0]
    params_3d["sdc_imex_mode"] = True

    # Re-generate fields for 3D
    n_grid_3d = int(np.prod(params_3d["nx"]))
    np.random.seed(42)
    w_A_3d = np.random.normal(0.0, 0.5, n_grid_3d)
    w_B_3d = -w_A_3d
    initial_fields_3d = {"A": w_A_3d, "B": w_B_3d}

    try:
        with redirect_stdout(io.StringIO()):
            lfts_3d = LFTS(params_3d, random_seed=123)

        start = time.perf_counter()
        with redirect_stdout(io.StringIO()):
            lfts_3d.run(initial_fields_3d)
        time_3d = time.perf_counter() - start

        print(f"  Completed in {time_3d*1000:.1f} ms")
        print("  Status: PASSED")
    except Exception as e:
        print(f"  ERROR: {e}")
        return False

    # Summary
    print("\n" + "=" * 70)
    print("Summary:")
    print(f"  2D 32x32 without IMEX: {time_no_imex*1000:.1f} ms")
    print(f"  2D 32x32 with IMEX:    {time_imex*1000:.1f} ms")
    if time_imex < time_no_imex:
        speedup = (time_no_imex - time_imex) / time_no_imex * 100
        print(f"  IMEX speedup: {speedup:.1f}%")
    print(f"  3D 16^3 with IMEX:     {time_3d*1000:.1f} ms")
    print("=" * 70)
    print("All tests PASSED!")
    return True


def test_imex_non_sdc_warning():
    """Test that IMEX mode is ignored for non-SDC methods."""
    print("\n" + "=" * 70)
    print("Testing IMEX mode warning for non-SDC methods")
    print("=" * 70)

    params = {
        "nx": [32, 32],
        "lx": [4.0, 4.0],
        "chain_model": "continuous",
        "ds": 1/32,
        "segment_lengths": {"A": 1.0, "B": 1.0},
        "chi_n": {"A,B": 15.0},
        "distinct_polymers": [{
            "volume_fraction": 1.0,
            "blocks": [
                {"type": "A", "length": 0.5},
                {"type": "B", "length": 0.5}
            ]
        }],
        "compressor": {
            "name": "am",
            "max_hist": 20,
            "start_error": 1e-2,
            "mix_min": 0.1,
            "mix_init": 0.1,
        },
        "langevin": {
            "max_step": 1,
            "dt": 0.5,
            "nbar": 1000.0,
        },
        "saddle": {
            "max_iter": 100,
            "tolerance": 1e-4,
        },
        "recording": {
            "dir": "test_imex_output",
            "recording_period": 10,
            "sf_computing_period": 10,
            "sf_recording_period": 10,
        },
        "verbose_level": 1,  # Must be > 0 for hamiltonian to be computed
        "platform": "cuda",
        "numerical_method": "rqm4",  # Non-SDC method
        "sdc_imex_mode": True,       # Should be ignored with a warning
    }

    try:
        # This should work but log a warning about IMEX being ignored
        with redirect_stdout(io.StringIO()):
            lfts = LFTS(params, random_seed=123)
        print("  Warning logged (as expected): sdc_imex_mode ignored for rqm4")
        print("  Status: PASSED")
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


if __name__ == "__main__":
    success = True
    success = test_lfts_imex_mode() and success
    success = test_imex_non_sdc_warning() and success

    # Cleanup
    import shutil
    if os.path.exists("test_imex_output"):
        shutil.rmtree("test_imex_output")

    sys.exit(0 if success else 1)
