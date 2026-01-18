#!/usr/bin/env python3
"""
Reproduce Fig. 1 and Fig. 2 from Song et al., Chinese J. Polym. Sci. 2018, 36, 488-496.

Fig. 1: Convergence in Ns (contour steps)
- AB diblock, f = 0.375, χN = 18, Gyroid phase
- Grid: 32³
- Vary Ns from 10 to 10000

Fig. 2: Convergence in Nx (spatial resolution)
- AB diblock, f = 0.32, χN = 40 and 80, Gyroid phase
- Vary Nx from 20 to 128
- Fixed Ns = 101

Usage:
    # Run all tests for a method
    python benchmark_song2018.py fig1 etdrk4
    python benchmark_song2018.py fig2 etdrk4

    # Run single test (for parallel jobs)
    python benchmark_song2018.py fig1 etdrk4 --Ns 100
    python benchmark_song2018.py fig2 etdrk4 --Nx 64 --chi_n 40
"""

import os
import sys
import time
import json
import numpy as np
from datetime import datetime
from copy import deepcopy

# Set OpenMP environment before importing polymerfts
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "0"
os.environ["OMP_NUM_THREADS"] = "4"

from polymerfts import scft


def create_gyroid_initial_field(nx, f=0.375):
    """Create gyroid phase initial field."""
    w_A = np.zeros(list(nx), dtype=np.float64)
    w_B = np.zeros(list(nx), dtype=np.float64)

    for i in range(nx[0]):
        xx = (i + 1) * 2 * np.pi / nx[0]
        for j in range(nx[1]):
            yy = (j + 1) * 2 * np.pi / nx[1]
            zz = np.arange(1, nx[2] + 1) * 2 * np.pi / nx[2]

            c1 = np.sqrt(8.0/3.0) * (
                np.cos(xx) * np.sin(yy) * np.sin(2.0*zz) +
                np.cos(yy) * np.sin(zz) * np.sin(2.0*xx) +
                np.cos(zz) * np.sin(xx) * np.sin(2.0*yy)
            )
            c2 = np.sqrt(4.0/3.0) * (
                np.cos(2.0*xx) * np.cos(2.0*yy) +
                np.cos(2.0*yy) * np.cos(2.0*zz) +
                np.cos(2.0*zz) * np.cos(2.0*xx)
            )
            # Amplitude based on volume fraction
            amp = 0.3 * (0.5 - f) / 0.125  # Scale amplitude
            w_A[i, j, :] = -amp * c1 + 0.1 * c2
            w_B[i, j, :] = amp * c1 - 0.1 * c2

    return w_A, w_B


def get_gyroid_box_size(chi_n, f):
    """Get optimal box size for Gyroid phase based on chi_n and f.

    These values are calibrated from converged SCFT calculations.
    The Gyroid unit cell size depends on both χN and f.
    """
    # Base box sizes from converged calculations
    if f == 0.375 and chi_n == 18.0:
        return 3.65  # Converged value for Fig 1 conditions
    elif f == 0.32:
        if chi_n == 40.0:
            return 3.85  # Converged value for Fig 2, χN=40
        elif chi_n == 80.0:
            return 4.00  # Converged value for Fig 2, χN=80
    # Default scaling for other cases
    return 3.5 * (chi_n / 18.0) ** 0.17


def get_gyroid_params(nx, ds, chi_n, f, numerical_method):
    """Get parameter dictionary for Gyroid phase simulation."""
    lx_base = get_gyroid_box_size(chi_n, f)

    return {
        "platform": "cuda",
        "nx": list(nx),
        "lx": [lx_base, lx_base, lx_base],
        "reduce_memory": False,
        "box_is_altering": False,  # Fixed box size (stress computation not supported for CN-ADI)
        "chain_model": "continuous",
        "ds": ds,
        "numerical_method": numerical_method,
        "segment_lengths": {"A": 1.0, "B": 1.0},
        "chi_n": {"A,B": chi_n},
        "distinct_polymers": [{
            "volume_fraction": 1.0,
            "blocks": [
                {"type": "A", "length": f},
                {"type": "B", "length": 1.0 - f},
            ],
        }],
        "optimizer": {
            "name": "am",
            "max_hist": 20,
            "start_error": 1e-2,
            "mix_min": 0.02,
            "mix_init": 0.02,
        },
        "max_iter": 2000,
        "tolerance": 1e-9,
    }


def run_single_test(params, initial_fields, max_iter=None):
    """Run a single SCFT calculation and return results."""
    result = {}

    try:
        params = deepcopy(params)
        if max_iter:
            params["max_iter"] = max_iter

        # Initialize and run
        t_start = time.perf_counter()
        calculation = scft.SCFT(params=params)
        t_init = time.perf_counter() - t_start

        t_run_start = time.perf_counter()
        final_result = calculation.run(initial_fields=initial_fields, return_result=True)
        t_run = time.perf_counter() - t_run_start

        result["success"] = True
        result["init_time_s"] = t_init
        result["run_time_s"] = t_run
        result["total_time_s"] = t_init + t_run
        result["iterations"] = final_result.iteration if hasattr(final_result, 'iteration') else params.get("max_iter", 0)
        result["final_error"] = final_result.error_level
        result["free_energy"] = final_result.free_energy
        result["converged"] = final_result.error_level < params["tolerance"]
        result["final_lx"] = list(final_result.lx)

    except Exception as e:
        result["success"] = False
        result["error"] = str(e)
        print(f"  ERROR: {e}")

    return result


def benchmark_fig1(method):
    """
    Reproduce Fig. 1: Convergence in Ns (contour steps).

    System: f = 0.375, χN = 18, Gyroid, M = 32³
    """
    print("="*70)
    print(f"FIG 1 BENCHMARK: Ns Convergence - {method.upper()}")
    print("="*70)
    print("System: f = 0.375, χN = 18, Gyroid phase, M = 32³")

    f = 0.375
    chi_n = 18.0
    nx = [32, 32, 32]

    # Ns values to test (ds = 1/Ns)
    # Note: f*Ns must be an integer so each block has integer contour steps.
    # For f = 0.375 = 3/8, this requires Ns to be divisible by 8.
    Ns_values = [40, 80, 160, 200, 320, 400, 640, 800, 1000, 2000, 4000]

    # Create initial field
    w_A, w_B = create_gyroid_initial_field(nx, f)
    initial_fields = {"A": w_A, "B": w_B}

    results = []

    for Ns in Ns_values:
        ds = 1.0 / Ns
        print(f"\n  Ns = {Ns} (ds = {ds:.6f}): ", end="", flush=True)

        params = get_gyroid_params(nx, ds, chi_n, f, method)
        result = run_single_test(params, initial_fields)

        result["Ns"] = Ns
        result["ds"] = ds
        result["method"] = method
        result["chi_n"] = chi_n
        result["f"] = f
        result["nx"] = nx

        if result["success"]:
            print(f"F = {result['free_energy']:.10f}, "
                  f"time = {result['total_time_s']:.2f}s, "
                  f"error = {result['final_error']:.2e}")

        results.append(result)

    return results


def benchmark_fig2(method):
    """
    Reproduce Fig. 2: Convergence in Nx (spatial resolution).

    System: f = 0.32, χN = 40 and 80, Gyroid, Ns = 101
    """
    print("="*70)
    print(f"FIG 2 BENCHMARK: Nx Convergence - {method.upper()}")
    print("="*70)
    print("System: f = 0.32, Gyroid phase, Ns = 101")

    f = 0.32
    Ns = 101
    ds = 1.0 / Ns

    # Chi_n values to test
    chi_n_values = [40.0, 80.0]

    # Nx values to test
    Nx_values = [24, 32, 48, 64, 80, 96, 112, 128]

    results = []

    for chi_n in chi_n_values:
        print(f"\n--- χN = {chi_n} ---")

        for Nx in Nx_values:
            nx = [Nx, Nx, Nx]
            print(f"\n  Nx = {Nx}: ", end="", flush=True)

            # Create initial field for this grid
            w_A, w_B = create_gyroid_initial_field(nx, f)
            initial_fields = {"A": w_A, "B": w_B}

            params = get_gyroid_params(nx, ds, chi_n, f, method)
            result = run_single_test(params, initial_fields)

            result["Ns"] = Ns
            result["ds"] = ds
            result["Nx"] = Nx
            result["method"] = method
            result["chi_n"] = chi_n
            result["f"] = f
            result["nx"] = nx

            if result["success"]:
                print(f"F = {result['free_energy']:.10f}, "
                      f"time = {result['total_time_s']:.2f}s, "
                      f"error = {result['final_error']:.2e}")

            results.append(result)

    return results


def convert_numpy(obj):
    """Convert numpy types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    return obj


def run_single_fig1_test(method, Ns):
    """Run a single Fig 1 test with specific Ns."""
    f = 0.375
    chi_n = 18.0
    nx = [32, 32, 32]
    ds = 1.0 / Ns

    print(f"Running Fig 1: method={method}, Ns={Ns}")

    w_A, w_B = create_gyroid_initial_field(nx, f)
    initial_fields = {"A": w_A, "B": w_B}

    params = get_gyroid_params(nx, ds, chi_n, f, method)
    result = run_single_test(params, initial_fields)

    result["Ns"] = Ns
    result["ds"] = ds
    result["method"] = method
    result["chi_n"] = chi_n
    result["f"] = f
    result["nx"] = nx

    if result["success"]:
        print(f"F = {result['free_energy']:.10f}, time = {result['total_time_s']:.2f}s")

    return [result]


def run_single_fig2_test(method, Nx, chi_n):
    """Run a single Fig 2 test with specific Nx and chi_n."""
    f = 0.32
    Ns = 101
    ds = 1.0 / Ns
    nx = [Nx, Nx, Nx]

    print(f"Running Fig 2: method={method}, Nx={Nx}, chi_n={chi_n}")

    w_A, w_B = create_gyroid_initial_field(nx, f)
    initial_fields = {"A": w_A, "B": w_B}

    params = get_gyroid_params(nx, ds, chi_n, f, method)
    result = run_single_test(params, initial_fields)

    result["Ns"] = Ns
    result["ds"] = ds
    result["Nx"] = Nx
    result["method"] = method
    result["chi_n"] = chi_n
    result["f"] = f
    result["nx"] = nx

    if result["success"]:
        print(f"F = {result['free_energy']:.10f}, time = {result['total_time_s']:.2f}s")

    return [result]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Song 2018 SCFT benchmark")
    parser.add_argument("test_type", choices=["fig1", "fig2"], help="Test type")
    parser.add_argument("method", choices=["rqm4", "rk2", "etdrk4", "cn-adi2", "cn-adi4-lr"], help="Numerical method")
    parser.add_argument("--Ns", type=int, default=None, help="Single Ns value for fig1")
    parser.add_argument("--Nx", type=int, default=None, help="Single Nx value for fig2")
    parser.add_argument("--chi_n", type=float, default=None, help="Single chi_n value for fig2")
    args = parser.parse_args()

    test_type = args.test_type
    method = args.method

    print("="*70)
    print("SCFT NUMERICAL METHODS BENCHMARK")
    print("Reproducing Song et al., Chinese J. Polym. Sci. 2018, 36, 488-496")
    print("="*70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Test: {test_type.upper()}")
    print(f"Method: {method.upper()}")
    print(f"Platform: CUDA")

    if test_type == "fig1":
        if args.Ns is not None:
            # Single test mode
            results = run_single_fig1_test(method, args.Ns)
            output_file = f"benchmark_fig1_{method}_Ns{args.Ns}.json"
        else:
            # Full benchmark
            results = benchmark_fig1(method)
            output_file = f"benchmark_fig1_{method}_results.json"
    elif test_type == "fig2":
        if args.Nx is not None and args.chi_n is not None:
            # Single test mode
            results = run_single_fig2_test(method, args.Nx, args.chi_n)
            output_file = f"benchmark_fig2_{method}_Nx{args.Nx}_chi{int(args.chi_n)}.json"
        else:
            # Full benchmark
            results = benchmark_fig2(method)
            output_file = f"benchmark_fig2_{method}_results.json"

    # Save results
    output_data = {
        "metadata": {
            "date": datetime.now().isoformat(),
            "test_type": test_type,
            "method": method,
            "platform": "cuda",
        },
        "results": results,
    }

    with open(output_file, 'w') as f:
        json.dump(convert_numpy(output_data), f, indent=2)

    print(f"\nResults saved to {output_file}")

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    successful = [r for r in results if r.get("success")]
    print(f"Successful tests: {len(successful)} / {len(results)}")

    if successful:
        if test_type == "fig1":
            print("\nNs\t\tFree Energy\t\tTime (s)")
            print("-" * 50)
            for r in successful:
                print(f"{r['Ns']}\t\t{r['free_energy']:.10f}\t{r['total_time_s']:.2f}")
        else:
            print("\nχN\tNx\tFree Energy\t\tTime (s)")
            print("-" * 60)
            for r in successful:
                print(f"{r['chi_n']}\t{r['Nx']}\t{r['free_energy']:.10f}\t{r['total_time_s']:.2f}")


if __name__ == "__main__":
    main()
