#!/usr/bin/env python3
"""
Test SDC solver performance and material conservation.

This script benchmarks the SDC (Spectral Deferred Correction) solver
against other numerical methods (RQM4, CN-ADI2) in terms of:
1. Material conservation (mass error)
2. Performance (time per iteration)
3. Convergence behavior (Q vs ds)
"""

import os
import sys
import time
import numpy as np
import io
from contextlib import redirect_stdout

os.environ["OMP_MAX_ACTIVE_LEVELS"] = "0"

from polymerfts import SCFT


def run_test(nx, lx, ds, method, platform, n_iter=1, verbose=False):
    """Run SCFT iteration and return results."""

    params = {
        "nx": nx,
        "lx": lx,
        "chain_model": "continuous",
        "ds": ds,
        "segment_lengths": {"A": 1.0, "B": 1.0},
        "chi_n": {"A,B": 20.0},
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
        "box_is_altering": False,
        "max_iter": n_iter,
        "tolerance": 1e-10,
        "numerical_method": method,
        "platform": platform,
    }

    # Suppress SCFT output unless verbose
    if verbose:
        scft = SCFT(params)
    else:
        with redirect_stdout(io.StringIO()):
            scft = SCFT(params)

    # Generate random fields with std ~ 5
    n_grid = int(np.prod(nx))
    np.random.seed(42)
    w_A = np.random.normal(0.0, 5.0, n_grid)
    w_B = -w_A

    # Run SCFT with return_result=True
    if verbose:
        start = time.perf_counter()
        result = scft.run({"A": w_A, "B": w_B}, return_result=True)
        elapsed = time.perf_counter() - start
    else:
        with redirect_stdout(io.StringIO()):
            start = time.perf_counter()
            result = scft.run({"A": w_A, "B": w_B}, return_result=True)
            elapsed = time.perf_counter() - start

    # Get partition function (first polymer type)
    Q = result.partition_functions[0]

    # Compute mass error from phi
    phi_total = result.phi["A"] + result.phi["B"]
    mass_error = np.mean(phi_total - 1.0)

    return {
        "Q": Q,
        "mass_error": mass_error,
        "time_ms": elapsed * 1000,
    }


def test_material_conservation():
    """Test material conservation for different methods."""
    print("\n" + "=" * 75)
    print("Material Conservation Test")
    print("=" * 75)

    configs = [
        {"nx": [32, 32], "lx": [4.0, 4.0], "ds": 1/64, "name": "2D 32x32"},
        {"nx": [32, 32, 32], "lx": [4.0, 4.0, 4.0], "ds": 1/64, "name": "3D 32^3"},
    ]

    methods = ["rqm4", "cn-adi2", "sdc-2", "sdc-4"]

    for cfg in configs:
        print(f"\n{cfg['name']} (ds={cfg['ds']:.4f}, periodic BC, CUDA)")
        print("-" * 75)
        print(f"{'Method':<12} {'Q':<18} {'Mass Error':<18} {'Time (ms)':<12}")
        print("-" * 75)

        for method in methods:
            try:
                result = run_test(cfg["nx"], cfg["lx"], cfg["ds"], method, "cuda")
                print(f"{method:<12} {result['Q']:<18.10f} {result['mass_error']:<18.2e} {result['time_ms']:<12.2f}")
            except Exception as e:
                print(f"{method:<12} ERROR: {str(e)[:50]}")


def test_convergence():
    """Test convergence order for different methods."""
    print("\n" + "=" * 75)
    print("Convergence Test (2D 32x32)")
    print("=" * 75)

    nx = [32, 32]
    lx = [4.0, 4.0]
    ds_values = [1/16, 1/32, 1/64, 1/128]
    methods = ["rqm4", "cn-adi2", "sdc-2", "sdc-4"]

    print("\nQ vs ds:")
    print("-" * 80)
    print(f"{'ds':<10}", end="")
    for method in methods:
        print(f"{method:<18}", end="")
    print()
    print("-" * 80)

    results = {m: [] for m in methods}
    for ds in ds_values:
        print(f"{ds:<10.4f}", end="")
        for method in methods:
            try:
                result = run_test(nx, lx, ds, method, "cuda")
                results[method].append(result["Q"])
                print(f"{result['Q']:<18.10f}", end="")
            except Exception as e:
                results[method].append(None)
                print(f"{'ERROR':<18}", end="")
        print()

    print("\nConvergence order (log(|dQ|)/log(ds_ratio)):")
    print("-" * 80)
    for method in methods:
        Q_list = results[method]
        if None in Q_list:
            print(f"{method}: ERROR")
            continue
        # Compute errors relative to finest resolution
        Q_ref = Q_list[-1]
        errors = [abs(Q - Q_ref) for Q in Q_list[:-1]]
        if all(e > 1e-15 for e in errors):
            orders = []
            for i in range(len(errors)-1):
                ratio = errors[i] / errors[i+1]
                order = np.log(ratio) / np.log(2)
                orders.append(order)
            print(f"{method:<12} {' -> '.join([f'{o:.2f}' for o in orders])}")
        else:
            print(f"{method:<12} Errors too small to compute order")


def test_performance():
    """Test performance across different grid sizes."""
    print("\n" + "=" * 75)
    print("Performance Test (CUDA, 5 runs)")
    print("=" * 75)

    configs = [
        {"nx": [32, 32], "lx": [4.0, 4.0], "ds": 1/64, "name": "2D 32x32"},
        {"nx": [64, 64], "lx": [8.0, 8.0], "ds": 1/64, "name": "2D 64x64"},
        {"nx": [32, 32, 32], "lx": [4.0, 4.0, 4.0], "ds": 1/64, "name": "3D 32^3"},
    ]

    methods = ["rqm4", "cn-adi2", "sdc-2", "sdc-4"]

    for cfg in configs:
        print(f"\n{cfg['name']}")
        print("-" * 50)
        print(f"{'Method':<12} {'Time (ms)':<20}")
        print("-" * 50)

        for method in methods:
            try:
                times = []
                for _ in range(5):
                    result = run_test(cfg["nx"], cfg["lx"], cfg["ds"], method, "cuda")
                    times.append(result["time_ms"])
                print(f"{method:<12} {np.mean(times):<8.2f} ± {np.std(times):<8.2f}")
            except Exception as e:
                print(f"{method:<12} ERROR: {str(e)[:35]}")


def main():
    print("=" * 75)
    print("SDC Solver Performance and Material Conservation Test")
    print("=" * 75)

    test_material_conservation()
    test_convergence()
    test_performance()

    print("\n" + "=" * 75)
    print("Summary:")
    print("- Mass error should be ~1e-13 to 1e-17 for good material conservation")
    print("- SDC-2 achieves ~2-3rd order convergence")
    print("- SDC-4 achieves ~3-4th order convergence")
    print("- SDC methods are slower than RQM4/CN-ADI due to iterative PCG solver")
    print("=" * 75)


if __name__ == "__main__":
    main()
