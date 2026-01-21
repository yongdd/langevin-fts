#!/usr/bin/env python3
"""
Benchmark SDC solver performance with warm-starting PCG.

This script measures the time for SDC propagator computation
to evaluate the performance impact of warm-starting.
"""

import os
import sys
import time
import numpy as np

# Set environment before importing
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "0"

from polymerfts import SCFT

def run_benchmark(nx, lx, ds, n_iterations=5):
    """Run SDC benchmark and return average time per SCFT iteration."""

    # Setup parameters for SCFT (simpler than LFTS)
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
        "max_iter": 1,  # Just 1 iteration for timing
        "tolerance": 1e-10,
        "numerical_method": "sdc-2",  # Use SDC solver
    }

    # Create SCFT solver
    scft = SCFT(params)

    # Generate random fields with moderate amplitude
    n_grid = np.prod(nx)
    np.random.seed(42)
    w_A = np.random.normal(0.0, 5.0, n_grid)
    w_B = -w_A

    # Warm-up run
    scft.run({"A": w_A, "B": w_B})

    # Timed runs
    times = []
    for i in range(n_iterations):
        # Slightly perturb fields
        w_A_perturbed = w_A + np.random.normal(0.0, 0.01, n_grid)
        w_B_perturbed = -w_A_perturbed

        start = time.perf_counter()
        scft.run({"A": w_A_perturbed, "B": w_B_perturbed})
        end = time.perf_counter()
        times.append(end - start)

    return np.mean(times), np.std(times)

def main():
    print("=" * 70)
    print("SDC Solver Performance Benchmark (Warm-starting PCG)")
    print("=" * 70)
    print()

    # Test configurations - focus on 3D where PCG is used and warm-start matters
    configs = [
        # 3D tests (PCG used) - larger grids to see warm-start benefit
        {"nx": [32, 32, 32], "lx": [4.0, 4.0, 4.0], "ds": 1/64, "name": "3D 32^3"},
        {"nx": [48, 48, 48], "lx": [6.0, 6.0, 6.0], "ds": 1/64, "name": "3D 48^3"},
        {"nx": [64, 64, 64], "lx": [8.0, 8.0, 8.0], "ds": 1/64, "name": "3D 64^3"},
    ]

    print(f"{'Configuration':<15} {'Grid Points':<12} {'Time (ms)':<12} {'Std (ms)':<10}")
    print("-" * 70)

    results = []
    for cfg in configs:
        n_grid = np.prod(cfg["nx"])
        try:
            mean_time, std_time = run_benchmark(
                cfg["nx"], cfg["lx"], cfg["ds"], n_iterations=5
            )
            results.append({
                "name": cfg["name"],
                "n_grid": n_grid,
                "mean_ms": mean_time * 1000,
                "std_ms": std_time * 1000
            })
            print(f"{cfg['name']:<15} {n_grid:<12} {mean_time*1000:<12.2f} {std_time*1000:<10.2f}")
        except Exception as e:
            print(f"{cfg['name']:<15} {n_grid:<12} ERROR: {e}")

    print()
    print("=" * 70)
    print("Current: Warm-start PCG (x0 = RHS)")
    print()
    print("Note: To compare, manually revert warm-start in CudaSolverSDC.cu")
    print("      and re-run this benchmark.")

if __name__ == "__main__":
    main()
