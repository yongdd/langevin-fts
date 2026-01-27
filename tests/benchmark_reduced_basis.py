#!/usr/bin/env python3
"""
Benchmark script to measure performance of reduced basis propagator storage.

Compares:
1. Standard mode (full grid propagators)
2. Reduced basis mode (space group symmetry)
"""

import os
import time
import numpy as np
from scipy.ndimage import gaussian_filter
from polymerfts import scft

# Suppress OpenMP threading for consistent measurements
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"

# Major Simulation params
f = 24/90       # A-fraction of major BCP chain, f

def create_initial_fields(nx, lx):
    """Create BCC initial fields."""
    w_A = np.zeros(list(nx), dtype=np.float64)
    w_B = np.zeros(list(nx), dtype=np.float64)

    n_unitcell = 1
    sphere_positions = []
    for i in range(0,n_unitcell):
        for j in range(0,n_unitcell):
            for k in range(0,n_unitcell):
                sphere_positions.append([i/n_unitcell,j/n_unitcell,k/n_unitcell])
                sphere_positions.append([(i+1/2)/n_unitcell,(j+1/2)/n_unitcell,(k+1/2)/n_unitcell])
    for x,y,z in sphere_positions:
        molecules, my, mz = np.round((np.array([x, y, z])*nx)).astype(np.int32)
        w_A[molecules,my,mz] = -1/(np.prod(lx)/np.prod(nx))
    w_A = gaussian_filter(w_A, sigma=np.min(nx)/15, mode='wrap')
    return w_A, w_B

# Common parameters for AB diblock in BCC phase
def get_params(use_space_group=False):
    params = {
        "nx": [32, 32, 32],
        "lx": [1.9, 1.9, 1.9],
        "reduce_memory": False,
        "box_is_altering": False,
        "chain_model": "continuous",
        "ds": 1/90,
        "segment_lengths": {"A": 1.0, "B": 1.0},
        "chi_n": {"A,B": 18.1},
        "distinct_polymers": [{
            "volume_fraction": 1.0,
            "blocks": [
                {"type": "A", "length": f, "v": 0, "u": 1},
                {"type": "B", "length": 1-f, "v": 1, "u": 2},
            ],
        }],
        "optimizer": {
            "name": "am",
            "max_hist": 20,
            "start_error": 1e-2,
            "mix_min": 0.1,
            "mix_init": 0.1,
        },
        "max_iter": 20,
        "tolerance": 1e-10,
        "verbose_level": 0,
        "platform": "cpu-fftw",
    }

    if use_space_group:
        params["space_group"] = {"symbol": "Im-3m", "number": 529}

    return params

def run_benchmark(use_space_group=False, n_iter=20):
    """Run SCFT simulation."""
    params = get_params(use_space_group)
    params["max_iter"] = n_iter

    w_A, w_B = create_initial_fields(params["nx"], params["lx"])

    simulation = scft.SCFT(params)
    simulation.run(initial_fields={"A": w_A, "B": w_B})

    return simulation

if __name__ == "__main__":
    print("=" * 60)
    print("Reduced Basis Propagator Performance Benchmark")
    print("=" * 60)
    print()

    n_runs = 3
    n_iter = 20  # Number of SCFT iterations per run

    # Warm-up run
    print("Warm-up run (without space group)...")
    run_benchmark(use_space_group=False, n_iter=5)

    # Benchmark without space group
    print(f"\nBenchmarking without space group ({n_runs} runs, {n_iter} iter each)...")
    times_no_sg = []
    for i in range(n_runs):
        start = time.perf_counter()
        sim = run_benchmark(use_space_group=False, n_iter=n_iter)
        elapsed = time.perf_counter() - start
        times_no_sg.append(elapsed)
        Q = sim.prop_solver.get_partition_function(0)
        print(f"  Run {i+1}: {elapsed:.3f}s, Q = {Q:.10f}")

    mean_no_sg = np.mean(times_no_sg)
    std_no_sg = np.std(times_no_sg)

    # Warm-up run
    print("\nWarm-up run (with space group Im-3m)...")
    run_benchmark(use_space_group=True, n_iter=5)

    # Benchmark with space group
    print(f"\nBenchmarking with space group Im-3m ({n_runs} runs, {n_iter} iter each)...")
    times_sg = []
    for i in range(n_runs):
        start = time.perf_counter()
        sim = run_benchmark(use_space_group=True, n_iter=n_iter)
        elapsed = time.perf_counter() - start
        times_sg.append(elapsed)
        Q = sim.prop_solver.get_partition_function(0)
        print(f"  Run {i+1}: {elapsed:.3f}s, Q = {Q:.10f}")

    mean_sg = np.mean(times_sg)
    std_sg = np.std(times_sg)

    # Summary
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Without space group: {mean_no_sg:.3f} ± {std_no_sg:.3f} s")
    print(f"With space group:    {mean_sg:.3f} ± {std_sg:.3f} s")
    overhead = ((mean_sg/mean_no_sg) - 1) * 100
    print(f"Overhead:            {overhead:+.1f}%")
    print()
    print("Memory usage (propagators):")
    print(f"  Full grid:      32 x 32 x 32 = 32768 points")
    print(f"  Reduced basis:  489 points (Im-3m)")
    print(f"  Memory savings: {(1 - 489/32768) * 100:.1f}%")
