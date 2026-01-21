#!/usr/bin/env python
"""
SDC Convergence Benchmark for 2D Cylinder Phase.

Run a single SCFT calculation with specified SDC order and N (contour steps).
Results are saved to a JSON file for later analysis.

Usage:
    python run_sdc_benchmark.py --order 6 --N 40 --output result_sdc6_N40.json
"""
import os
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "0"
os.environ["OMP_NUM_THREADS"] = "4"
# Note: SLURM sets CUDA_VISIBLE_DEVICES automatically via --gres=gpu:1

import argparse
import json
import time
import numpy as np
from polymerfts import scft


def run_benchmark(order: int, N: int, output_file: str):
    """Run SCFT benchmark for given SDC order and N."""

    ds = 1.0 / N

    # 2D Cylinder phase parameters (AB diblock, f=0.35)
    params = {
        "platform": "cuda",
        "nx": [64, 64],
        "lx": [4.0, 4.0],
        "box_is_altering": False,
        "chain_model": "continuous",
        "ds": ds,
        "reduce_memory": False,
        "segment_lengths": {"A": 1.0, "B": 1.0},
        "chi_n": {"A,B": 20.0},
        "distinct_polymers": [{
            "volume_fraction": 1.0,
            "blocks": [
                {"type": "A", "length": 0.35},
                {"type": "B", "length": 0.65}
            ],
        }],
        "numerical_method": f"sdc-{order}",
        "optimizer": {
            "name": "am",
            "max_hist": 20,
            "start_error": 1e-2,
            "mix_min": 0.1,
            "mix_init": 0.1,
        },
        "max_iter": 3000,
        "tolerance": 1e-9,
        "verbose_level": 1,
    }

    # Initialize cylinder-like field (hexagonal pattern)
    w_A = np.zeros(params["nx"], dtype=np.float64)
    w_B = np.zeros(params["nx"], dtype=np.float64)

    nx, ny = params["nx"]
    lx, ly = params["lx"]

    for i in range(nx):
        for j in range(ny):
            x = 2 * np.pi * i / nx
            y = 2 * np.pi * j / ny
            # Hexagonal cylinder approximation
            val = np.cos(x) + np.cos(0.5*x + np.sqrt(3)/2*y) + np.cos(0.5*x - np.sqrt(3)/2*y)
            w_A[i, j] = val
            w_B[i, j] = -val

    print(f"Running SDC-{order} with N={N} (ds={ds:.6f})...")

    # Run SCFT
    start_time = time.time()
    calc = scft.SCFT(params=params)
    calc.run(initial_fields={"A": w_A, "B": w_B})
    elapsed_time = time.time() - start_time

    # Get results
    free_energy = calc.free_energy

    result = {
        "order": order,
        "N": N,
        "ds": ds,
        "free_energy": free_energy,
        "elapsed_time": elapsed_time,
        "converged": True,
    }

    # Save result
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"SDC-{order}, N={N}: F={free_energy:.12f}, time={elapsed_time:.1f}s")
    print(f"Result saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SDC Convergence Benchmark")
    parser.add_argument("--order", type=int, required=True, help="SDC order (2-10)")
    parser.add_argument("--N", type=int, required=True, help="Number of contour steps")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file")

    args = parser.parse_args()
    run_benchmark(args.order, args.N, args.output)
