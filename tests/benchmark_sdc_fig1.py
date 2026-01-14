#!/usr/bin/env python3
"""
Benchmark SDC methods for Figure 1 in NumericalMethodsPerformance.md
Usage: python benchmark_sdc_fig1.py --method sdc-4 --ns 40
"""

import os
import sys
import time
import json
import argparse
import numpy as np

os.environ["OMP_MAX_ACTIVE_LEVELS"] = "0"

from polymerfts import scft

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True, help="sdc-4 or sdc-6")
    parser.add_argument("--ns", type=int, required=True, help="Number of contour steps")
    args = parser.parse_args()

    # Benchmark conditions from NumericalMethodsPerformance.md
    f = 0.375
    chi_n = 18.0
    nx = 32
    lx = 3.65
    N = args.ns
    ds = 1.0 / N

    print(f"{args.method.upper()} Gyroid Benchmark")
    print(f"=" * 50)
    print(f"f = {f}, χN = {chi_n}")
    print(f"Grid: {nx}³, L = {lx}")
    print(f"Ns = {N}, ds = {ds}")
    print(f"=" * 50)

    params = {
        "platform": "cuda",
        "nx": [nx, nx, nx],
        "lx": [lx, lx, lx],
        "chain_model": "continuous",
        "ds": ds,
        "segment_lengths": {"A": 1.0, "B": 1.0},
        "chi_n": {"A,B": chi_n},
        "distinct_polymers": [{
            "volume_fraction": 1.0,
            "blocks": [
                {"type": "A", "length": f},
                {"type": "B", "length": 1.0 - f}
            ]
        }],
        "numerical_method": args.method,
        "reduce_memory_usage": False,
        "box_is_altering": False,
        "optimizer": {
            "name": "am",
            "max_hist": 20,
            "start_error": 1e-2,
            "mix_min": 0.1,
            "mix_init": 0.1,
        },
        "max_iter": 2000,
        "tolerance": 1e-9,
        "verbose_level": 1,
    }

    # Initialize gyroid field (proper form from GyroidNoBoxChange.py)
    print("\nInitializing gyroid field...")
    w_A = np.zeros([nx, nx, nx], dtype=np.float64)
    w_B = np.zeros([nx, nx, nx], dtype=np.float64)

    for i in range(nx):
        xx = (i+1)*2*np.pi/nx
        for j in range(nx):
            yy = (j+1)*2*np.pi/nx
            zz = np.arange(1, nx+1)*2*np.pi/nx

            c1 = np.sqrt(8.0/3.0)*(np.cos(xx)*np.sin(yy)*np.sin(2.0*zz) +
                np.cos(yy)*np.sin(zz)*np.sin(2.0*xx)+np.cos(zz)*np.sin(xx)*np.sin(2.0*yy))
            c2 = np.sqrt(4.0/3.0)*(np.cos(2.0*xx)*np.cos(2.0*yy)+
                np.cos(2.0*yy)*np.cos(2.0*zz)+np.cos(2.0*zz)*np.cos(2.0*xx))
            w_A[i,j,:] = -0.3164*c1 + 0.1074*c2
            w_B[i,j,:] =  0.3164*c1 - 0.1074*c2

    initial_fields = {"A": w_A, "B": w_B}

    # Run SCFT
    print(f"\nRunning SCFT with {args.method}...")
    start_time = time.time()

    calculation = scft.SCFT(params=params)
    calculation.run(initial_fields=initial_fields)

    elapsed_time = time.time() - start_time

    # Results
    print(f"\n" + "=" * 50)
    print(f"Results:")
    print(f"  Free energy: {calculation.free_energy:.15f}")
    print(f"  Iterations: {calculation.iter}")
    print(f"  Error level: {calculation.error_level:.2e}")
    print(f"  Elapsed time: {elapsed_time:.1f} s")
    print(f"=" * 50)

    # Save results to JSON
    result = {
        "method": args.method,
        "ds": ds,
        "N": N,
        "chi_n": chi_n,
        "f": f,
        "nx": nx,
        "lx": lx,
        "dim": 3,
        "phase": "gyroid",
        "platform": "cuda",
        "free_energy": float(calculation.free_energy),
        "error_level": float(calculation.error_level),
        "iterations": int(calculation.iter),
        "converged": bool(calculation.error_level < 1e-9),
        "elapsed_time": float(elapsed_time)
    }

    output_file = f"tests/sdc_gyroid_benchmark/result_{args.method}_N{N}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()
