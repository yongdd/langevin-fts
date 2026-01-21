#!/usr/bin/env python3
"""
Test SDC-4 method on Gyroid phase (same conditions as benchmark).
Conditions: f = 0.375, χN = 18, M = 32³, L = 3.65, Ns = 40
"""

import os
import sys
import time
import numpy as np

# Set environment before importing polymerfts
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "0"

from polymerfts import scft

def main():
    # Benchmark conditions from NumericalMethodsPerformance.md
    f = 0.375
    chi_n = 18.0
    nx = 32
    lx = 3.65
    N = 80  # Ns = 80 (contour steps)
    ds = 1.0 / N

    print(f"SDC-4 Gyroid Benchmark")
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
        "numerical_method": "sdc-4",
        "reduce_memory": False,
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

    # Initialize gyroid field
    print("\nInitializing gyroid field...")
    x = np.linspace(0, 2*np.pi, nx, endpoint=False)
    y = np.linspace(0, 2*np.pi, nx, endpoint=False)
    z = np.linspace(0, 2*np.pi, nx, endpoint=False)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Gyroid approximation
    w_gyroid = (np.sin(X)*np.cos(Y) + np.sin(Y)*np.cos(Z) + np.sin(Z)*np.cos(X))
    w_gyroid = w_gyroid / np.std(w_gyroid) * 0.3

    initial_fields = {
        "A": w_gyroid,
        "B": -w_gyroid
    }

    # Run SCFT
    print("\nRunning SCFT with SDC-4...")
    start_time = time.time()

    calculation = scft.SCFT(params=params)
    calculation.run(initial_fields=initial_fields)

    elapsed_time = time.time() - start_time

    # Results
    print(f"\n" + "=" * 50)
    print(f"Results:")
    print(f"  Free energy: {calculation.free_energy:.10f}")
    print(f"  Iterations: {calculation.iter}")
    print(f"  Error level: {calculation.error_level:.2e}")
    print(f"  Elapsed time: {elapsed_time:.1f} s")
    print(f"=" * 50)

    # Compare with documented value
    expected_F = -0.47933271
    diff = abs(calculation.free_energy - expected_F)
    print(f"\nComparison with documented value:")
    print(f"  Expected F (SDC-4, Ns=40): {expected_F}")
    print(f"  Difference: {diff:.2e}")

if __name__ == "__main__":
    main()
