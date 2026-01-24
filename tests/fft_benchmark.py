#!/usr/bin/env python3
"""
FFT Backend Performance Comparison: MKL vs FFTW

This script compares the performance of MKL and FFTW backends for
FFT operations in polymer field theory simulations.

Usage:
    python fft_benchmark.py [--nx 64] [--iterations 100]
"""

import sys
import time
import argparse
import numpy as np

try:
    from polymerfts import PropagatorSolver, PlatformSelector
except ImportError:
    print("Error: polymerfts module not found. Please install it first.")
    sys.exit(1)

def run_benchmark(platform, nx, n_iterations, chain_model="continuous", bc_type="periodic"):
    """Run a single benchmark for a given platform."""

    # Check if platform is available
    available = PlatformSelector.avail_platforms()
    if platform not in available:
        print(f"  {platform}: Not available")
        return None

    # Setup simulation parameters
    lx = [10.0] * len(nx)
    ds = 0.01
    f = 0.3

    # Boundary conditions: 2 per dimension (low, high)
    bc = [bc_type] * (len(nx) * 2)

    # Bond lengths
    bond_lengths = {"A": 1.0, "B": 1.0}

    # Create solver
    solver = PropagatorSolver(
        nx=nx, lx=lx,
        ds=ds,
        bond_lengths=bond_lengths,
        bc=bc,
        chain_model=chain_model,
        numerical_method="rqm4",
        platform=platform,
        reduce_memory=False
    )

    # Add AB diblock polymer
    solver.add_polymer(
        volume_fraction=1.0,
        blocks=[["A", f, 0, 1], ["B", 1-f, 1, 2]]
    )

    n_grid = solver.n_grid

    # Create potential fields
    w = {}
    w_A = np.zeros(n_grid, dtype=np.float64)
    w_B = np.zeros(n_grid, dtype=np.float64)

    # Initialize with cosine field
    if len(nx) == 3:
        for i in range(nx[0]):
            for j in range(nx[1]):
                for k in range(nx[2]):
                    idx = i*nx[1]*nx[2] + j*nx[2] + k
                    w_A[idx] = np.cos(2.0*np.pi*i/4.68)*np.cos(2.0*np.pi*j/3.48)*np.cos(2.0*np.pi*k/2.74)*0.1
    else:
        w_A[:] = np.random.randn(n_grid) * 0.1
    w_B = -w_A.copy()

    # Warmup
    for _ in range(5):
        solver.compute_propagators({"A": w_A, "B": w_B})

    # Benchmark
    times = []
    for i in range(n_iterations):
        # Slightly modify fields to avoid caching
        w_A += np.random.randn(n_grid) * 0.001
        w_B = -w_A

        start = time.perf_counter()
        solver.compute_propagators({"A": w_A, "B": w_B})
        end = time.perf_counter()
        times.append(end - start)

    mean_time = np.mean(times)
    std_time = np.std(times)

    return mean_time, std_time

def main():
    parser = argparse.ArgumentParser(description="FFT Backend Performance Comparison")
    parser.add_argument("--nx", type=int, nargs='+', default=[64, 64, 64],
                        help="Grid dimensions (default: 64 64 64)")
    parser.add_argument("--iterations", type=int, default=50,
                        help="Number of iterations (default: 50)")
    parser.add_argument("--chain-model", choices=["continuous", "discrete"],
                        default="continuous", help="Chain model (default: continuous)")
    parser.add_argument("--bc", choices=["periodic", "reflecting", "absorbing"],
                        default="periodic", help="Boundary condition type (default: periodic)")
    args = parser.parse_args()

    nx = args.nx
    n_iterations = args.iterations
    chain_model = args.chain_model
    bc_type = args.bc

    print("=" * 60)
    print("FFT Backend Performance Comparison")
    print("=" * 60)
    print(f"Grid: {nx}")
    print(f"Chain model: {chain_model}")
    print(f"Boundary condition: {bc_type}")
    print(f"Iterations: {n_iterations}")
    print()

    # Get available platforms
    available = PlatformSelector.avail_platforms()
    print(f"Available platforms: {available}")
    print()

    results = {}

    # Test cpu-mkl
    if "cpu-mkl" in available:
        print("Testing cpu-mkl...")
        result = run_benchmark("cpu-mkl", nx, n_iterations, chain_model, bc_type)
        if result:
            results["cpu-mkl"] = result
            print(f"  cpu-mkl: {result[0]*1000:.2f} +/- {result[1]*1000:.2f} ms per iteration")

    # Test cpu-fftw
    if "cpu-fftw" in available:
        print("Testing cpu-fftw...")
        result = run_benchmark("cpu-fftw", nx, n_iterations, chain_model, bc_type)
        if result:
            results["cpu-fftw"] = result
            print(f"  cpu-fftw: {result[0]*1000:.2f} +/- {result[1]*1000:.2f} ms per iteration")

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)

    if len(results) >= 2:
        mkl_time = results.get("cpu-mkl", (None, None))[0]
        fftw_time = results.get("cpu-fftw", (None, None))[0]

        if mkl_time and fftw_time:
            speedup = fftw_time / mkl_time
            if speedup > 1:
                print(f"MKL is {speedup:.2f}x faster than FFTW")
            else:
                print(f"FFTW is {1/speedup:.2f}x faster than MKL")

    for name, (mean, std) in sorted(results.items(), key=lambda x: x[1][0]):
        print(f"  {name}: {mean*1000:.2f} +/- {std*1000:.2f} ms")

if __name__ == "__main__":
    main()
