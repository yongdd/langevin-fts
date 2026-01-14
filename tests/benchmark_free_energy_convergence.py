#!/usr/bin/env python3
"""
Free Energy Convergence Benchmark

This script benchmarks the convergence of the partition function Q (related to free energy)
for all available numerical methods as ds → 0.

Methods tested:
- Pseudo-Spectral:
  - RQM4: 4th-order Richardson extrapolation (Ranjan, Qin, Morse 2008)
  - ETDRK4: Exponential Time Differencing RK4 (Cox & Matthews 2002)

- Real-Space (CN-ADI based):
  - cn-adi2: 2nd-order Crank-Nicolson ADI
  - cn-adi4-lr: 4th-order CN-ADI with per-step Richardson extrapolation

The convergence order p is determined from the error scaling: error ∝ ds^p

References:
- RQM4: Ranjan, Qin & Morse, Macromolecules 41, 942-954 (2008)
- Pseudo-spectral benchmarks: Stasiak & Matsen, Eur. Phys. J. E 34, 110 (2011)
"""

import os
import sys
import time
import numpy as np
from datetime import datetime

os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
os.environ["OMP_NUM_THREADS"] = "4"

from polymerfts._core import PlatformSelector


def create_lamellar_field(nx, lx, chi_n=12.0, n_periods=3):
    """Create a lamellar-like potential field in z-direction."""
    dim = len(nx)

    if dim == 1:
        x = (np.arange(nx[0]) + 0.5) * lx[0] / nx[0]
        w = chi_n * 0.5 * np.cos(2 * np.pi * n_periods * x / lx[0])
    elif dim == 2:
        y = (np.arange(nx[1]) + 0.5) * lx[1] / nx[1]
        w = np.zeros(nx)
        for j in range(nx[1]):
            w[:, j] = chi_n * 0.5 * np.cos(2 * np.pi * n_periods * y[j] / lx[1])
        w = w.flatten()
    else:  # 3D
        z = (np.arange(nx[2]) + 0.5) * lx[2] / nx[2]
        w = np.zeros(nx)
        for k in range(nx[2]):
            w[:, :, k] = chi_n * 0.5 * np.cos(2 * np.pi * n_periods * z[k] / lx[2])
        w = w.flatten()

    return w


def benchmark_method(platform, numerical_method, nx, lx, ds, chi_n=12.0, n_warmup=2, n_runs=3):
    """
    Benchmark a specific numerical method.

    Returns partition function Q and computation time.
    """
    w_a = create_lamellar_field(nx, lx, chi_n=chi_n)

    bond_lengths = {"A": 1.0}
    factory = PlatformSelector.create_factory(platform, False)
    molecules = factory.create_molecules_information("continuous", ds, bond_lengths)
    molecules.add_polymer(1.0, [["A", 1.0, 0, 1]])

    # Use periodic BC for real-space methods
    if numerical_method in ["cn-adi2", "cn-adi4-lr"]:
        bc = ["periodic"] * (2 * len(nx))
        cb = factory.create_computation_box(nx=list(nx), lx=list(lx), bc=bc)
    else:
        cb = factory.create_computation_box(nx=list(nx), lx=list(lx), bc=[])

    prop_opt = factory.create_propagator_computation_optimizer(molecules, True)
    solver = factory.create_propagator_computation(cb, molecules, prop_opt, numerical_method)

    # Warmup
    for _ in range(n_warmup):
        solver.compute_propagators({"A": w_a})

    # Benchmark
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        solver.compute_propagators({"A": w_a})
        times.append(time.perf_counter() - t0)

    Q = solver.get_total_partition(0)
    avg_time = np.mean(times) * 1000  # ms

    return Q, avg_time


def compute_convergence_order(ds_values, Q_values, Q_ref):
    """Compute convergence order from error analysis."""
    errors = []
    ds_valid = []

    for i in range(len(ds_values) - 1):  # Exclude reference
        err = abs(Q_values[i] - Q_ref)
        if err > 1e-15:  # Skip machine-precision values
            errors.append(err)
            ds_valid.append(ds_values[i])

    if len(errors) < 2:
        return None, []

    # Compute order from consecutive pairs
    orders = []
    for i in range(len(errors) - 1):
        if errors[i] > 1e-15 and errors[i+1] > 1e-15:
            p = np.log(errors[i] / errors[i+1]) / np.log(ds_valid[i] / ds_valid[i+1])
            orders.append(p)

    return np.mean(orders) if orders else None, errors


def run_benchmark(platform="cuda"):
    """Run the full convergence benchmark."""

    print("=" * 80)
    print("FREE ENERGY CONVERGENCE BENCHMARK")
    print("=" * 80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Platform: {platform}")
    print()

    # Test configuration
    nx = [32, 32, 32]
    lx = [4.0, 4.0, 4.0]
    chi_n = 12.0

    print("Test configuration:")
    print(f"  Grid: {nx[0]} x {nx[1]} x {nx[2]}")
    print(f"  Box:  {lx[0]} x {lx[1]} x {lx[2]}")
    print(f"  chi_N = {chi_n}")
    print(f"  Lamellar field with 3 periods")
    print()

    # ds values to test
    ds_values = [1/20, 1/40, 1/80, 1/160, 1/320, 1/640]

    # Methods to benchmark
    methods = {
        "rqm4": "RQM4 (Pseudo-Spectral, 4th order)",
        "etdrk4": "ETDRK4 (Pseudo-Spectral, 4th order)",
        "cn-adi2": "CN-ADI2 (Real-Space, 2nd order)",
        "cn-adi4-lr": "CN-ADI4 (Real-Space, 4th order per-step)"
    }

    results = {}

    for method, description in methods.items():
        print(f"Testing {description}...")
        results[method] = {"Q": [], "time_ms": [], "description": description}

        for ds in ds_values:
            try:
                Q, t = benchmark_method(platform, method, nx, lx, ds, chi_n)
                results[method]["Q"].append(Q)
                results[method]["time_ms"].append(t)
                N = int(round(1.0/ds))
                print(f"  N={N:4d}: Q={Q:.12f}, time={t:.1f}ms")
            except Exception as e:
                print(f"  Failed for ds={ds}: {e}")
                results[method]["Q"].append(None)
                results[method]["time_ms"].append(None)
        print()

    # Print convergence table
    print("=" * 80)
    print("PARTITION FUNCTION Q vs CONTOUR STEPS N")
    print("=" * 80)

    header = f"{'N':>6}"
    for method in methods:
        header += f"  {method:>14}"
    print(header)
    print("-" * 80)

    for i, ds in enumerate(ds_values):
        N = int(round(1.0/ds))
        row = f"{N:>6}"
        for method in methods:
            Q = results[method]["Q"][i]
            if Q is not None:
                row += f"  {Q:>14.10f}"
            else:
                row += f"  {'N/A':>14}"
        print(row)

    # Compute and print convergence orders
    print()
    print("=" * 80)
    print("CONVERGENCE ORDER ANALYSIS")
    print("=" * 80)
    print()
    print("Error = |Q - Q_ref| where Q_ref is the finest discretization (N=640)")
    print("Convergence order p: Error ∝ ds^p")
    print()

    print(f"{'Method':<20} {'Order p':>10} {'Expected':>10} {'Q_ref':>20}")
    print("-" * 60)

    expected_orders = {
        "rqm4": 4.0,
        "etdrk4": 4.0,
        "cn-adi2": 2.0,
        "cn-adi4-lr": 4.0
    }

    for method in methods:
        Q_vals = [q for q in results[method]["Q"] if q is not None]
        if len(Q_vals) >= 2:
            Q_ref = Q_vals[-1]
            order, errors = compute_convergence_order(
                ds_values[:len(Q_vals)], Q_vals, Q_ref
            )

            if order is not None:
                print(f"{method:<20} {order:>10.2f} {expected_orders[method]:>10.1f} {Q_ref:>20.12f}")
            else:
                print(f"{method:<20} {'N/A':>10} {expected_orders[method]:>10.1f} {Q_ref:>20.12f}")
        else:
            print(f"{method:<20} {'N/A':>10} {expected_orders[method]:>10.1f} {'N/A':>20}")

    # Print error table
    print()
    print("=" * 80)
    print("ERROR |Q - Q_ref| FOR EACH METHOD")
    print("=" * 80)

    header = f"{'N':>6}"
    for method in methods:
        header += f"  {method:>12}"
    print(header)
    print("-" * 80)

    for i, ds in enumerate(ds_values[:-1]):  # Exclude reference
        N = int(round(1.0/ds))
        row = f"{N:>6}"
        for method in methods:
            Q_vals = [q for q in results[method]["Q"] if q is not None]
            if len(Q_vals) > i and Q_vals[i] is not None:
                Q_ref = Q_vals[-1]
                err = abs(Q_vals[i] - Q_ref)
                row += f"  {err:>12.2e}"
            else:
                row += f"  {'N/A':>12}"
        print(row)

    # Print timing comparison
    print()
    print("=" * 80)
    print("COMPUTATION TIME (ms)")
    print("=" * 80)

    header = f"{'N':>6}"
    for method in methods:
        header += f"  {method:>12}"
    print(header)
    print("-" * 80)

    for i, ds in enumerate(ds_values):
        N = int(round(1.0/ds))
        row = f"{N:>6}"
        for method in methods:
            t = results[method]["time_ms"][i]
            if t is not None:
                row += f"  {t:>12.1f}"
            else:
                row += f"  {'N/A':>12}"
        print(row)

    # Summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
Key findings:

1. PSEUDO-SPECTRAL METHODS (rqm4, etdrk4):
   - 4th-order convergence in ds
   - Best accuracy for periodic systems
   - Use FFT for O(N log N) efficiency

2. REAL-SPACE METHODS:
   - cn-adi2: 2nd-order, fastest per step but needs smaller ds for accuracy
   - cn-adi4-lr: 4th-order via per-step Richardson, 3x cost of cn-adi2

3. RECOMMENDATIONS:
   - For periodic systems: Use rqm4 (default) or etdrk4
   - For non-periodic BC: Use cn-adi4-lr for 4th-order accuracy
""")

    return results


if __name__ == "__main__":
    # Run on CUDA by default, fall back to CPU
    try:
        results = run_benchmark("cuda")
    except Exception as e:
        print(f"CUDA failed: {e}")
        print("Falling back to CPU...")
        results = run_benchmark("cpu-mkl")
