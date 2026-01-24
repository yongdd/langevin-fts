#!/usr/bin/env python3
"""
Benchmark: Numerical Methods with Gyroid and Fddd Phases

This script benchmarks all numerical methods on realistic polymer phases:
- Gyroid (Ia-3d): AB diblock, f=0.36, chi_N=20
- Fddd (O^70): AB diblock, f=0.43, chi_N=14

Methods tested:
- Pseudo-spectral: RQM4, ETDRK4
- Real-space: CN-ADI2, CN-ADI4

All methods are selected at runtime via the pseudo_method and realspace_method
parameters to PlatformSelector.create_factory().
"""

import os
import sys
import time
import numpy as np
import json
from datetime import datetime

# Set OpenMP environment
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
os.environ["OMP_NUM_THREADS"] = "4"

from polymerfts import _core


def create_gyroid_field(nx, chi_n=20.0):
    """
    Create Gyroid (Ia-3d) initial field for AB diblock.

    Reference: Hajduk et al., Macromolecules 1994, 27, 4063
    """
    w_A = np.zeros(nx, dtype=np.float64)
    w_B = np.zeros(nx, dtype=np.float64)

    for i in range(nx[0]):
        xx = (i + 1) * 2 * np.pi / nx[0]
        for j in range(nx[1]):
            yy = (j + 1) * 2 * np.pi / nx[1]
            for k in range(nx[2]):
                zz = (k + 1) * 2 * np.pi / nx[2]

                c1 = np.sqrt(8.0/3.0) * (
                    np.cos(xx) * np.sin(yy) * np.sin(2.0*zz) +
                    np.cos(yy) * np.sin(zz) * np.sin(2.0*xx) +
                    np.cos(zz) * np.sin(xx) * np.sin(2.0*yy))
                c2 = np.sqrt(4.0/3.0) * (
                    np.cos(2.0*xx) * np.cos(2.0*yy) +
                    np.cos(2.0*yy) * np.cos(2.0*zz) +
                    np.cos(2.0*zz) * np.cos(2.0*xx))

                w_A[i, j, k] = -0.3164 * c1 + 0.1074 * c2
                w_B[i, j, k] = 0.3164 * c1 - 0.1074 * c2

    return w_A.flatten(), w_B.flatten()


def create_fddd_field(nx, chi_n=14.0):
    """
    Create Fddd (O^70) initial field for AB diblock.

    Uses sinusoidal approximation for the orthorhombic network phase.
    """
    w_A = np.zeros(nx, dtype=np.float64)
    w_B = np.zeros(nx, dtype=np.float64)

    # Fddd has orthorhombic symmetry with specific wave vectors
    for i in range(nx[0]):
        x = (i + 0.5) * 2 * np.pi / nx[0]
        for j in range(nx[1]):
            y = (j + 0.5) * 2 * np.pi / nx[1]
            for k in range(nx[2]):
                z = (k + 0.5) * 2 * np.pi / nx[2]

                # Fddd structure factor approximation
                val = (np.sin(x) * np.sin(y) +
                       np.sin(y) * np.sin(z) +
                       np.sin(z) * np.sin(x))

                w_A[i, j, k] = -0.2 * val
                w_B[i, j, k] = 0.2 * val

    return w_A.flatten(), w_B.flatten()


def benchmark_method(platform, pseudo_method, realspace_method,
                     nx, lx, f, chi_n, ds, w_A, w_B,
                     solver_type="pseudospectral",
                     n_warmup=3, n_runs=5):
    """
    Benchmark a single numerical method configuration.

    Returns partition functions for both polymer blocks and timing.
    """
    # Create factory with specified numerical methods
    factory = _core.PlatformSelector.create_factory(
        platform, False, "real", pseudo_method, realspace_method)

    # Create molecules (AB diblock)
    bond_lengths = {"A": 1.0, "B": 1.0}
    molecules = factory.create_molecules_information("Continuous", ds, bond_lengths)
    molecules.add_polymer(1.0, [["A", f, 0, 1], ["B", 1.0 - f, 1, 2]])

    # Create computation box
    if solver_type == "pseudospectral":
        bc = []  # Periodic by default for pseudo-spectral
    else:
        bc = ["periodic"] * (2 * len(nx))  # Explicit periodic for real-space

    cb = factory.create_computation_box(nx=list(nx), lx=list(lx), bc=bc)

    # Create solver
    prop_opt = factory.create_propagator_computation_optimizer(molecules, True)
    if solver_type == "pseudospectral":
        solver = factory.create_propagator_computation(cb, molecules, prop_opt)
    else:
        solver = factory.create_propagator_computation(cb, molecules, prop_opt)

    # Create field dictionary (need to compute w from w_A, w_B for incompressible system)
    # For AB diblock: w_A = w+ + w-, w_B = w+ - w-
    # where w+ is pressure field, w- is exchange field
    fields = {"A": w_A.copy(), "B": w_B.copy()}

    # Warmup runs
    for _ in range(n_warmup):
        solver.compute_propagators(fields)

    # Benchmark runs
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        solver.compute_propagators(fields)
        times.append(time.perf_counter() - t0)

    # Get partition function
    Q = solver.get_total_partition(0)
    avg_time = np.mean(times) * 1000  # Convert to ms
    std_time = np.std(times) * 1000

    return {
        'Q': Q,
        'time_ms': avg_time,
        'time_std_ms': std_time
    }


def run_phase_benchmark(phase_name, nx, lx, f, chi_n, w_A, w_B,
                        platform="cuda", ds=0.01):
    """
    Run benchmark for a single phase with all four numerical methods.
    """
    print(f"\n{'='*70}")
    print(f"PHASE: {phase_name}")
    print(f"{'='*70}")
    print(f"  Grid: {nx[0]} x {nx[1]} x {nx[2]}")
    print(f"  Box:  {lx[0]:.3f} x {lx[1]:.3f} x {lx[2]:.3f}")
    print(f"  f = {f}, chi_N = {chi_n}, ds = {ds}")
    print(f"  Platform: {platform}")

    results = {}

    # Pseudo-spectral methods
    print(f"\n--- Pseudo-Spectral Methods ---")

    for method in ["rqm4", "etdrk4"]:
        print(f"\n  Testing {method.upper()}...", end=" ", flush=True)
        try:
            result = benchmark_method(
                platform, method, "cn-adi2",  # realspace_method doesn't matter here
                nx, lx, f, chi_n, ds, w_A, w_B,
                solver_type="pseudospectral",
                n_warmup=5, n_runs=10
            )
            results[method] = result
            print(f"Q = {result['Q']:.9f}, time = {result['time_ms']:.2f} ms")
        except Exception as e:
            print(f"FAILED: {e}")
            results[method] = None

    # Real-space methods
    print(f"\n--- Real-Space Methods ---")

    for method in ["cn-adi2", "cn-adi4-lr"]:
        print(f"\n  Testing {method.upper()}...", end=" ", flush=True)
        try:
            result = benchmark_method(
                platform, "rqm4", method,  # pseudo_method doesn't matter here
                nx, lx, f, chi_n, ds, w_A, w_B,
                solver_type="realspace",
                n_warmup=5, n_runs=10
            )
            results[method] = result
            print(f"Q = {result['Q']:.9f}, time = {result['time_ms']:.2f} ms")
        except Exception as e:
            print(f"FAILED: {e}")
            results[method] = None

    return results


def print_comparison_table(phase_name, results):
    """Print comparison table for a phase."""
    print(f"\n{'-'*70}")
    print(f"COMPARISON TABLE: {phase_name}")
    print(f"{'-'*70}")
    print(f"{'Method':<15} {'Solver':<15} {'Q':<18} {'Time (ms)':<12} {'Rel. Speed':<10}")
    print(f"{'-'*70}")

    # Find fastest time for relative speed calculation
    times = [r['time_ms'] for r in results.values() if r is not None]
    min_time = min(times) if times else 1.0

    method_info = {
        'rqm4': ('RQM4', 'Pseudo-Spectral'),
        'etdrk4': ('ETDRK4', 'Pseudo-Spectral'),
        'cn-adi2': ('CN-ADI2', 'Real-Space'),
        'cn-adi4-lr': ('CN-ADI4', 'Real-Space'),
    }

    for method, (name, solver) in method_info.items():
        if results.get(method) is not None:
            r = results[method]
            rel_speed = r['time_ms'] / min_time
            print(f"{name:<15} {solver:<15} {r['Q']:<18.9f} {r['time_ms']:<12.2f} {rel_speed:<10.2f}x")
        else:
            print(f"{name:<15} {solver:<15} {'FAILED':<18} {'-':<12} {'-':<10}")


def main():
    print("="*70)
    print("NUMERICAL METHODS BENCHMARK: GYROID AND FDDD PHASES")
    print("="*70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Detect platform from command line or default
    import sys
    if len(sys.argv) > 1 and sys.argv[1] in ["cuda", "cpu-fftw"]:
        platform = sys.argv[1]
    else:
        # Default to CUDA if available
        try:
            test_factory = _core.PlatformSelector.create_factory("cuda", False)
            platform = "cuda"
        except Exception:
            platform = "cpu-fftw"
    print(f"\nUsing platform: {platform}")

    all_results = {}
    ds = 0.01  # N = 100

    # Gyroid phase
    print("\n" + "="*70)
    print("SETTING UP GYROID PHASE")
    print("="*70)

    gyroid_nx = [32, 32, 32]
    gyroid_lx = [3.3, 3.3, 3.3]
    gyroid_f = 0.36
    gyroid_chi_n = 20.0

    w_A_gyroid, w_B_gyroid = create_gyroid_field(gyroid_nx, gyroid_chi_n)

    gyroid_results = run_phase_benchmark(
        "Gyroid (Ia-3d)", gyroid_nx, gyroid_lx, gyroid_f, gyroid_chi_n,
        w_A_gyroid, w_B_gyroid, platform, ds
    )
    all_results['gyroid'] = gyroid_results
    print_comparison_table("Gyroid", gyroid_results)

    # Fddd phase
    print("\n" + "="*70)
    print("SETTING UP FDDD PHASE")
    print("="*70)

    fddd_nx = [48, 32, 24]  # Slightly smaller than original for faster benchmarking
    fddd_lx = [5.58, 3.17, 1.59]
    fddd_f = 0.43
    fddd_chi_n = 14.0

    w_A_fddd, w_B_fddd = create_fddd_field(fddd_nx, fddd_chi_n)

    fddd_results = run_phase_benchmark(
        "Fddd (O^70)", fddd_nx, fddd_lx, fddd_f, fddd_chi_n,
        w_A_fddd, w_B_fddd, platform, ds
    )
    all_results['fddd'] = fddd_results
    print_comparison_table("Fddd", fddd_results)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    print("\n### Key Observations ###\n")

    # Compare RQM4 vs ETDRK4
    for phase in ['gyroid', 'fddd']:
        r = all_results[phase]
        if r.get('rqm4') and r.get('etdrk4'):
            q_diff = abs(r['rqm4']['Q'] - r['etdrk4']['Q'])
            t_ratio = r['etdrk4']['time_ms'] / r['rqm4']['time_ms']
            print(f"{phase.upper()}: RQM4 vs ETDRK4")
            print(f"  Q difference: {q_diff:.2e}")
            print(f"  ETDRK4 is {t_ratio:.2f}x {'slower' if t_ratio > 1 else 'faster'} than RQM4")

    print("\n### Pseudo-Spectral vs Real-Space ###\n")

    for phase in ['gyroid', 'fddd']:
        r = all_results[phase]
        if r.get('rqm4') and r.get('cn-adi2'):
            q_diff = abs(r['rqm4']['Q'] - r['cn-adi2']['Q']) / r['rqm4']['Q']
            t_ratio = r['cn-adi2']['time_ms'] / r['rqm4']['time_ms']
            print(f"{phase.upper()}: RQM4 vs CN-ADI2")
            print(f"  Q relative difference: {q_diff:.2e}")
            print(f"  CN-ADI2 is {t_ratio:.2f}x {'slower' if t_ratio > 1 else 'faster'} than RQM4")

    # Save results
    output_file = "benchmark_phases_results.json"

    def convert_types(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(v) for v in obj]
        return obj

    with open(output_file, 'w') as f:
        json.dump({
            'date': datetime.now().isoformat(),
            'platform': platform,
            'ds': ds,
            'gyroid': {
                'nx': gyroid_nx,
                'lx': gyroid_lx,
                'f': gyroid_f,
                'chi_n': gyroid_chi_n,
                'results': convert_types(gyroid_results)
            },
            'fddd': {
                'nx': fddd_nx,
                'lx': fddd_lx,
                'f': fddd_f,
                'chi_n': fddd_chi_n,
                'results': convert_types(fddd_results)
            }
        }, f, indent=2)

    print(f"\nResults saved to {output_file}")

    return all_results


if __name__ == "__main__":
    main()
