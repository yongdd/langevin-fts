#!/usr/bin/env python3
"""
Benchmark: ETDRK4 vs RQM4

Note: ETDRK4 is tested via C++ unit test (TestCpuETDRK4).
This script benchmarks the RQM4 pseudo-spectral solver
and compares CPU vs CUDA performance.

RQM4 (Ranjan-Qin-Morse 4th-order) uses Richardson extrapolation to achieve
4th-order temporal accuracy.

For ETDRK4 vs RQM4 comparison, run the C++ test:
    ./build/tests/TestCpuETDRK4

Both methods are 4th-order accurate in time, so they should produce
nearly identical results for the same problem.
"""

import os
import sys
import time
import numpy as np

# Set OpenMP environment
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
os.environ["OMP_NUM_THREADS"] = "4"

from polymerfts import _core


def create_lamellar_field(nx, lx, chi_n=12.0, n_periods=3):
    """Create a lamellar-like potential field."""
    M = np.prod(nx)
    dim = len(nx)

    if dim == 1:
        x = (np.arange(nx[0]) + 0.5) * lx[0] / nx[0]
        w = chi_n * 0.5 * np.cos(2 * np.pi * n_periods * x / lx[0])
    elif dim == 2:
        x = (np.arange(nx[0]) + 0.5) * lx[0] / nx[0]
        w = np.zeros(nx)
        for i in range(nx[0]):
            w[i, :] = chi_n * 0.5 * np.cos(2 * np.pi * n_periods * x[i] / lx[0])
        w = w.flatten()
    else:  # 3D
        x = (np.arange(nx[2]) + 0.5) * lx[2] / nx[2]
        w = np.zeros(nx)
        for k in range(nx[2]):
            w[:, :, k] = chi_n * 0.5 * np.cos(2 * np.pi * n_periods * x[k] / lx[2])
        w = w.flatten()

    return w


def benchmark_pseudospectral(platform, nx, lx, ds_values, n_warmup=2, n_runs=5):
    """Benchmark pseudo-spectral solver (RQM4)."""

    results = {
        'ds': ds_values,
        'Q': [],
        'time': []
    }

    M = np.prod(nx)
    w_a = create_lamellar_field(nx, lx)

    for ds in ds_values:
        print(f"\n  ds = {ds:.6f} (N = {int(1/ds)}):")

        bond_lengths = {"A": 1.0}
        factory = _core.PlatformSelector.create_factory(platform, False)
        molecules = factory.create_molecules_information("Continuous", ds, bond_lengths)
        molecules.add_polymer(1.0, [["A", 1.0, 0, 1]])
        cb = factory.create_computation_box(nx=list(nx), lx=list(lx), bc=[])
        prop_opt = factory.create_propagator_computation_optimizer(molecules, True)
        solver = factory.create_pseudospectral_solver(cb, molecules, prop_opt)

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
        avg_time = np.mean(times)

        results['Q'].append(Q)
        results['time'].append(avg_time)

        print(f"    Q = {Q:.12f}, time = {avg_time*1000:.2f} ms")

    return results


def test_convergence(platform="cpu-mkl"):
    """Test convergence order of pseudo-spectral method."""

    print(f"\n{'='*60}")
    print(f"Convergence Test ({platform})")
    print('='*60)

    nx = [32, 32, 32]
    lx = [4.0, 4.0, 4.0]

    ds_values = [1/10, 1/20, 1/40, 1/80, 1/160]

    w_a = create_lamellar_field(nx, lx)

    Q_values = []

    for ds in ds_values:
        print(f"\n  ds = {ds:.6f}:")

        bond_lengths = {"A": 1.0}
        factory = _core.PlatformSelector.create_factory(platform, False)
        molecules = factory.create_molecules_information("Continuous", ds, bond_lengths)
        molecules.add_polymer(1.0, [["A", 1.0, 0, 1]])
        cb = factory.create_computation_box(nx=list(nx), lx=list(lx), bc=[])
        prop_opt = factory.create_propagator_computation_optimizer(molecules, True)
        solver = factory.create_pseudospectral_solver(cb, molecules, prop_opt)

        solver.compute_propagators({"A": w_a})
        Q = solver.get_total_partition(0)
        Q_values.append(Q)

        print(f"    Q = {Q:.12f}")

    # Estimate convergence order
    print("\n  Convergence Analysis:")
    print("  " + "-"*50)

    Q_ref = Q_values[-1]

    print("\n  Partition function Q vs ds:")
    for i in range(len(ds_values)):
        error = abs(Q_values[i] - Q_ref)
        print(f"    ds={ds_values[i]:.5f}: Q = {Q_values[i]:.12f}, error = {error:.2e}")

    # Estimate order
    if len(ds_values) >= 3:
        errors = [abs(Q_values[i] - Q_ref) for i in range(len(ds_values) - 1)]
        if errors[0] > 1e-14 and errors[1] > 1e-14:
            order = np.log(errors[0] / errors[1]) / np.log(ds_values[0] / ds_values[1])
            print(f"\n  Estimated convergence order: p ≈ {order:.1f}")
            print(f"  (Expected: p ≈ 4.0 for RQM4)")

    return Q_values


def compare_cpu_cuda():
    """Compare CPU vs CUDA results."""

    print("\n" + "="*60)
    print("CPU vs CUDA Comparison")
    print("="*60)

    nx = [32, 32, 32]
    lx = [4.0, 4.0, 4.0]
    ds = 0.01

    w_a = create_lamellar_field(nx, lx)

    # CPU
    print("\nCPU (MKL):")
    factory_cpu = _core.PlatformSelector.create_factory("cpu-mkl", False)
    molecules_cpu = factory_cpu.create_molecules_information("Continuous", ds, {"A": 1.0})
    molecules_cpu.add_polymer(1.0, [["A", 1.0, 0, 1]])
    cb_cpu = factory_cpu.create_computation_box(nx=list(nx), lx=list(lx), bc=[])
    prop_opt_cpu = factory_cpu.create_propagator_computation_optimizer(molecules_cpu, True)
    solver_cpu = factory_cpu.create_pseudospectral_solver(cb_cpu, molecules_cpu, prop_opt_cpu)

    solver_cpu.compute_propagators({"A": w_a})
    Q_cpu = solver_cpu.get_total_partition(0)
    print(f"  Q = {Q_cpu:.12f}")

    # CUDA
    print("\nCUDA:")
    try:
        factory_cuda = _core.PlatformSelector.create_factory("cuda", False)
        molecules_cuda = factory_cuda.create_molecules_information("Continuous", ds, {"A": 1.0})
        molecules_cuda.add_polymer(1.0, [["A", 1.0, 0, 1]])
        cb_cuda = factory_cuda.create_computation_box(nx=list(nx), lx=list(lx), bc=[])
        prop_opt_cuda = factory_cuda.create_propagator_computation_optimizer(molecules_cuda, True)
        solver_cuda = factory_cuda.create_pseudospectral_solver(cb_cuda, molecules_cuda, prop_opt_cuda)

        solver_cuda.compute_propagators({"A": w_a})
        Q_cuda = solver_cuda.get_total_partition(0)
        print(f"  Q = {Q_cuda:.12f}")

        rel_diff = abs(Q_cpu - Q_cuda) / Q_cpu
        print(f"\nRelative difference: {rel_diff:.2e}")
        if rel_diff < 1e-10:
            print("  -> PASSED: CPU and CUDA results match")
        else:
            print("  -> WARNING: Results differ")
    except Exception as e:
        print(f"  CUDA not available: {e}")


def main():
    print("="*60)
    print("Pseudo-Spectral Solver Benchmark")
    print("="*60)
    print("\nNote: For ETDRK4 vs RQM4 comparison, run C++ test:")
    print("  ./build/tests/TestCpuETDRK4")

    # Test configurations
    ds_values = [1/20, 1/40, 1/80, 1/160]

    # 3D benchmark
    print("\n" + "="*60)
    print("3D Benchmark (32x32x32 grid)")
    print("="*60)

    nx_3d = [32, 32, 32]
    lx_3d = [4.0, 4.0, 4.0]

    # CPU benchmark
    print("\n--- CPU (MKL) ---")
    try:
        cpu_results = benchmark_pseudospectral("cpu-mkl", nx_3d, lx_3d, ds_values)
    except Exception as e:
        print(f"CPU benchmark failed: {e}")
        cpu_results = None

    # CUDA benchmark
    print("\n--- CUDA ---")
    try:
        cuda_results = benchmark_pseudospectral("cuda", nx_3d, lx_3d, ds_values, n_warmup=5, n_runs=10)
    except Exception as e:
        print(f"CUDA benchmark failed: {e}")
        cuda_results = None

    # Convergence test
    test_convergence("cpu-mkl")

    try:
        test_convergence("cuda")
    except Exception as e:
        print(f"CUDA convergence test failed: {e}")

    # CPU vs CUDA comparison
    compare_cpu_cuda()

    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)

    if cpu_results and cuda_results:
        # Compare Q values
        Q_diff = [abs(cpu_results['Q'][i] - cuda_results['Q'][i]) / cpu_results['Q'][i]
                  for i in range(len(ds_values))]
        max_diff = max(Q_diff)
        print(f"\nMax relative Q difference (CPU vs CUDA): {max_diff:.2e}")

        # Compare times
        print("\nPerformance comparison:")
        print(f"{'ds':>10} {'CPU (ms)':>12} {'CUDA (ms)':>12} {'Speedup':>10}")
        print("-"*46)
        for i, ds in enumerate(ds_values):
            speedup = cpu_results['time'][i] / cuda_results['time'][i]
            print(f"{ds:>10.4f} {cpu_results['time'][i]*1000:>12.2f} {cuda_results['time'][i]*1000:>12.2f} {speedup:>10.1f}x")


if __name__ == "__main__":
    main()
