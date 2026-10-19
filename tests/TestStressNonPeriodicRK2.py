#!/usr/bin/env python3
"""Stress with reflecting/absorbing boundary conditions for the RK2 method.

TestStressNonPeriodic.py covers RQM4 and the discrete chain. RK2 has its own
stress code path on the GPU, so it is checked here the same way:

1. Finite-difference validation of -d(ln Q)/dL in 1D and 2D on every platform.
2. CPU-vs-CUDA equivalence for mixed boundary conditions in 1D, 2D and 3D
   (the CPU side shares its stress code with RQM4, which is FD-validated).
"""

import os
import sys

os.environ["OMP_MAX_ACTIVE_LEVELS"] = "0"

import numpy as np
from scipy.ndimage import gaussian_filter
from polymerfts import _core
from polymerfts.propagator_solver import PropagatorSolver

DS = 0.01
F_A = 0.36
BOND = {"A": 1.0, "B": 1.1}

n_pass = 0
n_fail = 0


def check(cond, msg):
    global n_pass, n_fail
    if cond:
        n_pass += 1
        print(f"[PASS] {msg}")
    else:
        n_fail += 1
        print(f"[FAIL] {msg}")


def smooth_fields(nx, seed):
    rng = np.random.default_rng(seed)
    w = rng.normal(0.0, 1.0, size=(2, *nx))
    for c in range(2):
        w[c] = gaussian_filter(w[c], sigma=2.0, mode="nearest")
        w[c] *= 5.0 / max(w[c].std(), 1e-12)
    return w.reshape(2, -1)


def make_solver(nx, lx, bc, platform):
    solver = PropagatorSolver(
        nx=list(nx), lx=list(lx), ds=DS,
        bond_lengths=dict(BOND), bc=list(bc),
        chain_model="continuous", numerical_method="rk2",
        platform=platform)
    solver.add_polymer(1.0, [["A", F_A, 0, 1], ["B", 1.0 - F_A, 1, 2]])
    solver._initialize_solver()
    return solver


def run_fd_case(name, nx, lx, bc, platform, tol):
    w = smooth_fields(nx, 1234 + len(nx))
    wd = {"A": w[0], "B": w[1]}
    solver = make_solver(nx, lx, bc, platform)
    solver.compute_propagators(wd)
    solver.compute_stress()
    stress = np.array(solver.get_stress()).real

    def neg_lnq(lx_use):
        s2 = make_solver(nx, lx_use, bc, platform)
        s2.compute_propagators(wd)
        return -np.log(s2.get_partition_function(0))

    worst = 0.0
    for d in range(len(nx)):
        dl = lx[d] * 2e-4
        lx_p = list(lx); lx_p[d] += 0.5 * dl
        lx_m = list(lx); lx_m[d] -= 0.5 * dl
        fd = (neg_lnq(lx_p) - neg_lnq(lx_m)) / dl
        rel = abs(stress[d] - fd) / max(abs(fd), 1e-12)
        worst = max(worst, rel)
    check(worst < tol, f"{name} rk2 ({platform}): max FD mismatch {worst:.3e} < {tol}")


def main():
    platforms = _core.PlatformSelector.avail_platforms()
    cpu = next((p for p in platforms if p.startswith("cpu")), None)
    has_cuda = "cuda" in platforms
    if cpu is None and not has_cuda:
        print("SKIP: no computational platform available in this build.")
        sys.exit(0)

    # --- FD validation (the RK2 stress differs from the finite difference of its own
    #     Q by the O(ds^2) splitting error: 1.7e-2 at ds = 0.01 for the reflecting case,
    #     4e-3 at 0.005, 1e-3 at 0.0025; the same holds for periodic boundaries) ---
    test_platforms = ([cpu] if cpu else []) + (["cuda"] if has_cuda else [])
    for platform in test_platforms:
        run_fd_case("1D reflect", [48], [2.3], ["reflecting"] * 2, platform, 2.5e-2)
        run_fd_case("1D absorb", [48], [2.3], ["absorbing"] * 2, platform, 1.5e-2)
        run_fd_case("2D reflect/absorb", [24, 20], [2.1, 1.7],
                    ["reflecting", "reflecting", "absorbing", "absorbing"], platform, 1.5e-2)

    # --- Cross-platform equivalence ---
    if has_cuda and cpu is not None:
        cases = [
            ("1D reflect", [48], [2.3], ["reflecting"] * 2),
            ("1D absorb", [48], [2.3], ["absorbing"] * 2),
            ("2D r/a", [24, 20], [2.1, 1.7], ["reflecting", "reflecting", "absorbing", "absorbing"]),
            ("2D a/r", [24, 20], [2.1, 1.7], ["absorbing", "absorbing", "reflecting", "reflecting"]),
            ("3D r/a/r", [24, 20, 16], [2.1, 1.7, 1.4],
             ["reflecting", "reflecting", "absorbing", "absorbing", "reflecting", "reflecting"]),
            ("3D a/r/r", [24, 20, 16], [2.1, 1.7, 1.4],
             ["absorbing", "absorbing", "reflecting", "reflecting", "reflecting", "reflecting"]),
        ]
        for name, nx, lx, bc in cases:
            w = smooth_fields(nx, 777)
            wd = {"A": w[0], "B": w[1]}
            results = {}
            for platform in [cpu, "cuda"]:
                s = make_solver(nx, lx, bc, platform)
                s.compute_propagators(wd)
                s.compute_stress()
                results[platform] = (s.get_partition_function(0),
                                     np.array(s.get_stress()).real[:len(nx)])
            q_cpu, st_cpu = results[cpu]
            q_gpu, st_gpu = results["cuda"]
            rel_q = abs(q_gpu - q_cpu) / abs(q_cpu)
            rel_s = np.max(np.abs(st_gpu - st_cpu) / np.maximum(np.abs(st_cpu), 1e-12))
            check(rel_q < 1e-12, f"{name} rk2: Q cpu-vs-cuda {rel_q:.3e} < 1e-12")
            check(rel_s < 1e-10, f"{name} rk2: stress cpu-vs-cuda {rel_s:.3e} < 1e-10")

    print(f"\n{n_pass} passed, {n_fail} failed")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
