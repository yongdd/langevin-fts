#!/usr/bin/env python3
"""
Stress with non-periodic (reflecting/absorbing) boundary conditions.

1. Finite-difference validation: stress[d] must match the central difference
   of -lnQ with respect to L_d. For DISCRETE chains the stress formula is an
   exact derivative of the discretized lnQ, so agreement is limited only by
   the FD truncation (tolerance 1e-6). For CONTINUOUS chains the stress is a
   contour-quadrature approximation: at ds = 0.01 the periodic baseline
   mismatch is ~4e-4; reflecting converges like periodic (tolerance 1e-3),
   absorbing degrades to O(ds) from the wall boundary layers (tolerance
   1.5e-2). Do NOT tighten these without reducing ds accordingly.

2. Cross-platform equivalence for per-axis MIXED reflecting/absorbing BCs:
   Q and stress must agree between CPU and CUDA to machine precision. This
   guards the per-dimension BC plumbing (a squeeze bug here once built the
   CUDA Boltzmann/stress tables with the wrong axis assignment - wrong Q by
   O(1) - while all pure-BC cases passed).
"""

import os
import sys

os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
os.environ["OMP_NUM_THREADS"] = "2"

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


def make_solver(nx, lx, bc, model, platform):
    solver = PropagatorSolver(
        nx=list(nx), lx=list(lx), ds=DS,
        bond_lengths=dict(BOND), bc=list(bc),
        chain_model=model,
        numerical_method="rqm4" if model == "continuous" else None,
        platform=platform)
    solver.add_polymer(1.0, [["A", F_A, 0, 1], ["B", 1.0 - F_A, 1, 2]])
    solver._initialize_solver()
    return solver


def run_fd_case(name, nx, lx, bc, model, platform, tol):
    w = smooth_fields(nx, 1234 + len(nx))
    wd = {"A": w[0], "B": w[1]}
    solver = make_solver(nx, lx, bc, model, platform)
    solver.compute_propagators(wd)
    solver.compute_stress()
    stress = np.array(solver.get_stress()).real

    def neg_lnq(lx_use):
        s2 = make_solver(nx, lx_use, bc, model, platform)
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
    check(worst < tol,
          f"{name} [{model}] ({platform}): max FD mismatch {worst:.3e} < {tol}")


def main():
    platforms = _core.PlatformSelector.avail_platforms()
    cpu = next((p for p in platforms if p.startswith("cpu")), None)
    has_cuda = "cuda" in platforms
    if cpu is None and not has_cuda:
        print("SKIP: no computational platform available in this build.")
        sys.exit(0)

    # --- FD validation (discrete = exact derivative; continuous = quadrature) ---
    fd_cases = [
        ("1D reflect", [48], [2.3], ["reflecting"] * 2),
        ("1D absorb", [48], [2.3], ["absorbing"] * 2),
        ("3D mixed r/a/r", [24, 20, 16], [2.1, 1.7, 1.4],
         ["reflecting", "reflecting", "absorbing", "absorbing",
          "reflecting", "reflecting"]),
    ]
    test_platforms = ([cpu] if cpu else []) + (["cuda"] if has_cuda else [])
    for platform in test_platforms:
        for name, nx, lx, bc in fd_cases:
            run_fd_case(name, nx, lx, bc, "discrete", platform, 1e-6)
        run_fd_case("1D reflect", [48], [2.3], ["reflecting"] * 2,
                    "continuous", platform, 1e-3)
        run_fd_case("1D absorb", [48], [2.3], ["absorbing"] * 2,
                    "continuous", platform, 1.5e-2)

    # --- Cross-platform equivalence for mixed BCs ---
    if has_cuda and cpu is not None:
        nx = [24, 20, 16]
        lx = [2.1, 1.7, 1.4]
        w = smooth_fields(nx, 777)
        wd = {"A": w[0], "B": w[1]}
        mixed_bcs = [
            ("r/a/r", ["reflecting", "reflecting", "absorbing", "absorbing",
                       "reflecting", "reflecting"]),
            ("a/r/r", ["absorbing", "absorbing", "reflecting", "reflecting",
                       "reflecting", "reflecting"]),
            ("r/r/a", ["reflecting", "reflecting", "reflecting", "reflecting",
                       "absorbing", "absorbing"]),
        ]
        for model in ["continuous", "discrete"]:
            for bc_name, bc in mixed_bcs:
                results = {}
                for platform in [cpu, "cuda"]:
                    s = make_solver(nx, lx, bc, model, platform)
                    s.compute_propagators(wd)
                    s.compute_stress()
                    results[platform] = (s.get_partition_function(0),
                                         np.array(s.get_stress()).real[:3])
                q_cpu, st_cpu = results[cpu]
                q_gpu, st_gpu = results["cuda"]
                rel_q = abs(q_gpu - q_cpu) / abs(q_cpu)
                rel_s = np.max(np.abs(st_gpu - st_cpu)
                               / np.maximum(np.abs(st_cpu), 1e-12))
                check(rel_q < 1e-12,
                      f"mixed {bc_name} [{model}]: Q cpu-vs-cuda {rel_q:.3e} < 1e-12")
                check(rel_s < 1e-10,
                      f"mixed {bc_name} [{model}]: stress cpu-vs-cuda {rel_s:.3e} < 1e-10")

    print(f"\n{n_pass} passed, {n_fail} failed")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
