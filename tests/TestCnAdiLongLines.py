#!/usr/bin/env python3
"""CN-ADI lines longer than the shared memory of a CUDA block, CPU against CUDA.

The CUDA tridiagonal solver keeps the 3n coefficients of a line in shared memory
up to min(96000 bytes, the device limit), that is up to 4000 points; longer lines
read them from global memory (tridiagonal_global). Through PropagatorSolver with
cn-adi2, each case compares the partition function and the concentrations of
CUDA with CPU-FFTW (or CPU-MKL) on random fields of standard deviation 5:
- 1D: 4000 points (shared memory) and 4001 points (global memory) with
  reflecting walls, 6000 points with absorbing walls;
- 2D: a 4001-point reflecting axis next to a periodic one;
- 3D: a 4500-point absorbing z axis behind two periodic axes.
check_total_partition and the total concentration hold on CUDA as well.
"""

import contextlib
import io

import numpy as np
from polymerfts import PlatformSelector, PropagatorSolver

CASES = [
    ([4000], [40.0], ["reflecting"] * 2),
    ([4001], [40.0], ["reflecting"] * 2),
    ([6000], [60.0], ["absorbing"] * 2),
    ([4001, 3], [40.0, 1.0], ["reflecting", "reflecting", "periodic", "periodic"]),
    ([3, 2, 4500], [1.0, 1.0, 45.0], ["periodic", "periodic", "periodic", "periodic", "absorbing", "absorbing"]),
]


def run(platform, nx, lx, bc):
    with contextlib.redirect_stdout(io.StringIO()):
        s = PropagatorSolver(nx=nx, lx=lx, ds=0.02, bond_lengths={"A": 1.0, "B": 1.0}, bc=bc,
                             chain_model="continuous", numerical_method="cn-adi2", platform=platform)
        s.add_polymer(1.0, [["A", 0.5, 0, 1], ["B", 0.5, 1, 2]])
        rng = np.random.default_rng(3)
        n = int(np.prod(nx))
        s.compute_propagators({m: rng.normal(0.0, 5.0, n) for m in "AB"})
        s.compute_concentrations()
        partition_ok = s.check_total_partition()
    phi = {m: np.asarray(s.get_concentration(m)).ravel() for m in "AB"}
    return s.get_partition_function(0), phi, partition_ok, s.mean(phi["A"] + phi["B"])


def main():
    platforms = PlatformSelector.avail_platforms()
    cpu = [p for p in platforms if p.startswith("cpu")]
    if "cuda" not in platforms or not cpu:
        print("CUDA and a CPU platform are needed; skipped")
        return 77
    failures = 0
    for nx, lx, bc in CASES:
        q_cpu, phi_cpu, _, _ = run(cpu[0], nx, lx, bc)
        q_gpu, phi_gpu, partition_ok, total = run("cuda", nx, lx, bc)
        dq = abs(q_gpu - q_cpu) / abs(q_cpu)
        dphi = max(np.max(np.abs(phi_gpu[m] - phi_cpu[m])) for m in "AB")
        ok = dq < 1e-11 and dphi < 1e-10 and partition_ok and np.isfinite(total)
        failures += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] nx={nx} bc={bc[0]}/{bc[-1]}: Q rel diff {dq:.1e}, "
              f"phi max diff {dphi:.1e}, check_total_partition {partition_ok}, mean total phi {total:.6f}")
    print(f"\n{len(CASES) - failures} passed, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
