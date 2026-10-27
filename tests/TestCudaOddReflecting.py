#!/usr/bin/env python3
"""
CUDA runs boxes with reflecting walls on odd grid sizes, like the CPU backends.

CudaFFT used to refuse every odd non-periodic grid size, although the DCT-2 and
DCT-3 of CudaRealTransform take any length; only the DST-2 of absorbing axes
needs an even one. This test runs 1D, 2D and 3D boxes whose reflecting axes have
odd sizes (absorbing axes even) on CUDA and compares them with each CPU backend
on the same random field (std 5): partition function (relative 1e-10),
concentrations (absolute 1e-10) and stress (relative 1e-8), for continuous
(RQM4) and discrete chains. An odd absorbing axis must still be refused on
CUDA, and platform="auto" must choose a CPU backend for it (and for complex
fields of continuous chains with walls). Exit code 77 (skipped) without a CUDA
platform.
"""

import contextlib
import io
import sys

import numpy as np

from polymerfts import _core
from polymerfts.propagator_solver import PropagatorSolver

FAILURES = []


def check(condition, message):
    print(f"    [{'PASS' if condition else 'FAIL'}] {message}")
    if not condition:
        FAILURES.append(message)


def run(platform, chain_model, nx, lx, bc, w):
    with contextlib.redirect_stdout(io.StringIO()):
        solver = PropagatorSolver(
            nx=list(nx), lx=list(lx), ds=0.02, bond_lengths={"A": 1.0, "B": 1.0}, bc=list(bc),
            chain_model=chain_model, numerical_method="rqm4" if chain_model == "continuous" else None,
            platform=platform)
        solver.add_polymer(1.0, [["A", 0.5, 0, 1], ["B", 0.5, 1, 2]])
        solver.compute_propagators({m: w[m].copy() for m in "AB"})
        solver.compute_concentrations()
        phi = {m: np.asarray(solver.get_concentration(m)).flatten() for m in "AB"}
        solver.compute_stress()
        stress = np.asarray(solver.get_stress()).astype(float)[:len(nx)]
    return solver.get_partition_function(0), phi, stress


def main():
    platforms = _core.PlatformSelector.avail_platforms()
    cpus = [p for p in platforms if p.startswith("cpu")]
    print(f"Available platforms: {platforms}")
    if "cuda" not in platforms or not cpus:
        print("CUDA or a CPU platform is not available: skipped")
        return 77

    cases = [
        ([33], [3.0], ["reflecting", "reflecting"]),
        ([15, 12], [2.0, 1.6], ["reflecting", "reflecting", "absorbing", "absorbing"]),
        ([9, 8, 7], [1.5, 1.4, 1.3], ["reflecting", "reflecting", "absorbing", "absorbing", "reflecting", "reflecting"]),
    ]
    rng = np.random.default_rng(31)
    for nx, lx, bc in cases:
        w = {m: rng.normal(0.0, 5.0, int(np.prod(nx))) for m in "AB"}
        for chain_model in ("continuous", "discrete"):
            q_gpu, phi_gpu, s_gpu = run("cuda", chain_model, nx, lx, bc, w)
            for cpu in cpus:
                q_cpu, phi_cpu, s_cpu = run(cpu, chain_model, nx, lx, bc, w)
                tag = f"{nx} {bc[::2]} {chain_model}: cuda vs {cpu}"
                rel_q = abs(q_gpu - q_cpu) / abs(q_cpu)
                dphi = max(np.max(np.abs(phi_gpu[m] - phi_cpu[m])) for m in "AB")
                rel_s = np.max(np.abs(s_gpu - s_cpu) / np.abs(s_cpu))
                check(rel_q < 1e-10, f"{tag}: Q relative diff {rel_q:.2e}")
                check(dphi < 1e-10, f"{tag}: phi max abs diff {dphi:.2e}")
                check(rel_s < 1e-8, f"{tag}: stress relative diff {rel_s:.2e}")

    refused = False
    try:
        run("cuda", "continuous", [8, 7], [1.0, 1.0], ["reflecting", "reflecting", "absorbing", "absorbing"],
            {m: np.zeros(56) for m in "AB"})
    except Exception as exc:
        refused = "absorbing" in str(exc)
    check(refused, "cuda refuses an odd absorbing axis with a message naming it")

    # platform="auto" keeps the inputs that only the CPU backends run on the CPU
    walls = ["reflecting", "reflecting", "absorbing", "absorbing"]
    for nx, bc, chain_model, field_type, platform in (
            ([8, 7], walls, "continuous", "real", "cpu"), ([8, 7], walls, "discrete", "real", "cpu"),
            ([8, 8], walls, "discrete", "real", "cuda"), ([8, 8], walls, "continuous", "complex", "cpu"),
            ([8, 8], walls, "discrete", "complex", "cuda"), ([8, 7], ["periodic"]*4, "continuous", "complex", "cuda")):
        with contextlib.redirect_stdout(io.StringIO()):
            chosen = PropagatorSolver(nx=nx, lx=[1.0, 1.0], ds=0.02, bond_lengths={"A": 1.0}, bc=bc,
                                      chain_model=chain_model, field_type=field_type).platform
        check(chosen.startswith(platform), f"auto platform for {nx} {bc[::2]} {chain_model} {field_type}: {chosen}")

    if FAILURES:
        print(f"\nFAILED: {len(FAILURES)} check(s)")
        return 1
    print("\nOdd reflecting grids on CUDA: PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
