#!/usr/bin/env python3
"""SCFTResult returned by SCFT.run(return_result=True).

1D lamella on the CPU: the result must match the attributes of the run, its
fields must be on the full grid, and its w fields must restart the run at the
converged solution. 3D BCC with the Im-3m space group: one iteration must
give the same full-grid fields as the same iteration without the space group.
"""

import contextlib
import io
import sys

import numpy as np

from polymerfts import SCFT, SCFTResult, SpaceGroup, PlatformSelector


def cpu_platform():
    avail = PlatformSelector.avail_platforms()
    cpu = [p for p in avail if p.startswith("cpu")]
    return "cpu-mkl" if "cpu-mkl" in avail else cpu[0]


def params(nx, lx, **extra):
    p = {
        "nx": nx, "lx": lx,
        "chain_model": "continuous", "ds": 1/50,
        "segment_lengths": {"A": 1.0, "B": 1.0},
        "chi_n": {"A,B": 20.0},
        "distinct_polymers": [{"volume_fraction": 1.0,
                               "blocks": [{"type": "A", "length": 0.3}, {"type": "B", "length": 0.7}]}],
        "platform": cpu_platform(),
        "box_is_altering": False,
        "max_iter": 300, "tolerance": 1e-8,
        "optimizer": {"name": "am", "max_hist": 20, "start_error": 1e-2, "mix_min": 0.1, "mix_init": 0.1},
    }
    p.update(extra)
    return p


def main():
    failed = []

    # ---- 1D lamella: result = attributes, full-grid fields, restart from result.w
    x = (np.arange(64) + 0.5)/64
    with contextlib.redirect_stdout(io.StringIO()):
        calc = SCFT(params([64], [4.0]))
        result = calc.run(initial_fields={"A": np.cos(2*np.pi*x), "B": -np.cos(2*np.pi*x)}, return_result=True)
    if not isinstance(result, SCFTResult) or not result.converged:
        failed.append("1D run did not return a converged SCFTResult")
    if result.free_energy != calc.free_energy or result.iterations != calc.iter:
        failed.append("result differs from the run attributes")
    if sorted(result.phi) != ["A", "B"] or sorted(result.w) != ["A", "B"] \
            or any(np.size(v) != 64 for v in list(result.phi.values()) + list(result.w.values())):
        failed.append("1D fields are not {A, B} on the full grid")
    if abs(np.mean(result.phi["A"] + result.phi["B"]) - 1.0) > 1e-10:
        failed.append("1D concentrations do not sum to 1")

    with contextlib.redirect_stdout(io.StringIO()):
        restart = calc.run(initial_fields=result.w, return_result=True)
    if restart.iterations > 2 or abs(restart.free_energy - result.free_energy) > 1e-8:
        failed.append(f"restart from result.w took {restart.iterations} iterations, "
                      f"free energy {restart.free_energy} vs {result.free_energy}")

    # ---- 3D BCC: the space-group result equals the same iteration without the space group
    n = 16
    g = (np.arange(n) + 0.5)/n
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    w_a = sum(np.cos(2*np.pi*(s*u + t*v)) for u, v in ((X, Y), (Y, Z), (Z, X)) for s, t in ((1, 1), (1, -1)))
    w_a = SpaceGroup([n]*3, "Im-3m", 529).symmetrize(np.reshape(w_a, (1, -1)))[0]
    results = {}
    for name, sg in (("on", {"space_group": {"symbol": "Im-3m", "number": 529}}), ("off", {})):
        with contextlib.redirect_stdout(io.StringIO()):
            calc = SCFT(params([n]*3, [1.9]*3, max_iter=1, **sg))
            results[name] = calc.run(initial_fields={"A": w_a, "B": -w_a}, return_result=True)
    for kind in ("phi", "w"):
        for m in ("A", "B"):
            on, off = getattr(results["on"], kind)[m], getattr(results["off"], kind)[m]
            if np.size(on) != n**3:
                failed.append(f"BCC {kind}[{m}] is not on the full {n}^3 grid")
            elif not np.allclose(on, off, rtol=0, atol=1e-10):
                failed.append(f"BCC {kind}[{m}] differs from the run without space group by {np.max(np.abs(on - off)):.2e}")

    for f in failed:
        print("FAILED:", f)
    if failed:
        sys.exit(1)
    print("SCFTResult: OK")


if __name__ == "__main__":
    main()
