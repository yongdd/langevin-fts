#!/usr/bin/env python3
"""
Compressible SCFT (zeta_n): the converged fields must satisfy the compressible
field equations w_i = sum_j chi_ij phi_j + zeta (sum_j phi_j - 1).

Until 2026-10 the residual was the incompressible one for both models, so a
compressible run returned the incompressible solution (sum phi = 1 everywhere)
whatever zeta was, and its free energy was the compressible Hamiltonian away
from its saddle point (1D AB lamella, chi N = 15, as in this test: residual
9.68, F = -0.364 for zeta N = 5 and -0.322 for zeta N = 100; with
fts_consistent_energy residual 2.18, F = 1.886 and -0.060).

Checks (1D lamellae of AB diblocks and an ABC triblock):
1. The compressible residual is below 1e-6 and sum phi is not uniform.
2. The full CE free energy minus the ideal mixing term equals the field-dependent
   free energy -ln Q + <(1/2) phi.chi.phi + (zeta/2)(sum phi - 1)^2 - w.phi>
   value computed from the fields (AB; 1e-9). The ideal term is -1 for these
   pure melts with alpha = 1.
3. zeta -> infinity: the free energy and phi approach the incompressible solution
   as 1/zeta (the ratio of the differences at zeta N = 100 and 1000 is near 10).
"""

import os
import io
import sys
import contextlib

os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from polymerfts import _core, scft

NX = 64
x = np.arange(NX)

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


def cpu_platform():
    cpu = [p for p in _core.PlatformSelector.avail_platforms() if p.startswith("cpu")]
    return cpu[0] if cpu else "cuda"


def run(chi_n, blocks, lx, zeta_n, initial, extra=None):
    types = sorted({b[0] for b in blocks})
    params = {"nx": [NX], "lx": [lx], "box_is_altering": False, "chain_model": "continuous", "ds": 1 / 50,
              "segment_lengths": {t: 1.0 for t in types}, "chi_n": chi_n, "platform": cpu_platform(),
              "distinct_polymers": [{"volume_fraction": 1.0,
                                     "blocks": [{"type": t, "length": length} for t, length in blocks]}],
              "optimizer": {"name": "am", "max_hist": 20, "start_error": 1e-2, "mix_min": 0.1, "mix_init": 0.1},
              "max_iter": 3000, "tolerance": 1e-9}
    if zeta_n is not None:
        params["zeta_n"] = zeta_n
    if extra:
        params.update(extra)
    with contextlib.redirect_stdout(io.StringIO()):
        calc = scft.SCFT(params=params)
        calc.run(initial_fields=initial)
    phi = np.array([calc.phi[t] for t in types])
    w = np.array(calc.w)
    X = np.zeros((len(types), len(types)))
    for key, value in chi_n.items():
        i, j = [types.index(t) for t in key.split(",")]
        X[i, j] = X[j, i] = value
    zeta = 0.0 if zeta_n is None else zeta_n
    total = np.sum(phi, axis=0)
    residual = X @ phi + zeta * (total - 1.0) - w
    if zeta_n is None:
        residual -= np.mean(residual, axis=0)        # a common pressure field is free
        residual -= np.mean(residual, axis=1)[:, None]  # and so are the means of each field
    Q = calc.prop_solver.get_partition_function(0)
    F = -np.log(Q) + np.mean(0.5 * np.einsum("in,ij,jn->n", phi, X, phi) + 0.5 * zeta * (total - 1.0) ** 2
                             - np.sum(w * phi, axis=0))
    return {"calc": calc, "phi": phi, "residual": np.max(np.abs(residual)), "std_total": np.std(total), "F": F}


def main():
    ab_sym = ({"A,B": 15.0}, [("A", 0.5), ("B", 0.5)], 1.6)
    ab_asym = ({"A,B": 20.0}, [("A", 0.35), ("B", 0.65)], 1.7)
    ab_init = {"A": 3 * np.cos(2 * np.pi * x / NX), "B": -3 * np.cos(2 * np.pi * x / NX)}
    for name, (chi_n, blocks, lx) in (("AB f=0.5 chiN 15", ab_sym), ("AB f=0.35 chiN 20", ab_asym)):
        r = run(chi_n, blocks, lx, 5.0, ab_init)
        calc = r["calc"]
        check(calc.error_level < 1e-9 and r["residual"] < 1e-6 and r["std_total"] > 1e-2,
              f"{name}, zetaN 5: compressible residual {r['residual']:.1e}, std(sum phi) {r['std_total']:.1e}, "
              f"{calc.iter} iterations")
        F_code = calc.free_energy_absolute + 1.0
        check(abs(F_code - r["F"]) < 1e-9,
              f"{name}, zetaN 5: free energy {F_code:.12f} = mean-field value {r['F']:.12f}")

    # zeta -> infinity
    chi_n, blocks, lx = ab_asym
    inc = run(chi_n, blocks, lx, None, ab_init)
    diffs = {}
    for zeta_n in (100.0, 1000.0):
        r = run(chi_n, blocks, lx, zeta_n, ab_init)
        diffs[zeta_n] = (abs(r["F"] - inc["F"]), np.max(np.abs(r["phi"] - inc["phi"])))
    ratio_F = diffs[100.0][0] / diffs[1000.0][0]
    ratio_phi = diffs[100.0][1] / diffs[1000.0][1]
    check(5 < ratio_F < 20 and 5 < ratio_phi < 20 and diffs[1000.0][1] < 1e-2,
          f"zetaN 100 -> 1000: |F - F_inc| {diffs[100.0][0]:.2e} -> {diffs[1000.0][0]:.2e}, "
          f"max|phi - phi_inc| {diffs[100.0][1]:.2e} -> {diffs[1000.0][1]:.2e} (1/zeta)")

    # ABC triblock (three fields, the projector residual with M = 3)
    abc_init = {"A": 3 * np.cos(2 * np.pi * x / NX), "B": np.zeros(NX), "C": -3 * np.cos(2 * np.pi * x / NX)}
    r = run({"A,B": 15.0, "B,C": 15.0, "A,C": 25.0}, [("A", 0.3), ("B", 0.4), ("C", 0.3)], 1.8, 20.0, abc_init)
    check(r["calc"].error_level < 1e-9 and r["residual"] < 1e-6 and r["std_total"] > 1e-3,
          f"ABC triblock, zetaN 20: compressible residual {r['residual']:.1e}, std(sum phi) {r['std_total']:.1e}, "
          f"{r['calc'].iter} iterations")

    print(f"\n{n_pass} passed, {n_fail} failed")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
