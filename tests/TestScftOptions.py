#!/usr/bin/env python3
"""
SCFT options: angle relaxation, axis_ordering and the field-residual projector.

1. Angle relaxation (Oblique2D): hexagonal cylinders started from gamma = 100 deg
   relax to gamma = 120 deg with a = b. The angles enter the optimizer in radians,
   the unit of their residual -dH/dgamma; until 2026-10 they entered in degrees,
   which made the angle steps 57 times smaller (1493 iterations here with
   cpu-mkl, 1544 with cpu-fftw, instead of about 240). The check requires fewer
   than 600 iterations.
2. axis_ordering: a run whose nx, lx, bc and initial fields are given in the
   order ["b", "a"] reproduces the run in the standard order (results are in the
   standard order). Until 2026-10 the initial fields and the boundary conditions
   were not reordered (walls on the wrong axis: dF 9e-2 here).
3. Free energy: the mean-field free energy from the monomer fields, measured from
   the homogeneous state (1/2) phibar^T chi N phibar, does not depend on the
   convention of the field means (fts_consistent_energy on or off, 1e-10), is 0 in
   the disordered phase at any composition, and plus free_energy_homogeneous equals
   at the saddle point the field-theoretic Hamiltonian with its constant (as
   recorded by L-FTS). Until 2026-10 the free energy depended on the convention:
   with fts_consistent_energy it was -chi N (f - 1/2)^2 in the disordered phase. At 1^T (chi N)^-1 1 = 0 (chi_AB = chi_AC = 10, chi_BC = 40) the auxiliary
   field of the zero eigenvalue of the projected chi N has a linear term and acts as
   a uniform field: F and the box are the limit of chi_BC N = 40 -+ 1e-4 (their
   mean, to 1e-9), also with fts_consistent_energy.
4. Field residual: the incompressible and compressible SCFT converge for chi N
   matrices with c = 1^T (chi N)^-1 1 < 0 (AB with chi N = -5; ABC with chi_AB N =
   chi_BC N = 10, chi_AC N = 60, and with chi_AB N = chi_AC N = 10, chi_BC N = 50),
   near c = 0 (chi_BC N = 39.9 and 40.1: c = +5e-4 and -5e-4) and at small zeta N
   (0.5, 2). Until 2026-10 the residual projected with
   P = I - e e^T (chi N)^-1/c: its pressure part, (1, ..., 1) sum phi/c, had the
   wrong sign for c < 0 and grew without bound near c = 0, and the compressible
   pressure part was amplified for small zeta N; these runs ended in NaN or did
   not converge.
5. A singular chi N matrix: an ABC triblock whose B and C are chemically identical
   (chi_AB = chi_AC, chi_BC = 0) gives the free energy of the AB diblock, and an
   ABC triblock with a zero pair (chi_AB = 25, chi_BC = 20, chi_AC = 0) the limit
   chi_AC -> 0 (with box relaxation).
   Until 2026-10 incompressible SCFT rejected every singular chi N matrix, which
   includes any zero pair in a three-monomer system.
"""

import os
import io
import sys
import contextlib

os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from polymerfts import _core, scft, ValidationError

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


def base(nx, lx, chi_n, blocks, box):
    return {"nx": nx, "lx": lx, "box_is_altering": box, "chain_model": "continuous", "ds": 1 / 50,
            "segment_lengths": {t: 1.0 for t, _ in blocks}, "chi_n": chi_n, "platform": cpu_platform(),
            "distinct_polymers": [{"volume_fraction": 1.0,
                                   "blocks": [{"type": t, "length": length} for t, length in blocks]}],
            "optimizer": {"name": "am", "max_hist": 20, "start_error": 1e-2, "mix_min": 0.1, "mix_init": 0.1},
            "max_iter": 3000, "tolerance": 1e-8}


def run(params, fields):
    with contextlib.redirect_stdout(io.StringIO()):
        calc = scft.SCFT(params=params)
        calc.run(initial_fields=fields)
    return calc


def test_angle_relaxation():
    nx = [32, 32]
    x = (np.arange(32) + 0.5) / 32
    X, Y = np.meshgrid(x, x, indexing="ij")
    d2 = np.minimum(np.abs(X - 0.5), 1 - np.abs(X - 0.5)) ** 2 + np.minimum(np.abs(Y - 0.5), 1 - np.abs(Y - 0.5)) ** 2
    w_a = (-5.0 * np.exp(-d2 / 0.03)).ravel()
    params = base(nx, [1.8, 1.8], {"A,B": 20.0}, [("A", 0.3), ("B", 0.7)], True)
    params["angles"] = [90.0, 90.0, 100.0]
    calc = run(params, {"A": w_a, "B": -w_a})
    lx = calc.prop_solver.get_lx()
    gamma = calc.prop_solver.get_angles_degrees()[2]
    check(calc.error_level < 1e-8 and abs(gamma - 120.0) < 1e-4 and abs(lx[0] - lx[1]) < 1e-6 * lx[0] and calc.iter < 600,
          f"Oblique2D from 100 deg: gamma {gamma:.6f}, a {lx[0]:.6f}, b {lx[1]:.6f}, {calc.iter} iterations")


def test_axis_ordering():
    nx, lx = [24, 16], [2.4, 1.6]
    x = (np.arange(24) + 0.5) / 24
    y = (np.arange(16) + 0.5) / 16
    X, Y = np.meshgrid(x, y, indexing="ij")
    w_a = 3.0 * np.cos(2 * np.pi * X) + 1.0 * np.cos(2 * np.pi * Y)          # shape (24, 16): standard [a, b]
    blocks = [("A", 0.5), ("B", 0.5)]
    standard = run(base(nx, lx, {"A,B": 15.0}, blocks, False), {"A": w_a.ravel(), "B": -w_a.ravel()})
    params = base([16, 24], [1.6, 2.4], {"A,B": 15.0}, blocks, False)
    params["axis_ordering"] = ["b", "a"]
    reordered = run(params, {"A": w_a.T.copy(), "B": -w_a.T.copy()})               # given in the order [b, a]
    d_phi = np.max(np.abs(np.array(standard.phi["A"]) - np.array(reordered.phi["A"])))
    check(abs(standard.free_energy - reordered.free_energy) < 1e-10 and d_phi < 1e-8,
          f"axis_ordering ['b', 'a'] reproduces the standard run (dF {abs(standard.free_energy - reordered.free_energy):.1e}, "
          f"dphi {d_phi:.1e})")

    # Walls on the a axis only (mixed boundary conditions: real-space solver); the
    # field is not periodic along a, so that walls and periodic boundaries differ
    w_a = 3.0 * np.cos(np.pi * X) + 1.0 * np.cos(2 * np.pi * Y)
    params = base(nx, lx, {"A,B": 15.0}, blocks, False)
    params["numerical_method"] = "cn-adi2"
    params["bc"] = ["reflecting", "reflecting", "periodic", "periodic"]
    standard = run(params, {"A": w_a.ravel(), "B": -w_a.ravel()})
    params = base([16, 24], [1.6, 2.4], {"A,B": 15.0}, blocks, False)
    params["numerical_method"] = "cn-adi2"
    params["bc"] = ["periodic", "periodic", "reflecting", "reflecting"]
    params["axis_ordering"] = ["b", "a"]
    reordered = run(params, {"A": w_a.T.copy(), "B": -w_a.T.copy()})
    d_phi = np.max(np.abs(np.array(standard.phi["A"]) - np.array(reordered.phi["A"])))
    check(abs(standard.free_energy - reordered.free_energy) < 1e-10 and d_phi < 1e-8,
          f"axis_ordering ['b', 'a'] with walls on a reproduces the standard run "
          f"(dF {abs(standard.free_energy - reordered.free_energy):.1e}, dphi {d_phi:.1e})")


def test_free_energy():
    x = (np.arange(64) + 0.5) / 64
    wave = 3.0 * np.cos(2 * np.pi * x)
    blocks = [("A", 0.3), ("B", 0.7)]
    runs = {}
    for fts in (False, True):
        params = base([64], [1.6], {"A,B": 20.0}, blocks, False)
        params["tolerance"] = 1e-10
        params["fts_consistent_energy"] = fts
        runs[fts] = run(params, {"A": wave, "B": -wave})
    dF = abs(runs[False].free_energy - runs[True].free_energy)
    calc = runs[True]
    w_aux = calc.mpt.to_aux_fields(np.array(calc.w))
    total_partitions = [calc.prop_solver.get_partition_function(p) for p in range(calc.prop_solver.get_n_polymer_types())]
    H = calc.mpt.compute_hamiltonian(calc.prop_solver.molecules, w_aux, total_partitions, calc.cb, include_const_term=True)
    dH = abs(H - calc.free_energy - calc.free_energy_homogeneous)
    check(dF < 1e-10 and dH < 1e-8, f"AB f = 0.3, chi N = 20: F with zero field means = F with saddle means ({dF:.1e}); "
                                     f"F + free_energy_homogeneous = field-theoretic H with its constant ({dH:.1e})")
    for fts in (False, True):
        params = base([16], [1.6], {"A,B": 8.0}, blocks, False)
        params["fts_consistent_energy"] = fts
        disordered = run(params, {"A": np.zeros(16), "B": np.zeros(16)})
        check(abs(disordered.free_energy) < 1e-12 and abs(disordered.free_energy_homogeneous - 8.0 * 0.3 * 0.7) < 1e-12,
              f"disordered AB f = 0.3, chi N = 8{', fts_consistent_energy' if fts else ''}: F = 0 "
              f"({disordered.free_energy:.1e}), free_energy_homogeneous = chi N f(1 - f)")

    fields = {"A": 4.0 * np.cos(2 * np.pi * x), "B": np.zeros(64), "C": -4.0 * np.cos(2 * np.pi * x)}
    for fts in (False, True):
        results = []
        for chi_bc in (40.0, 40.0 + 1e-4, 40.0 - 1e-4):
            params = base([64], [1.3], {"A,B": 10.0, "A,C": 10.0, "B,C": chi_bc}, [("A", 0.3), ("B", 0.4), ("C", 0.3)], True)
            params["tolerance"] = 1e-10
            params["fts_consistent_energy"] = fts
            results.append(run(params, fields))
        F = [r.free_energy for r in results]
        L = [r.prop_solver.get_lx()[0] for r in results]
        dF, dL = abs(F[0] - 0.5 * (F[1] + F[2])), abs(L[0] - 0.5 * (L[1] + L[2]))
        check(results[0].error_level < 1e-10 and dF < 1e-9 and dL < 1e-8,
              f"1^T (chi N)^-1 1 = 0{' with fts_consistent_energy' if fts else ''}: F and the box are the limit of "
              f"the neighbors (dF {dF:.1e}, dL {dL:.1e})")


def test_field_residual():
    x = (np.arange(64) + 0.5) / 64
    wave = 3.0 * np.cos(2 * np.pi * x)
    ab = {"A": wave, "B": -wave}
    abc = {"A": wave, "B": np.zeros(64), "C": -wave}
    triblock = [("A", 0.3), ("B", 0.4), ("C", 0.3)]
    cases = [
        ("AB chi N = -5", {"A,B": -5.0}, [("A", 0.5), ("B", 0.5)], 1.6, ab, (None, 5.0, 100.0)),
        ("ABC chi N AB/BC/AC = 10/10/60", {"A,B": 10.0, "B,C": 10.0, "A,C": 60.0}, triblock, 1.8, abc, (None, 5.0, 100.0)),
        ("ABC chi N AB/AC/BC = 10/10/50", {"A,B": 10.0, "A,C": 10.0, "B,C": 50.0}, triblock, 1.8, abc, (None, 2.0, 100.0)),
        ("ABC chi N AB/AC/BC = 10/10/39.9", {"A,B": 10.0, "A,C": 10.0, "B,C": 39.9}, triblock, 1.8, abc, (None, 2.0, 100.0)),
        ("ABC chi N AB/AC/BC = 10/10/40.1", {"A,B": 10.0, "A,C": 10.0, "B,C": 40.1}, triblock, 1.8, abc, (None, 2.0, 100.0)),
        ("AB f = 0.3, chi N = 20", {"A,B": 20.0}, [("A", 0.3), ("B", 0.7)], 1.6, ab, (0.5,)),
        ("ABC chi N AB/BC/AC = 3/3/12.5", {"A,B": 3.0, "B,C": 3.0, "A,C": 12.5}, [("A", 0.45), ("B", 0.1), ("C", 0.45)], 1.8,
         abc, (0.5, 2.0)),
    ]
    for name, chi_n, blocks, lx, fields, models in cases:
        for zeta_n in models:
            params = base([64], [lx], chi_n, blocks, False)
            params["max_iter"] = 500
            if zeta_n is not None:
                params["zeta_n"] = zeta_n
            calc = run(params, fields)
            model = "incompressible" if zeta_n is None else f"compressible (zeta N {zeta_n:g})"
            check(calc.error_level < 1e-8, f"{name}, {model}: converges ({calc.iter} iterations, "
                  f"error {calc.error_level:.1e})")


def test_singular_chi():
    x = (np.arange(64) + 0.5) / 64
    wave = 4.0 * np.cos(2 * np.pi * x)
    for zeta_n in (None, 10.0):
        model = "incompressible" if zeta_n is None else f"compressible (zeta N {zeta_n:g})"
        params = base([64], [1.6], {"A,B": 20.0, "A,C": 20.0, "B,C": 0.0}, [("A", 0.3), ("B", 0.4), ("C", 0.3)], False)
        reference = base([64], [1.6], {"A,B": 20.0}, [("A", 0.3), ("B", 0.7)], False)
        for p in (params, reference):
            p["tolerance"] = 1e-10
            if zeta_n is not None:
                p["zeta_n"] = zeta_n
        identical = run(params, {"A": wave, "B": -wave, "C": -wave})
        diblock = run(reference, {"A": wave, "B": -wave})
        dF = abs(identical.free_energy - diblock.free_energy)
        check(identical.error_level < 1e-10 and dF < 1e-9,
              f"ABC with B = C, {model}: F = F of the AB diblock ({identical.free_energy:.12f}, dF {dF:.1e})")

        results = []
        for chi_ac in (0.0, 1e-6):
            p = base([64], [1.0], {"A,B": 25.0, "B,C": 20.0, "A,C": chi_ac}, [("A", 0.3), ("B", 0.4), ("C", 0.3)], True)
            p["tolerance"] = 1e-10
            if zeta_n is not None:
                p["zeta_n"] = zeta_n
            results.append(run(p, {"A": wave, "B": -wave, "C": 0.5 * wave}))
        dF = abs(results[0].free_energy - results[1].free_energy)
        dL = abs(results[0].prop_solver.get_lx()[0] - results[1].prop_solver.get_lx()[0])
        check(all(r.error_level < 1e-10 for r in results) and dF < 1e-6 and dL < 1e-6,
              f"ABC with chi_AC N = 0, {model}: the limit chi_AC N -> 0 (dF {dF:.1e} and dL {dL:.1e} to chi_AC N = 1e-6)")


def main():
    test_angle_relaxation()
    test_axis_ordering()
    test_free_energy()
    test_field_residual()
    test_singular_chi()
    print(f"\n{n_pass} passed, {n_fail} failed")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
