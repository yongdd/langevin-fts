"""
Stress and lattice updates on oblique cells with the ObliqueZ CrysFFT.

Cells with alpha = beta = 90 degrees and gamma != 90 whose space group has a
z-mirror use the ObliqueZ CrysFFT, with the z-mirror physical basis (identity
map) and with the irreducible basis (gather/scatter between the basis and the
physical grid). TestSpaceGroupPlatformEquivalence covers Q, phi and the
length derivatives of these runs; this test covers the rest of what a box
relaxation of such a cell consumes:

  1. All six stress components [dH/dL1, dH/dL2, dH/dL3, dH/dgamma, dH/dbeta,
     dH/dalpha], with the off-diagonal ones forced on, SG-on vs SG-off with
     the identical symmetrized fields on each platform. dH/dgamma must be
     clearly nonzero and dH/dbeta = dH/dalpha = 0 by the z-mirror symmetry.
  2. set_lattice_parameters() to new lengths and, for the monoclinic cell, a
     new gamma, then to a new gamma (monoclinic) or c (hexagonal) alone: Q,
     phi and the stress agree with a solver created at that lattice (the
     CrysFFT tables of the old cell must not survive, also when only an
     angle changes).
  3. CPU vs CUDA for the stress of each basis.

Cases: P6/mmm on a hexagonal 12x12x8 cell and P2/m with unique axis c (Hall
58) on an 11x10x8 cell with gamma = 100 -> 104 -> 97 degrees (odd nx), for
rqm4, rk2 and the discrete chain. Fields are strong (std 5) and symmetrized with the
full group. Runtime: a few seconds. Exit code 0 on success, 1 on failure.
"""

import sys
import time

import numpy as np

from polymerfts import _core
from polymerfts.propagator_solver import PropagatorSolver
from cuda_part import cuda_part_missing

DS = 0.025
BOND_LENGTHS = {"A": 1.0, "B": 1.2}
TOL_Q_REL = 1e-10
TOL_PHI_ABS = 1e-10
TOL_STRESS_REL = 1e-8      # relative to the largest stress component
TOL_ZERO_STRESS = 1e-12    # dH/dbeta and dH/dalpha

# (name, symbol, Hall number, nx, lattices visited in turn with set_lattice_parameters)
CASES = [
    ("P6/mmm", "P6/mmm", None, [12, 12, 8],
     [([2.0, 2.0, 1.5], [90.0, 90.0, 120.0]),
      ([2.1, 2.1, 1.45], [90.0, 90.0, 120.0]),
      ([2.1, 2.1, 1.55], [90.0, 90.0, 120.0])]),     # c only
    ("P2/m", "P2/m", 58, [11, 10, 8],
     [([1.5, 1.3, 1.2], [90.0, 90.0, 100.0]),
      ([1.55, 1.28, 1.22], [90.0, 90.0, 104.0]),
      ([1.55, 1.28, 1.22], [90.0, 90.0, 97.0])]),    # gamma only
]
METHODS = [("continuous", "rqm4"), ("continuous", "rk2"), ("discrete", None)]
BASES = ["none", "irreducible", "z-mirror"]

FAILURES = []


def check(condition, message):
    """Record a named assertion."""
    status = "PASS" if condition else "FAIL"
    print(f"    [{status}] {message}")
    if not condition:
        FAILURES.append(message)


def make_space_group(symbol, hall, nx, basis):
    sg = _core.SpaceGroup(nx, symbol, hall) if hall else _core.SpaceGroup(nx, symbol)
    if basis == "z-mirror":
        sg.enable_z_mirror_physical_basis()
    return sg


def run(platform, chain_model, method, case, basis, w_full, lattices):
    """Q, full-grid phi and the six stress components at each lattice in turn
    (set_lattice_parameters between them)."""
    _, symbol, hall, nx, _ = case
    sg = None if basis == "none" else make_space_group(symbol, hall, nx, basis)
    lx, angles = lattices[0]
    solver = PropagatorSolver(nx=nx, lx=lx, ds=DS, bond_lengths=dict(BOND_LENGTHS),
                              bc=["periodic"] * 6, chain_model=chain_model,
                              numerical_method=method, platform=platform,
                              angles=angles, space_group=sg)
    solver.add_polymer(1.0, [["A", 0.4, 0, 1], ["B", 0.6, 1, 2]])
    solver.set_force_off_diagonal_stress(True)
    w = {m: (sg.to_reduced_basis(w_full[m]) if sg is not None else w_full[m]) for m in "AB"}
    results = []
    for k, (lx_k, angles_k) in enumerate(lattices):
        if k > 0:
            solver.set_lattice_parameters(lx_k, angles_k)
        solver.compute_propagators(w)
        solver.compute_concentrations()
        solver.compute_stress()
        phi = {}
        for m in "AB":
            p = np.asarray(solver.get_concentration(m)).flatten()
            phi[m] = np.asarray(solver.from_reduced_basis(p) if sg is not None else p).flatten()
        results.append(dict(Q=solver.get_partition_function(0), phi=phi,
                            stress=np.asarray(solver.get_stress(), dtype=float),
                            chk=solver.check_total_partition()))
    return results


def compare(tag, a, b, tol_q=TOL_Q_REL, tol_phi=TOL_PHI_ABS, tol_stress=TOL_STRESS_REL):
    dq = abs(a["Q"] - b["Q"]) / abs(b["Q"])
    dphi = max(np.max(np.abs(a["phi"][m] - b["phi"][m])) for m in "AB")
    ds = np.max(np.abs(a["stress"] - b["stress"])) / np.max(np.abs(b["stress"]))
    check(dq < tol_q, f"{tag}: Q relative diff {dq:.2e} < {tol_q:.0e}")
    check(dphi < tol_phi, f"{tag}: phi max abs diff {dphi:.2e} < {tol_phi:.0e}")
    check(ds < tol_stress, f"{tag}: six stress components, max diff {ds:.2e} < {tol_stress:.0e} (relative)")


def main():
    t_start = time.time()
    platforms = list(_core.PlatformSelector.avail_platforms())
    print(f"Available platforms: {platforms}")

    for case in CASES:
        name, symbol, hall, nx, lattices = case
        sg_full = make_space_group(symbol, hall, nx, "irreducible")
        rng = np.random.default_rng(2024)
        w_full = {}
        for m in "AB":
            field = sg_full.symmetrize(rng.normal(0.0, 1.0, int(np.prod(nx))))
            w_full[m] = field * (5.0 / np.std(field))

        for chain_model, method in METHODS:
            tag_method = method or "discrete"
            print(f"\n{name} {nx}, {tag_method}")
            runs = {}
            for platform in platforms:
                for basis in BASES:
                    tag = f"{name}/{tag_method}/{platform}/{basis}"
                    r = run(platform, chain_model, method, case, basis, w_full, lattices)
                    runs[(platform, basis)] = r
                    for k, rk in enumerate(r):
                        s = rk["stress"]
                        check(rk["chk"], f"{tag}/lattice {k}: check_total_partition()")
                        check(abs(s[3]) > 1e-3 * np.max(np.abs(s[:3])),
                              f"{tag}/lattice {k}: dH/dgamma = {s[3]:.4e} is nonzero")
                        check(max(abs(s[4]), abs(s[5])) < TOL_ZERO_STRESS,
                              f"{tag}/lattice {k}: dH/dbeta, dH/dalpha = {s[4]:.1e}, {s[5]:.1e} vanish")
                    # The updated solver against solvers created at the later lattices
                    for k in range(1, len(lattices)):
                        fresh = run(platform, chain_model, method, case, basis, w_full, [lattices[k]])[0]
                        compare(f"{tag}/lattice {k}: set_lattice_parameters vs fresh solver", r[k], fresh)
                for basis in ["irreducible", "z-mirror"]:
                    for k in range(len(lattices)):
                        compare(f"{name}/{tag_method}/{platform}/lattice {k}: SG {basis} vs SG-off",
                                runs[(platform, basis)][k], runs[(platform, "none")][k])
            if "cuda" in platforms:
                for platform in platforms:
                    if platform == "cuda":
                        continue
                    for basis in BASES:
                        for k in range(len(lattices)):
                            compare(f"{name}/{tag_method}/{basis}/lattice {k}: {platform} vs cuda",
                                    runs[(platform, basis)][k], runs[("cuda", basis)][k])

    elapsed = time.time() - t_start
    print("\n" + "=" * 64)
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} assertion(s) failed (runtime {elapsed:.1f} s):")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    if cuda_part_missing():
        return 1
    print(f"All ObliqueZ stress tests PASSED (runtime {elapsed:.1f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
