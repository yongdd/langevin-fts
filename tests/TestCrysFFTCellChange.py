#!/usr/bin/env python3
"""CrysFFT solvers after a cell change: updated solver against a fresh one.

A solver whose lattice changes (set_lattice_parameters) rebuilds its CrysFFT
tables, including the k^2 tables that the stress uses. For each platform, each
case computes Q, the concentrations and the stress after one or two cell changes
and compares them with a solver created at the new cell (they agree to round-off;
the CUDA solvers share and recycle their tables). The changes include one axis
alone, so a cell comparison that ignored Ly or Lz would keep stale tables. CUDA
is also compared with the CPU, stress included. Cases (fields symmetrized and
rescaled to a standard deviation of 5, continuous and discrete chains):
- Pmmm 16x12x8, Pmmm physical basis and irreducible basis: the Pmmm DCT
  (nz/2 = 4); Ly alone, then Lx, Ly and Lz;
- Pmmm 32x24x16, irreducible basis: the recursive 3m CrysFFT (nz/2 = 8); Ly
  alone, then Lz alone;
- Im-3m 16^3, m3 physical basis: the recursive 3m CrysFFT on a cubic cell.
"""

import contextlib
import io

import numpy as np
from polymerfts import _core, PlatformSelector, PropagatorSolver
from cuda_part import cuda_part_missing

CASES = [
    ("Pmmm", None, [16, 12, 8], "pmmm", [[2.0, 1.8, 1.6], [2.0, 1.9, 1.6], [2.1, 1.95, 1.5]]),
    ("Pmmm", None, [16, 12, 8], "irreducible", [[2.0, 1.8, 1.6], [2.0, 1.9, 1.6], [2.1, 1.95, 1.5]]),
    ("Pmmm", None, [32, 24, 16], "irreducible", [[3.0, 2.4, 2.0], [3.0, 2.6, 2.0], [3.0, 2.6, 2.2]]),
    ("Im-3m", 529, [16, 16, 16], "m3", [[1.9] * 3, [2.0] * 3]),
]


def space_group(symbol, hall, nx, basis):
    sg = _core.SpaceGroup(nx, symbol, hall) if hall else _core.SpaceGroup(nx, symbol)
    if basis == "pmmm":
        sg.enable_pmmm_physical_basis()
    if basis == "m3":
        sg.enable_m3_physical_basis()
    return sg


def results(platform, chain, case, lattices, fields):
    """Q, phi_A and the stress after each lattice of `lattices` (one solver, updated in turn)."""
    symbol, hall, nx, basis, _ = case
    sg = space_group(symbol, hall, nx, basis)
    out = []
    with contextlib.redirect_stdout(io.StringIO()):
        s = PropagatorSolver(nx=nx, lx=lattices[0], ds=0.025, bond_lengths={"A": 1.0, "B": 1.2}, bc=["periodic"] * 6,
                             chain_model=chain, numerical_method="rqm4" if chain == "continuous" else None,
                             platform=platform, space_group=sg)
        s.add_polymer(1.0, [["A", 0.4, 0, 1], ["B", 0.6, 1, 2]])
        w = {m: sg.to_reduced_basis(fields[m]) for m in "AB"}
        for i, lx in enumerate(lattices):
            if i:
                s.set_lattice_parameters(lx, [90.0, 90.0, 90.0])
            s.compute_propagators(w)
            s.compute_concentrations()
            s.compute_stress()
            assert s.check_total_partition(), (platform, chain, symbol, basis, lx)
            phi = np.asarray(s.get_concentration("A")) + np.asarray(s.get_concentration("B"))
            assert abs(s.mean(phi) - 1.0) < 1e-10, (platform, chain, symbol, basis, lx)
            out.append((s.get_partition_function(0), np.asarray(s.get_concentration("A"), float).copy(),
                        np.asarray(s.get_stress(), float).copy()))
    return out


def max_rel(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))) / max(np.max(np.abs(b)), 1e-300))


def main():
    platforms = PlatformSelector.avail_platforms()
    cpu = [p for p in platforms if p.startswith("cpu")]
    failures = checks = 0
    for case in CASES:
        symbol, hall, nx, basis, lattices = case
        full = space_group(symbol, hall, nx, "irreducible")
        rng = np.random.default_rng(5)
        fields = {}
        for m in "AB":
            f = full.symmetrize(rng.normal(0.0, 1.0, int(np.prod(nx))))
            fields[m] = 5.0 * f / np.std(f)
        for chain in ("continuous", "discrete"):
            reference = {}
            for platform in platforms:
                updated = results(platform, chain, case, lattices, fields)
                for k in range(1, len(lattices)):
                    fresh = results(platform, chain, case, [lattices[k]], fields)[0]
                    d = max(abs(updated[k][0] - fresh[0]) / abs(fresh[0]), max_rel(updated[k][1], fresh[1]),
                            max_rel(updated[k][2], fresh[2]))
                    ok = d < 1e-12
                    checks += 1
                    failures += not ok
                    print(f"[{'PASS' if ok else 'FAIL'}] {platform} {symbol} {basis} {nx} {chain}: lattice {lattices[k]} "
                          f"updated vs fresh {d:.1e}")
                reference[platform] = updated
            if "cuda" in reference and cpu:
                d = max(max_rel(g[2], c[2]) for g, c in zip(reference["cuda"], reference[cpu[0]]))
                ok = d < 1e-10
                checks += 1
                failures += not ok
                print(f"[{'PASS' if ok else 'FAIL'}] cuda vs {cpu[0]} {symbol} {basis} {nx} {chain}: stress {d:.1e}")
    print(f"\n{checks - failures} passed, {failures} failed")
    return 1 if cuda_part_missing() or failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
