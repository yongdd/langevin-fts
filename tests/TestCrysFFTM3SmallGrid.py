#!/usr/bin/env python3
"""
The m3 physical basis on grids below the selector's nz/2 >= 8 size heuristic.

SpaceGroup.enable_m3_physical_basis() accepts any even grid whose m3 mapping
exists, but the CrysFFT selector applied nz/2 >= 8 (a performance heuristic
meant for choosing a basis) to an m3 basis that was already enabled, so
PropagatorSolver failed with "M3 physical basis is enabled but recursive 3m
CrysFFT is unavailable" on Im-3m 8^3 and 12^3. The recursive 3m engines are
exact for nz/2 >= 2. This test enables the m3 basis on those grids and compares
each platform's run with the full-grid run on the identical symmetrized field
(std 5): partition function (relative 1e-10), concentrations (absolute 1e-10),
stress[0:3] (relative 1e-8), check_total_partition() and material conservation,
for continuous (RQM4) and discrete chains. SpaceGroup.m3_crysfft_feasible(),
which SCFT uses to choose a basis, keeps the heuristic.
"""

import contextlib
import io
import sys

import numpy as np

from polymerfts import _core
from polymerfts.propagator_solver import PropagatorSolver
from cuda_part import cuda_part_missing

DS = 0.02
TOL_Q_REL = 1e-10
TOL_PHI_ABS = 1e-10
TOL_STRESS_REL = 1e-8
FAILURES = []


def check(condition, message):
    print(f"    [{'PASS' if condition else 'FAIL'}] {message}")
    if not condition:
        FAILURES.append(message)


def make_space_group(nx, m3_basis):
    with contextlib.redirect_stdout(io.StringIO()):
        sg = _core.SpaceGroup(nx, "Im-3m", 529)
        if m3_basis:
            sg.enable_m3_physical_basis()
    return sg


def run(platform, chain_model, nx, lx, sg, w_full):
    with contextlib.redirect_stdout(io.StringIO()):
        solver = PropagatorSolver(
            nx=list(nx), lx=list(lx), ds=DS, bond_lengths={"A": 1.0, "B": 1.0}, bc=["periodic"] * 6,
            chain_model=chain_model, numerical_method="rqm4" if chain_model == "continuous" else None,
            platform=platform, space_group=sg)
        solver.add_polymer(1.0, [["A", 0.5, 0, 1], ["B", 0.5, 1, 2]])
        if sg is None:
            w = {m: w_full[m] for m in "AB"}
        else:
            w = {m: sg.to_reduced_basis(w_full[m][None, :])[0].copy() for m in "AB"}
        solver.compute_propagators(w)
        chk = solver.check_total_partition()
        solver.compute_concentrations()
        phi = {}
        for m in "AB":
            p = np.asarray(solver.get_concentration(m)).flatten()
            phi[m] = p if sg is None else sg.from_reduced_basis(p[None, :])[0]
        solver.compute_stress()
        stress = np.asarray(solver.get_stress())[:3].astype(float)
    return dict(Q=solver.get_partition_function(0), chk=chk, phi=phi, stress=stress)


def main():
    platforms = _core.PlatformSelector.avail_platforms()
    print(f"Available platforms: {platforms}")
    rng = np.random.default_rng(29)
    for n in (8, 12):
        nx, lx = [n] * 3, [1.6] * 3
        sg_full = make_space_group(nx, m3_basis=False)
        sg = make_space_group(nx, m3_basis=True)
        check(sg.using_m3_physical_basis(), f"Im-3m {n}^3: the m3 basis is enabled")
        check(not sg_full.m3_crysfft_feasible(),
              f"Im-3m {n}^3: m3_crysfft_feasible() keeps the size heuristic (SCFT does not choose m3 here)")
        w_full = {}
        for m in "AB":
            f = sg_full.symmetrize(rng.normal(0.0, 1.0, n ** 3)).flatten()
            f = (f - f.mean()) / f.std() * 5.0
            w_full[m] = sg.from_reduced_basis(sg.to_reduced_basis(f[None, :]))[0].copy()
        for chain_model in ("continuous", "discrete"):
            for platform in platforms:
                tag = f"Im-3m {n}^3 m3 basis/{chain_model}/{platform}"
                try:
                    on = run(platform, chain_model, nx, lx, sg, w_full)
                except RuntimeError as exc:
                    check(False, f"{tag}: solver with the m3 basis raised: {str(exc).strip().splitlines()[-1]}")
                    continue
                off = run(platform, chain_model, nx, lx, None, w_full)
                rel_q = abs(on["Q"] - off["Q"]) / abs(off["Q"])
                dphi = max(np.max(np.abs(on["phi"][m] - off["phi"][m])) for m in "AB")
                rel_s = np.max(np.abs(on["stress"] - off["stress"]) / np.abs(off["stress"]))
                cons = abs(np.mean(on["phi"]["A"] + on["phi"]["B"]) - 1.0)
                check(on["chk"], f"{tag}: check_total_partition()")
                check(rel_q < TOL_Q_REL, f"{tag}: Q relative diff {rel_q:.2e} < {TOL_Q_REL}")
                check(dphi < TOL_PHI_ABS, f"{tag}: phi max abs diff {dphi:.2e} < {TOL_PHI_ABS}")
                check(rel_s < TOL_STRESS_REL, f"{tag}: stress[0:3] relative diff {rel_s:.2e} < {TOL_STRESS_REL}")
                check(cons < 1e-12, f"{tag}: |mean(phi_A + phi_B) - 1| = {cons:.2e} < 1e-12")
    if FAILURES:
        print(f"\nFAILED: {len(FAILURES)} check(s)")
        return 1
    if cuda_part_missing():
        return 1
    print("\nm3 basis on small grids: PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
