#!/usr/bin/env python3
"""
Space-group platform equivalence test.

For each case (Pmmm on an anisotropic grid, Im-3m/BCC exercising the
Recursive3m CrysFFT path since nz/2 = 8), each chain model (continuous and
discrete), and each available platform, this test verifies that turning
space-group (SG) reduction on does not change the physics relative to a
full-grid computation with the identical symmetrized field:

  1. Same-platform SG-on vs SG-off:
     - partition function Q       (relative difference < 1e-10)
     - stress[0:3]                (relative difference < 1e-8)
     - concentration phi expanded via from_reduced_basis, compared pointwise
       to the SG-off control      (max abs difference < 1e-10)
  2. Cross-platform CPU-SG vs CUDA-SG: Q (rel < 1e-10) and pointwise phi
     (max abs diff < 1e-10), when CUDA is available.
  3. check_total_partition() passes, and material conservation:
     mean over the FULL grid of (phi_A + phi_B) == 1 within 1e-12.
     (An unweighted mean over the reduced basis would be wrong because
     orbits have unequal sizes - phi is expanded to the full grid first.)
  4. Mask + space group, per the contract actually implemented (below).

Both space-group basis modes are covered:
  - "physical" basis (m3 -> pmmm auto-enable, mirroring scft.py), which uses
    the CrysFFT PmmmDct / Recursive3m solvers, and
  - "irreducible" basis (no physical basis enabled).

ACTUAL mask + space-group contract encoded by this test
-------------------------------------------------------
Masks are stored on the full grid in ComputationBox; when a space group is
set, ComputationBox::set_space_group converts the stored mask to the reduced
basis, and the solvers either apply it in the reduced basis (CrysFFT paths)
or expand it back to the full grid (non-CrysFFT SG paths).

  - continuous + SG + mask: SUPPORTED and numerically correct on cpu-mkl,
    cpu-fftw, and cuda (Q and pointwise phi match the SG-off masked control
    to machine precision). Asserted below.
  - discrete + SG + mask: SUPPORTED and numerically correct on cpu-mkl,
    cpu-fftw, and cuda. Asserted below.
  - The mask subtests go through the public PropagatorSolver wrapper's
    `mask=` constructor argument (masks are always supplied on the FULL
    grid, even with a space group; ComputationBox reduces the stored mask
    when the space group is set). This also covers the wrapper's mask size
    validation, which historically crashed before solver initialization.
  - Each masked run additionally asserts that the mask actually bites:
    the expanded concentration must vanish on every blocked grid point.

Field / mask symmetry requirement: input fields and masks must be symmetric
under the FULL declared space group. SpaceGroup.symmetrize() only averages
over the physical-basis subgroup once a physical basis is enabled (e.g. the
8 mirror operations for the m3 basis of Im-3m), which is NOT sufficient:
a merely mirror-symmetric field propagates correctly (Q, phi) but the
symmetrized SG stress path then legitimately disagrees with the full-grid
control. This test therefore symmetrizes with a separate full-group
(irreducible-basis) SpaceGroup instance before projecting into the reduced
basis of the SG object actually used for the computation.

Masks are exactly space-group-symmetric binary (0/1) spherical cavities,
constructed deterministically by orbit-averaging + thresholding.

Runtime: well under 90 s. All random seeds fixed. No GPU selection is done
here - the default CUDA device (or CUDA_VISIBLE_DEVICES, if set by the
environment) is used. Exit code 0 on success, 1 on any failure, with a
printed report.
"""

import sys
import time

import numpy as np

from polymerfts import _core
from polymerfts.propagator_solver import PropagatorSolver

DS = 0.02  # 1/50 contour steps
BOND_LENGTHS = {"A": 1.0, "B": 1.0}
BC = ["periodic"] * 6
TOL_Q_REL = 1e-10
TOL_STRESS_REL = 1e-8
TOL_PHI_ABS = 1e-10
TOL_CONSERVATION = 1e-12

FAILURES = []


def check(condition, message):
    """Record a named assertion."""
    status = "PASS" if condition else "FAIL"
    print(f"    [{status}] {message}")
    if not condition:
        FAILURES.append(message)


def create_space_group(case_name, nx):
    if case_name == "Pmmm":
        return _core.SpaceGroup(nx, "Pmmm")
    return _core.SpaceGroup(nx, "Im-3m", 529)


def make_space_group(case_name, nx, physical):
    """Create the SpaceGroup, optionally auto-enabling the physical basis
    exactly like scft.py does for orthogonal cells (m3 -> pmmm)."""
    sg = create_space_group(case_name, nx)
    mode = "irreducible"
    if physical:
        try:
            sg.enable_m3_physical_basis()
            mode = "m3-physical"
        except Exception:
            try:
                sg.enable_pmmm_physical_basis()
                mode = "pmmm-physical"
            except Exception:
                mode = "irreducible"
        # Both test cases are known to support a physical basis; if the
        # enable calls start throwing, the CrysFFT coverage would silently
        # degrade to a duplicate irreducible run - fail loudly instead.
        check(mode != "irreducible",
              f"{case_name}: physical basis must be available "
              "(CrysFFT coverage would silently disappear)")
    return sg, mode


def grid_coords(nx, lx):
    xs = [(np.arange(n) + 0.5) * (l / n) for n, l in zip(nx, lx)]
    return np.meshgrid(*xs, indexing="ij")


def make_fields(nx, lx, sg, sg_full, seed=42):
    """Smooth fields with std = 5, symmetric under the FULL space group
    (symmetrized with sg_full) and exactly representable in the reduced
    basis of `sg` (round-tripped through to/from_reduced_basis)."""
    X, Y, Z = grid_coords(nx, lx)
    rng = np.random.default_rng(seed)
    w_full = {}
    w_reduced = {}
    for monomer in ["A", "B"]:
        f = np.zeros(nx)
        for _ in range(4):
            kx, ky, kz = rng.integers(1, 4, size=3)
            amp = rng.normal(0.0, 1.0)
            f += amp * (np.cos(2.0 * np.pi * kx * X / lx[0])
                        * np.cos(2.0 * np.pi * ky * Y / lx[1])
                        * np.cos(2.0 * np.pi * kz * Z / lx[2]))
        f = sg_full.symmetrize(f.flatten()).flatten()  # full-group symmetric
        f = f - f.mean()
        f = f / f.std() * 5.0
        # Round-trip so the full-grid control field is the exact expansion
        # of the reduced-basis field fed to the SG-on run.
        f_red = sg.to_reduced_basis(f.reshape(1, -1))[0].copy()
        f_full = sg.from_reduced_basis(f_red.reshape(1, -1))[0].copy()
        resid = np.max(np.abs(f_full - f))
        if resid > 1e-12:
            raise RuntimeError(
                f"symmetrized field is not representable in the reduced "
                f"basis (residual {resid:.3e})")
        w_full[monomer] = f_full
        w_reduced[monomer] = f_red
    return w_full, w_reduced


def make_mask(nx, lx, sg_full):
    """Exactly full-group-symmetric binary mask: spherical cavity,
    orbit-averaged and thresholded so symmetrize(mask) == mask exactly."""
    X, Y, Z = grid_coords(nx, lx)
    cx, cy, cz = [l / 2.0 for l in lx]
    r = 0.3 * min(lx)
    d2 = (X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2
    m = np.where(d2 < r * r, 0.0, 1.0).flatten()
    m_avg = sg_full.symmetrize(m).flatten()
    m = np.where(m_avg > 0.5, 1.0, 0.0)
    resid = np.max(np.abs(sg_full.symmetrize(m).flatten() - m))
    if resid != 0.0:
        raise RuntimeError(f"mask construction is not exactly symmetric: {resid}")
    n_blocked = int(np.sum(m == 0.0))
    if n_blocked == 0 or n_blocked == m.size:
        raise RuntimeError("mask is trivial (all ones or all zeros)")
    return m


def run_propagator_solver(platform, chain_model, nx, lx, sg, w_full, w_reduced,
                          use_sg, reduce_memory=False):
    """Unmasked run through the public PropagatorSolver interface.
    Returns Q, expanded full-grid phi, check_total_partition, stress[0:3]."""
    solver = PropagatorSolver(
        nx=nx, lx=lx, ds=DS,
        bond_lengths=dict(BOND_LENGTHS),
        bc=list(BC),
        chain_model=chain_model,
        numerical_method="rqm4" if chain_model == "continuous" else None,
        platform=platform,
        reduce_memory=reduce_memory,
        space_group=sg if use_sg else None,
    )
    solver.add_polymer(1.0, [["A", 0.5, 0, 1], ["B", 0.5, 1, 2]])
    w_in = w_reduced if use_sg else w_full
    solver.compute_propagators({"A": w_in["A"], "B": w_in["B"]})
    Q = solver.get_partition_function(0)
    chk = solver.check_total_partition()
    solver.compute_concentrations()
    phi = {}
    for m in ["A", "B"]:
        p = np.asarray(solver.get_concentration(m)).flatten()
        if use_sg:
            p = solver.from_reduced_basis(p)
        phi[m] = np.asarray(p).flatten()
    solver.compute_stress()
    stress = np.asarray(solver.get_stress())[:3].astype(float)
    return dict(Q=Q, chk=chk, phi=phi, stress=stress)


def run_masked(platform, chain_model, nx, lx, sg, w_full, w_reduced, mask,
               use_sg):
    """Masked run through the public PropagatorSolver wrapper, exercising the
    `mask=` constructor argument (masks are always passed on the full grid;
    ComputationBox reduces the stored mask when the space group is set)."""
    solver = PropagatorSolver(
        nx=list(nx), lx=list(lx), ds=DS,
        bond_lengths=dict(BOND_LENGTHS), bc=list(BC),
        chain_model=chain_model, numerical_method="rqm4",
        platform=platform, mask=np.asarray(mask).flatten(),
        space_group=sg if use_sg else None)
    solver.add_polymer(1.0, [["A", 0.5, 0, 1], ["B", 0.5, 1, 2]])
    w_in = w_reduced if use_sg else w_full
    solver.compute_propagators({"A": w_in["A"].copy(), "B": w_in["B"].copy()})
    Q = solver.get_partition_function(0)
    chk = solver.check_total_partition()
    solver.compute_concentrations()
    phi = {}
    for m in ["A", "B"]:
        p = np.asarray(solver.get_concentration(m)).flatten()
        if use_sg:
            p = sg.from_reduced_basis(p.reshape(1, -1))[0]
        phi[m] = np.asarray(p).flatten()
    return dict(Q=Q, chk=chk, phi=phi)


def compare_runs(tag, control, sg_run, with_stress):
    rel_q = abs(control["Q"] - sg_run["Q"]) / abs(control["Q"])
    check(rel_q < TOL_Q_REL, f"{tag}: Q relative diff {rel_q:.3e} < {TOL_Q_REL}")
    max_dphi = max(np.max(np.abs(control["phi"][m] - sg_run["phi"][m]))
                   for m in ["A", "B"])
    check(max_dphi < TOL_PHI_ABS,
          f"{tag}: pointwise phi max abs diff {max_dphi:.3e} < {TOL_PHI_ABS}")
    if with_stress:
        rel_s = np.max(np.abs((control["stress"] - sg_run["stress"])
                              / np.abs(control["stress"])))
        check(rel_s < TOL_STRESS_REL,
              f"{tag}: stress[0:3] relative diff {rel_s:.3e} < {TOL_STRESS_REL}")


def main():
    t_start = time.time()
    platforms = _core.PlatformSelector.avail_platforms()
    cpu_platforms = [p for p in platforms if p.startswith("cpu")]
    has_cuda = "cuda" in platforms
    print(f"Available platforms: {platforms}")
    check(len(cpu_platforms) >= 1,
          "at least one CPU platform must be available "
          "(otherwise this test would pass while covering nothing)")
    if not has_cuda:
        print("NOTE: CUDA not available - CUDA subtests skipped.")

    cases = [
        ("Pmmm", [32, 24, 16], [3.0, 2.0, 1.5]),
        ("Im-3m", [16, 16, 16], [1.9, 1.9, 1.9]),  # Recursive3m: nz/2 = 8
    ]

    for case_name, nx, lx in cases:
        # Full-group (irreducible-basis) instance used ONLY to symmetrize
        # fields/masks over the complete space group (see module docstring).
        sg_full = create_space_group(case_name, nx)
        for physical in [True, False]:
            sg, mode = make_space_group(case_name, nx, physical)
            n_reduced = sg.get_n_reduced_basis()
            total_grid = int(np.prod(nx))
            print(f"\n=== Case {case_name} nx={nx} lx={lx} basis={mode} "
                  f"(n_reduced={n_reduced}, total={total_grid}) ===")
            w_full, w_reduced = make_fields(nx, lx, sg, sg_full, seed=42)
            for m in ["A", "B"]:
                assert abs(np.std(w_full[m]) - 5.0) < 0.5, "field std must be ~5"
            mask = make_mask(nx, lx, sg_full)
            print(f"  mask: {int(np.sum(mask == 0.0))}/{mask.size} "
                  f"blocked grid points (exactly symmetric)")

            for chain_model in ["continuous", "discrete"]:
                # ---- unmasked: SG-on vs SG-off per platform ----
                results = {}
                for platform in platforms:
                    tag = f"{case_name}/{mode}/{chain_model}/{platform}"
                    control = run_propagator_solver(
                        platform, chain_model, nx, lx, sg, w_full, w_reduced,
                        use_sg=False)
                    sg_run = run_propagator_solver(
                        platform, chain_model, nx, lx, sg, w_full, w_reduced,
                        use_sg=True)
                    results[platform] = (control, sg_run)
                    print(f"  -- {tag}: Q_off={control['Q']:.12e} "
                          f"Q_SG={sg_run['Q']:.12e}")
                    check(control["chk"], f"{tag}: SG-off check_total_partition()")
                    check(sg_run["chk"], f"{tag}: SG-on check_total_partition()")
                    compare_runs(f"{tag}: SG-on vs SG-off", control, sg_run,
                                 with_stress=True)
                    # reduce_memory (checkpointing) with SG must reproduce the
                    # standard SG run for both chain models on both platforms.
                    rm_run = run_propagator_solver(
                        platform, chain_model, nx, lx, sg, w_full, w_reduced,
                        use_sg=True, reduce_memory=True)
                    check(rm_run["chk"],
                          f"{tag}: SG-on reduce_memory check_total_partition()")
                    compare_runs(f"{tag}: SG-on reduce_memory vs standard",
                                 sg_run, rm_run, with_stress=True)
                    # Material conservation, expanded to the full grid first.
                    total_phi = sg_run["phi"]["A"] + sg_run["phi"]["B"]
                    cons = abs(np.mean(total_phi) - 1.0)
                    check(cons < TOL_CONSERVATION,
                          f"{tag}: SG-on material conservation "
                          f"|mean(phi_A+phi_B)-1| = {cons:.3e} "
                          f"< {TOL_CONSERVATION}")

                # ---- unmasked: cross-platform CPU-SG vs CUDA-SG ----
                if has_cuda:
                    _, cuda_sg = results["cuda"]
                    for cpu in cpu_platforms:
                        _, cpu_sg = results[cpu]
                        tag = (f"{case_name}/{mode}/{chain_model}/"
                               f"{cpu}-SG vs cuda-SG")
                        rel_q = abs(cpu_sg["Q"] - cuda_sg["Q"]) / abs(cuda_sg["Q"])
                        check(rel_q < TOL_Q_REL,
                              f"{tag}: Q relative diff {rel_q:.3e} < {TOL_Q_REL}")
                        max_dphi = max(
                            np.max(np.abs(cpu_sg["phi"][m] - cuda_sg["phi"][m]))
                            for m in ["A", "B"])
                        check(max_dphi < TOL_PHI_ABS,
                              f"{tag}: pointwise phi max abs diff "
                              f"{max_dphi:.3e} < {TOL_PHI_ABS}")

                # ---- mask + SG (supported for both chain models on all
                # ---- platforms; factory API, see module docstring) ----
                blocked = np.asarray(mask).flatten() == 0.0
                for platform in platforms:
                    tag = f"{case_name}/{mode}/{chain_model}/{platform}/mask"
                    control = run_masked(platform, chain_model, nx, lx, sg,
                                         w_full, w_reduced, mask, use_sg=False)
                    sg_run = run_masked(platform, chain_model, nx, lx, sg,
                                        w_full, w_reduced, mask, use_sg=True)
                    check(control["chk"], f"{tag}: SG-off check_total_partition()")
                    check(sg_run["chk"], f"{tag}: SG-on check_total_partition()")
                    compare_runs(f"{tag}: SG-on vs SG-off", control, sg_run,
                                 with_stress=False)
                    # The mask must actually bite: concentration vanishes on
                    # blocked grid points. (Guards against a regression that
                    # silently drops the mask on BOTH runs, which the SG-on
                    # vs SG-off comparison alone cannot detect.)
                    for run_name, run in (("SG-off", control), ("SG-on", sg_run)):
                        max_blocked_phi = max(
                            np.max(np.abs(run["phi"][m][blocked]))
                            for m in ["A", "B"])
                        check(max_blocked_phi < 1e-12,
                              f"{tag}: {run_name} phi vanishes on blocked "
                              f"points (max {max_blocked_phi:.3e} < 1e-12)")

    elapsed = time.time() - t_start
    print("\n" + "=" * 64)
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} assertion(s) failed "
              f"(runtime {elapsed:.1f} s):")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print(f"All space-group platform equivalence tests PASSED "
          f"(runtime {elapsed:.1f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
