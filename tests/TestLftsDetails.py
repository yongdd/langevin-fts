#!/usr/bin/env python3
"""
L-FTS details that change recorded results.

1. WTMD restart: a run continued from a checkpoint reproduces the uninterrupted
   run, including the WTMD statistics (u, U', I0, I1). Until 2026-10 the
   checkpoint had no I1 and no samples since the last update, so dH/dpsi = I1/I0
   was underestimated after a restart.
2. WTMD in a non-orthogonal cell uses the cell volume and |k| from the cell
   metric (the minimal alias, as the propagator solvers), and its order parameter
   from the half spectrum equals the full-spectrum sum, also for odd nz.
3. A step whose saddle point fails is retried, so every
   recording period holds its samples (until 2026-10 the step was skipped:
   no dH file at a recording step, two periods in the next one).
4. Compressible L-FTS: the uniform mode of the pressure-like field sits at its
   analytic saddle value -(c1 + sum_j A_jk <phi_j>)/(2 c2) (until 2026-10 it was
   pinned to whatever the first solve left, here 7.49993 instead of 7.5), and
   the recorded H belongs to the returned fields (a consistency check that also
   held before).
"""

import os
import io
import sys
import glob
import shutil
import tempfile
import contextlib

os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from scipy.io import loadmat
from polymerfts import _core, LFTS
from polymerfts.wtmd import WTMD
from polymerfts.smearing import Smearing

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


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def cpu_platform():
    cpu = [p for p in _core.PlatformSelector.avail_platforms() if p.startswith("cpu")]
    return cpu[0] if cpu else "cuda"


def params(work_dir, nx, lx, max_step, **extra):
    p = {"nx": nx, "lx": lx, "chain_model": "continuous", "ds": 1 / 20,
         "segment_lengths": {"A": 1.0, "B": 1.0}, "chi_n": {"A,B": 15.0},
         "polymers": [{"volume_fraction": 1.0, "blocks": [{"type": "A", "length": 0.5},
                                                                   {"type": "B", "length": 0.5}]}],
         "platform": cpu_platform(), "langevin": {"nbar": 10000, "dt": 8.0, "max_step": max_step},
         "recording": {"dir": work_dir, "checkpoint_interval": 4, "sample_interval": 1, "statistics_save_interval": 2},
         "saddle": {"max_iter": 200, "tolerance": 1e-4},
         "compressor": {"name": "am", "max_hist": 20, "start_error": 5e-1, "mix_min": 0.01, "mix_init": 0.01},
         "verbose_level": 1}
    p.update(extra)
    return p


def test_wtmd_restart(root):
    nx, lx = [8, 8, 8], [2.0, 2.0, 2.0]
    wtmd = {"ell": 4, "kc": 6.02, "delta_t": 5.0, "sigma_psi": 0.16, "psi_min": 0.0, "psi_max": 10.0,
            "dpsi": 1e-3, "update_freq": 3, "recording_period": 100}
    rng = np.random.default_rng(3)
    w0 = {"A": rng.normal(0.0, 0.5, 512), "B": rng.normal(0.0, 0.5, 512)}

    straight = quiet(LFTS, params(os.path.join(root, "straight"), nx, lx, 8, wtmd=wtmd), random_seed=1)
    quiet(straight.run, initial_fields=w0)
    first = quiet(LFTS, params(os.path.join(root, "first"), nx, lx, 4, wtmd=wtmd), random_seed=1)
    quiet(first.run, initial_fields=w0)
    second = quiet(LFTS, params(os.path.join(root, "second"), nx, lx, 8, wtmd=wtmd), random_seed=1)
    quiet(second.continue_run, os.path.join(root, "first", "fields_000004.mat"))

    a, b = straight.wtmd, second.wtmd
    diffs = [np.max(np.abs(a.u - b.u)), np.max(np.abs(a.up - b.up)), np.max(np.abs(a.I0 - b.I0))]
    diffs += [np.max(np.abs(a.I1[key] - b.I1.get(key, 0.0))) for key in a.I1]
    w_straight = loadmat(os.path.join(root, "straight", "fields_000008.mat"))["w_A"]
    w_second = loadmat(os.path.join(root, "second", "fields_000008.mat"))["w_A"]
    check(max(diffs) < 1e-12 and np.max(np.abs(w_straight - w_second)) < 1e-10 and len(a.I1) > 0,
          f"WTMD: continued run = uninterrupted run (statistics {max(diffs):.1e}, "
          f"fields {np.max(np.abs(w_straight - w_second)):.1e})")


def test_wtmd_cell():
    nx, lx, angles = [10, 10, 8], [1.8, 1.8, 2.3], [90.0, 90.0, 120.0]
    wt = quiet(WTMD, nx, lx, 10000, np.array([-15.0, 0.0]), [0], angles=angles)
    volume = lx[0] * lx[1] * lx[2] * np.sin(np.deg2rad(120.0))
    sm = quiet(Smearing, nx, lx, {"type": "gaussian", "a_int": 0.1}, angles=angles)
    k_ref = np.sqrt(sm.k_sq[:, :, :nx[2] // 2 + 1])
    fk_ref = 1.0 / (1.0 + np.exp(12.0 * (k_ref - wt.kc) / wt.kc))
    check(abs(wt.V - volume) < 1e-12 * volume and np.max(np.abs(wt.fk - fk_ref)) < 1e-12,
          "WTMD in a hexagonal cell: cell volume and |k| from the cell metric")

    # The half spectrum stands for the full one: with odd nz the plane kz = nz//2 is not its
    # own conjugate. Coarse grids, where fk does not vanish on that plane, show it
    rng = np.random.default_rng(7)
    for nx, lx in (([6, 6, 6], [2.0] * 3), ([6, 6, 5], [2.0] * 3), ([12, 12, 9], [4.0] * 3)):
        wt = quiet(WTMD, nx, lx, 10000, np.array([-12.0, 200.0]), [0])
        w = rng.normal(0, 5, nx)
        psi = wt.compute_order_parameter(0, np.array([w.ravel(), np.zeros(w.size)]))
        k = np.meshgrid(*[2 * np.pi / lx[d] * np.fft.fftfreq(nx[d], 1.0 / nx[d]) for d in range(3)], indexing="ij")
        fk = 1.0 / (1.0 + np.exp(12.0 * (np.sqrt(sum(kd**2 for kd in k)) - wt.kc) / wt.kc))
        psi_full = np.sum(np.abs(np.fft.fftn(w))**4 * fk)**0.25 / np.prod(nx)
        check(abs(psi - psi_full) < 1e-12 * psi_full, f"WTMD order parameter from the half spectrum, nx = {nx}")


class FailingLFTS(LFTS):
    """LFTS whose saddle-point solve reports failure on chosen calls."""
    fail_calls = ()

    def find_saddle_point(self, w_aux):
        self._calls = getattr(self, "_calls", 0) + 1
        phi, hamiltonian, saddle_iter, error_level = super().find_saddle_point(w_aux)
        if self._calls in self.fail_calls:
            error_level = np.inf
        return phi, hamiltonian, saddle_iter, error_level


def test_saddle_failure(root):
    work = os.path.join(root, "failing")
    # Call 1 is the initial solve, call 2 step 1, call 3 the first try of step 2 (a recording step)
    FailingLFTS.fail_calls = (3,)
    sim = quiet(FailingLFTS, params(work, [16, 16], [3.0, 3.0], 4), random_seed=1)
    rng = np.random.default_rng(3)
    quiet(sim.run, initial_fields={"A": rng.normal(0.0, 0.5, 256), "B": rng.normal(0.0, 0.5, 256)})
    counts = {os.path.basename(f): np.size(loadmat(f, squeeze_me=True)["H_history"])
              for f in sorted(glob.glob(os.path.join(work, "dH_*.mat")))}
    sf_files = sorted(os.path.basename(f) for f in glob.glob(os.path.join(work, "structure_function_*.mat")))
    check(counts == {"dH_000002.mat": 2, "dH_000004.mat": 2}
          and sf_files == ["structure_function_000002.mat", "structure_function_000004.mat"],
          f"failed saddle point at a recording step: the step is repeated, records {counts}")


def test_compressible_mean(root):
    work = os.path.join(root, "compressible")
    sim = quiet(LFTS, params(work, [16, 16], [3.0, 3.0], 4, zeta_n=100.0), random_seed=1)
    rng = np.random.default_rng(3)
    # Both fields start with a mean of 3: a uniform offset of the pressure-like field
    w0 = {"A": 3.0 + rng.normal(0.0, 0.5, 256), "B": 3.0 + rng.normal(0.0, 0.5, 256)}
    quiet(sim.run, initial_fields=w0)
    data = loadmat(os.path.join(work, "fields_000004.mat"), squeeze_me=True)
    w = np.array([np.ravel(data["w_A"]), np.ravel(data["w_B"])])
    w_aux = sim.mpt.to_aux_fields(w)
    mpt = sim.mpt
    errors = []
    for k in mpt.aux_fields_imag_idx:
        target = -(mpt.h_coef_mu1[k] + np.dot(mpt.matrix_a[:, k], sim.mean_concentrations())) / (2 * mpt.h_coef_mu2[k])
        errors.append(abs(np.mean(w_aux[k]) - target))
    # H recorded at step 4 against H recomputed from the saved fields
    quiet(sim.prop_solver.compute_propagators, {"A": w[0], "B": w[1]})
    q = [sim.prop_solver.get_partition_function(p) for p in range(sim.prop_solver.get_n_polymer_types())]
    h = mpt.compute_hamiltonian(sim.prop_solver.molecules, w_aux, q, sim.cb, include_const_term=True)
    h_recorded = np.ravel(loadmat(os.path.join(work, "dH_000004.mat"), squeeze_me=True)["H_history"])[-1]
    check(max(errors) < 1e-12 and abs(h - h_recorded) < 1e-10,
          f"compressible L-FTS: mean of the pressure-like field at its saddle value ({max(errors):.1e}), "
          f"recorded H = H of the saved fields ({abs(h - h_recorded):.1e})")


def main():
    root = tempfile.mkdtemp()
    try:
        test_wtmd_restart(root)
        test_wtmd_cell()
        test_saddle_failure(root)
        test_compressible_mean(root)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print(f"\n{n_pass} passed, {n_fail} failed")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
