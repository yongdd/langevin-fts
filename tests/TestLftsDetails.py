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
5. WTMD hills: every sample deposits a hill of the same weight. Samples beyond
   the Ψ range deposit nothing, I1/I0 is dH/dχN also when half of the samples
   lie outside, the hills do not wrap around the Ψ axis, and a sample just
   outside the range deposits its tail inside. Until 2026-10 the histogram was
   divided by the samples inside the range (NaN when there were none, I1/I0 =
   1.25 instead of 2.5 here) and convolved circularly (a hill at 1.99 reached Ψ = 0).
"""

import os
import io
import sys
import glob
import shutil
import stat
import tempfile
import contextlib

os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from scipy.io import loadmat
from polymerfts import _core, LFTS, ValidationError
from polymerfts.utils import atomic_savemat
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
    # The checkpoint also holds the bias arrays without the factor CV: they restore bit for bit
    check(all(np.array_equal(x, y) for x, y in ((a.u, b.u), (a.up, b.up), (a.I0, b.I0)))
          and a.I1.keys() == b.I1.keys() and all(np.array_equal(a.I1[key], b.I1[key]) for key in a.I1),
          "WTMD: continued run restores the bias arrays bitwise")

    # An older checkpoint without the unscaled copies restores the scaled arrays divided by CV
    data = {k: v for k, v in loadmat(os.path.join(root, "first", "fields_000004.mat")).items()
            if not k.startswith("__")}
    unscaled = [key for key in data if key.startswith("wtmd_unscaled_")]
    for key in unscaled:
        del data[key]
    atomic_savemat(os.path.join(root, "legacy.mat"), data, long_field_names=True, do_compression=True)
    legacy = quiet(LFTS, params(os.path.join(root, "legacy"), nx, lx, 8, wtmd=wtmd), random_seed=1)
    quiet(legacy.continue_run, os.path.join(root, "legacy.mat"))
    c = legacy.wtmd
    legacy_diffs = [np.max(np.abs(a.u - c.u)), np.max(np.abs(a.up - c.up)), np.max(np.abs(a.I0 - c.I0))]
    legacy_diffs += [np.max(np.abs(a.I1[key] - c.I1.get(key, 0.0))) for key in a.I1]
    w_legacy = loadmat(os.path.join(root, "legacy", "fields_000008.mat"))["w_A"]
    check(len(unscaled) == 2 + len(a.I1) and max(legacy_diffs) < 1e-12
          and np.max(np.abs(w_straight - w_legacy)) < 1e-10,
          f"WTMD: continued run from a checkpoint without the unscaled arrays (statistics "
          f"{max(legacy_diffs):.1e}, fields {np.max(np.abs(w_straight - w_legacy)):.1e})")


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

    # A zero exchange field (Ψ = 0, as in a start from w = 0) gets no bias, not 0 * inf = NaN
    wt.up[:] = 1.0
    psi = wt.compute_order_parameter(0, np.zeros([2, int(np.prod(nx))]))
    langevin = np.zeros([1, int(np.prod(nx))])
    quiet(wt.add_bias_to_langevin, psi, langevin)
    check(psi == 0.0 and np.array_equal(langevin, np.zeros_like(langevin)), "WTMD: no bias at a zero exchange field")

    # The constructor scales copies of the given statistics, not the caller's arrays
    I1 = {"A,B": np.ones(wt.bins)}
    wt = quiet(WTMD, nx, lx, 10000, np.array([-12.0, 200.0]), [0], I1=I1)
    check(np.array_equal(I1["A,B"], np.ones(wt.bins)) and np.allclose(wt.I1["A,B"], 1.0 / wt.CV, rtol=1e-15, atol=0),
          "WTMD: the given I1 arrays are left unchanged")


def test_wtmd_hills():
    def update(psis):
        wt = quiet(WTMD, [4, 4, 4], [1.0, 1.0, 1.0], 10000, np.array([-12.0, 200.0]), [0],
                   sigma_psi=0.05, psi_min=0.0, psi_max=2.0, dpsi=1e-3)
        for psi in psis:
            wt.store_order_parameter(psi, {"A,B": 2.5})
        quiet(wt.update_statistics)
        return wt

    wt = update(np.full(20, 5.0))
    check(np.array_equal(wt.u, np.zeros(wt.bins)) and np.array_equal(wt.I0, np.zeros(wt.bins)),
          "WTMD: samples beyond the Psi range deposit nothing")
    wt = update(np.concatenate([np.full(20, 1.0), np.full(20, 5.0)]))
    inside = wt.I0 > 1e-3 * wt.I0.max()
    ratio = wt.I1["A,B"][inside] / wt.I0[inside]
    check(np.allclose(ratio, 2.5, rtol=1e-12, atol=0) and abs(wt.I0.max() - 0.5) < 1e-12,
          f"WTMD: half of the samples outside the range: I1/I0 = {ratio.min():.12f}, peak of I0 {wt.I0.max():.12f} (0.5)")
    wt = update(np.full(20, 1.99))
    check(np.max(np.abs(wt.u[:200])) < 1e-12 * np.max(wt.u),
          f"WTMD: a hill at 1.99 does not wrap around to Psi = 0 ({np.max(np.abs(wt.u[:200])) / np.max(wt.u):.1e})")
    wt = update(np.full(20, 2.02))
    tail = np.exp(-0.5 * ((wt.psi_range[-1] - 2.02) / 0.05) ** 2)
    check(abs(wt.I0[-1] - tail) < 1e-9 * tail,
          f"WTMD: a sample at 2.02 deposits its tail inside the range [0, 2) (I0 {wt.I0[-1]:.9f}, Gaussian {tail:.9f})")


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


def test_lr_walls(root):
    """LR and LRAM measure and apply the Jacobian in the cosine series along reflecting walls:
    from random fields (std 2) they reach the saddle point in about as few iterations as in a
    periodic box (5 for LR there), with periodic axes too (cn-adi2). Absorbing walls are refused."""
    R, P = "reflecting", "periodic"
    nx, lx = [16, 12, 10], [3.0, 2.4, 2.0]
    n = int(np.prod(nx))
    w = np.zeros([2, n])
    w[0] = np.random.default_rng(5).normal(0.0, 2.0, n)
    for bc, extra in (([R] * 6, {}), ([R] * 6, {"smearing": {"type": "gaussian", "a_int": 0.1}}),
                      ([P, P, R, R, P, P], {"numerical_method": "cn-adi2"})):
        for name in ("lr", "lram"):
            p = params(os.path.join(root, "lr_walls"), nx, lx, 2, bc=bc, **extra)
            p["saddle"] = {"max_iter": 100, "tolerance": 1e-6}
            p["compressor"] = {"name": name, "max_hist": 20, "start_error": 5e-1, "mix_min": 0.01, "mix_init": 0.01}
            sim = quiet(LFTS, p, random_seed=1)
            _, _, iterations, error = quiet(sim.find_saddle_point, w.copy())
            check(iterations <= 10 and error < 1e-6,
                  f"{name} with walls {bc[::2]} {list(extra)}: saddle point in {iterations} iterations ({error:.1e})")
    p = params(os.path.join(root, "lr_walls"), nx, lx, 2, bc=[R, R, "absorbing", "absorbing", R, R])
    p["compressor"] = {"name": "lr"}
    try:
        quiet(LFTS, p, random_seed=1)
        refused = False
    except ValidationError as exc:
        refused = "absorbing" in str(exc)
    check(refused, "lr refused in a box with absorbing walls")


def test_file_permissions(root):
    """Checkpoints and run_status.json get the permissions of any new file (0666 less the
    umask), not the 0600 of a temporary file; a umask of 022 tells the two apart."""
    if os.name != "posix":
        return
    work = os.path.join(root, "permissions")
    rng = np.random.default_rng(3)
    previous = os.umask(0o022)
    try:
        sim = quiet(LFTS, params(work, [16], [3.0], 4), random_seed=1)
        quiet(sim.run, initial_fields={"A": rng.normal(0.0, 0.5, 16), "B": rng.normal(0.0, 0.5, 16)})
    finally:
        os.umask(previous)
    modes = {name: oct(stat.S_IMODE(os.stat(os.path.join(work, name)).st_mode))
             for name in ("fields_000004.mat", "run_status.json")}
    check(all(mode == oct(0o644) for mode in modes.values()), f"result files follow the umask {modes}")


def main():
    root = tempfile.mkdtemp()
    try:
        test_wtmd_restart(root)
        test_wtmd_cell()
        test_wtmd_hills()
        test_saddle_failure(root)
        test_compressible_mean(root)
        test_lr_walls(root)
        test_file_permissions(root)
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print(f"\n{n_pass} passed, {n_fail} failed")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
