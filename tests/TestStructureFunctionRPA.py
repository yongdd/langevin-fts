#!/usr/bin/env python3
"""
Structure function of L-FTS and CL-FTS against RPA (disordered AB diblock).

At chi N = 6 (far below the ODT) and N̄ = 1e6 the fluctuations are Gaussian, so
the recorded S_ij(k) must agree with the random-phase approximation:
  incompressible (L-FTS):  N/S_AA = g_tot/(g_A g_B - g_AB^2) - 2 chi N  (Leibler),
  compressible (CL-FTS):   S/N = (G^-1 + X)^-1, G = [[g_A, g_AB], [g_AB, g_B]],
                           X = [[zeta N, zeta N + chi N], [zeta N + chi N, zeta N]],
with Debye functions g(f, x) = 2 (f x + exp(-f x) - 1)/x^2 and x = k^2 R_g^2.
With Gaussian smearing the interactions act on the smeared densities: chi N and
X are multiplied by Gamma(k)^2, and the recorded S(k) is that of the densities.
The checks use the first eight wavevectors of a 1D box: the mean ratio to RPA must
lie within 5 % of 1 and every single ratio within 15 % (sampling noise).

From 2024-09-01 to 2026-10 the estimator used the inverse of the auxiliary-field
matrix instead of the matrix itself and gave S/M (one half for AB: L-FTS 0.50 of
RPA here). CL-FTS also multiplied by conj(phi(k)) instead of phi(-k) (they differ
for complex fields), paired the concentrations of one step with the fields of the
next, and left out the fluctuating compressible field: 0.43 of RPA here. With
smearing the estimator gives Gamma(k) S(k); it was recorded without dividing by
Gamma (with the factor 1/2: 0.23 to 0.49 of RPA in the smeared L-FTS case below).
"""

import os
import io
import sys
import glob
import contextlib
import tempfile
import shutil

os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from scipy.io import loadmat
from polymerfts import _core, LFTS, CLFTS

CHI_N, F, NX, LX, NBAR, STEPS = 6.0, 0.5, 32, 4.0, 1e6, 20000
N_MODES = 8

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


def debye(f, x):
    return 2.0 * (f * x + np.exp(-f * x) - 1.0) / x ** 2


def rpa(zeta_n, a_int=None):
    """S/N for the first N_MODES wavevectors: dict pair -> array."""
    result = {"A,A": [], "A,B": [], "B,B": []}
    for n in range(1, N_MODES + 1):
        k = 2 * np.pi * n / LX
        x = k ** 2 / 6.0
        gamma_sq = 1.0 if a_int is None else np.exp(-a_int ** 2 * k ** 2)
        g_a, g_b, g_tot = debye(F, x), debye(1 - F, x), debye(1.0, x)
        g_ab = 0.5 * (g_tot - g_a - g_b)
        if zeta_n is None:
            s_aa = 1.0 / (g_tot / (g_a * g_b - g_ab ** 2) - 2 * CHI_N * gamma_sq)
            s = np.array([[s_aa, -s_aa], [-s_aa, s_aa]])
        else:
            G = np.array([[g_a, g_ab], [g_ab, g_b]])
            X = np.array([[zeta_n, zeta_n + CHI_N], [zeta_n + CHI_N, zeta_n]])
            s = np.linalg.inv(np.linalg.inv(G) + gamma_sq * X)
        result["A,A"].append(s[0, 0]); result["A,B"].append(s[0, 1]); result["B,B"].append(s[1, 1])
    return {key: np.array(v) for key, v in result.items()}


def params(work_dir, dt):
    return {"nx": [NX], "lx": [LX], "chain_model": "continuous", "ds": 1 / 50,
            "segment_lengths": {"A": 1.0, "B": 1.0}, "chi_n": {"A,B": CHI_N},
            "distinct_polymers": [{"volume_fraction": 1.0, "blocks": [{"type": "A", "length": F},
                                                                      {"type": "B", "length": 1 - F}]}],
            "platform": cpu_platform(), "langevin": {"nbar": NBAR, "dt": dt, "max_step": STEPS},
            "recording": {"dir": work_dir, "recording_period": STEPS, "sf_computing_period": 1,
                          "sf_recording_period": STEPS},
            "verbose_level": 1}


def compare(name, data, suffix, reference):
    for key, ref in reference.items():
        pair = key.replace(",", "_")
        s = np.real(np.ravel(data["structure_function_" + pair + suffix]))[1:N_MODES + 1]
        ratio = s / ref
        check(abs(np.mean(ratio) - 1.0) < 0.05 and np.all(np.abs(ratio - 1.0) < 0.15),
              f"{name} S_{pair}/RPA: mean {np.mean(ratio):.3f}, range [{ratio.min():.3f}, {ratio.max():.3f}]")


def main():
    root = tempfile.mkdtemp()
    try:
        # L-FTS, incompressible
        work = os.path.join(root, "lfts")
        p = params(work, 0.5)
        p["saddle"] = {"max_iter": 200, "tolerance": 1e-6}
        p["compressor"] = {"name": "lram", "max_hist": 20, "start_error": 5e-1, "mix_min": 0.01, "mix_init": 0.01}
        with contextlib.redirect_stdout(io.StringIO()):
            sim = LFTS(p, random_seed=7)
            sim.run(initial_fields={"A": np.zeros(NX), "B": np.zeros(NX)})
        data = loadmat(sorted(glob.glob(os.path.join(work, "structure_function_*.mat")))[-1])
        compare("L-FTS", data, "", rpa(None))

        # CL-FTS, compressible
        work = os.path.join(root, "clfts")
        p = params(work, 1.0)
        p["zeta_n"] = 100.0
        with contextlib.redirect_stdout(io.StringIO()):
            sim = CLFTS(p, random_seed=7)
            sim.run(initial_fields={"A": np.zeros(NX), "B": np.zeros(NX)})
        data = loadmat(sorted(glob.glob(os.path.join(work, "structure_function_*.mat")))[-1])
        compare("CL-FTS (zeta N = 100)", data, "_real", rpa(100.0))

        # Smeared interactions (Gaussian, a_int = 0.1: Gamma = 0.45 at the eighth mode)
        smearing = {"type": "gaussian", "a_int": 0.1}
        work = os.path.join(root, "lfts_smeared")
        p = params(work, 0.5)
        p["saddle"] = {"max_iter": 200, "tolerance": 1e-6}
        p["compressor"] = {"name": "lram", "max_hist": 20, "start_error": 5e-1, "mix_min": 0.01, "mix_init": 0.01}
        p["smearing"] = smearing
        with contextlib.redirect_stdout(io.StringIO()):
            sim = LFTS(p, random_seed=7)
            sim.run(initial_fields={"A": np.zeros(NX), "B": np.zeros(NX)})
        data = loadmat(sorted(glob.glob(os.path.join(work, "structure_function_*.mat")))[-1])
        compare("L-FTS smeared", data, "", rpa(None, smearing["a_int"]))

        work = os.path.join(root, "clfts_smeared")
        p = params(work, 1.0)
        p["zeta_n"] = 100.0
        p["smearing"] = smearing
        with contextlib.redirect_stdout(io.StringIO()):
            sim = CLFTS(p, random_seed=7)
            sim.run(initial_fields={"A": np.zeros(NX), "B": np.zeros(NX)})
        data = loadmat(sorted(glob.glob(os.path.join(work, "structure_function_*.mat")))[-1])
        compare("CL-FTS smeared (zeta N = 100)", data, "_real", rpa(100.0, smearing["a_int"]))
    finally:
        shutil.rmtree(root, ignore_errors=True)

    print(f"\n{n_pass} passed, {n_fail} failed")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
