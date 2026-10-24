#!/usr/bin/env python3
"""
Structure functions of charged CL-FTS against the RPA (Debye-Hueckel level).

A melt of AB diblocks (discrete chains, N = 20, f = 0.5) whose A segments carry the
valence z_A = 0.2, neutralized by monovalent counterions C (one segment), with
chi N = 6, zeta N = 100, per-type Gaussian smearing and the Coulomb coupling
E = 1e4, at Nbar = 1e6 where the fluctuations are Gaussian, in a 2D box. The RPA is

  S/(rho0 N) = [G^-1 + Gamma (X + zeta N e e^T) Gamma + (E/k^2) (Gamma z)(Gamma z)^T]^-1,

with the single-chain correlations G_ij = sum_p phi_p/(n_p N) sum_{s in i, s' in j}
exp(-k^2 |s - s'|/(6N)) of molecules p with n_p segments, the smearing functions
Gamma_i(k) = exp(-a_i^2 k^2/2) and the valences z. S(k) is averaged over the shells
|k| = 2 pi n/L, n = 1..8 (four or more wavevectors each). For S_AA, S_BB and S_AB the
mean ratio to the RPA must lie within 5 % of 1 and every shell within 20 %. Screening
lowers S_CC at the first four shells to 0.16-0.72 of the RPA without charges, which
makes its estimator noisy (a standard error of 0.3 at n = 1 from 4 x 40000 steps in
1D); only its mean over n = 2..8 is checked, within 10 %, and the first four shells
must lie below 0.85 of the RPA without charges.

A melt of polyampholytes E5K5E5K5 (valences -1 and +1, no chi, zeta N = 10, E = 300)
has a composition mode E - K with zero eigenvalue and no auxiliary field; its
structure function comes from the electrostatic potential (Section 6.1 of the L-FTS
theory). The density and the charge structure functions, S_EE + S_KK +- 2 S_EK, must
meet the criteria of S_AA above, and the charge structure function must lie below the
RPA without charges at the first three shells. In both systems, the recorded structure
function of the smeared charge density (charge_structure_function, a combination of the
mixed and field-field estimators of psi) must meet the same criteria.
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
from polymerfts import _core, CLFTS

CHI_N, ZETA_N, N, N_A, NX, LX, NBAR, STEPS = 6.0, 100.0, 20, 10, 32, 4.0, 1e6, 30000
Z_A, COUPLING = 0.2, 1e4
PHI_C = Z_A * N_A / N / (1 + Z_A * N_A / N)   # z_A phi_A = phi_C
RADII = {"A": 0.1, "B": 0.1, "C": 0.15}
N_MODES = 8
PA_SEQUENCE, PA_ZETA_N, PA_COUPLING, PA_RADIUS = "EEEEEKKKKKEEEEEKKKKK", 10.0, 300.0, 0.1

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


def rpa(coupling):
    """S/(rho0 N) of the pairs AA, BB, AB, CC and of the smeared charge density (key "charge")."""
    types, z = ["A", "B", "C"], np.array([Z_A, 0.0, -1.0])
    segments = np.arange(N)
    a, b = segments < N_A, segments >= N_A
    result = {"A,A": [], "B,B": [], "A,B": [], "C,C": [], "charge": []}
    for n in range(1, N_MODES + 1):
        k = 2 * np.pi * n / LX
        bonds = np.exp(-k ** 2 / (6 * N) * np.abs(segments[:, None] - segments[None, :]))
        G = np.zeros((3, 3))
        G[0, 0] = (1 - PHI_C) / N ** 2 * bonds[np.ix_(a, a)].sum()
        G[1, 1] = (1 - PHI_C) / N ** 2 * bonds[np.ix_(b, b)].sum()
        G[0, 1] = G[1, 0] = (1 - PHI_C) / N ** 2 * bonds[np.ix_(a, b)].sum()
        G[2, 2] = PHI_C / N
        gamma = np.array([np.exp(-RADII[t] ** 2 * k ** 2 / 2) for t in types])
        X = np.array([[0.0, CHI_N, 0.0], [CHI_N, 0.0, 0.0], [0.0, 0.0, 0.0]]) + ZETA_N
        U = np.outer(gamma, gamma) * (X + coupling / k ** 2 * np.outer(z, z))
        s = np.linalg.inv(np.linalg.inv(G) + U)
        result["A,A"].append(s[0, 0]); result["B,B"].append(s[1, 1])
        result["A,B"].append(s[0, 1]); result["C,C"].append(s[2, 2])
        result["charge"].append((gamma * z) @ s @ (gamma * z))
    return {key: np.array(v) for key, v in result.items()}


def params(work_dir):
    return {"nx": [NX, NX], "lx": [LX, LX], "chain_model": "discrete", "ds": 1 / N,
            "segment_lengths": {"A": 1.0, "B": 1.0, "C": 1.0}, "chi_n": {"A,B": CHI_N}, "zeta_n": ZETA_N,
            "polymers": [{"volume_fraction": 1 - PHI_C, "blocks": [{"type": "A", "length": N_A / N},
                                                                    {"type": "B", "length": 1 - N_A / N}]},
                         {"volume_fraction": PHI_C, "blocks": [{"type": "C", "length": 1 / N}]}],
            "smearing": {"type": "gaussian", "a_int": RADII},
            "electrostatics": {"charges": {"A": Z_A, "C": -1.0}, "coupling": COUPLING},
            "platform": cpu_platform(), "langevin": {"nbar": NBAR, "dt": 1.0, "max_step": STEPS},
            "recording": {"dir": work_dir, "checkpoint_interval": STEPS, "sample_interval": 1,
                          "statistics_save_interval": STEPS},
            "verbose_level": 1}


def rpa_polyampholyte(coupling):
    """S/(rho0 N) of the bead density, the charge density and the smeared charge density."""
    z = np.array([-1.0 if c == "E" else 1.0 for c in PA_SEQUENCE])
    segments = np.arange(N)
    types, valences = [z < 0, z > 0], np.array([-1.0, 1.0])
    density, charge, smeared = [], [], []
    for n in range(1, N_MODES + 1):
        k = 2 * np.pi * n / LX
        bonds = np.exp(-k ** 2 / (6 * N) * np.abs(segments[:, None] - segments[None, :]))
        G = np.array([[bonds[np.ix_(ti, tj)].sum() for tj in types] for ti in types]) / N ** 2
        U = np.exp(-PA_RADIUS ** 2 * k ** 2) * (PA_ZETA_N * np.ones((2, 2)) + coupling / k ** 2 * np.outer(valences, valences))
        s = np.linalg.inv(np.linalg.inv(G) + U)
        density.append(s.sum())
        charge.append(valences @ s @ valences)
        smeared.append(np.exp(-PA_RADIUS ** 2 * k ** 2) * charge[-1])
    return np.array(density), np.array(charge), np.array(smeared)


def params_polyampholyte(work_dir):
    blocks, start = [], 0
    for i in range(1, N + 1):   # consecutive segments of one type form a block
        if i == N or PA_SEQUENCE[i] != PA_SEQUENCE[start]:
            blocks.append({"type": PA_SEQUENCE[start], "length": (i - start) / N})
            start = i
    return {"nx": [NX, NX], "lx": [LX, LX], "chain_model": "discrete", "ds": 1 / N,
            "segment_lengths": {"E": 1.0, "K": 1.0}, "chi_n": {}, "zeta_n": PA_ZETA_N,
            "polymers": [{"volume_fraction": 1.0, "blocks": blocks}],
            "smearing": {"type": "gaussian", "a_int": PA_RADIUS},
            "electrostatics": {"charges": {"E": -1.0, "K": 1.0}, "coupling": PA_COUPLING},
            "platform": cpu_platform(), "langevin": {"nbar": NBAR, "dt": 1.0, "max_step": STEPS},
            "recording": {"dir": work_dir, "checkpoint_interval": STEPS, "sample_interval": 1,
                          "statistics_save_interval": STEPS},
            "verbose_level": 1}


def shells(s):
    """Mean of S over the wavevectors with |m|^2 = n^2, n = 1..N_MODES (m: integer frequencies)."""
    m = np.fft.fftfreq(NX, 1.0 / NX)
    m_sq = m[:, None] ** 2 + m[None, :] ** 2
    return np.array([np.mean(s[m_sq == n * n]) for n in range(1, N_MODES + 1)])


def run(make_params, seed):
    """Shell-averaged S of all pairs, and of the smeared charge density from psi ("charge"), of one run."""
    root = tempfile.mkdtemp()
    try:
        work = os.path.join(root, "charged")
        p = make_params(work)
        with contextlib.redirect_stdout(io.StringIO()):
            sim = CLFTS(p, random_seed=seed)
            sim.run(initial_fields={t: np.zeros(NX * NX) for t in p["segment_lengths"]})
        data = loadmat(sorted(glob.glob(os.path.join(work, "structure_function_*.mat")))[-1])
        names = sorted(p["segment_lengths"])
        result = {i + "," + j: shells(np.real(np.reshape(data["structure_function_" + i + "_" + j + "_real"], (NX, NX))))
                  for a, i in enumerate(names) for j in names[a:]}
        result["charge"] = shells(np.real(np.reshape(data["charge_structure_function_real"], (NX, NX))))
        return result
    finally:
        shutil.rmtree(root, ignore_errors=True)


def main():
    measured = run(params, 7)
    reference, neutral = rpa(COUPLING), rpa(0.0)
    for key in ("A,A", "B,B", "A,B"):
        ratio = measured[key] / reference[key]
        check(abs(np.mean(ratio) - 1.0) < 0.05 and np.all(np.abs(ratio - 1.0) < 0.2),
              f"charged CL-FTS S_{key.replace(',', '_')}/RPA: mean {np.mean(ratio):.3f}, "
              f"range [{ratio.min():.3f}, {ratio.max():.3f}]")
    ratio = measured["C,C"] / reference["C,C"]
    check(abs(np.mean(ratio[1:]) - 1.0) < 0.1,
          f"charged CL-FTS S_C_C/RPA: mean of n = 2..8 {np.mean(ratio[1:]):.3f}, all {np.array2string(ratio, precision=2)}")
    ratio = measured["C,C"][:4] / neutral["C,C"][:4]
    check(np.all(ratio < 0.85), f"S_CC below the RPA without charges: ratios {np.array2string(ratio, precision=3)}")
    ratio = measured["charge"] / reference["charge"]
    check(abs(np.mean(ratio) - 1.0) < 0.05 and np.all(np.abs(ratio - 1.0) < 0.2),
          f"charge structure function (psi estimators)/RPA: mean {np.mean(ratio):.3f}, range [{ratio.min():.3f}, {ratio.max():.3f}]")

    # Polyampholyte: the charge direction has no auxiliary field
    measured = run(params_polyampholyte, 7)
    reference, neutral = rpa_polyampholyte(PA_COUPLING), rpa_polyampholyte(0.0)
    density = measured["E,E"] + measured["K,K"] + 2 * measured["E,K"]
    charge = measured["E,E"] + measured["K,K"] - 2 * measured["E,K"]
    for name, s, ref in (("density", density, reference[0]), ("charge", charge, reference[1]),
                         ("charge (psi estimators)", measured["charge"], reference[2])):
        ratio = s / ref
        check(abs(np.mean(ratio) - 1.0) < 0.05 and np.all(np.abs(ratio - 1.0) < 0.2),
              f"polyampholyte {name} S/RPA: mean {np.mean(ratio):.3f}, range [{ratio.min():.3f}, {ratio.max():.3f}]")
    ratio = charge[:3] / neutral[1][:3]
    check(np.all(ratio < 0.9), f"polyampholyte charge S below the RPA without charges: ratios {np.array2string(ratio, precision=3)}")

    print(f"\n{n_pass} passed, {n_fail} failed")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
