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
must lie below 0.85 of the RPA without charges (the margin check of the errors below
covers n = 1..3: at n = 4 the RPA ratio, 0.72, lies only two standard errors below the
bound). S_BC (n = 1..5, where it is not small) must lie
within 10 % of the RPA on average and 30 % at every shell: Gamma_B/Gamma_C grows to
1.47 there, so dividing a pair by the smearing function of the wrong type fails.

A melt of polyampholytes E5K5E5K5 (valences -1 and +1, no chi, zeta N = 10, E = 300)
has a composition mode E - K with zero eigenvalue and no auxiliary field; its
structure function comes from the electrostatic potential (Section 6.1 of the L-FTS
theory). The density and the charge structure functions, S_EE + S_KK +- 2 S_EK, must
meet the criteria of S_AA above, the charge structure function with a mean within
1.5 % of the RPA, and the charge structure function must lie below the RPA without
charges at the first three shells.

In both systems the structure function S_c of the smeared charge density is also
recorded from psi alone (Section 6 of docs/theory/StructureFunctionEstimators.md):
- The combination of least variance (charge_structure_function) must meet the criteria
  of S_AA. In the Gaussian regime it follows from the linear response of the chains and
  hardly depends on the sampling of psi: a psi noise scaled by 2 leaves it at the RPA.
- The mixed form tests the sampling (the same noise error gives 4 times the RPA). In
  the polyampholyte its mean must lie within 1.5 % of the RPA and every shell within
  20 %; in the diblock, where it is noisier (standard errors up to 0.07 per shell),
  within 10 % on average and 30 % per shell.
- The field-field form tests it too, at the shells of strong screening,
  (E/k^2) S_c > 0.5 (n = 1..3 and 1..2), with the criteria of S_AA. Elsewhere it is the
  small difference of two large terms and too noisy for them.
Before psi took the averaged noise of the LM integrator, the explicit charge density of
its step put the mixed form (1.023) and S_EE + S_KK - 2 S_EK (1.020) of the polyampholyte
above the RPA and the field-field form up to 30 % below it at large k; both forms now
agree with the RPA within their errors (mixed form 1.005 +- 0.004, S_EE + S_KK - 2 S_EK
1.002 +- 0.005). The 1.5 % criteria of the polyampholyte fail with the former step;
TestClftsElectrostatics checks the step directly.

Errors: the runs record their statistics in BLOCKS blocks of STEPS/BLOCKS steps, far
longer than the correlation times (at most a few hundred steps, the screened psi modes).
The first block, which starts from zero fields, is discarded; the means are those of the
other blocks, and the standard errors follow from the spread of the block means. Every tolerance must exceed 2 standard errors, so that a pass is not
one by chance; the printed errors give the actual margins. For the bounds against the
RPA without charges, the margin between the bound and the RPA ratio must exceed 2
standard errors.
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
BLOCKS, ERRORS = 30, 2.0   # blocks of the statistics; tolerances must exceed ERRORS standard errors
Z_A, COUPLING = 0.2, 1e4
PHI_C = Z_A * N_A / N / (1 + Z_A * N_A / N)   # z_A phi_A = phi_C
RADII = {"A": 0.1, "B": 0.1, "C": 0.15}
N_MODES = 8
PA_SEQUENCE, PA_ZETA_N, PA_COUPLING, PA_RADIUS = "EEEEEKKKKKEEEEEKKKKK", 10.0, 300.0, 0.1
CHARGE_FORMS = {"charge": "combination", "charge_mixed": "mixed form", "charge_field_field": "field-field form"}

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


def estimate(blocks):
    """Mean over the blocks (axis 0) and its standard error."""
    return np.mean(blocks, axis=0), np.std(blocks, axis=0, ddof=1) / np.sqrt(len(blocks))


def check_ratio(name, blocks, reference, mean_tol=0.05, shell_tol=0.2):
    """Mean ratio to the RPA within mean_tol of 1 and, unless shell_tol is None, every shell within
    shell_tol; then that the tolerances exceed ERRORS standard errors."""
    ratio = blocks / reference
    shell, shell_error = estimate(ratio)
    mean, mean_error = estimate(np.mean(ratio, axis=1))
    agree = abs(mean - 1.0) < mean_tol
    resolved = ERRORS * mean_error < mean_tol
    text = f"{name}/RPA: mean {mean:.3f} +- {mean_error:.3f}"
    if shell_tol is not None:
        agree = agree and np.all(np.abs(shell - 1.0) < shell_tol)
        resolved = resolved and np.all(ERRORS * shell_error < shell_tol)
        text += f", range [{shell.min():.3f}, {shell.max():.3f}], largest shell error {shell_error.max():.3f}"
    check(agree, text)
    check(resolved, f"{name}: the tolerances exceed {ERRORS:g} standard errors")


def check_below(name, blocks, neutral, expected, bound, margin_shells=None):
    """Every shell of S/S_neutral below bound, where the RPA gives `expected`; then that the margin
    of the bound exceeds ERRORS standard errors (at the first margin_shells shells, or all)."""
    ratio, error = estimate(blocks / neutral)
    check(np.all(ratio < bound), f"{name} below {bound} of the RPA without charges: ratios "
          f"{np.array2string(ratio, precision=3)} +- {np.array2string(error, precision=3)}")
    n = len(ratio) if margin_shells is None else margin_shells
    check(np.all(ERRORS * error[:n] < bound - expected[:n]), f"{name}: the margins to {bound} exceed {ERRORS:g} "
          f"standard errors at n = 1..{n} (RPA ratios {np.array2string(expected, precision=3)})")


def strong_screening(s_charge, coupling):
    """Shells with (E/k^2) S_c > 0.5, where the field-field form is the better one."""
    k_sq = (2 * np.pi * np.arange(1, N_MODES + 1) / LX) ** 2
    return s_charge * coupling / k_sq > 0.5


def cpu_platform():
    cpu = [p for p in _core.PlatformSelector.avail_platforms() if p.startswith("cpu")]
    return cpu[0] if cpu else "cuda"


def rpa(coupling):
    """S/(rho0 N) of the pairs AA, BB, AB, CC, BC and of the smeared charge density (key "charge")."""
    types, z = ["A", "B", "C"], np.array([Z_A, 0.0, -1.0])
    segments = np.arange(N)
    a, b = segments < N_A, segments >= N_A
    result = {"A,A": [], "B,B": [], "A,B": [], "C,C": [], "B,C": [], "charge": []}
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
        result["B,C"].append(s[1, 2])
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
                          "statistics_save_interval": STEPS // BLOCKS},
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
                          "statistics_save_interval": STEPS // BLOCKS},
            "verbose_level": 1}


def shells(s):
    """Mean of S over the wavevectors with |m|^2 = n^2, n = 1..N_MODES (m: integer frequencies)."""
    m = np.fft.fftfreq(NX, 1.0 / NX)
    m_sq = m[:, None] ** 2 + m[None, :] ** 2
    return np.array([np.mean(s[m_sq == n * n]) for n in range(1, N_MODES + 1)])


def run(make_params, seed):
    """Shell-averaged S of all pairs and of the smeared charge density from psi (keys of CHARGE_FORMS)
    in each block of STEPS/BLOCKS steps after the first (burn-in), shape (BLOCKS - 1, N_MODES)."""
    root = tempfile.mkdtemp()
    try:
        work = os.path.join(root, "charged")
        p = make_params(work)
        with contextlib.redirect_stdout(io.StringIO()):
            sim = CLFTS(p, random_seed=seed)
            sim.run(initial_fields={t: np.zeros(NX * NX) for t in p["segment_lengths"]})
        files = sorted(glob.glob(os.path.join(work, "structure_function_*.mat")))
        assert len(files) == BLOCKS, files
        names = sorted(p["segment_lengths"])
        result = {}
        for file_name in files[1:]:
            data = loadmat(file_name)
            assert int(np.squeeze(data["sample_count"])) == STEPS // BLOCKS
            keys = {i + "," + j: "structure_function_" + i + "_" + j for a, i in enumerate(names) for j in names[a:]}
            keys.update({key: key.replace("charge", "charge_structure_function") for key in CHARGE_FORMS})
            for key, name in keys.items():
                result.setdefault(key, []).append(shells(np.real(np.reshape(data[name + "_real"], (NX, NX)))))
        return {key: np.array(blocks) for key, blocks in result.items()}
    finally:
        shutil.rmtree(root, ignore_errors=True)


def main():
    measured = run(params, 7)
    reference, neutral = rpa(COUPLING), rpa(0.0)
    for key in ("A,A", "B,B", "A,B"):
        check_ratio(f"charged CL-FTS S_{key.replace(',', '_')}", measured[key], reference[key])
    check_ratio("charged CL-FTS S_B_C (n = 1..5)", measured["B,C"][:, :5], reference["B,C"][:5],
                mean_tol=0.1, shell_tol=0.3)
    check_ratio("charged CL-FTS S_C_C (mean of n = 2..8)", measured["C,C"][:, 1:], reference["C,C"][1:],
                mean_tol=0.1, shell_tol=None)
    check_below("S_CC", measured["C,C"][:, :4], neutral["C,C"][:4], reference["C,C"][:4] / neutral["C,C"][:4], 0.85,
                margin_shells=3)
    strong = strong_screening(reference["charge"], COUPLING)
    check_ratio("charge structure function from psi, combination", measured["charge"], reference["charge"])
    check_ratio("charge structure function from psi, mixed form", measured["charge_mixed"], reference["charge"],
                mean_tol=0.1, shell_tol=0.3)
    check_ratio(f"charge structure function from psi, field-field form (n = 1..{np.sum(strong)})",
                measured["charge_field_field"][:, strong], reference["charge"][strong])

    # Polyampholyte: the charge direction has no auxiliary field
    measured = run(params_polyampholyte, 7)
    reference, neutral = rpa_polyampholyte(PA_COUPLING), rpa_polyampholyte(0.0)
    density = measured["E,E"] + measured["K,K"] + 2 * measured["E,K"]
    charge = measured["E,E"] + measured["K,K"] - 2 * measured["E,K"]
    check_ratio("polyampholyte density S", density, reference[0])
    # 1.5 %: the former psi step put both 2 % above the RPA
    check_ratio("polyampholyte charge S", charge, reference[1], mean_tol=0.015)
    strong = strong_screening(reference[2], PA_COUPLING)
    check_ratio("polyampholyte charge S from psi, combination", measured["charge"], reference[2])
    check_ratio("polyampholyte charge S from psi, mixed form", measured["charge_mixed"], reference[2], mean_tol=0.015)
    check_ratio(f"polyampholyte charge S from psi, field-field form (n = 1..{np.sum(strong)})",
                measured["charge_field_field"][:, strong], reference[2][strong])
    check_below("polyampholyte charge S", charge[:, :3], neutral[1][:3], reference[1][:3] / neutral[1][:3], 0.9)

    print(f"\n{n_pass} passed, {n_fail} failed")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
