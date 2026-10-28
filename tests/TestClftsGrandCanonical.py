#!/usr/bin/env python3
"""Grand-canonical species of CL-FTS: polymers and point ions with a chemical potential.

A canonical AB diblock (volume fraction 0.6, charged A blocks, discrete chains, N = 20), a
grand-canonical B homopolymer of length 0.5 and grand-canonical Na+ and Cl- point ions:
1. Input: a polymer with both or neither of volume_fraction and chemical_potential, canonical volume
   fractions above 1, grand-canonical polymers in SCFT and L-FTS, grafted grand-canonical polymers,
   an ion with both concentration and chemical_potential and a grand-canonical counterion are
   rejected. Charged grand-canonical species need no electroneutral input and free the uniform mode
   of psi; a neutral grand-canonical polymer, or a grand-canonical polyampholyte without net
   charge, keeps it at zero.
2. Ensembles agree on the same (real) fields: with e^(beta mu) = phi_h/(alpha_h Q_h) the
   grand-canonical homopolymer has the partition function, concentrations and number of chains of
   the canonical one with volume fraction phi_h, and the chemical potentials of the two ensembles
   agree.
3. Forces of psi (uniform direction included) and of the auxiliary fields against central
   differences of H with -e^(beta mu_p) Q_p and -e^(beta mu_s) Q_s, on random fields (standard
   deviation 5 for the auxiliary fields), with a charged homopolymer.
4. Mean-field limit (nbar 1e28, LM and etd steps from zero fields): psi_0 reaches the Donnan
   potential, the homopolymer the Flory-Huggins density; the recorded chemical potentials, numbers
   of chains and ions and the osmotic pressure equal their Flory-Huggins values.
5. The osmotic pressure equals -dH/dV at fixed chemical potentials and fixed numbers of canonical
   chains, at fixed grid values, with and without screening.
6. Restarts are bitwise.
"""

import contextlib
import io
import os
import tempfile
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from scipy.io import loadmat
from scipy.optimize import brentq
from polymerfts import CLFTS, LFTS, PlatformSelector, ValidationError
from polymerfts.validation import validate_scft_params

NX, LX, NBAR, N = [12, 10], [2.0, 1.8], 1e4, 20
CHI, ZETA, Z_A = 2.0, 50.0, 0.2
ALPHA_H, MU_H = 0.5, 0.1
MU_ION = float(np.log(N * 0.05))   # c = 0.05 per segment at psi = 0


def cpu_platform():
    return next(p for p in PlatformSelector.avail_platforms() if p.startswith("cpu"))


def parameters(directory, steps=2, charges=None, ions=None, screening_length=None, **langevin):
    """Canonical AB diblock (A charged), grand-canonical B homopolymer, Na+ and Cl- at fixed chemical potentials."""
    params = {
        "nx": NX, "lx": LX, "ds": 1 / N, "chain_model": "discrete",
        "segment_lengths": {"A": 1.0, "B": 1.0}, "chi_n": {"A,B": CHI}, "zeta_n": ZETA,
        "polymers": [{"volume_fraction": 0.6, "blocks": [{"type": "A", "length": 0.5}, {"type": "B", "length": 0.5}]},
                     {"chemical_potential": MU_H, "blocks": [{"type": "B", "length": ALPHA_H}]}],
        "smearing": {"type": "gaussian", "a_int": 0.15},
        "electrostatics": {"charges": charges if charges is not None else {"A": Z_A}, "coupling": 2e3,
                           "psi_dt_scaling": 20.0,
                           "ions": ions if ions is not None else
                           {"Na": {"valence": 1.0, "chemical_potential": MU_ION, "smearing_length": 0.1},
                            "Cl": {"valence": -1.0, "chemical_potential": MU_ION, "smearing_length": 0.12}}},
        "platform": cpu_platform(),
        "langevin": {"nbar": NBAR, "dt": 0.5, "max_step": steps, **langevin},
        "recording": {"dir": str(directory), "checkpoint_interval": 1, "sample_interval": 1,
                      "statistics_save_interval": 2, "pressure": True},
        "verbose_level": 0,
    }
    if screening_length is not None:
        params["electrostatics"]["screening_length"] = screening_length
    return params


def quiet(make):
    with contextlib.redirect_stdout(io.StringIO()):
        return make()


def expect_validation_error(make, message):
    try:
        quiet(make)
    except ValidationError as error:
        assert message in str(error), f"expected '{message}', got '{error}'"
        return
    raise AssertionError(f"expected a ValidationError with '{message}'")


def smooth(x, std):
    k = np.meshgrid(*[np.fft.fftfreq(m) * m for m in NX], indexing="ij")
    y = np.real(np.fft.ifftn(np.fft.fftn(np.reshape(x, NX)) * np.exp(-0.08 * sum(kk ** 2 for kk in k)))).ravel()
    return std * (y - y.mean()) / y.std()


def inputs(root):
    def changed(change):
        params = parameters(root / "inputs")
        change(params)
        return lambda: CLFTS(params)

    def both(params):
        params["polymers"][1]["volume_fraction"] = 0.4
    expect_validation_error(changed(both), "'chemical_potential' in place of 'volume_fraction'")

    def overfull(params):
        params["polymers"].append({"volume_fraction": 0.5, "blocks": [{"type": "B", "length": 1.0}]})
    expect_validation_error(changed(overfull), "must not exceed 1.0")

    def grafted(params):
        params["polymers"][1]["initial_conditions"] = {"(0)": "G"}
    expect_validation_error(changed(grafted), "cannot be grafted")

    def two_amounts(params):
        params["electrostatics"]["ions"]["Na"]["concentration"] = 0.05
    expect_validation_error(changed(two_amounts), "one of 'concentration' and 'chemical_potential'")

    def open_counterion(params):
        params["electrostatics"]["counterion"] = "Cl"
    expect_validation_error(changed(open_counterion), "is not one of the canonical")

    params = parameters(root / "inputs")
    params.pop("electrostatics")
    expect_validation_error(lambda: validate_scft_params({**params, "optimizer": {"name": "am"}, "relax_cell": False}),
                            "CL-FTS only, not in SCFT")
    params.update(saddle={"max_iter": 10, "tolerance": 1e-4},
                  compressor={"name": "am", "max_hist": 20, "start_error": 1e-1, "mix_min": 0.1, "mix_init": 0.1})
    params["recording"].pop("pressure")
    expect_validation_error(lambda: LFTS(params), "CL-FTS only, not in L-FTS")

    # Grand-canonical ions, or a charged grand-canonical polymer, free the uniform mode of psi; the
    # canonical charge then need not be neutral
    sim = quiet(lambda: CLFTS(parameters(root / "inputs")))
    assert not sim.electrostatics.zero_uniform_mode, "grand-canonical ions"
    sim = quiet(lambda: CLFTS(parameters(root / "inputs", charges={"A": Z_A, "B": -0.1}, ions={})))
    assert not sim.electrostatics.zero_uniform_mode, "charged grand-canonical polymer"
    expect_validation_error(lambda: CLFTS(parameters(root / "inputs", ions={})), "not electroneutral")
    canonical_ions = {"Na": {"valence": 1.0, "concentration": 0.05, "smearing_length": 0.1},
                      "Cl": {"valence": -1.0, "concentration": 0.05, "smearing_length": 0.12}}
    sim = quiet(lambda: CLFTS(parameters(root / "inputs", ions=canonical_ions)))
    assert sim.electrostatics.zero_uniform_mode, "neutral grand-canonical polymer"
    assert abs(sim.electrostatics.ions[1]["concentration"] - (0.05 + 0.6 * 0.5 * Z_A)) < 1e-14, "counterions"
    # A grand-canonical polyampholyte without net charge leaves the charge of the system unchanged
    params = parameters(root / "inputs", charges={"A": Z_A, "B": -Z_A}, ions={})
    params["polymers"][1]["blocks"] = [{"type": "A", "length": 0.25}, {"type": "B", "length": 0.25}]
    sim = quiet(lambda: CLFTS(params))
    assert sim.electrostatics.zero_uniform_mode, "grand-canonical polyampholyte without net charge"


def ensembles(root):
    """The grand-canonical homopolymer equals the canonical one with e^(beta mu) = phi_h/(alpha_h Q_h)."""
    rng = np.random.default_rng(11)
    n_grid = int(np.prod(NX))
    w_aux = np.array([smooth(rng.normal(0, 1, n_grid), 5.0) for _ in range(2)], dtype=np.complex128)
    params = parameters(root / "ensembles")
    params.pop("electrostatics")
    canonical = {**params, "polymers": [params["polymers"][0], {"volume_fraction": 0.4, "blocks": params["polymers"][1]["blocks"]}]}
    sim_canonical = quiet(lambda: CLFTS(canonical))
    sim_canonical.mpt.pin_dropped_fields(w_aux)
    phi, _ = sim_canonical.compute_concentrations(w_aux)
    q = [sim_canonical.prop_solver.get_partition_function(p) for p in range(2)]
    assert abs(q[1].imag) < 1e-14 * abs(q[1]), "real fields give a real Q"
    params["polymers"][1]["chemical_potential"] = float(np.log(0.4 / (ALPHA_H * q[1].real)))
    sim = quiet(lambda: CLFTS(params))
    phi_grand, _ = sim.compute_concentrations(w_aux)
    q_grand = [sim.prop_solver.get_partition_function(p) for p in range(2)]
    assert np.allclose(q_grand, q, rtol=1e-12, atol=0), "partition functions"
    for m in "AB":
        assert np.max(np.abs(phi_grand[m] - phi[m])) < 1e-12, f"concentration {m}"
    assert np.allclose(sim.number_densities(q_grand), [0.6, 0.4 / ALPHA_H], rtol=1e-12, atol=0), "numbers of chains"
    assert np.allclose(sim.chemical_potentials(q_grand), sim_canonical.chemical_potentials(q), rtol=0, atol=1e-12), \
        "chemical potentials"


def hamiltonian(sim, w_aux):
    phi, _ = sim.compute_concentrations(w_aux)
    phi_smeared = sim.smearing.apply_to_dict(phi)
    partitions = [sim.prop_solver.get_partition_function(p) for p in range(sim.prop_solver.get_n_polymer_types())]
    h = sim.mpt.compute_hamiltonian(sim.prop_solver.molecules, w_aux, partitions, sim.cb,
                                    include_const_term=True, phi=phi_smeared)
    return h + sim.electrostatics.hamiltonian() + sim.grand_canonical_hamiltonian(partitions), phi_smeared


def forces(root):
    sim = quiet(lambda: CLFTS(parameters(root / "forces", charges={"A": Z_A, "B": -0.05})))
    es, n_grid, M = sim.electrostatics, int(np.prod(NX)), 2
    rng = np.random.default_rng(23)
    w_aux = rng.normal(0.0, 5.0, (M, n_grid)) + 1j * rng.normal(0.0, 1.0, (M, n_grid))
    sim.mpt.pin_dropped_fields(w_aux)
    es.psi = 2.0 + rng.normal(0.0, 1.0, n_grid) + 1j * rng.normal(0.0, 1.0, n_grid)
    _, phi_smeared = hamiltonian(sim, w_aux)
    psi_force = es.force(es.charge_density(phi_smeared))
    aux_force = sim.mpt.compute_func_deriv(w_aux, phi_smeared, range(M))
    eps, psi = 1e-5, es.psi.copy()
    for direction in (rng.normal(0.0, 1.0, n_grid), np.ones(n_grid)):
        es.psi = psi + eps * direction
        h_plus, _ = hamiltonian(sim, w_aux)
        es.psi = psi - eps * direction
        h_minus, _ = hamiltonian(sim, w_aux)
        es.psi = psi
        expected = np.mean(psi_force * direction)
        assert abs((h_plus - h_minus) / (2 * eps) - expected) < 1e-6 * max(1.0, abs(expected)), "psi force"
    direction = rng.normal(0.0, 1.0, n_grid)
    for i in range(M):
        if i in sim.mpt.aux_fields_dropped_idx:
            continue
        sign = 1.0 if i in sim.mpt.aux_fields_real_idx else -1.0
        shifted = w_aux.copy()
        shifted[i] += eps * direction
        h_plus, _ = hamiltonian(sim, shifted)
        shifted[i] -= 2 * eps * direction
        h_minus, _ = hamiltonian(sim, shifted)
        expected = sign * np.mean(aux_force[i] * direction)
        assert abs((h_plus - h_minus) / (2 * eps) - expected) < 1e-6 * max(1.0, abs(expected)), f"aux force {i}"


def mean_field(root):
    """Flory-Huggins state: phi_A = 0.3, phi_B = 0.3 + phi_h, Donnan psi_0 and ions c_+- = 0.05 x^(+-1)."""
    def residual(phi_h):
        w_b = CHI * 0.3 + ZETA * (0.6 + phi_h - 1.0)
        return phi_h - ALPHA_H * np.exp(MU_H - ALPHA_H * w_b)
    phi_h = brentq(residual, 1e-6, 1.0, xtol=1e-15)
    phi_a, phi_b = 0.3, 0.3 + phi_h
    excess = ZETA * (phi_a + phi_b - 1.0)
    w_a, w_b = CHI * phi_b + excess, CHI * phi_a + excess
    # Neutrality Z_A phi_A + 0.05 (x - 1/x) = 0 with x = exp(-psi_0/N)
    x = (-Z_A * phi_a / 0.05 + np.sqrt((Z_A * phi_a / 0.05) ** 2 + 4.0)) / 2.0
    psi_0, c_plus, c_minus = -N * np.log(x), 0.05 * x, 0.05 / x
    mu = [np.log(0.6) + 0.5 * (w_a + Z_A * psi_0) + 0.5 * w_b, MU_H, MU_ION, MU_ION]
    numbers = [0.6, phi_h / ALPHA_H, N * c_plus, N * c_minus]
    pressure = (0.6 + phi_h / ALPHA_H + CHI * phi_a * phi_b + 0.5 * ZETA * ((phi_a + phi_b) ** 2 - 1.0)
                + N * (c_plus + c_minus))

    for integrator, dt, steps in (("lm", 0.5, 400), ("etd", 5.0, 100)):
        directory = root / f"mean_field_{integrator}"
        params = parameters(directory, steps=steps, nbar=1e28, dt=dt, integrator=integrator)
        params["recording"].update(checkpoint_interval=steps, statistics_save_interval=steps)
        averages = {"psi_0": [], "phi_h": []}
        holder = {}

        def record(sample, averages=averages, holder=holder, steps=steps):
            if sample["step"] > steps - 50:
                averages["psi_0"].append(np.mean(holder["sim"].electrostatics.psi))
                averages["phi_h"].append(np.mean(sample["phi"]["B"]) - phi_a)
        sim = quiet(lambda: CLFTS(params, random_seed=3, callbacks={"mean": record}))
        holder["sim"] = sim
        with contextlib.redirect_stdout(io.StringIO()):
            sim.run(initial_fields={m: np.zeros(int(np.prod(NX))) for m in "AB"})
        data = loadmat(directory / f"dH_{steps:06d}.mat", squeeze_me=True)
        last = slice(-50, None)
        recorded = {name: np.mean((data[name + "_history_real"] + 1j * data[name + "_history_imag"])[last], axis=0)
                    for name in ("chemical_potential", "number_density", "pressure")}
        checks = ((np.mean(averages["psi_0"]), psi_0, "psi_0 (Donnan potential)"),
                  (np.mean(averages["phi_h"]), phi_h, "homopolymer density"),
                  (recorded["chemical_potential"], mu, "chemical potentials"),
                  (recorded["number_density"], numbers, "numbers of chains and ions"),
                  (recorded["pressure"], pressure, "osmotic pressure"))
        for value, expected, name in checks:
            error = np.max(np.abs(np.asarray(value) - np.asarray(expected)) / np.maximum(1.0, np.abs(expected)))
            assert error < 1e-6, f"{integrator}: {name} {value} against {expected} (error {error:.1e})"


def pressure(root):
    """The operator against -dH/dV - d ln N/dV at fixed grid values, fixed chemical potentials and
    fixed numbers of canonical chains."""
    C = np.sqrt(NBAR)
    for screening_length in (None, 0.3):
        def make(scale=1.0):
            p = parameters(root / "pressure", charges={"A": Z_A, "B": -0.05}, screening_length=screening_length)
            p["lx"] = [length * scale ** 0.5 for length in LX]
            return quiet(lambda: CLFTS(p))
        rng = np.random.default_rng(7)
        sim = make()
        n_grid, M = int(np.prod(NX)), len(sim.monomer_types)
        w_aux = np.array([smooth(rng.normal(0, 1, n_grid), 5.0) + 1j * smooth(rng.normal(0, 1, n_grid), 1.0)
                          for _ in range(M)])
        psi = 1.0 + smooth(rng.normal(0, 1, n_grid), 0.5) + 1j * smooth(rng.normal(0, 1, n_grid), 0.5)
        volume = sim.prop_solver.get_volume()

        def state(s):
            pinned = w_aux.copy()
            s.mpt.pin_dropped_fields(pinned)
            s.electrostatics.psi = psi
            phi, _ = s.compute_concentrations(pinned)
            q = [s.prop_solver.get_partition_function(p) for p in range(2)]
            h_fields = sum(s.mpt.h_coef_mu2[i] * np.mean(pinned[i] ** 2) + s.mpt.h_coef_mu1[i] * np.mean(pinned[i])
                           for i in range(M)) + s.mpt.h_const
            # Terms that scale with V: the fields, psi with the grand-canonical ions, the homopolymer
            h_volume = h_fields + s.electrostatics.hamiltonian() + s.grand_canonical_hamiltonian(q)
            return np.log(q[0]), h_volume, phi, pinned

        _, _, phi, pinned = state(sim)
        operator = sim.pressure(pinned, phi) - 0.6

        def dilated(scale):
            s = make(scale)
            log_q, h_volume, _, _ = state(s)
            v = s.prop_solver.get_volume()
            h = C * (-volume * 0.6 * log_q + v * h_volume)
            # ln N: n_grid/2 ln V per kept auxiliary field and 1/2 ln(V (k^2 + kappa^2)) per psi mode
            es = s.electrostatics
            modes = es.screened_k_sq > 0
            log_norm = (-0.5 * (s.mpt.R + s.mpt.I) * n_grid * np.log(v)
                        - 0.5 * np.sum(np.log(v * es.screened_k_sq[modes])))
            return h, log_norm, v
        (h_plus, n_plus, v_plus), (h_minus, n_minus, v_minus) = dilated(1 + 1e-5), dilated(1 - 1e-5)
        expected = -((h_plus - h_minus) + (n_plus - n_minus)) / (v_plus - v_minus) / C
        assert abs(operator - expected) < 1e-8 * abs(expected), \
            f"pressure (screening {screening_length}): operator {operator:.10f}, expected {expected:.10f}"


def restart(root):
    def run(directory, steps, continue_from=None):
        sim = quiet(lambda: CLFTS(parameters(directory, steps=steps, charges={"A": Z_A, "B": -0.05}), random_seed=7))
        rng = np.random.default_rng(3)
        with contextlib.redirect_stdout(io.StringIO()):
            if continue_from is None:
                sim.run(initial_fields={m: rng.normal(0.0, 1.0, int(np.prod(NX))) for m in "AB"})
            else:
                sim.continue_run(continue_from)
    run(root / "full", 4)
    run(root / "part", 3)
    run(root / "part", 4, continue_from=root / "part" / "fields_000003.mat")
    for name in ("fields_000004.mat", "dH_000004.mat", "structure_function_000004.mat"):
        a, b = loadmat(root / "full" / name, squeeze_me=True), loadmat(root / "part" / name, squeeze_me=True)
        keys = [k for k in a if not k.startswith("__") and isinstance(a[k], np.ndarray) and a[k].dtype.kind in "fc"
                and a[k].size]
        assert keys and max(np.max(np.abs(a[k] - b[k])) for k in keys) == 0.0, f"continued run, {name}"
    data = loadmat(root / "part" / "dH_000004.mat", squeeze_me=True)
    assert np.shape(data["number_density_history_real"]) == (2, 4), "pending numbers of chains and ions"


def main():
    if not any(p.startswith("cpu") for p in PlatformSelector.avail_platforms()):
        return 77
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name, check in (("inputs", inputs), ("ensembles", ensembles), ("forces", forces),
                            ("mean-field limit", mean_field), ("pressure", pressure), ("restart", restart)):
            check(root)
            print(f"[PASS] {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
