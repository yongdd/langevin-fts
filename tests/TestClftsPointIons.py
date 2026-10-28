#!/usr/bin/env python3
"""Point ions of CL-FTS (electrostatics.ions): no volume, no chi, only their smeared charge.

An AB diblock with charged A blocks (discrete chains, N = 20) in a salt of point ions Na+ and Cl-:
1. Input: the counterions neutralize the polymer charge (Cl- receives the polymer charge on top of
   the salt; without an ion of the opposite sign there are none); an ambiguous counterion, a
   counterion of the wrong sign, no counterions without screening, an ion named like a monomer
   type, a missing smearing length and continuous chains without n_segments are rejected; with
   n_segments, continuous chains accept bjerrum_length.
2. Forces of psi and of the auxiliary fields against central differences of H, which contains
   -N c_s ln Q_s of the ions, on random fields (standard deviation 5 for the auxiliary fields).
3. Linear response: the smeared ion charge of psi = i eps cos(k.r) is -sum_s z_s^2 c_s Gamma_s(k)^2 psi/N.
4. Chemical potentials: ln(N c_s) at psi = 0 and in the records, after the polymers; restarts are
   bitwise and refuse the ions in another order; the osmotic pressure equals -dH/dV at fixed grid
   values and fixed numbers of ions, with and without screening.
5. Debye-Hueckel statistics: with neutral chains and weak coupling psi is Gaussian with the screening
   operator (k^2 + kappa^2)/E + sum_s z_s^2 c_s Gamma_s^2/N: V Nbar^(1/2) times this operator times
   <|psi(k)/n|^2> is 1 and the holomorphic <psi(k) psi(-k)> -1, mode by mode on average (etd steps).
"""

import contextlib
import io
import os
import tempfile
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from scipy.io import loadmat
from polymerfts import CLFTS, PlatformSelector, ValidationError

NX, LX, NBAR, N = [12, 10], [2.0, 1.8], 1e4, 20
Z_A = 0.2


def cpu_platform():
    return next(p for p in PlatformSelector.avail_platforms() if p.startswith("cpu"))


def parameters(directory, steps=2, screening_length=None, ions=None, **electrostatics):
    """AB diblock (A charged) with Na+ and Cl- point ions; the counterions are Cl-."""
    params = {
        "nx": NX, "lx": LX, "ds": 1 / N, "chain_model": "discrete",
        "segment_lengths": {"A": 1.0, "B": 1.0}, "chi_n": {"A,B": 12.0}, "zeta_n": 50.0,
        "polymers": [{"volume_fraction": 1.0, "blocks": [{"type": "A", "length": 0.5}, {"type": "B", "length": 0.5}]}],
        "smearing": {"type": "gaussian", "a_int": {"A": 0.15, "B": 0.15}},
        "electrostatics": {"charges": {"A": Z_A}, "coupling": 2e3, "psi_dt_scaling": 0.8,
                           "ions": ions if ions is not None else
                           {"Na": {"valence": 1.0, "concentration": 0.05, "smearing_length": 0.1},
                            "Cl": {"valence": -1.0, "concentration": 0.05, "smearing_length": 0.12}}},
        "platform": cpu_platform(),
        "langevin": {"nbar": NBAR, "dt": 0.5, "max_step": steps},
        "recording": {"dir": str(directory), "checkpoint_interval": 1, "sample_interval": 1,
                      "statistics_save_interval": 2, "pressure": True},
        "verbose_level": 0,
    }
    params["electrostatics"].update(electrostatics)
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
    sim = quiet(lambda: CLFTS(parameters(root / "inputs")))
    concentrations = {ion["name"]: ion["concentration"] for ion in sim.electrostatics.ions}
    # A: half of every chain carries 0.2 per segment, a net charge 0.1 per segment
    assert abs(concentrations["Na"] - 0.05) < 1e-14 and abs(concentrations["Cl"] - 0.15) < 1e-14, concentrations

    def changed(**change):
        return lambda: CLFTS(parameters(root / "inputs", **change))
    two_anions = {"Na": {"valence": 1.0, "concentration": 0.05, "smearing_length": 0.1},
                  "Cl": {"valence": -1.0, "concentration": 0.05, "smearing_length": 0.1},
                  "SO4": {"valence": -2.0, "concentration": 0.0, "smearing_length": 0.1}}
    expect_validation_error(changed(ions=two_anions), "name the counterion")
    quiet(changed(ions=two_anions, counterion="SO4"))
    expect_validation_error(changed(counterion="Na"), "has the sign of the net charge")
    expect_validation_error(changed(counterion=None), "not electroneutral")
    quiet(changed(counterion=None, screening_length=0.3))   # an implicit electrolyte screens a net charge
    # Without a canonical ion of the opposite sign there are no counterions
    cations = {"Na": {"valence": 1.0, "concentration": 0.05, "smearing_length": 0.1}}
    quiet(changed(ions=cations, screening_length=0.3))
    expect_validation_error(changed(ions=cations), "not electroneutral")
    expect_validation_error(changed(ions={"A": {"valence": 1.0, "concentration": 0.1, "smearing_length": 0.1}}),
                            "not a monomer type")
    expect_validation_error(changed(ions={"Na": {"valence": 1.0, "concentration": 0.1}}), "smearing_length")
    expect_validation_error(changed(ions={"Na": {"valence": 1.0, "concentration": 0.1, "smearing_length": 0.0}}),
                            "smearing_length")
    expect_validation_error(changed(n_segments=30), "1/ds")

    def continuous(**electrostatics):
        params = parameters(root / "inputs", **electrostatics)
        params.update(chain_model="continuous", ds=1 / 50)
        if "bjerrum_length" in electrostatics:
            params["electrostatics"].pop("coupling")
        return lambda: CLFTS(params)
    expect_validation_error(continuous(), "n_segments")
    sim = quiet(continuous(n_segments=N, bjerrum_length=0.01))
    expected = 4 * np.pi * 0.01 * N ** 2 * np.sqrt(NBAR)
    assert abs(sim.electrostatics.coupling - expected) < 1e-12 * expected, "bjerrum_length with n_segments"


def hamiltonian(sim, w_aux):
    phi, _ = sim.compute_concentrations(w_aux)
    phi_smeared = sim.smearing.apply_to_dict(phi)
    partitions = [sim.prop_solver.get_partition_function(p) for p in range(sim.prop_solver.get_n_polymer_types())]
    h = sim.mpt.compute_hamiltonian(sim.prop_solver.molecules, w_aux, partitions, sim.cb,
                                    include_const_term=True, phi=phi_smeared)
    return h + sim.electrostatics.hamiltonian(), phi_smeared


def forces(root):
    sim = quiet(lambda: CLFTS(parameters(root / "forces")))
    es, n_grid, M = sim.electrostatics, int(np.prod(NX)), 2
    rng = np.random.default_rng(23)
    w_aux = rng.normal(0.0, 5.0, (M, n_grid)) + 1j * rng.normal(0.0, 1.0, (M, n_grid))
    sim.mpt.pin_dropped_fields(w_aux)
    es.psi = rng.normal(0.0, 1.0, n_grid) + 1j * rng.normal(0.0, 1.0, n_grid)
    _, phi_smeared = hamiltonian(sim, w_aux)
    psi_force = es.force(es.charge_density(phi_smeared))
    aux_force = sim.mpt.compute_func_deriv(w_aux, phi_smeared, range(M))
    eps, direction, psi = 1e-5, rng.normal(0.0, 1.0, n_grid), es.psi.copy()
    es.psi = psi + eps * direction
    h_plus, _ = hamiltonian(sim, w_aux)
    es.psi = psi - eps * direction
    h_minus, _ = hamiltonian(sim, w_aux)
    es.psi = psi
    expected = np.mean(psi_force * direction)
    assert abs((h_plus - h_minus) / (2 * eps) - expected) < 1e-6 * max(1.0, abs(expected)), "psi force with point ions"
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


def linear_response(root):
    sim = quiet(lambda: CLFTS(parameters(root / "response")))
    es, eps = sim.electrostatics, 1e-20
    x = np.meshgrid(*[np.arange(n) / n for n in NX], indexing="ij")
    for m in ((1, 0), (2, 3), (0, 5)):
        mode = np.cos(2 * np.pi * (m[0] * x[0] + m[1] * x[1])).ravel()
        es.psi = 1j * eps * mode
        response = np.imag(es.ion_charge()) / eps
        k_sq = sum((2 * np.pi * mi / li) ** 2 for mi, li in zip(m, LX))
        expected = -sum(ion["valence"] ** 2 * ion["concentration"] * np.exp(-ion["smearing_length"] ** 2 * k_sq)
                        for ion in es.ions) / N * mode
        assert np.max(np.abs(response - expected)) < 1e-12 * np.max(np.abs(expected)), f"ion response at {m}"


def records(root):
    sim = quiet(lambda: CLFTS(parameters(root / "records")))
    es, n_grid = sim.electrostatics, int(np.prod(NX))
    mu = es.ion_chemical_potentials()
    assert np.allclose(mu, [np.log(N * ion["concentration"]) for ion in es.ions], rtol=0, atol=1e-14), "mu at psi = 0"
    rng = np.random.default_rng(3)
    with contextlib.redirect_stdout(io.StringIO()):
        sim.run(initial_fields={m: rng.normal(0.0, 1.0, n_grid) for m in "AB"})
    data = loadmat(root / "records" / "dH_000002.mat", squeeze_me=True)
    recorded = np.atleast_2d(data["chemical_potential_history_real"] + 1j * data["chemical_potential_history_imag"])
    assert recorded.shape == (2, 1 + len(es.ions)), recorded.shape
    assert np.allclose(recorded[-1, 1:], es.ion_chemical_potentials(), rtol=0, atol=1e-12), "recorded ion mu"


def restart(root):
    def run(directory, steps, continue_from=None):
        sim = quiet(lambda: CLFTS(parameters(directory, steps=steps), random_seed=7))
        rng = np.random.default_rng(3)
        with contextlib.redirect_stdout(io.StringIO()):
            if continue_from is None:
                sim.run(initial_fields={m: rng.normal(0.0, 1.0, int(np.prod(NX))) for m in "AB"})
            else:
                sim.continue_run(continue_from)
    run(root / "full", 4)
    run(root / "part", 2)
    run(root / "part", 4, continue_from=root / "part" / "fields_000002.mat")
    for name in ("fields_000004.mat", "dH_000004.mat", "structure_function_000004.mat"):
        a, b = loadmat(root / "full" / name, squeeze_me=True), loadmat(root / "part" / name, squeeze_me=True)
        keys = [k for k in a if not k.startswith("__") and isinstance(a[k], np.ndarray) and a[k].dtype.kind in "fc"
                and a[k].size]
        assert keys and max(np.max(np.abs(a[k] - b[k])) for k in keys) == 0.0, f"continued run with point ions, {name}"
    # The record columns follow the order of the ions, which a continuation must keep
    params = parameters(root / "part", steps=4)
    params["electrostatics"]["ions"] = dict(reversed(list(params["electrostatics"]["ions"].items())))
    sim = quiet(lambda: CLFTS(params, random_seed=7))
    expect_validation_error(lambda: sim.continue_run(root / "part" / "fields_000002.mat"), "another order")


def pressure(root):
    """The operator against -dH/dV - d ln N/dV at fixed grid values and fixed numbers of chains and ions."""
    C = np.sqrt(NBAR)
    for screening_length in (None, 0.3):
        params = parameters(root / "pressure", screening_length=screening_length)
        rng = np.random.default_rng(7)
        sim = quiet(lambda: CLFTS(params))
        n_grid, M = int(np.prod(NX)), len(sim.monomer_types)
        w_aux = np.array([smooth(rng.normal(0, 1, n_grid), 5.0) + 1j * smooth(rng.normal(0, 1, n_grid), 1.0)
                          for _ in range(M)])
        psi = smooth(rng.normal(0, 1, n_grid), 0.5) + 1j * smooth(rng.normal(0, 1, n_grid), 0.5)
        volume = sim.prop_solver.get_volume()
        ions = np.array([N * ion["concentration"] for ion in sim.electrostatics.ions])

        def state(s):
            pinned = w_aux.copy()
            s.mpt.pin_dropped_fields(pinned)
            s.electrostatics.psi = psi
            phi, _ = s.compute_concentrations(pinned)
            es = s.electrostatics
            log_q = np.log([s.prop_solver.get_partition_function(p) for p in range(len(s.chain_densities))])
            _, ion_q = es.ion_densities()
            h_fields = sum(s.mpt.h_coef_mu2[i] * np.mean(pinned[i] ** 2) + s.mpt.h_coef_mu1[i] * np.mean(pinned[i])
                           for i in range(M)) + s.mpt.h_const
            # psi term alone: H_el without the ions, whose term is -N c_s ln Q_s at fixed numbers
            h_psi = es.hamiltonian() + np.sum(ions * np.log(np.asarray(ion_q, dtype=complex)))
            return log_q, np.log(np.asarray(ion_q, dtype=complex)), h_fields + h_psi, phi, pinned

        _, _, _, phi, pinned = state(sim)
        operator = sim.pressure(pinned, phi) - np.sum(sim.chain_densities) - np.sum(ions)

        def dilated(scale):
            p = parameters(root / "pressure", screening_length=screening_length)
            p["lx"] = [length * scale ** 0.5 for length in LX]
            s = quiet(lambda: CLFTS(p))
            log_q, log_ion_q, h_fields, _, _ = state(s)
            v = s.prop_solver.get_volume()
            h = C * (volume * (-np.sum(s.chain_densities * log_q) - np.sum(ions * log_ion_q)) + v * h_fields)
            # ln N: n_grid/2 ln V per kept auxiliary field and 1/2 ln(V (k^2 + kappa^2)) per psi mode
            es = s.electrostatics
            modes = es.screened_k_sq > 0
            log_norm = (-0.5 * (s.mpt.R + s.mpt.I) * n_grid * np.log(v)
                        - 0.5 * np.sum(np.log(v * es.screened_k_sq[modes])))
            return h, log_norm, v
        (h_plus, n_plus, v_plus), (h_minus, n_minus, v_minus) = dilated(1 + 1e-5), dilated(1 - 1e-5)
        expected = -((h_plus - h_minus) + (n_plus - n_minus)) / (v_plus - v_minus) / C
        assert abs(operator - expected) < 1e-8 * abs(expected), \
            f"pressure with point ions (screening {screening_length}): operator {operator:.10f}, expected {expected:.10f}"


def debye_hueckel(root):
    for screening_length in (None, 0.4):
        params = parameters(root / "debye", steps=3200, screening_length=screening_length)
        params["electrostatics"]["charges"] = {"A": 0.0}
        params["chi_n"] = {"A,B": 2.0}
        params["langevin"].update(dt=20.0, nbar=1e10, integrator="etd")
        params["recording"].update(checkpoint_interval=3200, statistics_save_interval=3200, pressure=False)
        sums = {"abs": np.zeros(NX), "holomorphic": np.zeros(NX, dtype=complex), "count": 0}
        holder = {}

        def record(sample, sums=sums, holder=holder):
            if sample["step"] > 200:
                es = holder["es"]
                sums["abs"] += np.abs(es.psi_k) ** 2
                sums["holomorphic"] += es.psi_k * es.psi_minus_k()
                sums["count"] += 1
        sim = quiet(lambda: CLFTS(params, random_seed=5, callbacks={"psi": record}))
        es, n_grid = sim.electrostatics, int(np.prod(NX))
        holder["es"] = es
        with contextlib.redirect_stdout(io.StringIO()):
            sim.run(initial_fields={m: np.zeros(n_grid) for m in "AB"})
        operator = es.screened_k_sq / es.coupling + sum(
            ion["valence"] ** 2 * ion["concentration"] * ion["gamma"] ** 2 for ion in es.ions) / N
        scale = sim.prop_solver.get_volume() * np.sqrt(sim.langevin["nbar"]) * operator / n_grid ** 2
        modes = es.screened_k_sq > 0   # without screening the uniform mode of psi is zero
        variance = np.mean((scale * sums["abs"] / sums["count"])[modes])
        holomorphic = np.mean((scale * sums["holomorphic"] / sums["count"]).real[modes])
        assert abs(variance - 1.0) < 0.03 and abs(holomorphic + 1.0) < 0.03, \
            f"Debye-Hueckel (screening {screening_length}): <|psi|^2> {variance:.4f}, <psi psi> {holomorphic:.4f}"


def main():
    if not any(p.startswith("cpu") for p in PlatformSelector.avail_platforms()):
        return 77
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name, check in (("inputs", inputs), ("forces", forces), ("linear response", linear_response),
                            ("records", records), ("restart", restart), ("pressure", pressure),
                            ("Debye-Hueckel statistics", debye_hueckel)):
            check(root)
            print(f"[PASS] {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
