#!/usr/bin/env python3
"""CL-FTS with smeared Coulomb interactions (electrostatics.py).

1. Neutral reductions: with all valences zero, the auxiliary fields and concentrations
   equal those of CL-FTS without electrostatics (psi draws its own noise); with
   charges but coupling E = 0, H and the measurements are equal as well. Per-type
   smearing lengths that are all equal reproduce one common length bit for bit.
2. Forces: the force of psi, lap(psi)/E + c, and the forces of the auxiliary fields
   are the derivatives of H, by central differences around random complex fields
   (standard deviation 5 for the real parts).
3. Step of psi: without its noise, one step is the exact flow of
   d psi/d tau = s dH/d psi over dt (fine RK4 steps of the force). The stationary variance
   of every mode is that of exp(-H), V Nbar^(1/2) (k^2/E + R) <|psi(k)|^2> = 1, without
   charges and, with the averaged noise of the LM integrator, also for a charge density
   c = -R psi held fixed during each step; at time steps far beyond the stability limit of
   Euler steps too.
4. Restart: a run continued from a checkpoint with pending measurements reproduces
   the uninterrupted run bit for bit, including psi and its noise stream.
5. Input checks: electroneutrality, monomer types, coupling, smearing, boundaries,
   random copolymers and the methods other than CL-FTS.
"""
import contextlib
import io
import json
import os
import tempfile
import warnings
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from scipy.io import loadmat
from polymerfts import LFTS, CLFTS, SCFT, PlatformSelector, ValidationError
from polymerfts.utils import validate_checkpoint, _measurement_key

Z_A = 0.2
PHI_C = Z_A * 0.5 / (1 + Z_A * 0.5)   # counterions: z_A phi_A = phi_C
NX, LX = [12, 10], [2.0, 1.8]


def parameters(directory, steps=6, electrostatics=True):
    """AB diblock with charged A blocks and monovalent counterions C, 2D, discrete chains."""
    params = {
        "nx": NX, "lx": LX, "ds": 1 / 20, "chain_model": "discrete",
        "segment_lengths": {"A": 1.0, "B": 1.0, "C": 1.0}, "chi_n": {"A,B": 12.0}, "zeta_n": 50.0,
        "polymers": [{"volume_fraction": 1 - PHI_C, "blocks": [{"type": "A", "length": 0.5},
                                                                {"type": "B", "length": 0.5}]},
                     {"volume_fraction": PHI_C, "blocks": [{"type": "C", "length": 0.05}]}],
        "smearing": {"type": "gaussian", "a_int": {"A": 0.15, "B": 0.15, "C": 0.2}},
        "platform": next(p for p in PlatformSelector.avail_platforms() if p.startswith("cpu")),
        "langevin": {"nbar": 1e4, "dt": 0.5, "max_step": steps},
        "recording": {"dir": str(directory), "checkpoint_interval": 3, "sample_interval": 1,
                      "statistics_save_interval": 6},
        "verbose_level": 0,
    }
    if electrostatics:
        params["electrostatics"] = {"charges": {"A": Z_A, "C": -1.0}, "coupling": 2e3, "psi_dt_scaling": 0.8}
    return params


def initial_fields():
    rng = np.random.default_rng(11)
    return {m: rng.normal(0.0, 1.0, int(np.prod(NX))) for m in ("A", "B", "C")}


def run(params, *, continue_from=None):
    with contextlib.redirect_stdout(io.StringIO()):
        sim = CLFTS(params, random_seed=5)
        if continue_from is None:
            sim.run(initial_fields=initial_fields())
        else:
            sim.continue_run(continue_from)
    return sim


def mat(directory, name):
    return loadmat(Path(directory) / name, squeeze_me=True)


def max_difference(a, b, prefixes):
    keys = [k for k in a if k.startswith(prefixes)]
    assert keys, prefixes
    return max(np.max(np.abs(np.asarray(a[k], dtype=complex) - np.asarray(b[k], dtype=complex))) for k in keys)


def neutral_reductions(root):
    # Valences zero, E > 0: the same fields and concentrations as without electrostatics
    plain = parameters(root / "plain", electrostatics=False)
    neutral = parameters(root / "neutral")
    neutral["electrostatics"]["charges"] = {"A": 0.0, "C": 0.0}
    run(plain), run(neutral)
    a, b = mat(root / "plain", "fields_000006.mat"), mat(root / "neutral", "fields_000006.mat")
    assert max_difference(a, b, ("w_", "phi_")) == 0.0
    assert np.max(np.abs(b["psi_real"])) > 0.0   # psi fluctuates, decoupled

    # Charges, E = 0: also the same H and structure functions
    unscreened = parameters(root / "unscreened")
    unscreened["electrostatics"]["coupling"] = 0.0
    run(unscreened)
    b = mat(root / "unscreened", "fields_000006.mat")
    assert max_difference(a, b, ("w_", "phi_")) == 0.0
    assert max_difference(mat(root / "plain", "dH_000006.mat"), mat(root / "unscreened", "dH_000006.mat"),
                          ("H_history", "dH_history")) == 0.0
    assert max_difference(mat(root / "plain", "structure_function_000006.mat"),
                          mat(root / "unscreened", "structure_function_000006.mat"), ("structure_function_",)) == 0.0

    # Per-type smearing lengths that are all equal: the run of one common length
    common, per_type = parameters(root / "common"), parameters(root / "per_type")
    common["smearing"]["a_int"] = 0.15
    per_type["smearing"]["a_int"] = dict.fromkeys(("A", "B", "C"), 0.15)
    run(common), run(per_type)
    for name, prefixes in (("fields_000006.mat", ("w_", "phi_", "psi_real", "psi_imag", "psi_k_")),
                           ("structure_function_000006.mat", ("structure_function_", "charge_structure_function"))):
        assert max_difference(mat(root / "common", name), mat(root / "per_type", name), prefixes) == 0.0


def hamiltonian(sim, w_aux):
    phi, _ = sim.compute_concentrations(w_aux)
    phi_smeared = sim.smearing.apply_to_dict(phi)
    partitions = [sim.prop_solver.get_partition_function(p) for p in range(sim.prop_solver.get_n_polymer_types())]
    h = sim.mpt.compute_hamiltonian(sim.prop_solver.molecules, w_aux, partitions, sim.cb,
                                    include_const_term=True, phi=phi_smeared)
    return h + sim.electrostatics.hamiltonian(), phi_smeared


def forces(root):
    with contextlib.redirect_stdout(io.StringIO()):
        sim = CLFTS(parameters(root / "forces"), random_seed=5)
    es, n_grid, M = sim.electrostatics, int(np.prod(NX)), 3
    rng = np.random.default_rng(23)
    w_aux = rng.normal(0.0, 5.0, (M, n_grid)) + 1j * rng.normal(0.0, 1.0, (M, n_grid))
    sim.mpt.pin_dropped_fields(w_aux)
    es.psi = rng.normal(0.0, 1.0, n_grid) + 1j * rng.normal(0.0, 1.0, n_grid)
    # psi is complex: psi(-k) is the index-reversed psi(k), FFT[conj psi](k) conjugated
    psi_grid = np.reshape(es.psi, NX)
    assert np.allclose(es.psi_minus_k(), np.conj(np.fft.fftn(np.conj(psi_grid))), rtol=0, atol=1e-10), "psi(-k)"
    _, phi_smeared = hamiltonian(sim, w_aux)
    psi_force = es.force(es.charge_density(phi_smeared))
    aux_force = sim.mpt.compute_func_deriv(w_aux, phi_smeared, range(M))
    eps = 1e-5

    # dH = mean(force * delta) along a random real direction delta
    direction = rng.normal(0.0, 1.0, n_grid)
    psi = es.psi.copy()
    es.psi = psi + eps * direction
    h_plus, _ = hamiltonian(sim, w_aux)
    es.psi = psi - eps * direction
    h_minus, _ = hamiltonian(sim, w_aux)
    es.psi = psi
    expected = np.mean(psi_force * direction)
    assert abs((h_plus - h_minus) / (2 * eps) - expected) < 1e-6 * max(1.0, abs(expected)), "psi force"

    # Auxiliary fields: +dH/dOmega for real fields, -dH/dOmega for imaginary fields
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


def psi_step(root):
    with contextlib.redirect_stdout(io.StringIO()):
        sim = CLFTS(parameters(root / "step"), random_seed=5)
    es, n_grid = sim.electrostatics, int(np.prod(NX))
    dt, s = sim.langevin["dt"], es.psi_dt_scaling
    rng = np.random.default_rng(31)

    def zero_mean(x):
        return x - np.mean(x)
    psi0 = zero_mean(rng.normal(0.0, 1.0, n_grid) + 1j * rng.normal(0.0, 1.0, n_grid))
    c = zero_mean(rng.normal(0.0, 1.0, n_grid) + 1j * rng.normal(0.0, 1.0, n_grid))

    # The same noise from psi = 0 and c = 0: the difference is the deterministic part of the step
    state, previous = es.random_bg.state, rng.normal(0.0, es.sigma, n_grid)
    es.psi, es.previous_noise = psi0, previous.copy()
    es.step(c)
    stepped = es.psi.copy()
    es.random_bg.state = state
    es.psi, es.previous_noise = np.zeros(n_grid), previous.copy()
    es.step(np.zeros(n_grid))
    noise = es.psi.copy()

    def drift(psi):
        es.psi = psi
        return s * es.force(c)
    flow, n_sub = psi0.copy(), 200
    h = dt / n_sub
    for _ in range(n_sub):
        k1 = drift(flow)
        k2 = drift(flow + h / 2 * k1)
        k3 = drift(flow + h / 2 * k2)
        k4 = drift(flow + h * k3)
        flow = flow + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    assert np.max(np.abs(stepped - noise - flow)) < 1e-10 * np.max(np.abs(flow)), "deterministic part of the psi step"

    # Stationary variance against exp(-H) of psi with a charge density that follows it linearly,
    # c = -R psi (the Gaussian response of the chains; R = 0: no charges):
    # V Nbar^(1/2) (k^2/E + R) <|psi(k)/n|^2> = 1 for every mode. The mean over the modes uses 20
    # blocks of 1000 steps after 10 relaxation times of the slowest free mode. At dt = 20 the
    # fastest modes have gamma dt = 5, beyond 2, where Euler steps diverge. The averaged noise of
    # the LM integrator is exact also with R > 0; fresh noise (Euler integrator) is exact for R = 0
    # only (with R = 0.1 its variance is 9 % too large at dt = 2 and 3 times at dt = 20)
    for dt, integrator, response in ((2.0, "lm", 0.0), (20.0, "lm", 0.0), (20.0, "euler", 0.0),
                                     (2.0, "lm", 0.1), (20.0, "lm", 0.1)):
        params = parameters(root / "equipartition")
        params["langevin"].update(dt=dt, integrator=integrator)
        params["electrostatics"]["charges"] = {"A": 0.0, "C": 0.0}
        with contextlib.redirect_stdout(io.StringIO()):
            sim = CLFTS(params, random_seed=5)
        es = sim.electrostatics
        rate = es.k_sq / es.coupling
        scale = sim.prop_solver.get_volume() * np.sqrt(sim.langevin["nbar"]) * (rate + response) / n_grid ** 2
        gamma_dt = rate * es.psi_dt_scaling * dt
        for _ in range(int(np.ceil(10 / gamma_dt[rate > 0].min()))):
            es.step(-response * es.psi)
        blocks = np.zeros(20)
        for b in range(len(blocks)):
            total = np.zeros(NX)
            for _ in range(1000):
                es.step(-response * es.psi)
                total += np.abs(es.psi_k) ** 2
            blocks[b] = np.mean((scale * total / 1000)[rate > 0])
        mean, error = np.mean(blocks), np.std(blocks, ddof=1) / np.sqrt(len(blocks))
        assert abs(mean - 1.0) < 0.02 and 3 * error < 0.02, \
            f"stationary variance of psi ({integrator}, dt = {dt}, R = {response}): {mean:.4f} +- {error:.4f} of exp(-H)"

    # psi takes the noise scheme its integrator declares (averaged_noise); a custom integrator
    # must declare it (input_checks)
    class Custom:
        name, settings, averaged_noise = "custom", {}, True

        def step(self, fields, force, **kwargs):
            return fields
    for integrator, averaged in (("lm", True), (Custom(), True), ("euler", False)):
        with contextlib.redirect_stdout(io.StringIO()):
            sim = CLFTS(parameters(root / "integrators"), random_seed=5, integrator=integrator)
        assert sim.electrostatics.averaged_noise == averaged, f"averaged psi noise with the integrator {integrator}"


def restart(root):
    sim = run(parameters(root / "full"))
    # psi draws from the jumped stream of the auxiliary fields, recorded in the run metadata
    metadata = json.loads(str(mat(root / "full", "dH_000006.mat")["simulation_metadata_json"]))
    assert metadata["initial_psi_rng_state"] == np.random.PCG64(5).jumped().state
    # A checkpoint without psi is rejected
    expect_validation_error(lambda: sim.electrostatics.restore({"psi_real": np.zeros(int(np.prod(NX)))}),
                            "no electrostatic potential")
    run(parameters(root / "part", steps=3))
    run(parameters(root / "part"), continue_from=root / "part" / "fields_000003.mat")
    a, b = mat(root / "full", "fields_000006.mat"), mat(root / "part", "fields_000006.mat")
    assert max_difference(a, b, ("w_", "phi_", "psi_real", "psi_imag", "normal_noise_prev", "psi_noise_prev")) == 0.0
    assert str(a["psi_rng_state_json"]) == str(b["psi_rng_state_json"])
    for name in ("dH_000006.mat", "structure_function_000006.mat"):
        assert max_difference(mat(root / "full", name), mat(root / "part", name),
                              ("H_history", "dH_history", "structure_function_", "charge_structure_function")) == 0.0
    # The drift of psi is recorded apart, and its pending samples continue after a restart
    a, b = mat(root / "full", "dH_000006.mat"), mat(root / "part", "dH_000006.mat")
    assert np.size(a["psi_drift_max_history"]) == 6 and np.all(a["psi_drift_max_history"] > 0)
    assert max_difference(a, b, ("psi_drift_max_history",)) == 0.0

    # A run continued by hand takes psi through initial_psi; without it psi restarts at zero
    fields = mat(root / "part", "fields_000003.mat")
    psi = fields["psi_real"] + 1j * fields["psi_imag"]
    initial = {m: fields["w_" + m + "_real"] + 1j * fields["w_" + m + "_imag"] for m in ("A", "B", "C")}
    for given in (psi, None):
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sim = CLFTS(parameters(root / "manual", steps=4), random_seed=5)
            es, first = sim.electrostatics, []

            def step(c, step=es.step):
                first.append(es.psi.copy())
                step(c)
            es.step = step
            sim.run(initial, normal_noise_prev=fields["normal_noise_prev"], start_langevin_step=4, initial_psi=given)
        assert np.array_equal(first[0], np.zeros_like(psi) if given is None else psi), "psi at the start of step 4"
        assert any("initial_psi" in str(w.message) for w in caught) == (given is None), "warning without initial_psi"
    expect_validation_error(lambda: CLFTS(parameters(root / "manual", steps=4)).run(
        initial, start_langevin_step=4, initial_psi=np.zeros(3)), "initial_psi")
    expect_validation_error(lambda: CLFTS(parameters(root / "manual", steps=4, electrostatics=False)).run(
        initial, initial_psi=psi), "needs 'electrostatics'")
    # Without Coulomb coupling psi does not move: a nonzero initial_psi is rejected
    uncoupled = parameters(root / "manual", steps=4)
    uncoupled["electrostatics"]["coupling"] = 0.0
    expect_validation_error(lambda: CLFTS(uncoupled).run(initial, start_langevin_step=4, initial_psi=psi),
                            "without Coulomb coupling")

    # A checkpoint must hold psi, psi(k), the previous noise, the stream of psi, the pending
    # drifts of psi and the pending charge structure sums
    data = loadmat(root / "part" / "fields_000003.mat", squeeze_me=True)
    with contextlib.redirect_stdout(io.StringIO()):
        sim = CLFTS(parameters(root / "part"), random_seed=5)
    validate_checkpoint(sim, data)
    pending = np.size(data["record_psi_drift_history"])
    assert pending == 3, pending
    for change, message in ((lambda d: d.pop("psi_k_real"), "lacks electrostatic data"),
                            (lambda d: d.pop("record_psi_drift_history"), "lacks electrostatic data"),
                            (lambda d: d.update(psi_noise_prev=np.zeros(3)), "electrostatic array 'psi_noise_prev'"),
                            (lambda d: d.update(psi_rng_state_json="{}"), "random-number state of psi"),
                            (lambda d: d.update(record_psi_drift_history=np.full(pending, np.nan)),
                             "record_psi_drift_history"),
                            (lambda d: d.pop(_measurement_key("sf", "charge_mixed")), "charge structure sum")):
        modified = dict(data)
        change(modified)
        expect_validation_error(lambda: validate_checkpoint(sim, modified), message)


def expect_validation_error(make, message):
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            make()
    except ValidationError as exc:
        assert message in str(exc), (message, str(exc))
        return
    raise AssertionError(f"Expected ValidationError: {message}")


def input_checks(root):
    def changed(change):
        params = parameters(root / "inputs")
        change(params)
        return lambda: CLFTS(params)

    expect_validation_error(changed(lambda p: p["electrostatics"]["charges"].update(A=0.3)), "not electroneutral")
    expect_validation_error(changed(lambda p: p["electrostatics"]["charges"].update(B=0.1)), "not electroneutral")
    expect_validation_error(changed(lambda p: p["electrostatics"]["charges"].update(D=1.0)), "unknown monomer types")
    expect_validation_error(changed(lambda p: p["electrostatics"].update(bjerrum_length=0.01)), "exactly one")
    expect_validation_error(changed(lambda p: p["electrostatics"].pop("coupling")), "exactly one")
    expect_validation_error(changed(lambda p: p["electrostatics"].update(psi_dt_scaling=0.0)), "psi_dt_scaling")
    expect_validation_error(changed(lambda p: p["electrostatics"].update(psi_dt_scale=0.2)), "unknown keys")
    expect_validation_error(changed(lambda p: p.pop("smearing")), "needs 'smearing'")
    expect_validation_error(changed(lambda p: p["smearing"]["a_int"].update(C=0.0)), "positive smearing length")
    expect_validation_error(changed(lambda p: p.update(bc=["reflecting"] * 4)), "periodic boxes only")
    expect_validation_error(changed(lambda p: p["smearing"]["a_int"].pop("C")), "one length for each monomer type")

    def bjerrum(chain_model, ds):
        def change(p):
            p["electrostatics"] = {"charges": {"A": Z_A, "C": -1.0}, "bjerrum_length": 0.01}
            p["chain_model"], p["ds"] = chain_model, ds
        return change
    expect_validation_error(changed(bjerrum("continuous", 0.05)), "needs discrete chains")
    expect_validation_error(changed(bjerrum("discrete", 0.03)), "integer 1/ds")

    def lfts(per_type):
        params = parameters(root / "inputs", electrostatics=not per_type)
        if not per_type:
            params["smearing"] = {"type": "gaussian", "a_int": 0.15}
        params["saddle"] = {"max_iter": 10, "tolerance": 1e-4}
        params["compressor"] = {"name": "am", "max_hist": 5, "start_error": 0.5, "mix_min": 0.01, "mix_init": 0.01}
        return lambda: LFTS(params)
    expect_validation_error(lfts(False), "CL-FTS only")
    expect_validation_error(lfts(True), "CL-FTS only")
    scft = parameters(root / "inputs")
    scft.update(optimizer={"name": "am"}, relax_cell=False)
    expect_validation_error(lambda: SCFT(scft), "CL-FTS only")

    # The coupling of bjerrum_length: E = 4 pi (l_B/R0) N^2 Nbar^(1/2)
    params = parameters(root / "inputs")
    params["electrostatics"] = {"charges": {"A": Z_A, "C": -1.0}, "bjerrum_length": 0.01}
    with contextlib.redirect_stdout(io.StringIO()):
        sim = CLFTS(params)
    assert np.isclose(sim.electrostatics.coupling, 4 * np.pi * 0.01 * 20 ** 2 * 100.0)

    # A custom integrator must declare the noise scheme that psi takes (psi_step: declared ones)
    class Undeclared:
        name, settings = "undeclared", {}

        def step(self, fields, force, **kwargs):
            return fields
    expect_validation_error(lambda: CLFTS(parameters(root / "inputs"), integrator=Undeclared()),
                            "must declare averaged_noise")
    fresh = Undeclared()
    fresh.averaged_noise = False
    with contextlib.redirect_stdout(io.StringIO()):
        sim = CLFTS(parameters(root / "inputs"), integrator=fresh)
    assert sim.electrostatics.averaged_noise is False


def main():
    if not any(p.startswith("cpu") for p in PlatformSelector.avail_platforms()):
        return 77
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name, check in (("neutral reductions", neutral_reductions), ("forces", forces), ("psi step", psi_step),
                            ("restart", restart), ("input checks", input_checks)):
            check(root)
            print(f"[PASS] {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
