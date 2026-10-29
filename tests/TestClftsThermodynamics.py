#!/usr/bin/env python3
"""Chemical potential and osmotic pressure operators of CL-FTS (CLFTS.chemical_potentials, CLFTS.pressure,
CLFTS.pressure_virial).

1. Pressure as a volume derivative: for random complex fields (standard deviation 5 for the
   real parts) and psi, the operator without the ideal term equals -(dH/dV + d ln N/dV)/C at
   fixed grid values and numbers of chains, where N holds the Gaussian normalizations of the
   field integrals; both derivatives are central differences over dilations of the box, and
   ln N is evaluated from the dilated grids (its V dependence only). Cases: a 3D grid with
   discrete chains, without smearing, with one Gaussian length per type, with sigmoidal
   smearing, with smeared charges with and without screening; continuous chains (RQM4,
   within the contour error of the stress); a 2D hexagonal cell; a 1D grid; a polyampholyte
   whose composition mode has no field (zero eigenvalue), screened with a net charge.
2. Mean-field limit: at the homogeneous saddle point the operators give the Flory-Huggins
   chemical potentials and osmotic pressure, plus E q^2/(2 kappa^2) for a screened system with
   the net charge q; the dilation operator after removing the Gaussian normalizations, the virial
   form as it is (all forces vanish there, and its added terms remove the normalizations).
3. Records: the chemical potentials and the virial pressures in the dH files equal the operators
   recomputed from the saved state of the same step.
4. Virial form: in CL runs of 2000 steps with and without screening, the Schwinger-Dyson terms
   that pressure_virial adds to the dilation operator average to zero within four standard errors
   (block averages), and the virial form fluctuates less than half as much.
5. Restart: a continued run reproduces the pending records bit for bit; a checkpoint without
   them (earlier versions) gives NaN for its pending samples.
6. Input checks: recording.pressure must be a bool, is CL-FTS only and needs free chains in a
   periodic box; with continuous chains and rk2 it warns about the monomer types whose contour
   step b sqrt(ds) exceeds their smearing length, and only then (a random block with its
   effective segment length).
"""
import contextlib
import io
import os
import tempfile
import warnings
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from scipy.io import loadmat, savemat
from polymerfts import LFTS, CLFTS, PlatformSelector, ValidationError

NX, LX = [8, 6, 7], [1.6, 1.3, 1.45]
Z_A = 0.2
PHI_C = Z_A * 0.3 / (1 + Z_A * 0.3)   # counterions: z_A phi_A = phi_C
NBAR, ZETA_N = 400.0, 30.0


def parameters(directory, lx=LX, charged=True, screening_length=None, smearing="per_type", steps=2,
               nx=NX, angles=None, system="diblock", chain_model="discrete", valence_k=1.0):
    """AB diblock (A fraction 0.3, A charged) with counterions C, or an EKEK polyampholyte without chi."""
    params = {
        "nx": list(nx), "lx": list(lx), "ds": 1 / 20 if chain_model == "discrete" else 1 / 50, "chain_model": chain_model,
        "segment_lengths": {"A": 1.0, "B": 1.3, "C": 1.0}, "chi_n": {"A,B": 9.0, "B,C": 2.0}, "zeta_n": ZETA_N,
        "polymers": [{"volume_fraction": 1 - PHI_C, "blocks": [{"type": "A", "length": 0.3}, {"type": "B", "length": 0.7}]},
                     {"volume_fraction": PHI_C, "blocks": [{"type": "C", "length": 0.05}]}],
        "platform": next(p for p in PlatformSelector.avail_platforms() if p.startswith("cpu")),
        "langevin": {"nbar": NBAR, "dt": 0.5, "max_step": steps},
        "recording": {"dir": str(directory), "checkpoint_interval": 1, "sample_interval": 1,
                      "statistics_save_interval": 2, "pressure": True},
        "verbose_level": 0,
    }
    if angles is not None:
        params["angles"] = angles
    if smearing == "per_type":
        params["smearing"] = {"type": "gaussian", "a_int": {"A": 0.12, "B": 0.15, "C": 0.2}}
    elif smearing == "sigmoidal":
        params["smearing"] = {"type": "sigmoidal", "k_int": 20.0, "dk_int": 4.0}
    if charged:
        params["electrostatics"] = {"charges": {"A": Z_A, "C": -1.0}, "coupling": 500.0}
    if system == "polyampholyte":
        params.update(segment_lengths={"E": 1.0, "K": 1.0}, chi_n={},
                      polymers=[{"volume_fraction": 1.0, "blocks": [{"type": t, "length": 0.25} for t in "EKEK"]}])
        if smearing is not None:
            params["smearing"] = {"type": "gaussian", "a_int": 0.15}
        params["electrostatics"]["charges"] = {"E": -1.0, "K": valence_k}
    if charged and screening_length is not None:
        params["electrostatics"]["screening_length"] = screening_length
    return params


def quiet(make):
    with contextlib.redirect_stdout(io.StringIO()):
        return make()


def smooth(x, nx, std):
    """Smooth random field with the given standard deviation."""
    k = np.meshgrid(*[np.fft.fftfreq(m) * m for m in nx], indexing="ij")
    y = np.real(np.fft.ifftn(np.fft.fftn(np.reshape(x, nx)) * np.exp(-0.08 * sum(kk ** 2 for kk in k)))).ravel()
    return std * (y - y.mean()) / y.std()


def normalization(sim):
    """Normalizations of the Gaussian integrals over the auxiliary fields and psi, per C/R0^3."""
    n_grid, n_chains = sim.prop_solver.n_grid, np.sqrt(NBAR) * sim.prop_solver.get_volume()
    value = 0.5 * (sim.mpt.R + sim.mpt.I) * n_grid / n_chains
    es = sim.electrostatics
    if es is not None:
        modes = es.screened_k_sq > 0
        value += 0.5 * np.sum(1 - (2 / sim.prop_solver.dim) * es.k_sq[modes] / es.screened_k_sq[modes]) / n_chains
    return value


def log_normalization(sim, params):
    """V-dependent part of ln N, the Gaussian normalization of the field integrals, from the grid itself.

    An auxiliary field with a nonzero eigenvalue of chi N + zeta N (counted here from the input) has n_grid
    grid values whose quadratic coefficient is proportional to V; a mode of psi has one proportional to
    V (k^2 + kappa^2), and k^2 is that of this (dilated) grid.
    """
    types = sorted(params["segment_lengths"])
    u = np.full((len(types), len(types)), params["zeta_n"])
    for key, value in params["chi_n"].items():
        i, j = (types.index(t) for t in key.split(","))
        u[i, j] += value
        u[j, i] += value
    eigenvalues = np.linalg.eigvalsh(u)
    kept = int(np.sum(np.abs(eigenvalues) > 1e-8 * np.max(np.abs(eigenvalues))))
    volume = sim.prop_solver.get_volume()
    value = -0.5 * kept * sim.prop_solver.n_grid * np.log(volume)
    es = sim.electrostatics
    if es is not None and es.coupling > 0:
        modes = es.screened_k_sq > 0
        value -= 0.5 * np.sum(np.log(volume * es.screened_k_sq[modes]))
    return value


def state(sim, w_aux, psi):
    """ln Q_p, the field part of H per reference chain plus H_el, and phi at the given grid values."""
    w_aux = w_aux.copy()
    sim.mpt.pin_dropped_fields(w_aux)
    if sim.electrostatics is not None:
        sim.electrostatics.psi = psi
    phi, _ = sim.compute_concentrations(w_aux)
    log_q = np.array([np.log(sim.prop_solver.get_partition_function(p)) for p in range(len(sim.chain_densities))])
    h_field = sum(sim.mpt.h_coef_mu2[i] * np.mean(w_aux[i] ** 2) + sim.mpt.h_coef_mu1[i] * np.mean(w_aux[i])
                  for i in range(len(sim.monomer_types))) + sim.mpt.h_const
    h_el = 0.0 if sim.electrostatics is None else sim.electrostatics.hamiltonian()
    return log_q, h_field + h_el, phi, w_aux


def pressure_derivative(root):
    cases = [({"charged": False, "smearing": None}, 1e-8), ({"charged": False}, 1e-8),
             ({"charged": False, "smearing": "sigmoidal"}, 1e-8), ({"charged": True}, 1e-8),
             ({"charged": True, "screening_length": 0.25}, 1e-8),
             ({"charged": True, "chain_model": "continuous"}, 1e-4),
             ({"charged": True, "screening_length": 0.25, "nx": [10, 10], "lx": [1.5, 1.5], "angles": [120.0]}, 1e-8),
             ({"charged": True, "nx": [24], "lx": [2.5]}, 1e-8),
             ({"charged": True, "screening_length": 0.25, "nx": [10, 8], "lx": [1.7, 1.4], "system": "polyampholyte",
               "valence_k": 0.6}, 1e-8)]
    C = np.sqrt(NBAR)
    for options, tolerance in cases:
        nx, lx = options.get("nx", NX), options.get("lx", LX)
        dim, n_grid = len(nx), int(np.prod(nx))
        rng = np.random.default_rng(7)
        params = parameters(root / "derivative", **options)
        sim = quiet(lambda: CLFTS(params))
        M = len(sim.monomer_types)
        w_aux = np.array([smooth(rng.normal(0, 1, n_grid), nx, 5.0) + 1j * smooth(rng.normal(0, 1, n_grid), nx, 1.0)
                          for _ in range(M)])
        psi = smooth(rng.normal(0, 1, n_grid), nx, 0.5) + 1j * smooth(rng.normal(0, 1, n_grid), nx, 0.5)
        _, _, phi, pinned = state(sim, w_aux, psi)
        operator = sim.pressure(pinned, phi) - np.sum(sim.chain_densities)
        volume = sim.prop_solver.get_volume()

        # H and ln N at fixed numbers of chains n_p = (phi_p/alpha_p) C V0 and fixed grid values
        def dilated(scale):
            dilated_params = parameters(root / "derivative", **dict(options, lx=[length * scale ** (1 / dim) for length in lx]))
            dilated_sim = quiet(lambda: CLFTS(dilated_params))
            log_q, h_fields, _, _ = state(dilated_sim, w_aux, psi)
            v = dilated_sim.prop_solver.get_volume()
            return C * (volume * -np.sum(sim.chain_densities * log_q) + v * h_fields), log_normalization(dilated_sim, params), v
        (h_plus, n_plus, v_plus), (h_minus, n_minus, v_minus) = dilated(1 + 1e-5), dilated(1 - 1e-5)
        expected = -((h_plus - h_minus) + (n_plus - n_minus)) / (v_plus - v_minus) / C
        assert abs(operator - expected) < tolerance * abs(expected), \
            f"pressure {options}: operator {operator:.10f}, -(dH/dV + d ln N/dV)/C {expected:.10f}"


def mean_field(root):
    for options in ({}, {"system": "polyampholyte", "nx": [10, 8], "lx": [1.7, 1.4], "screening_length": 0.25,
                         "valence_k": 0.6}):
        params = parameters(root / "mean_field", **options)
        sim = quiet(lambda: CLFTS(params))
        n_grid, M = sim.prop_solver.n_grid, len(sim.monomer_types)
        phi_mean = np.zeros(M)
        for polymer in sim.polymers:
            length = sum(block["length"] for block in polymer["blocks"])
            for block in polymer["blocks"]:
                phi_mean[sim.monomer_types.index(block["type"])] += polymer["volume_fraction"] * block["length"] / length
        X = sim.mpt.matrix_chi
        w = X @ phi_mean + ZETA_N * (np.sum(phi_mean) - 1.0)   # saddle-point potentials of the homogeneous state
        es = sim.electrostatics
        z = np.array([es.charges[t] for t in sim.monomer_types])
        q = z @ phi_mean                                        # net charge per segment
        psi_0 = 0.0 if es.screening_length is None else es.coupling * q / es.kappa_sq
        w_aux = sim.mpt.to_aux_fields(np.repeat(w[:, None], n_grid, axis=1)).astype(complex)
        _, _, phi, w_aux = state(sim, w_aux, np.full(n_grid, psi_0, dtype=complex))
        flory_huggins = np.sum(sim.chain_densities) + 0.5 * phi_mean @ X @ phi_mean + 0.5 * ZETA_N * (np.sum(phi_mean) ** 2 - 1)
        if es.screening_length is not None:
            assert abs(q) > 0.05, "a net charge"
            flory_huggins += es.coupling * q ** 2 / (2 * es.kappa_sq)
        pressure = sim.pressure(w_aux, phi) - normalization(sim)
        assert abs(pressure - flory_huggins) < 1e-10 * flory_huggins, f"mean-field pressure {pressure} vs {flory_huggins}"
        pressure = sim.pressure_virial(w_aux, phi)
        assert abs(pressure - flory_huggins) < 1e-10 * flory_huggins, f"mean-field virial pressure {pressure} vs {flory_huggins}"
        mu = sim.chemical_potentials([sim.prop_solver.get_partition_function(p) for p in range(len(sim.chain_densities))])
        for p, polymer in enumerate(sim.polymers):
            shift = sum(block["length"] * (w + z * psi_0)[sim.monomer_types.index(block["type"])] for block in polymer["blocks"])
            expected = np.log(sim.chain_densities[p]) + shift
            assert abs(mu[p] - expected) < 1e-10, f"mean-field chemical potential {p}: {mu[p]} vs {expected}"


def records(root):
    params = parameters(root / "records", screening_length=0.3)
    params["recording"]["statistics_save_interval"] = 1
    quiet(lambda: CLFTS(params, random_seed=3).run(initial_fields={m: np.zeros(int(np.prod(NX))) for m in "ABC"}))
    sim = quiet(lambda: CLFTS(params, random_seed=3))
    saved = loadmat(root / "records" / "fields_000002.mat", squeeze_me=True)
    recorded = loadmat(root / "records" / "dH_000002.mat", squeeze_me=True)
    sim.electrostatics.psi = saved["psi_real"] + 1j * saved["psi_imag"]
    w_aux = np.reshape(saved["w_aux"], (3, -1))
    phi, _ = sim.compute_concentrations(w_aux)
    pressure = sim.pressure_virial(w_aux, phi)
    mu = sim.chemical_potentials([sim.prop_solver.get_partition_function(p) for p in range(2)])
    measured = complex(recorded["pressure_history_real"]) + 1j * complex(recorded["pressure_history_imag"])
    assert abs(measured - pressure) < 1e-10 * abs(pressure), f"recorded pressure {measured} vs {pressure}"
    measured = recorded["chemical_potential_history_real"] + 1j * recorded["chemical_potential_history_imag"]
    assert np.max(np.abs(measured - mu)) < 1e-10 * np.max(np.abs(mu)), "recorded chemical potentials"


def virial(root):
    def block(x, n_blocks=20):
        x = x[len(x) % n_blocks:].reshape(n_blocks, -1).mean(axis=1)
        return x.mean(), x.std(ddof=1) / np.sqrt(n_blocks)

    steps, skip = 2000, 200
    for screening_length in (0.3, None):
        params = parameters(root / "virial", screening_length=screening_length, steps=steps)
        params["recording"].update(checkpoint_interval=steps, statistics_save_interval=steps)
        dilation = []

        def record_dilation(sample):
            phi = {name: np.asarray(value) for name, value in sample["phi"].items()}
            dilation.append(sim.pressure(np.asarray(sample["w_aux"]), phi))
        sim = quiet(lambda: CLFTS(params, random_seed=5, callbacks={"dilation": record_dilation}))
        quiet(lambda: sim.run(initial_fields={m: np.zeros(int(np.prod(NX))) for m in "ABC"}))
        recorded = loadmat(root / "virial" / f"dH_{steps:06d}.mat", squeeze_me=True)
        virial_form = (recorded["pressure_history_real"] + 1j * recorded["pressure_history_imag"])[skip:]
        dilation_form = np.array(dilation)[skip:]
        for part in (np.real, np.imag):
            mean, error = block(part(virial_form - dilation_form))
            assert abs(mean) < 4 * error, f"Schwinger-Dyson terms (screening {screening_length}): {mean} +- {error}"
        ratio = np.std(np.real(virial_form)) / np.std(np.real(dilation_form))
        assert ratio < 0.5, f"virial form (screening {screening_length}): standard deviation ratio {ratio}"


def restart(root):
    def run(directory, steps, continue_from=None):
        params = parameters(directory, steps=steps)
        params["recording"].update(checkpoint_interval=3, statistics_save_interval=4)
        sim = quiet(lambda: CLFTS(params, random_seed=3))
        with contextlib.redirect_stdout(io.StringIO()):
            if continue_from is None:
                sim.run(initial_fields={m: np.zeros(int(np.prod(NX))) for m in "ABC"})
            else:
                sim.continue_run(continue_from)
    run(root / "full", 8)
    run(root / "part", 3)
    run(root / "part", 8, root / "part" / "fields_000003.mat")
    a, b = loadmat(root / "full" / "dH_000004.mat"), loadmat(root / "part" / "dH_000004.mat")
    for key in ("chemical_potential_history_real", "chemical_potential_history_imag", "pressure_history_real",
                "pressure_history_imag"):
        assert np.array_equal(a[key], b[key]), f"continued record {key}"

    # A checkpoint without the pending records: NaN for the three pending samples of steps 1-3
    data = loadmat(root / "part" / "fields_000003.mat", squeeze_me=True)
    for key in ("record_chemical_potential_history", "record_pressure_history"):
        data.pop(key)
    savemat(root / "old.mat", {k: v for k, v in data.items() if not k.startswith("__")},
            long_field_names=True, do_compression=True)
    run(root / "old", 4, root / "old.mat")
    records = loadmat(root / "old" / "dH_000004.mat")
    for key, shape in (("chemical_potential_history_real", (4, 2)), ("pressure_history_real", (1, 4))):
        values = np.atleast_2d(records[key])
        assert values.shape == shape and np.all(np.isnan(values[:3] if shape[0] == 4 else values[0, :3])) \
            and np.all(np.isfinite(values[3] if shape[0] == 4 else values[0, 3])), f"NaN for samples without records: {key}"


def expect_validation_error(make, message):
    try:
        quiet(make)
    except ValidationError as exc:
        assert message in str(exc), (message, str(exc))
        return
    raise AssertionError(f"Expected ValidationError: {message}")


def input_checks(root):
    def changed(change):
        params = parameters(root / "inputs")
        change(params)
        return lambda: CLFTS(params)
    expect_validation_error(changed(lambda p: p["recording"].update(pressure="yes")), "True or False")
    expect_validation_error(changed(lambda p: p["polymers"][1].update(initial_conditions={0: "G"})), "free chains")
    walls = parameters(root / "inputs", charged=False)
    walls["bc"] = ["reflecting"] * 6
    expect_validation_error(lambda: CLFTS(walls), "periodic box")
    lfts = parameters(root / "inputs", charged=False, smearing=None)
    lfts.pop("zeta_n")
    lfts["saddle"] = {"max_iter": 10, "tolerance": 1e-4}
    lfts["compressor"] = {"name": "am", "max_hist": 5, "start_error": 0.5, "mix_min": 0.01, "mix_init": 0.01}
    expect_validation_error(lambda: LFTS(lfts), "CL-FTS only")

    # rk2 warns about the pressure where one contour step b sqrt(ds) exceeds the smearing length: with
    # ds = 1/50 the steps of A, B and C are 0.141, 0.184 and 0.141 against the lengths 0.12, 0.15 and 0.2
    def rk2_warnings(change, **options):
        params = parameters(root / "inputs", chain_model="continuous", **options)
        params["numerical_method"] = "rk2"
        change(params)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            quiet(lambda: CLFTS(params))
        return [str(w.message) for w in caught if "rk2" in str(w.message)]
    messages = rk2_warnings(lambda p: None)
    assert len(messages) == 1 and "A: step" in messages[0] and "B: step" in messages[0] and "C: step" not in messages[0], messages
    assert len(rk2_warnings(lambda p: None, charged=False, smearing=None)) == 1
    assert not rk2_warnings(lambda p: p["smearing"].update(a_int=0.2))
    assert not rk2_warnings(lambda p: p.update(numerical_method="rqm4"))
    assert not rk2_warnings(lambda p: p["recording"].update(pressure=False))
    # A random block of C (1.0) and D (1.5) has the step 1.27 sqrt(ds) = 0.180 < 0.19, while D alone would
    # have 0.212: only the effective segment length counts
    def random_block(p):
        p["segment_lengths"].update(D=1.5)
        p["polymers"][1]["blocks"] = [{"type": "R", "length": 1.0, "monomer_fractions": {"C": 0.5, "D": 0.5}}]
        p["smearing"] = {"type": "gaussian", "a_int": 0.19}
    assert not rk2_warnings(random_block, charged=False)


def main():
    if not any(p.startswith("cpu") for p in PlatformSelector.avail_platforms()):
        return 77
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name, check in (("pressure as a volume derivative", pressure_derivative), ("mean-field limit", mean_field),
                            ("records", records), ("virial form", virial), ("restart", restart),
                            ("input checks", input_checks)):
            check(root)
            print(f"[PASS] {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
