#!/usr/bin/env python3
"""Electrostatics in closed boxes (walls on every side) with fixed charges, in SCFT and CL-FTS.

With walls the transforms are cosine series: psi has no normal derivative at the walls.
1. Input: a box that mixes periodic and wall axes, a fixed charge of the wrong size, box
   relaxation of a charged SCFT with walls and a mask (SCFT and CL-FTS) are rejected; the
   counterions neutralize the fixed charge, and 'fixed_charge': None means none.
2. Poisson-Boltzmann: a neutral diblock melt with Na+ and Cl- point ions between two negatively
   charged layers near the walls (1D, 64 points). psi of SCFT equals the solution of the
   Poisson-Boltzmann equation with the same smearing, solved independently in the cosine basis of
   scipy.fft.dct (1e-9); the CL-FTS mean-field limit (nbar 1e28) reaches it within its noise.
3. CL-FTS forces of psi and of the auxiliary fields against central differences of H on random
   fields in a closed 2D box with a charged diblock, a fixed charge, canonical and grand-canonical
   point ions (standard deviation 5 for the auxiliary fields).
4. A continued CL-FTS run in the closed box reproduces the uninterrupted run bit for bit.
5. Fixed charges in a periodic box, held at their grid values when the box dilates: the SCFT
   stress equals dF/dL of a charged lamella, and the CL-FTS osmotic pressure equals -dH/dV
   (with the normalizations of the Gaussian integrals) on random fields.
6. SCFT in a closed box: the stress (cosine modes of psi and of the ions) equals dF/dL with point
   ions and a fixed charge held at its grid values, and charged chains grafted to a wall (q_init)
   converge to the Poisson equation.
"""

import contextlib
import io
import os
import tempfile
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from scipy.fft import dct, idct
from scipy.io import loadmat
from scipy.optimize import root as solve_root
from polymerfts import CLFTS, SCFT, PlatformSelector, ValidationError

N, E = 20, 200.0


def cpu_platform():
    return next(p for p in PlatformSelector.avail_platforms() if p.startswith("cpu"))


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


def layers(nx, lx):
    """Negative Gaussian charge layers 0.4 R0 from both walls along the first axis."""
    x = (np.arange(nx[0]) + 0.5) * lx[0] / nx[0]
    profile = -0.3 * (np.exp(-(x - 0.4) ** 2 / 0.045) + np.exp(-(x - lx[0] + 0.4) ** 2 / 0.045))
    return np.ravel(np.broadcast_to(np.reshape(profile, (-1,) + (1,) * (len(nx) - 1)), nx))


def parameters(nx, lx, charge_a=0.0, ions=None, **extra):
    params = {"nx": nx, "lx": lx, "ds": 1 / N, "chain_model": "discrete", "bc": ["reflecting"] * (2 * len(nx)),
              "segment_lengths": {"A": 1.0, "B": 1.0}, "chi_n": {"A,B": 0.0}, "zeta_n": 50.0,
              "polymers": [{"volume_fraction": 1.0, "blocks": [{"type": "A", "length": 0.5}, {"type": "B", "length": 0.5}]}],
              "smearing": {"type": "gaussian", "a_int": 0.1}, "platform": cpu_platform(),
              "electrostatics": {"charges": {"A": charge_a}, "coupling": E, "fixed_charge": layers(nx, lx),
                                 "ions": ions if ions is not None else
                                 {"Na": {"valence": 1.0, "concentration": 0.01, "smearing_length": 0.1},
                                  "Cl": {"valence": -1.0, "concentration": 0.01, "smearing_length": 0.15}}}}
    params.update(extra)
    return params


def scft_options(**extra):
    options = {"optimizer": {"name": "am", "max_hist": 20, "start_error": 1e-2, "mix_min": 0.1, "mix_init": 0.1},
               "relax_cell": False, "max_iter": 2000, "tolerance": 1e-11}
    options.update(extra)
    return options


def clfts_options(directory, steps, nbar=1e28):
    return {"langevin": {"nbar": nbar, "dt": 0.5, "max_step": steps},
            "recording": {"dir": str(directory), "checkpoint_interval": steps, "sample_interval": 1,
                          "statistics_save_interval": steps},
            "verbose_level": 0}


def inputs(root):
    nx, lx = [64], [4.0]
    params = parameters(nx, lx, **scft_options())
    params["bc"] = ["periodic", "periodic"]
    quiet(lambda: SCFT(params))
    params = parameters([8, 8], [2.0, 2.0], **scft_options())
    params["bc"] = ["reflecting", "reflecting", "periodic", "periodic"]
    expect_validation_error(lambda: SCFT(params), "not both")
    params = parameters(nx, lx, **scft_options())
    params["electrostatics"]["fixed_charge"] = np.zeros(63)
    expect_validation_error(lambda: SCFT(params), "one real, finite charge density per grid point")
    expect_validation_error(lambda: SCFT(parameters(nx, lx, **scft_options(relax_cell=True))), "needs a periodic box")
    # Impenetrable regions are not supported by the simulation classes
    mask = np.ones(nx[0])
    expect_validation_error(lambda: SCFT(parameters(nx, lx, mask=mask, **scft_options())), "does not take a 'mask'")
    expect_validation_error(lambda: CLFTS(parameters(nx, lx, mask=mask, **clfts_options(root / "mask", 1))),
                            "does not take a 'mask'")
    sim = quiet(lambda: SCFT(parameters(nx, lx, **scft_options())))
    na = next(ion for ion in sim.electrostatics.ions if ion["name"] == "Na")
    assert abs(na["concentration"] - (0.01 - np.mean(layers(nx, lx)))) < 1e-14, "counterions of the fixed charge"
    params = parameters(nx, lx, **scft_options())
    params["electrostatics"]["fixed_charge"] = None
    sim = quiet(lambda: SCFT(params))
    assert sim.electrostatics.fixed_charge is None, "'fixed_charge': None"


def poisson_boltzmann(root):
    nx, lx = [64], [4.0]
    fixed = layers(nx, lx)
    k = np.pi * np.arange(nx[0]) / lx[0]
    concentrations = ((1.0, 0.01 - np.mean(fixed), 0.1), (-1.0, 0.01, 0.15))

    def smear(f, a):
        return idct(np.exp(-a * a * k * k / 2) * dct(f, norm="ortho"), norm="ortho")

    def residual(modes):
        psi = idct(np.r_[0.0, modes], norm="ortho")
        charge = fixed.copy()
        for z, c, a in concentrations:
            weight = np.exp(-z * smear(psi, a) / N)
            charge += z * smear(c * weight / np.mean(weight), a)
        return (dct(charge, norm="ortho") - k * k * np.r_[0.0, modes] / E)[1:]
    solution = solve_root(residual, np.zeros(nx[0] - 1), method="hybr", tol=1e-14)
    assert np.max(np.abs(residual(solution.x))) < 1e-13, "Poisson-Boltzmann reference"
    psi = idct(np.r_[0.0, solution.x], norm="ortho")

    sim = quiet(lambda: SCFT(parameters(nx, lx, **scft_options())))
    result = quiet(lambda: sim.run(initial_fields={"A": np.zeros(64), "B": np.zeros(64)}, return_result=True))
    assert result.converged and np.max(np.abs(result.psi - psi)) < 1e-9 * np.max(np.abs(psi)), \
        f"SCFT: max |psi - PB| = {np.max(np.abs(result.psi - psi)):.2e}"

    params = parameters(nx, lx, **clfts_options(root / "pb", 400))
    params["electrostatics"]["psi_dt_scaling"] = 5.0
    sim = quiet(lambda: CLFTS(params, random_seed=1))
    with contextlib.redirect_stdout(io.StringIO()):
        sim.run({"A": np.zeros(64), "B": np.zeros(64)})
    error = np.max(np.abs(sim.electrostatics.psi - psi))
    assert error < 1e-5 * np.max(np.abs(psi)), f"CL-FTS mean field: max |psi - PB| = {error:.2e}"


def hamiltonian(sim, w_aux):
    phi, _ = sim.compute_concentrations(w_aux)
    phi_smeared = sim.smearing.apply_to_dict(phi)
    partitions = [sim.prop_solver.get_partition_function(p) for p in range(sim.prop_solver.get_n_polymer_types())]
    h = sim.mpt.compute_hamiltonian(sim.prop_solver.molecules, w_aux, partitions, sim.cb,
                                    include_const_term=True, phi=phi_smeared)
    return h + sim.electrostatics.hamiltonian(), phi_smeared


def forces(root):
    nx, lx = [10, 8], [2.0, 1.6]
    ions = {"Na": {"valence": 1.0, "concentration": 0.02, "smearing_length": 0.1},
            "Cl": {"valence": -1.0, "concentration": 0.02, "smearing_length": 0.15},
            "K": {"valence": 1.0, "chemical_potential": float(np.log(N * 0.01)), "smearing_length": 0.12},
            "Br": {"valence": -1.0, "chemical_potential": float(np.log(N * 0.01)), "smearing_length": 0.12}}
    params = parameters(nx, lx, charge_a=0.1, ions=ions, **clfts_options(root / "forces", 2, nbar=1e4))
    params["chi_n"] = {"A,B": 12.0}
    sim = quiet(lambda: CLFTS(params))
    es, n_grid, M = sim.electrostatics, int(np.prod(nx)), 2
    rng = np.random.default_rng(29)
    w_aux = rng.normal(0.0, 5.0, (M, n_grid)) + 1j * rng.normal(0.0, 1.0, (M, n_grid))
    sim.mpt.pin_dropped_fields(w_aux)
    es.psi = 1.0 + rng.normal(0.0, 1.0, n_grid) + 1j * rng.normal(0.0, 1.0, n_grid)
    _, phi_smeared = quiet(lambda: hamiltonian(sim, w_aux))
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


def restart(root):
    nx, lx = [10, 8], [2.0, 1.6]

    def run(directory, steps, continue_from=None):
        params = parameters(nx, lx, charge_a=0.1, **clfts_options(directory, steps, nbar=1e4))
        params["recording"].update(checkpoint_interval=1, statistics_save_interval=2)
        sim = quiet(lambda: CLFTS(params, random_seed=7))
        sim.draw_architectures = False
        rng = np.random.default_rng(3)
        with contextlib.redirect_stdout(io.StringIO()):
            if continue_from is None:
                sim.run(initial_fields={m: rng.normal(0.0, 1.0, int(np.prod(nx))) for m in "AB"})
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


def periodic(root):
    nx, lx = [32], [4.0]
    electrostatics = {"charges": {"A": 0.2}, "coupling": 500.0, "fixed_charge": 0.5 * layers(nx, lx),
                      "ions": {"Na": {"valence": 1.0, "concentration": 0.01, "smearing_length": 0.1},
                               "Cl": {"valence": -1.0, "concentration": 0.01, "smearing_length": 0.15}}}
    x = np.arange(nx[0]) / nx[0]

    def scft(length, fields=None, psi=None):
        params = parameters(nx, [length], **scft_options(tolerance=1e-10))
        params.update(bc=["periodic"] * 2, chi_n={"A,B": 15.0}, electrostatics=electrostatics)
        sim = quiet(lambda: SCFT(params))
        fields = fields or {"A": 3 * np.cos(2 * np.pi * x), "B": -3 * np.cos(2 * np.pi * x)}
        return sim, quiet(lambda: sim.run(initial_fields=fields, initial_psi=psi, return_result=True))
    sim, result = scft(lx[0])
    assert result.converged
    w = np.array([result.w[m] for m in sim.monomer_types])
    quiet(lambda: sim.compute_concentrations(w))
    derivative = sim.compute_stress(w, result.phi)[0]
    delta, energies = 1e-4, []
    for sign in (1.0, -1.0):
        energies.append(scft(lx[0] + sign * delta, dict(result.w), result.psi)[1].free_energy_absolute)
    expected = (energies[0] - energies[1]) / (2 * delta)
    assert abs(derivative - expected) < 1e-5 * abs(expected), f"SCFT stress {derivative:.10e}, dF/dL {expected:.10e}"

    # CL-FTS: the operator without the ideal terms against -dH/dV - d ln N/dV at fixed grid values
    nx, lx, C = [12, 10], [2.0, 1.8], 100.0
    rng = np.random.default_rng(5)
    n_grid = int(np.prod(nx))
    fixed = 0.3 * rng.normal(0.0, 1.0, n_grid)
    fixed -= fixed.mean()

    def clfts(scale):
        params = parameters(nx, [length * scale ** 0.5 for length in lx], charge_a=0.1,
                            **clfts_options(root / "pressure", 2, nbar=C ** 2))
        params["bc"] = ["periodic"] * 4
        params["electrostatics"]["fixed_charge"] = fixed
        params["recording"]["pressure"] = True
        return quiet(lambda: CLFTS(params))
    sim = clfts(1.0)
    M = 2
    w_aux = np.array([rng.normal(0, 2.0, n_grid) + 1j * rng.normal(0, 0.5, n_grid) for _ in range(M)])
    psi = rng.normal(0, 0.3, n_grid) + 1j * rng.normal(0, 0.3, n_grid)
    volume = sim.prop_solver.get_volume()
    ions = np.array([N * ion["concentration"] for ion in sim.electrostatics.ions])

    def state(s):
        pinned = w_aux.copy()
        s.mpt.pin_dropped_fields(pinned)
        s.electrostatics.psi = psi
        phi, _ = s.compute_concentrations(pinned)
        es = s.electrostatics
        log_q = np.log(s.prop_solver.get_partition_function(0))
        _, ion_q = es.ion_densities()
        log_ion_q = np.log(np.asarray(ion_q, dtype=complex))
        h_fields = sum(s.mpt.h_coef_mu2[i] * np.mean(pinned[i] ** 2) + s.mpt.h_coef_mu1[i] * np.mean(pinned[i])
                       for i in range(M)) + s.mpt.h_const
        # psi and the fixed charge: H_el without the ions, whose term -N c_s ln Q_s has fixed numbers
        h_volume = h_fields + es.hamiltonian() + np.sum(ions * log_ion_q)
        return log_q, log_ion_q, h_volume, phi, pinned
    _, _, _, phi, pinned = quiet(lambda: state(sim))
    operator = sim.pressure(pinned, phi) - 1.0 - np.sum(ions)

    def dilated(scale):
        s = clfts(scale)
        log_q, log_ion_q, h_volume, _, _ = quiet(lambda: state(s))
        v = s.prop_solver.get_volume()
        h = C * (volume * (-log_q - np.sum(ions * log_ion_q)) + v * h_volume)
        es = s.electrostatics
        modes = es.screened_k_sq > 0
        log_norm = -0.5 * (s.mpt.R + s.mpt.I) * n_grid * np.log(v) - 0.5 * np.sum(np.log(v * es.screened_k_sq[modes]))
        return h, log_norm, v
    (h_plus, n_plus, v_plus), (h_minus, n_minus, v_minus) = dilated(1 + 1e-5), dilated(1 - 1e-5)
    expected = -((h_plus - h_minus) + (n_plus - n_minus)) / (v_plus - v_minus) / C
    assert abs(operator - expected) < 1e-8 * abs(expected), f"pressure {operator:.10f}, expected {expected:.10f}"


def closed_scft(root):
    nx, lx = [32], [3.0]
    fixed = layers(nx, lx)
    ions = {"Na": {"valence": 1.0, "concentration": 0.02, "smearing_length": 0.12},
            "Cl": {"valence": -1.0, "concentration": 0.02, "smearing_length": 0.15}}
    x = (np.arange(nx[0]) + 0.5) / nx[0]

    def run(length, initial_fields, initial_psi=None):
        params = parameters(nx, [length], charge_a=0.2, ions=ions, **scft_options(max_iter=4000))
        params["chi_n"] = {"A,B": 25.0}
        params["electrostatics"]["charges"]["B"] = -0.1
        params["electrostatics"]["fixed_charge"] = fixed
        sim = quiet(lambda: SCFT(params))
        result = quiet(lambda: sim.run(initial_fields=initial_fields, initial_psi=initial_psi, return_result=True))
        assert result.converged, "closed-box SCFT did not converge"
        return sim, result
    sim, result = run(lx[0], {"A": 3 * np.cos(2 * np.pi * x), "B": -3 * np.cos(2 * np.pi * x)})
    w = np.array([result.w[m] for m in sim.monomer_types])
    quiet(lambda: sim.compute_concentrations(w))
    stress = sim.compute_stress(w, result.phi)[0]
    eps = 1e-4
    f_plus, f_minus = (run(lx[0] + sign * eps, dict(result.w), result.psi)[1].free_energy_absolute for sign in (1, -1))
    derivative = (f_plus - f_minus) / (2 * eps)
    assert abs(stress - derivative) < 1e-6 * max(1.0, abs(derivative)), \
        f"closed-box stress {stress:.10f}, dF/dL {derivative:.10f}"

    # A charged brush grafted to the low wall among free B chains, with Na+ Cl- and counterions
    params = parameters(nx, lx, charge_a=0.1, **scft_options(max_iter=4000, tolerance=1e-9))
    params["chi_n"] = {"A,B": 5.0}
    params["electrostatics"]["fixed_charge"] = None
    params["polymers"] = [{"volume_fraction": 0.3, "blocks": [{"type": "A", "length": 1.0, "v": 0, "u": 1}],
                           "initial_conditions": {0: "G"}},
                          {"volume_fraction": 0.7, "blocks": [{"type": "B", "length": 1.0}]}]
    q_init = np.zeros(nx[0])
    q_init[0] = nx[0] / lx[0]
    sim = quiet(lambda: SCFT(params))
    result = quiet(lambda: sim.run(initial_fields={"A": np.zeros(nx[0]), "B": np.zeros(nx[0])}, q_init={"G": q_init},
                                   return_result=True))
    error = sim.electrostatics.poisson_error(sim.smearing.apply_to_dict(result.phi), sim.psi)
    assert result.converged and error < 1e-8, f"grafted chains: Poisson error {error:.2e}"


def main():
    if not any(p.startswith("cpu") for p in PlatformSelector.avail_platforms()):
        return 77
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name, check in (("inputs", inputs), ("Poisson-Boltzmann", poisson_boltzmann), ("forces", forces),
                            ("restart", restart), ("fixed charges in a periodic box", periodic),
                            ("closed-box SCFT", closed_scft)):
            check(root)
            print(f"[PASS] {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
