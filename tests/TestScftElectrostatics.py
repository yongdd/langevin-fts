#!/usr/bin/env python3
"""Mean-field electrostatics of SCFT: psi from the Poisson equation, point ions, Donnan potential.

AB diblocks with charged blocks in 1D periodic boxes (Gaussian smearing 0.05):
1. Input: bjerrum_length (SCFT has no N̄), psi_dt_scaling, a zero coupling, grand-canonical ions
   that cannot neutralize the system and a smearing without a length are rejected, and so is an
   initial_psi without electrostatics or with complex or non-finite values.
2. Zero valences reproduce SCFT without electrostatics.
3. Strong screening: E/(k^2 + kappa^2) -> E/kappa^2 turns the charges into the contact interaction
   chi_eff N = chi N - E (z_A - z_B)^2/(2 kappa^2). The free energy and the relaxed period of a
   lamella approach those of the neutral diblock with chi_eff as 1/kappa^2, and the Richardson
   extrapolation agrees with them.
4. The stress equals the derivative of the free energy with respect to the box length, with
   canonical point ions (unscreened) and with grand-canonical ions and screening.
5. Homogeneous states: grand-canonical salt gives the Donnan potential, a screened net charge
   psi_0 = E q/kappa^2, and both have the free energy of the homogeneous reference; canonical
   point ions add their ideal terms N c_s (ln N c_s - 1) to it, and their charge to q.
6. Ensembles: grand-canonical ions with the chemical potentials of a canonical lamella reach the
   same state, with the free energy F - sum_s N c_s mu_s; coexistence_quantities() refuses
   results with electrostatics.
7. Axis ordering: a fixed charge given in the axis order of nx (2D, axis_ordering ['b', 'a'])
   gives the state of the same input in the standard order.
8. Platforms: the lamella with canonical ions agrees on every available platform.
"""

import contextlib
import io
import os

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from polymerfts import SCFT, PlatformSelector, ValidationError
from cuda_part import cuda_part_missing

NX, N = 64, 50
X = np.arange(NX) / NX


def cpu_platform():
    return next(p for p in PlatformSelector.avail_platforms() if p.startswith("cpu"))


def parameters(chi_n, electrostatics=None, lx=4.3, relax_cell=False, chain_model="discrete", platform=None):
    params = {"nx": [NX], "lx": [lx], "ds": 1 / N, "chain_model": chain_model,
              "segment_lengths": {"A": 1.0, "B": 1.0}, "chi_n": {"A,B": chi_n},
              "polymers": [{"volume_fraction": 1.0, "blocks": [{"type": "A", "length": 0.5}, {"type": "B", "length": 0.5}]}],
              "smearing": {"type": "gaussian", "a_int": 0.05}, "platform": platform or cpu_platform(),
              "optimizer": {"name": "am", "max_hist": 20, "start_error": 1e-2, "mix_min": 0.1, "mix_init": 0.1},
              "relax_cell": relax_cell, "max_iter": 3000, "tolerance": 1e-10}
    if electrostatics is not None:
        params["electrostatics"] = electrostatics
    return params


def ions(amount="concentration", values=(0.02, 0.02)):
    return {"Na": {"valence": 1.0, amount: values[0], "smearing_length": 0.05},
            "Cl": {"valence": -1.0, amount: values[1], "smearing_length": 0.08}}


def run(params, initial_fields=None, initial_psi=None):
    initial_fields = initial_fields or {"A": 3.0 * np.cos(2 * np.pi * X), "B": -3.0 * np.cos(2 * np.pi * X)}
    with contextlib.redirect_stdout(io.StringIO()):
        sim = SCFT(params)
        result = sim.run(initial_fields=initial_fields, initial_psi=initial_psi, return_result=True)
    assert result.converged, "SCFT did not converge"
    return sim, result


def expect_validation_error(make, message):
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            make()
    except ValidationError as error:
        assert message in str(error), f"expected '{message}', got '{error}'"
        return
    raise AssertionError(f"expected a ValidationError with '{message}'")


def inputs():
    expect_validation_error(lambda: SCFT(parameters(18.0, {"charges": {"A": 0.2}, "bjerrum_length": 0.01,
                                                          "ions": ions()})), "SCFT has no")
    expect_validation_error(lambda: SCFT(parameters(18.0, {"charges": {"A": 0.2}, "coupling": 2e3, "psi_dt_scaling": 1.0,
                                                          "ions": ions()})), "CL-FTS option")
    expect_validation_error(lambda: SCFT(parameters(18.0, {"charges": {"A": 0.2}, "coupling": 0.0, "ions": ions()})),
                            "positive 'electrostatics.coupling'")
    # Grand-canonical cations cannot neutralize the positive charge of A
    cations = {"Na": {"valence": 1.0, "chemical_potential": 0.0, "smearing_length": 0.05}}
    expect_validation_error(lambda: SCFT(parameters(18.0, {"charges": {"A": 0.2}, "coupling": 2e3, "ions": cations})),
                            "cannot neutralize")
    # The smearing is checked before the charges use its length
    params = parameters(18.0, {"charges": {"A": 0.2}, "coupling": 2e3, "ions": ions()})
    params["smearing"] = {"type": "gaussian"}
    expect_validation_error(lambda: SCFT(params), "smearing.a_int")
    # initial_psi: only with electrostatics, real and finite
    fields = {"A": np.zeros(NX), "B": np.zeros(NX)}
    expect_validation_error(lambda: SCFT(parameters(18.0)).run(initial_fields=fields, initial_psi=np.zeros(NX)),
                            "initial_psi needs")
    charged = parameters(18.0, {"charges": {"A": 0.2}, "coupling": 2e3, "ions": ions()})
    for psi in (np.ones(NX) + 1j, np.full(NX, np.nan), np.zeros(NX + 1)):
        expect_validation_error(lambda: SCFT(charged).run(initial_fields=fields, initial_psi=psi), "real, finite")


def neutral_reduction():
    _, plain = run(parameters(18.0))
    _, zero = run(parameters(18.0, {"charges": {"A": 0.0}, "coupling": 2e3}))
    assert abs(zero.free_energy - plain.free_energy) < 1e-12 and zero.iterations == plain.iterations, \
        f"zero valences: {zero.free_energy} ({zero.iterations}) against {plain.free_energy} ({plain.iterations})"
    assert np.max(np.abs(zero.psi)) == 0.0, "psi of neutral chains"


def strong_screening():
    """Continuous chains, incompressible: chi N = 25 with E (z_A - z_B)^2/(2 kappa^2) = 5 against chi N = 20."""
    def lamella(chi_n, electrostatics=None):
        _, result = run(parameters(chi_n, electrostatics, lx=4.0, relax_cell=True, chain_model="continuous"))
        return result.free_energy, result.lx[0]
    f_neutral, lx_neutral = lamella(20.0)
    values = []
    for screening_length in (0.01, 0.005):
        coupling = 10.0 / screening_length ** 2
        values.append(lamella(25.0, {"charges": {"A": 0.5, "B": -0.5}, "coupling": coupling,
                                     "screening_length": screening_length}))
    (f1, lx1), (f2, lx2) = values
    ratio = (f1 - f_neutral) / (f2 - f_neutral)
    assert 3.9 < ratio < 4.1, f"free energy difference does not scale as 1/kappa^2 (ratio {ratio:.3f})"
    f_limit, lx_limit = f2 - (f1 - f2) / 3, lx2 - (lx1 - lx2) / 3
    assert abs(f_limit - f_neutral) < 1e-5 and abs(lx_limit - lx_neutral) < 1e-5 * lx_neutral, \
        f"limit: F {f_limit:.8f} against {f_neutral:.8f}, lx {lx_limit:.8f} against {lx_neutral:.8f}"


def stress():
    cases = (("canonical ions", {"charges": {"A": 0.2}, "coupling": 2e3, "ions": ions()}),
             ("grand-canonical ions with screening",
              {"charges": {"A": 0.2, "B": -0.05}, "coupling": 2e3, "screening_length": 0.5,
               "ions": ions("chemical_potential", (np.log(N * 0.03), np.log(N * 0.03)))}))
    for name, electrostatics in cases:
        sim, result = run(parameters(18.0, electrostatics))
        w = np.array([result.w[m] for m in sim.monomer_types])
        with contextlib.redirect_stdout(io.StringIO()):
            sim.compute_concentrations(w)
        derivative = sim.compute_stress(w, result.phi)[0]
        delta, energies = 1e-4, []
        for sign in (1.0, -1.0):
            _, shifted = run(parameters(18.0, electrostatics, lx=4.3 + sign * delta), dict(result.w), result.psi)
            energies.append(shifted.free_energy_absolute)
        expected = (energies[0] - energies[1]) / (2 * delta)
        assert abs(derivative - expected) < 1e-5 * abs(expected), f"{name}: stress {derivative:.10e}, dF/dL {expected:.10e}"


def homogeneous():
    # Donnan: 0.1 + c0 (x - 1/x) = 0 for the A charge 0.2 at phi_A = 0.5, x = exp(-psi_0/N)
    c0 = 0.02
    sim, result = run(parameters(5.0, {"charges": {"A": 0.2}, "coupling": 2e3,
                                       "ions": ions("chemical_potential", (np.log(N * c0),) * 2)}))
    x = (-0.1 / c0 + np.sqrt((0.1 / c0) ** 2 + 4.0)) / 2.0
    assert np.max(np.abs(result.psi + N * np.log(x))) < 1e-8 * N, "Donnan potential"
    numbers = sim.electrostatics.ion_number_densities(sim.psi)
    assert np.allclose(numbers, [N * c0 * x, N * c0 / x], rtol=1e-8), "Donnan ion numbers"
    assert abs(result.free_energy) < 1e-10, f"Donnan free energy {result.free_energy}"
    # Screened net charge q = 0.1: psi_0 = E q/kappa^2, F = E q^2/(2 kappa^2) on top of Flory-Huggins
    coupling, screening_length = 2e3, 0.5
    _, result = run(parameters(5.0, {"charges": {"A": 0.2}, "coupling": coupling, "screening_length": screening_length}))
    assert np.max(np.abs(result.psi - coupling * 0.1 * screening_length ** 2)) < 1e-8, "screened psi_0"
    assert abs(result.free_energy) < 1e-10, f"screened free energy {result.free_energy}"
    assert abs(result.free_energy_homogeneous - (5.0 / 4 + 0.5 * coupling * 0.01 * screening_length ** 2
                                                 + np.log(1.0) - 1.0)) < 1e-10, "homogeneous free energy"
    # Canonical salt with counterions (0.1 Cl- for the A charge): psi_0 = 0 and the ideal terms of the ions
    def ideal(*concentrations):
        return sum(N * c * (np.log(N * c) - 1.0) for c in concentrations)
    _, result = run(parameters(5.0, {"charges": {"A": 0.2}, "coupling": coupling, "ions": ions()}))
    assert np.max(np.abs(result.psi)) < 1e-8, "psi_0 of a neutral canonical system"
    assert abs(result.free_energy) < 1e-10, f"free energy with canonical ions {result.free_energy}"
    assert abs(result.free_energy_homogeneous - (5.0 / 4 + np.log(1.0) - 1.0 + ideal(0.02, 0.12))) < 1e-10, \
        "homogeneous free energy with canonical ions"
    # Screened, without counterions: the ions add their charge to q = 0.1 + 0.03 - 0.01
    _, result = run(parameters(5.0, {"charges": {"A": 0.2}, "coupling": coupling, "screening_length": screening_length,
                                     "ions": ions(values=(0.03, 0.01)), "counterion": None}))
    assert np.max(np.abs(result.psi - coupling * 0.12 * screening_length ** 2)) < 1e-8, "screened psi_0 with ions"
    assert abs(result.free_energy) < 1e-10, f"screened free energy with canonical ions {result.free_energy}"
    assert abs(result.free_energy_homogeneous - (5.0 / 4 + 0.5 * coupling * 0.12 ** 2 * screening_length ** 2
                                                 + np.log(1.0) - 1.0 + ideal(0.03, 0.01))) < 1e-10, \
        "homogeneous free energy with screened canonical ions"


def ensembles():
    sim, canonical = run(parameters(18.0, {"charges": {"A": 0.2}, "coupling": 2e3, "ions": ions()}))
    mu = canonical.ion_chemical_potentials
    numbers = sim.electrostatics.ion_number_densities(sim.psi)
    sim, grand = run(parameters(18.0, {"charges": {"A": 0.2}, "coupling": 2e3, "ions": ions("chemical_potential", mu)}))
    assert np.max(np.abs(grand.phi["A"] - canonical.phi["A"])) < 1e-8, "concentrations"
    assert np.allclose(sim.electrostatics.ion_number_densities(sim.psi), numbers, rtol=1e-8), "numbers of ions"
    expected = canonical.free_energy_absolute - float(np.dot(numbers, mu))
    assert abs(grand.free_energy_absolute - expected) < 1e-8, \
        f"grand free energy {grand.free_energy_absolute:.10f} against {expected:.10f}"
    try:
        canonical.coexistence_quantities()
    except ValueError as error:
        assert "electrostatics" in str(error), str(error)
    else:
        raise AssertionError("coexistence_quantities() accepted a result with electrostatics")


def axis_ordering():
    """Neutral diblock with Na+ Cl- and a fixed charge that varies along both axes, nx [16, 8]."""
    xa, xb = np.meshgrid((np.arange(16) + 0.5) / 16, (np.arange(8) + 0.5) / 8, indexing="ij")
    fixed = 0.3 * np.cos(2 * np.pi * xa) + 0.1 * np.cos(2 * np.pi * xb)

    def params(nx, lx, fixed_charge):
        return {"nx": nx, "lx": lx, "ds": 1 / 20, "chain_model": "discrete",
                "segment_lengths": {"A": 1.0, "B": 1.0}, "chi_n": {"A,B": 12.0},
                "polymers": [{"volume_fraction": 1.0, "blocks": [{"type": "A", "length": 0.5}, {"type": "B", "length": 0.5}]}],
                "smearing": {"type": "gaussian", "a_int": 0.1}, "platform": cpu_platform(),
                "optimizer": {"name": "am", "max_hist": 20, "start_error": 1e-2, "mix_min": 0.1, "mix_init": 0.1},
                "relax_cell": False, "max_iter": 3000, "tolerance": 1e-10,
                "electrostatics": {"charges": {"A": 0.1, "B": -0.1}, "coupling": 300.0, "fixed_charge": fixed_charge,
                                   "ions": ions(values=(0.05, 0.05))}}
    zero = {"A": np.zeros(128), "B": np.zeros(128)}
    _, standard = run(params([16, 8], [3.0, 2.0], fixed.ravel()), zero)
    transposed = params([8, 16], [2.0, 3.0], fixed.T.ravel())
    transposed["axis_ordering"] = ["b", "a"]
    _, ordered = run(transposed, zero)
    assert abs(ordered.free_energy - standard.free_energy) < 1e-10 and np.max(np.abs(ordered.psi - standard.psi)) < 1e-8, \
        f"axis ordering: F {ordered.free_energy:.12f} against {standard.free_energy:.12f}"


def platforms():
    electrostatics = {"charges": {"A": 0.2}, "coupling": 2e3, "ions": ions()}
    _, reference = run(parameters(18.0, electrostatics))
    for platform in PlatformSelector.avail_platforms():
        _, result = run(parameters(18.0, electrostatics, platform=platform))
        assert abs(result.free_energy - reference.free_energy) < 1e-9 and \
            np.max(np.abs(result.psi - reference.psi)) < 1e-7 and \
            np.allclose(result.ion_chemical_potentials, reference.ion_chemical_potentials, rtol=0, atol=1e-9), \
            f"{platform}: F {result.free_energy:.12f} against {reference.free_energy:.12f}"
        print(f"  {platform}: F = {result.free_energy:.10f}")


def main():
    if not any(p.startswith("cpu") for p in PlatformSelector.avail_platforms()):
        return 77
    for name, check in (("inputs", inputs), ("neutral reduction", neutral_reduction),
                        ("strong screening", strong_screening), ("stress", stress),
                        ("homogeneous states", homogeneous), ("ensembles", ensembles), ("axis ordering", axis_ordering),
                        ("platforms", platforms)):
        check()
        print(f"[PASS] {name}")
    return 1 if cuda_part_missing() else 0


if __name__ == "__main__":
    raise SystemExit(main())
