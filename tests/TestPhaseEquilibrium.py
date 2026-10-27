#!/usr/bin/env python3
"""CE thermodynamics: FH limits, field means, composition derivatives and binodals."""

import contextlib
import copy
import io
import os
import tempfile
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"

import numpy as np
from polymerfts import (SCFT, PlatformSelector, coexistence_quantities,
                        homogeneous_thermodynamics, find_binary_coexistence)


def test_common_tangent():
    def phase(x):
        return homogeneous_thermodynamics([1-x, x], [1.0, 2.0], np.eye(2), [[0, 2], [2, 0]])

    kwargs = dict(initial_compositions=[0.06, 0.85], composition_bounds=((0.01, 0.25), (0.65, 0.98)),
                  tolerance=1e-11)
    result = find_binary_coexistence(phase, phase, **kwargs)
    np.testing.assert_allclose(result["compositions"], [0.059566504584, 0.849059139039], rtol=0, atol=1e-10)
    other = find_binary_coexistence(phase, phase, reference_polymer=1, **kwargs)
    np.testing.assert_allclose(other["compositions"], result["compositions"], rtol=0, atol=1e-10)
    for x in result["compositions"]:
        state = phase(x)
        shifted = copy.deepcopy(state)
        shifted["chemical_potentials"] = (np.array(state["chemical_potentials"])
                                           + 3.7*np.array(state["polymer_lengths"])).tolist()
        a, b = coexistence_quantities(state), coexistence_quantities(shifted)
        np.testing.assert_allclose(a["exchange_chemical_potentials"], b["exchange_chemical_potentials"], rtol=0, atol=1e-14)
        np.testing.assert_allclose(a["reduced_grand_potential"], b["reduced_grand_potential"], rtol=0, atol=1e-14)

    # A change of species standard states is linear in composition and cannot move binodals.
    def new_standard(x):
        state = phase(x)
        constants = np.array([0.9, -0.4])
        offset = constants @ np.array(state["polymer_volume_fractions"])
        state["free_energy_absolute"] += offset
        state["free_energy_homogeneous"] += offset
        state["chemical_potentials"] = (np.array(state["chemical_potentials"])
                                         + constants*np.array(state["polymer_lengths"])).tolist()
        return state
    shifted_result = find_binary_coexistence(new_standard, new_standard, **kwargs)
    np.testing.assert_allclose(shifted_result["compositions"], result["compositions"], rtol=0, atol=1e-10)

    # Boundary names are case-insensitive, as in the solver
    state = phase(0.2)
    state["params"] = dict(state.get("params", {}), bc=["Periodic", "PERIODIC"])
    coexistence_quantities(state)

    # Reject failed or compressible states, missing species and incorrect callback compositions.
    for updates in ({"converged": False}, {"params": {"zeta_n": 10}},
                    {"params": {"bc": ["reflecting", "reflecting"]}},
                    {"free_energy_absolute": None}):
        state = phase(0.2)
        state.update(updates)
        try:
            coexistence_quantities(state)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid state accepted for bulk coexistence")
    endpoint = phase(0.0)
    assert endpoint["free_energy"] == 0.0
    assert np.isfinite(endpoint["free_energy_absolute"]) and np.isfinite(endpoint["grand_potential"])
    try:
        coexistence_quantities(endpoint)
    except ValueError:
        pass
    else:
        raise AssertionError("Pure endpoint accepted with finite exchange mu")
    try:
        find_binary_coexistence(lambda x: phase(0.2), phase, **kwargs)
    except ValueError:
        pass
    else:
        raise AssertionError("Incorrect callback composition accepted")
    try:
        find_binary_coexistence(phase, phase, [0.1, 0.1])
    except RuntimeError:
        pass
    else:
        raise AssertionError("Coincident compositions accepted as a binodal")


def test_scft():
    cpu = [p for p in PlatformSelector.avail_platforms() if p.startswith("cpu")]
    if not cpu:
        print("SCFT thermodynamics skipped: no CPU backend")
        return
    base = {
        "nx": [32], "lx": [2.0], "chain_model": "continuous", "ds": 0.02,
        "segment_lengths": {"A": 1.0, "B": 1.0}, "chi_n": {"A,B": 20.0},
        "platform": cpu[0], "relax_cell": False, "max_iter": 1500, "tolerance": 1e-11,
        "optimizer": {"name": "am", "max_hist": 20, "start_error": 1e-2, "mix_min": 0.1, "mix_init": 0.1},
    }

    def run(polymers, fields, chi=20.0):
        params = dict(base, polymers=polymers, chi_n={"A,B": chi})
        progress = []
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            calc = SCFT(params)
            result = calc.run(fields, return_result=True, on_iteration=lambda info: progress.append(info.energy))
        assert result.converged
        assert progress[-1] == result.free_energy
        assert "Free energy relative to the homogeneous state:" in output.getvalue()
        assert "Full canonical free energy" in output.getvalue()
        np.testing.assert_allclose(result.free_energy + result.free_energy_homogeneous,
                                   result.free_energy_absolute, rtol=0, atol=1e-12)
        return calc, result

    zeros = {"A": np.zeros(32), "B": np.zeros(32)}
    for model in ("continuous", "discrete"):
        base["chain_model"] = model
        for fraction in (0.3, 0.0):
            polymers = [{"volume_fraction": fraction, "blocks": [{"type": "A", "length": 0.5}]},
                        {"volume_fraction": 1-fraction, "blocks": [{"type": "B", "length": 2.0}]}]
            calc, result = run(polymers, zeros, chi=8.0)
            expected = homogeneous_thermodynamics([fraction, 1-fraction], [0.5, 2.0], np.eye(2), [[0, 8], [8, 0]])
            np.testing.assert_allclose(result.free_energy, expected["free_energy"], rtol=0, atol=1e-12)
            np.testing.assert_allclose(result.free_energy_absolute, expected["free_energy_absolute"], rtol=0, atol=1e-12)
            np.testing.assert_allclose(result.free_energy_homogeneous, result.free_energy_absolute, rtol=0, atol=1e-12)
            np.testing.assert_allclose(result.chemical_potentials, expected["chemical_potentials"], rtol=0, atol=1e-12)
            np.testing.assert_allclose(result.grand_potential, expected["grand_potential"], rtol=0, atol=1e-12)
    base["chain_model"] = "continuous"

    # The molecular entropy is per polymer species; random blocks enter the
    # monomer composition and field-mean correction with their fractions.
    polymers = [{"volume_fraction": 0.4, "blocks": [{"type": "A", "length": 0.3}, {"type": "B", "length": 0.7}]},
                {"volume_fraction": 0.6, "blocks": [{"type": "R", "length": 0.5, "monomer_fractions": {"A": 0.2, "B": 0.8}}]}]
    _, random = run(polymers, zeros, chi=8.0)
    expected = homogeneous_thermodynamics([0.4, 0.6], [1.0, 0.5], [[0.3, 0.7], [0.2, 0.8]], [[0, 8], [8, 0]])
    np.testing.assert_allclose(random.free_energy, expected["free_energy"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(random.free_energy_absolute, expected["free_energy_absolute"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(random.chemical_potentials, expected["chemical_potentials"], rtol=0, atol=1e-12)

    # A nonuniform diblock/homopolymer blend: g is the constrained derivative dF/dphi_h.
    wave = np.sqrt(2)*5*np.cos(2*np.pi*(np.arange(32)+0.5)/32)
    center = None
    derivative_errors = []
    # Density quadrature and derivatives of the RQM4 partition function agree
    # in the contour-continuum limit. Retain ds=0.02 and check refinement as well.
    for ds in (0.02, 0.01, 0.005):
        base["ds"] = ds
        energies = []
        for x in (0.15-1e-4, 0.15, 0.15+1e-4):
            polymers = [{"volume_fraction": 1-x, "blocks": [{"type": "A", "length": 0.5}, {"type": "B", "length": 0.5}]},
                        {"volume_fraction": x, "blocks": [{"type": "A", "length": 0.5}]}]
            calc, result = run(polymers, {"A": wave, "B": -wave})
            assert np.std(result.phi["A"]) > 0.1
            energies.append(result.free_energy_absolute)
            if x == 0.15:
                center = calc, result
        g = center[1].coexistence_quantities()["exchange_chemical_potentials"][1]
        derivative_errors.append(abs((energies[2]-energies[0])/2e-4-g))
    assert derivative_errors[1] < derivative_errors[0]/4
    assert derivative_errors[2] < derivative_errors[1]/4
    calc, result = center
    np.testing.assert_allclose(derivative_errors[-1], 0.0, rtol=0, atol=2e-6)

    # Independent shifts of monomer means must cancel in the reconstructed chemical potentials.
    shifted_w = calc.w + np.array([1.3, -0.8])[:, None]
    with contextlib.redirect_stdout(io.StringIO()):
        calc.prop_solver.compute_propagators(dict(zip(calc.monomer_types, shifted_w)))
    shifted_q = [calc.prop_solver.get_partition_function(p) for p in range(2)]
    np.testing.assert_allclose(calc.compute_chemical_potentials(shifted_w, shifted_q),
                               result.chemical_potentials, rtol=0, atol=1e-11)

    # Saved thermodynamics describe the final CE snapshot, even if the solver was subsequently used.
    import json
    calc.save_results("thermodynamics.json")
    saved = json.loads(Path("thermodynamics.json").read_text())
    np.testing.assert_allclose(saved["chemical_potentials"], result.chemical_potentials, rtol=0, atol=0)
    np.testing.assert_allclose(saved["partition_functions"], result.partition_functions, rtol=0, atol=0)
    assert saved["free_energy"] == result.free_energy
    assert saved["free_energy_absolute"] == result.free_energy_absolute
    np.testing.assert_allclose(result.grand_potential,
                               result.free_energy_absolute - np.dot(np.array(result.polymer_volume_fractions)
                                                                   / result.polymer_lengths, result.chemical_potentials),
                               rtol=0, atol=1e-12)

    # A reduced physical basis must use the same weighted field means as the full grid.
    axis = (np.arange(8)+0.5)/8
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    field = np.cos(4*np.pi*x)+np.cos(4*np.pi*y)+np.cos(4*np.pi*z)
    field *= 5/np.std(field)
    p = dict(base, nx=[8, 8, 8], lx=[2.0, 2.0, 2.0], max_iter=1,
             polymers=[{"volume_fraction": 1.0, "blocks": [{"type": "A", "length": 0.3}, {"type": "B", "length": 0.7}]}])
    states = []
    for symmetry in (False, True):
        options = dict(p)
        if symmetry:
            options["space_group"] = {"symbol": "Im-3m", "hall_number": 529}
        with contextlib.redirect_stdout(io.StringIO()):
            calc = SCFT(options)
            states.append(calc.run({"A": field, "B": -field}, return_result=True))
    np.testing.assert_allclose(states[0].chemical_potentials, states[1].chemical_potentials, rtol=0, atol=1e-11)
    np.testing.assert_allclose(states[0].free_energy, states[1].free_energy, rtol=0, atol=1e-11)
    np.testing.assert_allclose(states[0].free_energy_absolute, states[1].free_energy_absolute, rtol=0, atol=1e-11)
    # The fields of a result are on the full grid, also with a space group
    np.testing.assert_allclose(calc.compute_chemical_potentials(states[1].w, states[1].partition_functions),
                               states[1].chemical_potentials, rtol=0, atol=1e-11)


if __name__ == "__main__":
    test_common_tangent()
    previous = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="polymerfts-ce-") as scratch:
        try:
            os.chdir(scratch)
            test_scft()
        finally:
            os.chdir(previous)
    print("CE thermodynamics and phase coexistence passed")
