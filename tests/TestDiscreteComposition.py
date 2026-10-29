#!/usr/bin/env python3
"""Discrete composition and field shifts use the lengths accepted by the solver.

The native discrete model snaps accepted contour-length roundoff to N*ds.
Analytic uniform states check the Python composition, the uniform saddle modes,
and chemical potentials in a shifted field gauge against that same model.
"""
import contextlib
import copy
import io
import os
import sys
import tempfile

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from scipy.io import loadmat
from polymerfts import CLFTS, LFTS, PlatformSelector, SCFT, ValidationError
from polymerfts.validation import (net_polymer_charge, open_charges, point_ion_concentrations,
                                   polymer_block_lengths, validate_scft_params)
from cuda_part import cuda_part_missing


def quiet(function, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return function(*args, **kwargs)


def cases():
    # Absolute length acceptance permits a visible relative correction here.
    yield ("short_homopolymer", 1e-6, {"A": 1.0}, {},
           [{"volume_fraction": 1.0, "blocks": [{"type": "A", "length": 1.0009e-6}]}],
           [1.0], [np.log(1e6)], [5e5])
    # Ordinary ds, multiple blocks, and opposite accepted length errors.
    yield ("diblock", 0.01, {"A": 1.0, "B": 1.0}, {"A,B": 10.0},
           [{"volume_fraction": 1.0,
             "blocks": [{"type": "A", "length": 0.5 + 4e-10},
                        {"type": "B", "length": 0.5 - 4e-10}]}],
           [0.5, 0.5], [5.0], [5.0, -4.0])
    # A random block must distribute the native length among its real monomers.
    yield ("random_blend", 0.01, {"A": 1.0, "B": 1.0}, {"A,B": 10.0},
           [{"volume_fraction": 0.5,
             "blocks": [{"type": "R", "length": 0.5 + 4e-10,
                         "monomer_fractions": {"A": 0.25, "B": 0.75}}]},
            {"volume_fraction": 0.5, "blocks": [{"type": "A", "length": 0.5}]}],
           [0.625, 0.375], [2.8125, 1.875], [5.0, -4.0])


def assert_input_lengths(sim, params):
    # The native canonicalization must not rewrite user/checkpoint parameters.
    for original, stored, chain in zip(params["polymers"], sim.params["polymers"], sim.polymers):
        expected = [block["length"] for block in original["blocks"]]
        assert [block["length"] for block in stored["blocks"]] == expected
        assert [block["length"] for block in chain["blocks"]] == expected


def check_scft(params, expected_mean, expected_mu, shifted_means):
    params = copy.deepcopy(params)
    params.update(optimizer={"name": "am", "max_hist": 5, "start_error": 0.01,
                             "mix_min": 0.1, "mix_init": 0.1},
                  relax_cell=False, max_iter=2, tolerance=1e-8)
    original = copy.deepcopy(params)
    sim = quiet(SCFT, params)
    quiet(sim.run, {monomer: np.zeros(4) for monomer in sim.monomer_types})
    assert params == original
    assert_input_lengths(sim, original)
    assert sim.error_level < sim.tolerance
    actual_phi = np.stack([sim.phi[monomer] for monomer in sim.monomer_types])
    np.testing.assert_allclose(sim.mean_concentrations(), expected_mean, rtol=0, atol=2e-13)
    np.testing.assert_allclose(np.mean(actual_phi, axis=1), expected_mean, rtol=0, atol=2e-13)
    # Include the uniform residual omitted from SCFT's reported convergence norm.
    saddle_fields = sim.matrix_chi @ actual_phi + sim.zeta_n * (np.sum(actual_phi, axis=0) - 1.0)
    np.testing.assert_allclose(sim.w, saddle_fields, rtol=0, atol=5e-11)
    np.testing.assert_allclose(sim.compute_chemical_potentials(), expected_mu, rtol=0, atol=2e-12)

    # Recompute Q in an arbitrary constant field gauge. Its exponential shift
    # uses the native contour length and must cancel the chemical-potential correction.
    shifted_fields = np.repeat(np.asarray(shifted_means)[:, None], 4, axis=1)
    quiet(sim.compute_concentrations, shifted_fields)
    partitions = [sim.prop_solver.get_partition_function(p)
                  for p in range(sim.prop_solver.get_n_polymer_types())]
    np.testing.assert_allclose(sim.compute_chemical_potentials(shifted_fields, partitions),
                               expected_mu, rtol=0, atol=2e-12)
    assert sim.prop_solver.check_total_partition()


def check_lfts(params, expected_mean, directory):
    params = copy.deepcopy(params)
    params.update(langevin={"nbar": 1e8, "dt": 0.01, "max_step": 1},
                  compressor={"name": "am", "max_hist": 5, "start_error": 0.01,
                              "mix_min": 0.1, "mix_init": 0.1},
                  saddle={"max_iter": 2, "tolerance": 1e-8},
                  recording={"dir": directory, "checkpoint_interval": 1,
                             "sample_interval": 1, "statistics_save_interval": 1})
    original = copy.deepcopy(params)
    sim = quiet(LFTS, params, random_seed=5)
    fields = np.zeros((len(sim.monomer_types), 4))
    phi, hamiltonian, _, error = quiet(sim.find_saddle_point, fields)
    assert params == original
    assert_input_lengths(sim, original)
    assert np.isfinite(hamiltonian) and error < sim.saddle["tolerance"]
    np.testing.assert_allclose(sim.mean_concentrations(), expected_mean, rtol=0, atol=2e-13)
    actual_mean = [sim.cb.mean(phi[monomer]) for monomer in sim.monomer_types]
    np.testing.assert_allclose(actual_mean, expected_mean, rtol=0, atol=2e-13)
    # LFTS reports the spatial standard deviation; explicitly check the uniform mode too.
    derivative = sim.mpt.compute_func_deriv(fields, phi, sim.mpt.aux_fields_imag_idx)
    np.testing.assert_allclose(derivative, 0.0, rtol=0, atol=2e-12)
    assert sim.prop_solver.check_total_partition()


def electrostatic_cases():
    yield "neutral_short", "discrete", 1e-6, [1.0009e-6, 1e-6], [1e-6, 1e-6]
    yield "charged_short", "discrete", 1e-6, [2.0009e-6, 1e-6], [2e-6, 1e-6]
    yield "neutral_ordinary", "discrete", 0.01, [0.5 + 4e-10, 0.5], [0.5, 0.5]
    yield "continuous_control", "continuous", 0.01, [0.5 + 4e-10, 0.5], [0.5 + 4e-10, 0.5]


def charged_parameters(platform, chain_model, ds, lengths):
    params = {"nx": [4], "lx": [2.0], "chain_model": chain_model, "ds": ds,
              "segment_lengths": {"A": 1.0, "B": 1.0}, "chi_n": {"A,B": 2.0}, "zeta_n": 100.0,
              "polymers": [{"volume_fraction": 1.0,
                            "blocks": [{"type": "A", "length": lengths[0]},
                                       {"type": "B", "length": lengths[1]}]}],
              "smearing": {"type": "gaussian", "a_int": 0.1},
              "electrostatics": {"charges": {"A": 1.0, "B": -1.0}, "coupling": 1.0,
                                 "ions": {"X": {"valence": -1.0, "concentration": 0.0,
                                                "smearing_length": 0.1}}, "counterion": "X"},
              "platform": platform, "verbose_level": 0}
    if chain_model == "continuous":
        params["electrostatics"]["n_segments"] = 100
    return params


def check_counterions(platform, directory):
    for name, model, ds, lengths, effective in electrostatic_cases():
        params = charged_parameters(platform, model, ds, lengths)
        original = copy.deepcopy(params)
        np.testing.assert_array_equal(polymer_block_lengths(params, params["polymers"][0]), effective)
        expected_phi = np.array(effective) / sum(effective)
        expected_charge = expected_phi[0] - expected_phi[1]
        np.testing.assert_allclose(net_polymer_charge(params), expected_charge, rtol=0, atol=2e-16)
        np.testing.assert_allclose(point_ion_concentrations(params)["X"], expected_charge, rtol=0, atol=2e-16)
        assert not open_charges(params)
        assert params == original

        scft_params = copy.deepcopy(params)
        scft_params.update(optimizer={"name": "am", "max_hist": 5, "start_error": 0.01,
                                      "mix_min": 0.1, "mix_init": 0.1},
                           relax_cell=False, max_iter=5, tolerance=1e-8)
        sim = quiet(SCFT, scft_params)
        quiet(sim.run, {monomer: np.zeros(4) for monomer in "AB"})
        assert sim.error_level < sim.tolerance
        es = sim.electrostatics
        phi_smeared = sim.smearing.apply_to_dict(sim.phi)
        charge = es.charge_density(phi_smeared, sim.psi)
        # Include k=0 explicitly: the reported error projects out that mode in
        # an unscreened canonical system and previously hid incorrect counterions.
        full_residual = charge - es._fourier.backward(
            es.screened_k_sq * es._fourier.forward(sim.psi)) / es.coupling
        np.testing.assert_allclose(full_residual, 0.0, rtol=0, atol=2e-12)
        np.testing.assert_allclose(np.mean(charge), 0.0, rtol=0, atol=2e-13)
        np.testing.assert_allclose([np.mean(sim.phi[m]) for m in "AB"], expected_phi, rtol=0, atol=2e-13)
        np.testing.assert_allclose(es.ions[0]["concentration"], expected_charge, rtol=0, atol=2e-16)
        assert_input_lengths(sim, params)
        assert sim.prop_solver.check_total_partition()

        clfts_params = copy.deepcopy(params)
        clfts_directory = os.path.join(directory, name)
        clfts_params.update(langevin={"nbar": 1e8, "dt": 0.01, "max_step": 1},
                            recording={"dir": clfts_directory, "checkpoint_interval": 1,
                                       "sample_interval": 1, "statistics_save_interval": 1})
        sim = quiet(CLFTS, clfts_params, random_seed=19)
        sim.draw_architectures = False
        phi, _ = quiet(sim.compute_concentrations, np.zeros((2, 4), dtype=complex))
        es = sim.electrostatics
        charge = es.charge_density(sim.smearing.apply_to_dict(phi))
        np.testing.assert_allclose(es.force(charge), 0.0, rtol=0, atol=2e-12)
        np.testing.assert_allclose(np.mean(charge), 0.0, rtol=0, atol=2e-13)
        np.testing.assert_allclose(es.ions[0]["concentration"], expected_charge, rtol=0, atol=2e-16)
        assert es.zero_uniform_mode
        assert_input_lengths(sim, params)
        assert sim.prop_solver.check_total_partition()
        if name == "neutral_short":
            status = quiet(sim.run, {monomer: np.zeros(4) for monomer in "AB"})
            assert status["status"] == "completed" and status["last_completed_step"] == 1
            data = loadmat(os.path.join(clfts_directory, "fields_000001.mat"), squeeze_me=True)
            phi = {m: data[f"phi_{m}_real"] + 1j * data[f"phi_{m}_imag"] for m in "AB"}
            charge = es.charge_density(sim.smearing.apply_to_dict(phi))
            np.testing.assert_allclose(np.mean(charge), 0.0, rtol=0, atol=2e-13)
            assert_input_lengths(sim, params)
        assert params == original


def check_grand_canonical_charge(platform, directory):
    for charged in (False, True):
        # The first GCE chain is neutral after snapping when it has one bead of
        # each type. The control has two A beads and a separate negative GCE chain.
        params = charged_parameters(platform, "discrete", 1e-6,
                                    [2.0009e-6 if charged else 1.0009e-6, 1e-6])
        grand = params["polymers"][0]
        grand.pop("volume_fraction")
        grand["chemical_potential"] = 0.0
        params["segment_lengths"]["C"] = 1.0
        params["chi_n"].update({"A,C": 2.0, "B,C": 2.0})
        params["polymers"].insert(0, {"volume_fraction": 1.0,
                                      "blocks": [{"type": "C", "length": 1e-6}]})
        if charged:
            params["polymers"].append({"chemical_potential": 0.0,
                                        "blocks": [{"type": "B", "length": 1e-6}]})
        params["electrostatics"].pop("ions")
        params["electrostatics"].pop("counterion")
        params.update(langevin={"nbar": 1e8, "dt": 0.01, "max_step": 1},
                      recording={"dir": os.path.join(directory, "grand-" + str(charged)),
                                 "checkpoint_interval": 1, "sample_interval": 1,
                                 "statistics_save_interval": 1})
        original = copy.deepcopy(params)
        assert open_charges(params) is charged
        sim = quiet(CLFTS, params, random_seed=23)
        sim.draw_architectures = False
        assert sim.electrostatics.zero_uniform_mode == (not charged)
        fields = np.zeros((3, 4), dtype=complex)
        phi, _ = quiet(sim.compute_concentrations, fields)
        charge = sim.electrostatics.charge_density(sim.smearing.apply_to_dict(phi))
        np.testing.assert_allclose(np.mean(charge), 0.0, rtol=0, atol=2e-13)
        np.testing.assert_allclose(sim.electrostatics.force(charge), 0.0, rtol=0, atol=2e-12)
        assert sim.prop_solver.check_total_partition()
        assert params == original
        assert_input_lengths(sim, params)


def check_invalid_discrete_length():
    params = charged_parameters("auto", "discrete", 0.01, [0.505, 0.5])
    params.update(optimizer={"name": "am", "max_hist": 5, "start_error": 0.01,
                             "mix_min": 0.1, "mix_init": 0.1}, relax_cell=False)
    original = copy.deepcopy(params)
    try:
        validate_scft_params(params)
    except ValidationError as error:
        assert "polymers[0].blocks[0].length" in str(error), error
        assert "integer multiple" in str(error), error
    else:
        raise AssertionError("Invalid discrete length was not rejected during parameter validation")
    assert params == original


if __name__ == "__main__":
    check_invalid_discrete_length()
    for platform in PlatformSelector.avail_platforms():
        for name, ds, segments, chi_n, polymers, mean, mu, shifted in cases():
            params = {"nx": [4], "lx": [2.0], "chain_model": "discrete", "ds": ds,
                      "segment_lengths": segments, "chi_n": chi_n, "zeta_n": 100.0,
                      "polymers": polymers, "platform": platform, "verbose_level": 0}
            check_scft(params, mean, mu, shifted)
            with tempfile.TemporaryDirectory(prefix="polymerfts_discrete_composition_") as directory:
                check_lfts(params, mean, directory)
            print(platform, name, "passed")
        with tempfile.TemporaryDirectory(prefix="polymerfts_discrete_charge_") as directory:
            check_counterions(platform, directory)
            check_grand_canonical_charge(platform, directory)
        print(platform, "discrete charge validation and counterions passed")
    print("Test passed: native discrete composition and uniform field corrections")
    sys.exit(int(cuda_part_missing()))
