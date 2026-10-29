#!/usr/bin/env python3
"""Exact smearing/WTMD aliases on anisotropic oblique grids.

A fixed one-period search missed the metric minimum on a 4x32 hexagonal grid,
made Gamma(k) differ from Gamma(-k), and broke the CLFTS Hamiltonian gradient.
The reference searches a wider window using Cartesian reciprocal vectors.
"""
import contextlib
import io
import os
import tempfile

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from polymerfts import CLFTS, PlatformSelector, PropagatorSolver
from polymerfts.smearing import Smearing
from polymerfts.wtmd import WTMD
from cuda_part import cuda_part_missing


def quiet(function, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return function(*args, **kwargs)


def reference_k_squared(nx, lx, angles):
    alpha, beta, gamma = np.deg2rad(angles)
    lattice = np.zeros((len(nx), len(nx)))
    lattice[0, 0] = lx[0]
    lattice[1, :2] = lx[1] * np.array([np.cos(gamma), np.sin(gamma)])
    if len(nx) == 3:
        cx = np.cos(beta)
        cy = (np.cos(alpha) - cx * np.cos(gamma)) / np.sin(gamma)
        lattice[2] = lx[2] * np.array([cx, cy, np.sqrt(1 - cx**2 - cy**2)])
    reciprocal = 2 * np.pi * np.linalg.inv(lattice).T
    modes = np.array(np.meshgrid(*[np.fft.fftfreq(n) * n for n in nx], indexing="ij")).reshape(len(nx), -1)
    result = np.full(int(np.prod(nx)), np.inf)
    shifts = [range(-6, 7) if n > 1 else (0,) for n in nx]
    for shift in np.array(np.meshgrid(*shifts, indexing="ij")).reshape(len(nx), -1).T:
        wavevectors = (modes + (shift * nx)[:, None]).T @ reciprocal
        result = np.minimum(result, np.sum(wavevectors**2, axis=1))
    return result.reshape(nx)


def aliases_and_transforms():
    cases = [([4, 32], [2.0, 2.0], [90.0, 90.0, 120.0]),
             ([4, 32, 4], [2.0, 2.0, 3.0], [90.0, 90.0, 120.0]),
             ([1, 16, 4], [1.6, 1.9, 2.2], [75.0, 100.0, 70.0])]
    for nx, lx, angles in cases:
        params = {"type": "gaussian", "a_int": 0.05}
        smearing = quiet(Smearing, nx, lx, params, angles=angles)
        reference = reference_k_squared(nx, lx, angles)
        np.testing.assert_allclose(smearing.k_sq, reference, rtol=1e-12, atol=1e-11)
        reverse = np.ix_(*[(-np.arange(n)) % n for n in nx])
        np.testing.assert_allclose(smearing.fourier, smearing.fourier[reverse], rtol=1e-14, atol=1e-14)
        field = np.random.default_rng(17).normal(0.0, 5.0, int(np.prod(nx)))
        real = smearing.apply(field)
        complex_value = smearing.apply(field.astype(complex))
        np.testing.assert_allclose(real, complex_value, rtol=1e-12, atol=1e-12)
        for platform in PlatformSelector.avail_platforms():
            for field_type in ("real", "complex"):
                solver = quiet(PropagatorSolver, nx=nx, lx=lx, angles=angles, ds=0.1,
                               bond_lengths={"A": 1.0}, bc=["periodic"] * (2 * len(nx)),
                               chain_model="continuous", platform=platform, field_type=field_type)
                native_k = solver.fourier.k_squared()
                selection = tuple(slice(0, n) for n in native_k.shape)
                np.testing.assert_allclose(native_k, reference[selection], rtol=1e-12, atol=1e-11)
                native = quiet(Smearing, nx, lx, params, angles=angles, transform=solver.fourier)
                value = native.apply(field if field_type == "real" else field.astype(complex))
                np.testing.assert_allclose(value, real, rtol=1e-12, atol=1e-12)


def lattice_derivatives():
    nx, lx, angles = [4, 32], [1.6, 1.9], [90.0, 90.0, 110.0]
    smearing = quiet(Smearing, nx, lx, {"type": "gaussian", "a_int": 0.05}, angles=angles)
    field = np.random.default_rng(9).normal(0.0, 5.0, int(np.prod(nx)))
    # H = <w Gamma*w>/2: stress(w, w/2) is its metric derivative.
    analytic = smearing.stress([field], [0.5 * field])[:3]
    epsilon = 1e-5
    numeric = []
    for slot in range(3):
        energies = []
        for sign in (1, -1):
            lengths, cell_angles = list(lx), list(angles)
            if slot < 2:
                lengths[slot] += sign * epsilon
            else:
                cell_angles[2] += sign * np.rad2deg(epsilon)
            smearing.set_box(lengths, cell_angles)
            energies.append(0.5 * np.mean(field * smearing.apply(field)))
        numeric.append((energies[0] - energies[1]) / (2 * epsilon))
    np.testing.assert_allclose(analytic, numeric, rtol=2e-7, atol=2e-8)
    # set_box must invalidate both multipliers and tied-alias derivative moments.
    smearing.set_box(lx, angles)
    np.testing.assert_allclose(smearing.stress([field], [0.5 * field])[:3], analytic, rtol=0, atol=0)


def wtmd_aliases():
    nx, lx, angles = [4, 32, 4], [20.0, 20.0, 3.0], [90.0, 90.0, 120.0]
    wtmd = quiet(WTMD, nx, lx, 10000, np.array([-15.0, 0.0]), [0], angles=angles)
    k_squared = reference_k_squared(nx, lx, angles)
    cutoff = 1 / (1 + np.exp(12 * (np.sqrt(k_squared) - wtmd.kc) / wtmd.kc))
    np.testing.assert_allclose(wtmd.fk, cutoff[:, :, :nx[2] // 2 + 1], rtol=1e-12, atol=1e-14)
    field = np.random.default_rng(17).normal(0.0, 5.0, nx)
    fields = np.stack([field.ravel(), np.zeros(field.size)])
    psi = wtmd.compute_order_parameter(0, fields)
    expected = np.sum(np.abs(np.fft.fftn(field))**wtmd.l * cutoff)**(1 / wtmd.l) / field.size
    np.testing.assert_allclose(psi, expected, rtol=1e-12, atol=0)
    wtmd.up[:] = 0.7
    force = np.zeros((1, field.size))
    quiet(wtmd.add_bias_to_langevin, psi, force)
    direction = np.random.default_rng(11).normal(size=field.size)
    epsilon = 1e-5
    displaced = []
    for sign in (1, -1):
        trial = fields.copy()
        trial[0] += sign * epsilon * direction
        displaced.append(wtmd.compute_order_parameter(0, trial))
    np.testing.assert_allclose(np.mean(force[0] * direction), 0.7 * (displaced[0] - displaced[1]) / (2 * epsilon),
                               rtol=1e-7, atol=1e-10)


def clfts_gradient(directory):
    nx = [4, 32]
    rng = np.random.default_rng(17)
    monomer_fields = rng.normal(0.0, 5.0, (2, int(np.prod(nx)))).astype(complex)
    direction = rng.normal(size=monomer_fields.shape)
    for platform in PlatformSelector.avail_platforms():
        params = {"nx": nx, "lx": [2.0, 2.0], "angles": [90.0, 90.0, 120.0],
                  "chain_model": "discrete", "ds": 1.0, "segment_lengths": {"A": 1.0, "B": 1.0},
                  "chi_n": {"A,B": 15.0}, "zeta_n": 100.0,
                  "polymers": [{"volume_fraction": 0.5, "blocks": [{"type": m, "length": 1.0}]} for m in ("A", "B")],
                  "smearing": {"type": "gaussian", "a_int": 0.05}, "platform": platform,
                  "langevin": {"nbar": 10000, "dt": 0.01, "max_step": 1}, "verbose_level": 0,
                  "recording": {"dir": directory, "checkpoint_interval": 1, "sample_interval": 1,
                                "statistics_save_interval": 1}}
        sim = quiet(CLFTS, params, random_seed=5)
        fields = sim.mpt.to_aux_fields(monomer_fields)

        def evaluate(candidate):
            phi, _ = quiet(sim.compute_concentrations, candidate)
            phi_smeared = sim.smearing.apply_to_dict(phi)
            partitions = [sim.prop_solver.get_partition_function(p) for p in range(2)]
            energy = sim.mpt.compute_hamiltonian(sim.prop_solver.molecules, candidate, partitions, sim.cb,
                                                include_const_term=True, phi=phi_smeared)
            force = sim.mpt.compute_func_deriv(candidate, phi_smeared, range(2))
            return energy, force, partitions

        epsilon = 1e-5
        plus, _, _ = evaluate(fields + epsilon * direction)
        minus, _, _ = evaluate(fields - epsilon * direction)
        energy, force, partitions = evaluate(fields)
        assert abs(energy.imag) < 1e-11 and np.max(np.abs(np.imag(partitions))) < 1e-11, platform
        force[sim.mpt.aux_fields_imag_idx] *= -1
        analytic = np.mean(np.sum(force * direction, axis=0))
        np.testing.assert_allclose(analytic, (plus - minus) / (2 * epsilon), rtol=1e-7, atol=1e-9)


if __name__ == "__main__":
    aliases_and_transforms()
    lattice_derivatives()
    wtmd_aliases()
    with tempfile.TemporaryDirectory(prefix="polymerfts_aliases_") as directory:
        clfts_gradient(directory)
    print("Test passed: exact smearing and WTMD aliases, lattice derivatives, and CLFTS gradient")
    raise SystemExit(int(cuda_part_missing()))
