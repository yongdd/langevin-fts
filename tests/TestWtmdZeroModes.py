#!/usr/bin/env python3
"""WTMD ell>=1 at zero spectral coefficients, including a nonzero uniform field.

Zero bias must add exactly zero. With bias, ell>1 has a zero derivative at each
zero coefficient; ell=1 selects its zero subgradient. Exponents below one are
not differentiable there and are rejected by both public construction paths.
"""
import contextlib
import io
import os
import tempfile

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from polymerfts import LFTS, PlatformSelector, PropagatorSolver, ValidationError
from polymerfts.validation import validate_lfts_params
from polymerfts.wtmd import WTMD
from cuda_part import cuda_part_missing


def quiet(function, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return function(*args, **kwargs)


def parameters(directory, ell):
    return {"nx": [4, 4, 4], "lx": [2.0, 2.0, 2.0], "chain_model": "continuous", "ds": 0.1,
            "segment_lengths": {"A": 1.0, "B": 1.0}, "chi_n": {"A,B": 15.0},
            "polymers": [{"volume_fraction": 1.0, "blocks": [{"type": "A", "length": 0.5},
                                                             {"type": "B", "length": 0.5}]}],
            "langevin": {"nbar": 1e8, "dt": 0.01, "max_step": 1}, "verbose_level": 0,
            "compressor": {"name": "am", "max_hist": 20, "start_error": 0.5, "mix_min": 0.01, "mix_init": 0.01},
            "saddle": {"max_iter": 100, "tolerance": 1e-6},
            "recording": {"dir": directory, "checkpoint_interval": 1, "sample_interval": 1,
                          "statistics_save_interval": 1}, "wtmd": {"ell": ell}}


def zero_modes(fourier=None):
    nx = [4, 4, 4]
    n_grid = int(np.prod(nx))
    for ell in (1, 1.5, 2, 4):
        wtmd = quiet(WTMD, nx, [2.0] * 3, 10000, np.array([-15.0, 0.0]), [0], l=ell, fourier=fourier)
        fields = np.stack([np.ones(n_grid), np.zeros(n_grid)])
        psi = wtmd.compute_order_parameter(0, fields)
        assert psi > 0 and np.any(wtmd.w_aux_k == 0), ell
        force = np.random.default_rng(7).normal(size=(1, n_grid))
        original = force.copy()
        with np.errstate(divide="raise", invalid="raise", over="raise"):
            quiet(wtmd.add_bias_to_langevin, psi, force)
        np.testing.assert_array_equal(force, original)

        wtmd.up[:] = 0.7
        force[:] = 0
        with np.errstate(divide="raise", invalid="raise", over="raise"):
            quiet(wtmd.add_bias_to_langevin, psi, force)
        expected = 0.7 * wtmd.fk[0, 0, 0]**(1 / ell)
        np.testing.assert_allclose(force, expected, rtol=1e-12, atol=1e-12)

        psi = wtmd.compute_order_parameter(0, np.zeros_like(fields))
        force[:] = 0
        quiet(wtmd.add_bias_to_langevin, psi, force)
        np.testing.assert_array_equal(force, np.zeros_like(force))


if __name__ == "__main__":
    for construct in (lambda: WTMD([4, 4, 4], [2.0] * 3, 10000, np.array([-15.0, 0.0]), [0], l=0.5),
                      lambda: validate_lfts_params(parameters("unused", 0.5))):
        try:
            quiet(construct)
        except ValidationError as error:
            assert "ell" in str(error) and "at least 1" in str(error)
        else:
            raise AssertionError("ell below one was accepted")
    zero_modes()
    for platform in PlatformSelector.avail_platforms():
        solver = quiet(PropagatorSolver, nx=[4, 4, 4], lx=[2.0] * 3, ds=0.1,
                       bond_lengths={"A": 1.0}, bc=["periodic"] * 6,
                       chain_model="continuous", platform=platform)
        zero_modes(solver.fourier)
        with tempfile.TemporaryDirectory(prefix="polymerfts_wtmd_zero_") as directory:
            params = parameters(directory, 1)
            params["platform"] = platform
            sim = quiet(LFTS, params, random_seed=5)
            result = quiet(sim.run, {"A": np.ones(64), "B": -np.ones(64)})
            assert result["status"] == "completed" and result["last_completed_step"] == 1, result
    print("Test passed: WTMD zero Fourier coefficients and ell validation")
    raise SystemExit(int(cuda_part_missing()))
