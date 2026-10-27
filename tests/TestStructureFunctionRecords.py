#!/usr/bin/env python3
"""Recorded structure functions against a recomputation from the saved fields.

S_ij(k) = V sqrt(Nbar) mu_i(k) phi_j(-k) for one sample, mu_i = sum_k A_ik w_k / lambda_k over the
fields with a nonzero eigenvalue (docs/theory/StructureFunctionEstimators.md); with smearing in a
periodic box the record is divided by Gamma_i. Two steps with every recording interval 1, on every
available platform, agreement to 1e-10 relative:
1. L-FTS of an ABC triblock (two fluctuating fields) on a 5x6x7 grid: periodic with Gaussian
   smearing, and with reflecting walls.
2. CL-FTS of an AB diblock (discrete chains) with per-type smearing lengths on a 12x9 grid:
   periodic, and with reflecting walls (no division by Gamma).
Periodic boxes transform with PropagatorSolver.fourier; boxes with walls record the Fourier
coefficients of numpy.fft, not the cosine series of the transform, whose coefficients have the same
shape for complex fields.
"""

import contextlib
import io
import os
import tempfile

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from scipy.io import loadmat
from polymerfts import CLFTS, LFTS, PlatformSelector
from cuda_part import cuda_part_missing

failures = 0


def check(name, ok, detail=""):
    global failures
    failures += not ok
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


def recorded_pairs(sf, complex_fields):
    """{(i, j): S_ij} from a structure_function file."""
    pairs = {}
    for key in sf:
        if not key.startswith("structure_function_"):
            continue
        names = key[len("structure_function_"):]
        if complex_fields:
            if not names.endswith("_real"):
                continue
            names = names[:-len("_real")]
            value = sf[key] + 1j * sf["structure_function_" + names + "_imag"]
        else:
            value = sf[key]
        pairs[tuple(names.split("_"))] = np.asarray(value)
    return pairs


def compare(sim, directory, complex_fields, gamma):
    """Largest relative difference between the recorded and the recomputed S_ij of step 2."""
    fields = loadmat(os.path.join(directory, "fields_000002.mat"), squeeze_me=True)
    sf = loadmat(os.path.join(directory, "structure_function_000002.mat"), squeeze_me=True)
    names, nx = sim.monomer_types, tuple(sim.prop_solver.nx)
    n, M = int(np.prod(nx)), len(sim.monomer_types)
    w_aux = np.reshape(fields["w_aux"], (M, n))
    # L-FTS: the pressure field (the last one) does not fluctuate; CL-FTS: all fields do
    kept = [k for k in range(M if complex_fields else M - 1) if not np.isclose(sim.mpt.eigenvalues[k], 0.0)]
    if complex_fields:
        transform = lambda f: np.fft.fftn(np.reshape(f, nx))
        phi = {m: fields["phi_" + m + "_real"] + 1j * fields["phi_" + m + "_imag"] for m in names}
        phi_minus_k = {m: np.conj(transform(np.conj(phi[m]))) / n for m in names}
    else:
        transform = lambda f: np.fft.rfftn(np.reshape(f, nx))
        phi_minus_k = {m: np.conj(transform(fields["phi_" + m])) / n for m in names}
    mu = {m: sum(transform(w_aux[k]) * sim.mpt.matrix_a[i, k] / sim.mpt.eigenvalues[k] / n for k in kept)
          for i, m in enumerate(names)}
    norm = sim.prop_solver.get_volume() * np.sqrt(sim.langevin["nbar"])
    worst, pairs = 0.0, recorded_pairs(sf, complex_fields)
    for (i, j), recorded in pairs.items():
        expected = norm * mu[i] * phi_minus_k[j]
        if gamma is not None:
            g = gamma(i)
            expected = np.divide(expected, g, out=np.zeros_like(expected), where=g > 1e-12)
        worst = max(worst, np.max(np.abs(recorded - expected)) / np.max(np.abs(expected)))
    return worst, len(pairs)


def lfts(platform, root, bc):
    nx, lx = [5, 6, 7], [1.5, 1.6, 1.7]
    params = {"nx": nx, "lx": lx, "ds": 1 / 20, "chain_model": "continuous", "bc": [bc] * 6,
              "segment_lengths": {"A": 1.0, "B": 1.0, "C": 1.0}, "chi_n": {"A,B": 12.0, "B,C": 10.0, "A,C": 14.0},
              "polymers": [{"volume_fraction": 1.0, "blocks": [{"type": "A", "length": 0.3}, {"type": "B", "length": 0.4},
                                                                {"type": "C", "length": 0.3}]}],
              "platform": platform, "langevin": {"nbar": 1e4, "dt": 0.5, "max_step": 2},
              "recording": {"dir": root, "checkpoint_interval": 1, "sample_interval": 1, "statistics_save_interval": 1},
              "saddle": {"max_iter": 200, "tolerance": 1e-6},
              "compressor": {"name": "am", "max_hist": 20, "start_error": 5e-1, "mix_min": 0.01, "mix_init": 0.01},
              "verbose_level": 0}
    if bc == "periodic":
        params["smearing"] = {"type": "gaussian", "a_int": 0.1}
    rng = np.random.default_rng(3)
    with contextlib.redirect_stdout(io.StringIO()):
        sim = LFTS(params, random_seed=5)
        sim.run(initial_fields={m: rng.normal(0.0, 1.0, int(np.prod(nx))) for m in "ABC"})
    half = nx[-1] // 2 + 1
    gamma = (lambda name: sim.smearing.fourier[..., :half]) if bc == "periodic" else None
    worst, count = compare(sim, root, False, gamma)
    check(f"{platform} L-FTS ABC {bc}{' with smearing' if gamma else ''}: recorded S_ij = recomputation",
          worst < 1e-10 and count == 6, f"({count} pairs, max rel diff {worst:.1e})")


def clfts(platform, root, bc):
    nx, lx = [12, 9], [2.0, 1.8]
    # Discrete chains: the CUDA continuous-chain solvers take complex fields in periodic boxes only
    params = {"nx": nx, "lx": lx, "ds": 1 / 20, "chain_model": "discrete", "bc": [bc] * 4,
              "segment_lengths": {"A": 1.0, "B": 1.0}, "chi_n": {"A,B": 12.0}, "zeta_n": 50.0,
              "polymers": [{"volume_fraction": 1.0, "blocks": [{"type": "A", "length": 0.5}, {"type": "B", "length": 0.5}]}],
              "smearing": {"type": "gaussian", "a_int": {"A": 0.15, "B": 0.2}},
              "platform": platform, "langevin": {"nbar": 1e4, "dt": 0.5, "max_step": 2},
              "recording": {"dir": root, "checkpoint_interval": 1, "sample_interval": 1, "statistics_save_interval": 1},
              "verbose_level": 0}
    rng = np.random.default_rng(3)
    with contextlib.redirect_stdout(io.StringIO()):
        sim = CLFTS(params, random_seed=5)
        sim.run(initial_fields={m: rng.normal(0.0, 1.0, int(np.prod(nx))) for m in "AB"})
    gamma = (lambda name: sim.smearing.kernel(name)) if bc == "periodic" else None
    worst, count = compare(sim, root, True, gamma)
    check(f"{platform} CL-FTS AB {bc} with per-type smearing: recorded S_ij = recomputation",
          worst < 1e-10 and count == 3, f"({count} pairs, max rel diff {worst:.1e})")


def main():
    with tempfile.TemporaryDirectory() as tmp:
        for platform in PlatformSelector.avail_platforms():
            for bc in ("periodic", "reflecting"):
                lfts(platform, os.path.join(tmp, f"l_{platform}_{bc}"), bc)
                clfts(platform, os.path.join(tmp, f"cl_{platform}_{bc}"), bc)
    print(f"\n{failures} failed")
    return 1 if cuda_part_missing() or failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
