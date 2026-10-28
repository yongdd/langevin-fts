#!/usr/bin/env python3
"""The "etd" integrator: Leimkuhler-Matthews steps with the exact linear response (integrators.py).

1. Linear force F = J w with a synthetic Jacobian that couples a real-type field to two
   imaginary-type fields (complex symmetric in the variables of real noise): without
   noise one step is the exact flow exp(-S J dt) w, at time steps far beyond the stability
   limit of LM steps too; the stationary covariance of every mode is that of exp(-H),
   <x(k) x(-k)^T> = (sigma^2 n / (2 dt)) K^-1, mode by mode; a zero Jacobian gives the LM
   step; at a wavevector with an unstable homogeneous state only the unstable eigen-direction
   loses its exact step; where the Cholesky factor of phi_1 would be large, the noise factor
   is the symmetric square root, of the size of phi_1^(1/2); an asymmetric Jacobian is rejected.
   A matrix multiplier of the transform of complex fields acts on real noise as NumPy does
   (even and odd sizes, 1D to 3D).
2. Jacobian of CL-FTS: the complex-step columns equal central differences of the forces of all
   fields, psi included, for a charged diblock with counterions (with and without screening)
   and for a diblock on a 67-point grid, whose FFTs leave imaginary rounding errors; continuous
   chains with unequal segment lengths, whose measured Jacobian is asymmetric by the contour
   error, run with its symmetric part.
3. Runs: a continued run reproduces the uninterrupted one bit for bit (fields, psi, noise
   streams, measurements); psi keeps a zero uniform mode without screening; with zero
   valences psi has the Gaussian statistics of exp(-H) in a run at a time step beyond the LM
   limit of psi, <|psi(k)|^2> and the holomorphic <psi(k) psi(-k)> (which tells the rotated
   contour); an ordered charged lamella with strong screening, which diverged when unstable
   wavevectors took the plain LM step for all fields, runs like LM; a field dropped at a zero
   eigenvalue stays fixed; simulations sharing one integrator object keep their own response.
4. Input checks: L-FTS and walls reject the integrator; a step without the linear
   response, or with another time step, raises, and so does a Fourier transform of another
   grid, of real fields or of a box with walls; a copy of a prepared integrator keeps its
   settings but has no linear response.
"""
import contextlib
import copy
import io
import os
import tempfile
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import scipy.linalg
from scipy.io import loadmat
from polymerfts import LFTS, CLFTS, PlatformSelector, PropagatorSolver, ValidationError
from polymerfts.integrators import ExponentialTimeDifferencing, LeimkuhlerMatthews

Z_A = 0.2
PHI_C = Z_A * 0.5 / (1 + Z_A * 0.5)   # counterions: z_A phi_A = phi_C
NX, LX = [12, 10], [2.0, 1.8]


def cpu_platform():
    return next(p for p in PlatformSelector.avail_platforms() if p.startswith("cpu"))


def parameters(directory, steps=6, screening_length=None):
    """AB diblock with charged A blocks and monovalent counterions C, 2D, discrete chains."""
    params = {
        "nx": NX, "lx": LX, "ds": 1 / 20, "chain_model": "discrete",
        "segment_lengths": {"A": 1.0, "B": 1.0, "C": 1.0}, "chi_n": {"A,B": 12.0}, "zeta_n": 50.0,
        "polymers": [{"volume_fraction": 1 - PHI_C, "blocks": [{"type": "A", "length": 0.5},
                                                                {"type": "B", "length": 0.5}]},
                     {"volume_fraction": PHI_C, "blocks": [{"type": "C", "length": 0.05}]}],
        "smearing": {"type": "gaussian", "a_int": {"A": 0.15, "B": 0.15, "C": 0.2}},
        "electrostatics": {"charges": {"A": Z_A, "C": -1.0}, "coupling": 2e3, "psi_dt_scaling": 0.8},
        "platform": cpu_platform(),
        "langevin": {"nbar": 1e4, "dt": 2.0, "max_step": steps, "integrator": "etd"},
        "recording": {"dir": str(directory), "checkpoint_interval": 3, "sample_interval": 1,
                      "statistics_save_interval": 4},
        "verbose_level": 0,
    }
    if screening_length is not None:
        params["electrostatics"]["screening_length"] = screening_length
    return params


def initial_fields():
    rng = np.random.default_rng(11)
    return {m: rng.normal(0.0, 1.0, int(np.prod(NX))) for m in ("A", "B", "C")}


def quiet(make):
    with contextlib.redirect_stdout(io.StringIO()):
        return make()


def transform(shape):
    """PropagatorSolver.fourier of complex fields on a periodic grid of this shape."""
    solver = quiet(lambda: PropagatorSolver(nx=list(shape), lx=[1.0] * len(shape), ds=0.1, bond_lengths={"A": 1.0},
                                            bc=["periodic"] * (2 * len(shape)), chain_model="continuous",
                                            platform=cpu_platform(), field_type="complex"))
    return solver.fourier


def linear():
    rng = np.random.default_rng(1)
    n = 24
    k = 2 * np.pi * np.fft.fftfreq(n, d=1.0 / n) / 3.0
    direction = np.array([1.0, 1j, 1j])
    mobility = np.array([1.0, 0.4, 0.8])
    # Symmetric within a type, antisymmetric between the types, as for CL-FTS (TestClftsEtd 2)
    J = np.zeros((3, 3, n), dtype=complex)
    J[0, 0], J[1, 1], J[2, 2] = 0.3 + 0.05 * k ** 2, 1.0 + 0.2 * np.exp(-0.1 * k ** 2), 0.5 + 0.02 * k ** 2
    J[0, 1] = 0.15 * np.exp(-0.05 * k ** 2)
    J[1, 0] = -J[0, 1]
    J[1, 2] = J[2, 1] = 0.2 * np.exp(-0.05 * k ** 2)
    K = J * direction[None, :, None] / direction[:, None, None]

    def force(w, jacobian=J):
        return np.fft.ifft(np.einsum("ijk,jk->ik", jacobian, np.fft.fft(w, axis=1)), axis=1)

    def step(integrator, w, f, noise, previous, dt):
        return integrator.step(w, f, dt=dt, mobility=mobility, noise=noise, previous_noise=previous,
                               noise_direction=direction, evaluate_force=None)

    zero = np.zeros((3, n))
    ft = transform((n,))
    # The largest rate s λ is 31.9 (at the largest k): LM steps diverge beyond dt = 0.063, so both
    # time steps lie beyond the limit, dt = 20 by a factor of 300
    for dt in (0.5, 20.0):
        etd = ExponentialTimeDifferencing()
        quiet(lambda: etd.set_linear_response(J, dt=dt, mobility=mobility, noise_direction=direction, fourier=ft))
        w0 = rng.normal(0, 1, (3, n)) + 1j * rng.normal(0, 1, (3, n))
        w0_k = np.fft.fft(w0, axis=1)
        exact = np.fft.ifft(np.stack([scipy.linalg.expm(-dt * mobility[:, None] * J[:, :, q]) @ w0_k[:, q]
                                      for q in range(n)], axis=1), axis=1)
        stepped = step(etd, w0, force(w0), zero, zero, dt)
        assert np.max(np.abs(stepped - exact)) < 1e-10 * np.max(np.abs(exact)), f"exact flow at dt = {dt}"

        # Stationary covariance in x = w/n after 2000 relaxation steps (300 relaxation times of the slowest
        # mode, rate 0.3, at dt = 0.5)
        sigma = 0.3 * np.sqrt(dt)
        w, previous = np.zeros((3, n), dtype=complex), zero
        total = np.zeros((3, 3, n), dtype=complex)
        steps = 30000
        for s in range(steps + 2000):
            noise = rng.normal(0, sigma, (3, n))
            w = step(etd, w, force(w), noise, previous, dt)
            previous = noise
            if s >= 2000:
                x_k = np.fft.fft(w / direction[:, None], axis=1)
                total += np.einsum("ik,jk->ijk", x_k, np.roll(np.flip(x_k, axis=1), 1, axis=1))
        expected = np.stack([np.linalg.inv(K[:, :, q]) for q in range(n)], axis=2) * sigma ** 2 * n / (2 * dt)
        measured = total / steps
        diagonal = np.real(np.diagonal(measured, axis1=0, axis2=1) / np.diagonal(expected, axis1=0, axis2=1))
        # Every element of every mode, relative to the largest diagonal element of that mode
        scale = np.max(np.abs(np.diagonal(expected, axis1=0, axis2=1)), axis=1)
        elements = np.max(np.abs(measured - expected), axis=(0, 1)) / scale
        assert abs(diagonal.mean() - 1.0) < 0.02 and np.all(np.abs(diagonal - 1.0) < 0.1) and np.all(elements < 0.1), \
            (f"stationary covariance at dt = {dt}: diagonal mean {diagonal.mean():.4f}, range "
             f"[{diagonal.min():.3f}, {diagonal.max():.3f}], largest element error {elements.max():.3f}")

    # A zero Jacobian gives the LM step; a mode with a negative eigenvalue takes the LM step
    dt = 0.5
    w0 = rng.normal(0, 1, (3, n)) + 1j * rng.normal(0, 1, (3, n))
    noise, previous = rng.normal(0, 1, (3, n)), rng.normal(0, 1, (3, n))
    lm = step(LeimkuhlerMatthews(), w0, force(w0), noise, previous, dt)
    etd = ExponentialTimeDifferencing()
    quiet(lambda: etd.set_linear_response(np.zeros_like(J), dt=dt, mobility=mobility, noise_direction=direction,
                                          fourier=ft))
    assert np.max(np.abs(step(etd, w0, force(w0), noise, previous, dt) - lm)) < 1e-13, "zero Jacobian: LM step"

    # The step transforms real noise with the transform of complex fields: a matrix multiplier
    # on real fields against NumPy, for even and odd sizes in 1D to 3D
    for shape in ((8,), (7,), (6, 5), (4, 5, 6), (5, 4, 3)):
        probe = transform(shape)
        axes = tuple(range(1, len(shape) + 1))
        real = rng.normal(0, 1, (2, int(np.prod(shape))))
        matrix = rng.normal(size=(2, 2) + shape) + 1j * rng.normal(size=(2, 2) + shape)
        expected = np.fft.ifftn(np.einsum("ij...,j...->i...", matrix, np.fft.fftn(real.reshape((2,) + shape), axes=axes)),
                                axes=axes).reshape(2, -1)
        result = probe.apply([(probe.set_multiplier(matrix), real)])
        assert np.max(np.abs(result - expected)) < 1e-12 * np.max(np.abs(expected)), f"multiplier on real noise {shape}"

    # One unstable eigen-direction at q = 3 (and -3): it takes the LM step, the stable
    # directions of that wavevector keep their exact step
    unstable = J.copy()
    unstable[0, 0, 3] = unstable[0, 0, n - 3] = -0.5
    quiet(lambda: etd.set_linear_response(unstable, dt=dt, mobility=mobility, noise_direction=direction, fourier=ft))
    drift, _, _ = etd._matrices(unstable, dt=dt, mobility=mobility, noise_direction=direction)
    root = np.sqrt(mobility)
    for q in (3, n - 3):
        kq = root[:, None] * (unstable[:, :, q] * direction[None, :] / direction[:, None]) * root[None, :]
        values, vectors = np.linalg.eig(kq)
        assert np.sum(values.real < 0) == 1
        phi1 = np.where(values.real < 0, 1.0, -np.expm1(-values * dt) / (values * dt))
        expected = vectors @ np.diag(phi1) @ np.linalg.inv(vectors)
        expected = dt * direction[:, None] * root[:, None] * expected * root[None, :] / direction[None, :]
        assert np.allclose(drift[:, :, q], expected, rtol=0, atol=1e-12), "unstable direction alone takes the LM step"

    # phi_1 of K = [[1, ib], [ib, 1]] (real fields, dt = 1) has a small (0,0) element near b = 4.0846; its
    # Cholesky factor would be large, the symmetric square root is not
    ft1 = transform((1,))
    for b in (4.08, 4.085, 4.0846):
        jac = np.zeros((2, 2, 1), dtype=complex)
        jac[:, :, 0] = [[1.0, 1j * b], [1j * b, 1.0]]
        quiet(lambda: etd.set_linear_response(jac, dt=1.0, mobility=np.ones(2), noise_direction=np.ones(2), fourier=ft1))
        drift, noise, _ = etd._matrices(jac, dt=1.0, mobility=np.ones(2), noise_direction=np.ones(2))
        factor, phi1 = noise[:, :, 0], drift[:, :, 0]
        assert np.allclose(factor @ factor.T, phi1, rtol=0, atol=1e-10), "noise covariance phi_1"
        assert np.max(np.abs(factor)) < 10 * np.sqrt(np.max(np.abs(phi1))), f"noise factor of size phi_1^(1/2) at b = {b}"

    # An asymmetric Jacobian is an inconsistent measurement
    asymmetric = J.copy()
    asymmetric[1, 2] *= 1.01
    expect_validation_error(lambda: etd.set_linear_response(asymmetric, dt=dt, mobility=mobility,
                                                            noise_direction=direction, fourier=ft), "not symmetric")


def jacobian(root):
    for screening_length in (None, 0.3):
        sim = quiet(lambda: CLFTS(parameters(root / "jacobian", screening_length=screening_length), random_seed=5))
        J = sim._force_jacobian()
        M, n_grid, eps = 3, int(np.prod(NX)), 1e-6
        assert J.shape == (M + 1, M + 1) + tuple(NX)
        scale, worst = np.max(np.abs(J)), 0.0
        for j in range(M + 1):
            if j < M and sim.dt_scaling[j] == 0:
                continue
            kick = np.zeros((M + 1, n_grid), dtype=complex)
            kick[j, 0] = eps
            difference = (sim._evaluate_force_with_psi(kick) - sim._evaluate_force_with_psi(-kick)) / (2 * eps)
            for i in range(M + 1):
                worst = max(worst, np.max(np.abs(np.fft.fftn(np.reshape(difference[i], NX)) - J[i, j])) / scale)
        assert worst < 1e-6, f"Jacobian against central differences (screening_length {screening_length}): {worst:.2e}"

    # 67 grid points: the FFTs of real data leave imaginary rounding errors of about 1e-15
    params = parameters(root / "prime")
    params.pop("electrostatics")
    params.update(nx=[67], lx=[4.0 * 67 / 64], chain_model="continuous", ds=1 / 40, chi_n={"A,B": 8.0},
                  segment_lengths={"A": 1.0, "B": 1.0}, smearing=None,
                  polymers=[{"volume_fraction": 1.0, "blocks": [{"type": "A", "length": 0.3}, {"type": "B", "length": 0.7}]}])
    sim = quiet(lambda: CLFTS(params, random_seed=5))
    J, eps = sim._force_jacobian(), 1e-6
    worst = 0.0
    for j in range(2):
        kick = np.zeros((2, 67), dtype=complex)
        kick[j, 0] = eps
        difference = (sim.evaluate_force(kick) - sim.evaluate_force(-kick)) / (2 * eps)
        for i in range(2):
            worst = max(worst, np.max(np.abs(np.fft.fft(difference[i]) - J[i, j])) / np.max(np.abs(J)))
    assert worst < 1e-6, f"Jacobian on 67 grid points against central differences: {worst:.2e}"

    # Continuous chains with unequal segment lengths: their forces are the gradient of the discretized
    # H only up to the contour error, so the measured Jacobian is slightly asymmetric; the simulation
    # passes its symmetric part to the integrator
    params.update(nx=[64], lx=[4.0], numerical_method="rk2", ds=1 / 40, chi_n={"A,B": 14.0},
                  segment_lengths={"A": 1.0, "B": 1.2},
                  polymers=[{"volume_fraction": 1.0, "blocks": [{"type": "A", "length": 0.5}, {"type": "B", "length": 0.5}]}])
    sim = quiet(lambda: CLFTS(params, random_seed=5))
    J = sim._force_jacobian()
    ratio = (sim._integrated_noise_direction()[None, :] / sim._integrated_noise_direction()[:, None])[:, :, None]
    K, K_symmetric = J * ratio, sim._symmetric_jacobian(J) * ratio
    asymmetry = np.max(np.abs(K - np.swapaxes(K, 0, 1))) / np.max(np.abs(K))
    assert 1e-5 < asymmetry < 1e-3, f"asymmetry of the measured Jacobian {asymmetry:.1e}"
    assert np.max(np.abs(K_symmetric - 0.5 * (K + np.swapaxes(K, 0, 1)))) < 1e-14 * np.max(np.abs(K)), \
        "symmetric part of the Jacobian"


def runs(root):
    def run(params, continue_from=None):
        sim = quiet(lambda: CLFTS(params, random_seed=5))
        with contextlib.redirect_stdout(io.StringIO()):
            if continue_from is None:
                sim.run(initial_fields=initial_fields())
            else:
                sim.continue_run(continue_from)
        return sim

    for screening_length in (None, 0.4):
        full, part = root / f"full_{screening_length}", root / f"part_{screening_length}"
        sim = run(parameters(full, steps=8, screening_length=screening_length))
        uniform = sim.electrostatics.psi_k[(0,) * len(NX)]
        assert (uniform == 0.0) == (screening_length is None), "uniform mode of psi"
        run(parameters(part, steps=3, screening_length=screening_length))
        run(parameters(part, steps=8, screening_length=screening_length), continue_from=part / "fields_000003.mat")
        for name in ("fields_000006.mat", "dH_000008.mat", "structure_function_000008.mat"):
            a, b = loadmat(full / name, squeeze_me=True), loadmat(part / name, squeeze_me=True)
            keys = [k for k in a if not k.startswith("__") and isinstance(a[k], np.ndarray) and a[k].dtype.kind in "fc"]
            assert max(np.max(np.abs(a[k] - b[k])) for k in keys) == 0.0, f"continued run, {name}"

    # Zero valences: psi is decoupled and Gaussian. At dt = 20 the fastest psi mode has
    # s (k^2 + kappa^2)/E dt = 5.3, beyond the LM limit of 2; chi N = 2 keeps the auxiliary fields
    # Gaussian too. V Nbar^(1/2) ((k^2 + kappa^2)/E) <|psi(k)/n|^2> = 1 and the holomorphic
    # <psi(k) psi(-k)> = -1 (psi = i phi on the rotated contour), mode by mode on average
    for screening_length in (None, 0.4):
        params = parameters(root / "psi_statistics", steps=3200, screening_length=screening_length)
        params["electrostatics"]["charges"] = {"A": 0.0, "C": 0.0}
        params["chi_n"] = {"A,B": 2.0}
        params["langevin"]["dt"] = 20.0
        params["recording"].update(checkpoint_interval=3200, statistics_save_interval=3200)
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
            sim.run(initial_fields={m: np.zeros(n_grid) for m in ("A", "B", "C")})
        scale = sim.prop_solver.get_volume() * np.sqrt(sim.langevin["nbar"]) * es.screened_k_sq / es.coupling / n_grid ** 2
        modes = es.screened_k_sq > 0
        assert np.max(es.screened_k_sq) / es.coupling * es.psi_dt_scaling * 20.0 > 5.0, "beyond the LM limit of psi"
        variance = np.mean((scale * sums["abs"] / sums["count"])[modes])
        holomorphic = np.mean((scale * sums["holomorphic"] / sums["count"]).real[modes])
        assert abs(variance - 1.0) < 0.03 and abs(holomorphic + 1.0) < 0.03, \
            f"psi statistics (screening_length {screening_length}): <|psi|^2> {variance:.4f}, <psi psi> {holomorphic:.4f}"

    # Ordered charged lamella with strong screening: at its 4 unstable wavevectors the psi step of
    # LM has |1 - dt s (k^2 + kappa^2)/E| = 1.8; when the whole wavevector took the plain LM step,
    # psi diverged within 200 steps, while lm and etd now run
    n_seg, z_a = 20, 0.5
    phi_c = z_a * 0.5 / (1 + z_a * 0.5)
    params = parameters(root / "ordered", steps=200)
    params.update(nx=[32], lx=[4.0], chi_n={"A,B": 25.0}, zeta_n=100.0, smearing={"type": "gaussian", "a_int": 0.15},
                  polymers=[{"volume_fraction": 1 - phi_c, "blocks": [{"type": "A", "length": 0.5}, {"type": "B", "length": 0.5}]},
                            {"volume_fraction": phi_c, "blocks": [{"type": "C", "length": 1 / n_seg}]}])
    params["electrostatics"] = {"charges": {"A": z_a, "C": -1.0}, "coupling": 300.0, "screening_length": 0.05}
    params["langevin"].update(nbar=1e6, dt=2.0)
    params["recording"].update(checkpoint_interval=200, statistics_save_interval=200)
    x = np.linspace(0, 4.0, 32, endpoint=False)
    w_a = 5.0 * np.cos(4 * np.pi * x / 4.0)
    sim = quiet(lambda: CLFTS(params, random_seed=1))
    with contextlib.redirect_stdout(io.StringIO()):
        sim.run(initial_fields={"A": w_a, "B": -w_a, "C": np.zeros(32)})
    assert np.max(np.abs(sim.electrostatics.psi)) < 1.0, "ordered charged lamella with etd"

    # One integrator object for two simulations: each keeps the response of its own system
    shared = ExponentialTimeDifferencing()
    first, second = parameters(root / "shared_1"), parameters(root / "shared_2")
    second["chi_n"] = {"A,B": 20.0}
    sims = [quiet(lambda: CLFTS(p, random_seed=5, integrator=shared)) for p in (first, second)]
    assert sims[0].integrator is not sims[1].integrator and shared._prepared is None
    # Their steps from the same fields and forces, without noise, differ
    increments = []
    for sim in sims:
        prepared = sim.integrator._prepared
        zero = np.zeros((prepared["mobility"].size, int(np.prod(prepared["nx"]))))
        force = np.random.default_rng(2).normal(size=zero.shape) + 0j
        increments.append(sim.integrator.step(zero + 0j, force, dt=prepared["dt"], mobility=prepared["mobility"], noise=zero,
                                              previous_noise=zero, noise_direction=prepared["direction"], evaluate_force=None))
    assert not np.allclose(increments[0], increments[1]), "each simulation keeps the response of its own system"

    # Polyampholyte without chi: the composition mode of E and K has no field (zero eigenvalue)
    params = parameters(root / "dropped", steps=4)
    params.update(segment_lengths={"E": 1.0, "K": 1.0}, chi_n={},
                  polymers=[{"volume_fraction": 1.0, "blocks": [{"type": "E", "length": 0.25}, {"type": "K", "length": 0.25},
                                                                 {"type": "E", "length": 0.25}, {"type": "K", "length": 0.25}]}],
                  smearing={"type": "gaussian", "a_int": 0.15})
    params["electrostatics"]["charges"] = {"E": -1.0, "K": 1.0}
    sim = quiet(lambda: CLFTS(params, random_seed=5))
    dropped = sim.mpt.aux_fields_dropped_idx
    assert dropped, "polyampholyte without chi has a dropped field"
    with contextlib.redirect_stdout(io.StringIO()):
        sim.run(initial_fields={"E": np.zeros(int(np.prod(NX))), "K": np.zeros(int(np.prod(NX)))})
    w_aux = loadmat(root / "dropped" / "fields_000003.mat", squeeze_me=True)["w_aux"]
    expected = sim.mpt._vector_large_s[dropped[0]] / 2
    assert np.all(np.reshape(w_aux, (2, -1))[dropped[0]] == expected), "dropped field stays fixed"


def expect_validation_error(make, message):
    try:
        quiet(make)
    except ValidationError as exc:
        assert message in str(exc), (message, str(exc))
        return
    raise AssertionError(f"Expected ValidationError: {message}")


def input_checks(root):
    params = parameters(root / "inputs")
    params.pop("electrostatics")
    params["smearing"] = {"type": "gaussian", "a_int": 0.15}
    params["saddle"] = {"max_iter": 10, "tolerance": 1e-4}
    params["compressor"] = {"name": "am", "max_hist": 5, "start_error": 0.5, "mix_min": 0.01, "mix_init": 0.01}
    params.pop("zeta_n")
    expect_validation_error(lambda: LFTS(params), "CL-FTS only")
    walls = parameters(root / "inputs")
    walls.pop("electrostatics")
    walls["bc"] = ["reflecting"] * 4
    expect_validation_error(lambda: CLFTS(walls), "periodic box")

    etd, zero = ExponentialTimeDifferencing(), np.zeros((2, 8))
    arguments = dict(dt=1.0, mobility=np.ones(2), noise=zero, previous_noise=zero,
                     noise_direction=np.array([1.0, 1j]), evaluate_force=None)
    expect_validation_error(lambda: etd.step(zero, zero, **arguments), "needs the linear response")
    expect_validation_error(lambda: etd.set_linear_response(np.zeros((2, 2, 8)), dt=0.5, mobility=np.ones(2),
                                                            noise_direction=np.array([1.0, 1j]), fourier=transform((9,))),
                            "grid of the Jacobian")
    quiet(lambda: etd.set_linear_response(np.zeros((2, 2, 8)), dt=0.5, mobility=np.ones(2),
                                          noise_direction=np.array([1.0, 1j]), fourier=transform((8,))))
    expect_validation_error(lambda: etd.step(zero, zero, **arguments), "another time step")
    for name, wrong in (("real fields", lambda: quiet(lambda: PropagatorSolver(
                             nx=[8], lx=[1.0], ds=0.1, bond_lengths={"A": 1.0}, bc=["periodic"] * 2,
                             chain_model="continuous", platform=cpu_platform())).fourier),
                        ("walls", lambda: quiet(lambda: PropagatorSolver(
                             nx=[8], lx=[1.0], ds=0.1, bond_lengths={"A": 1.0}, bc=["reflecting"] * 2,
                             chain_model="continuous", platform=cpu_platform(), field_type="complex")).fourier)):
        expect_validation_error(lambda: etd.set_linear_response(np.zeros((2, 2, 8)), dt=0.5, mobility=np.ones(2),
                                                                noise_direction=np.array([1.0, 1j]), fourier=wrong()),
                                "periodic box")
    # A copy (each simulation copies its integrator) keeps its settings but not the multipliers
    etd.settings = {"version": 1, "probe": 0.5}
    copied = copy.deepcopy(etd)
    assert copied._prepared is None and copied.settings == {"version": 1, "probe": 0.5} and etd._prepared is not None, \
        "a copy of a prepared etd integrator keeps its settings but has no linear response"


def main():
    if not any(p.startswith("cpu") for p in PlatformSelector.avail_platforms()):
        return 77
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name, check in (("linear force", lambda: linear()), ("Jacobian", lambda: jacobian(root)),
                            ("runs", lambda: runs(root)), ("input checks", lambda: input_checks(root))):
            check()
            print(f"[PASS] {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
