#!/usr/bin/env python3
"""
Complex fields (CL-FTS) with non-periodic boundary conditions: mirror test.

An all-reflecting box is exactly equivalent to a periodic box of twice the
length per non-periodic axis with an even-mirrored field: the DCT-II basis
of the reflecting problem is the even subspace of the doubled periodic
problem, and the discrete bond convolution respects the mirror symmetry
exactly. The periodic complex path (cufft/MKL/FFTW c2c) is independently
validated, so it serves as a machine-precision reference for the
non-periodic complex path.

This guards against a bug where the DCT/DST pipelines silently transformed
only the REAL part of complex propagators (imaginary part zeroed on every
bond convolution), which corrupted complex-Langevin runs in wall boxes
while leaving all real-field (SCFT/L-FTS) and single-segment-species paths
intact. The complex+non-periodic transforms now process the real and
imaginary parts independently (the DCT/DST is real-linear) with
interleaved complex coefficients.

Checks per available platform (cpu-mkl / cpu-fftw: discrete + continuous;
cuda: discrete only — RQM4/RK2 CUDA complex+non-periodic intentionally
raise):
  1. Q(reflecting) == Q(mirrored periodic) to machine precision.
  2. phi(reflecting) == phi(mirrored periodic, first half) to machine
     precision, with genuinely complex fields (std ~ 3, nonzero Im(phi)).
  3. The mirrored periodic solution is itself mirror-symmetric.
"""

import os
import sys

os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
os.environ["OMP_NUM_THREADS"] = "2"

import numpy as np
from polymerfts import _core

TOL = 1e-12


def solve(platform, chain_model, ds, nx, lx, bc, w, method="rqm4",
          diblock=False, w2=None):
    factory = _core.PlatformSelector.create_factory(platform, False, "complex")
    cb = factory.create_computation_box(nx, lx, bc=bc)
    types = {"A": 1.0, "B": 1.0} if diblock else {"A": 1.0}
    molecules = factory.create_molecules_information(chain_model, ds, types)
    if diblock:
        # A-B junction exercises the discrete half-bond steps
        molecules.add_polymer(1.0, [["A", 0.5, 0, 1], ["B", 0.5, 1, 2]])
    else:
        molecules.add_polymer(1.0, [["A", 1.0, 0, 1]])
    optimizer = factory.create_propagator_computation_optimizer(molecules, True)
    solver = factory.create_propagator_computation(cb, molecules, optimizer, method)
    w_in = {"A": np.ascontiguousarray(w.reshape(-1))}
    if diblock:
        w_in["B"] = np.ascontiguousarray(w2.reshape(-1))
    solver.compute_propagators(w_in)
    solver.compute_concentrations()
    phi = np.array(solver.get_total_concentration("A")).reshape(nx)
    return phi, solver.get_total_partition(0)


def run_case(platform, chain_model, method="rqm4", diblock=False):
    nz = 16
    nx_a = [4, 4, nz]
    nx_b = [4, 4, 2 * nz]
    lx_a = [1.0, 1.0, 2.0]
    lx_b = [1.0, 1.0, 4.0]
    ds = 0.05

    rng = np.random.default_rng(7)
    wz = rng.normal(0, 3.0, nz) + 1j * rng.normal(0, 3.0, nz)
    w_a = np.empty(nx_a, dtype=np.complex128)
    w_a[...] = wz[None, None, :]
    w_b = np.empty(nx_b, dtype=np.complex128)
    w_b[..., :nz] = wz[None, None, :]
    w_b[..., nz:] = wz[None, None, ::-1]

    if diblock:
        wz2 = rng.normal(0, 3.0, nz) + 1j * rng.normal(0, 3.0, nz)
        w2_a = np.empty(nx_a, dtype=np.complex128)
        w2_a[...] = wz2[None, None, :]
        w2_b = np.empty(nx_b, dtype=np.complex128)
        w2_b[..., :nz] = wz2[None, None, :]
        w2_b[..., nz:] = wz2[None, None, ::-1]
    else:
        w2_a = w2_b = None
    phi_a, q_a = solve(platform, chain_model, ds, nx_a, lx_a, ["reflecting"] * 6,
                       w_a, method, diblock, w2_a)
    phi_b, q_b = solve(platform, chain_model, ds, nx_b, lx_b, ["periodic"] * 6,
                       w_b, method, diblock, w2_b)

    q_err = abs(q_a - q_b)
    sym_err = np.abs(phi_b[..., :nz] - phi_b[..., ::-1][..., :nz]).max()
    eq_err = np.abs(phi_a - phi_b[..., :nz]).max()
    im_max = np.abs(phi_a.imag).max()

    tag = chain_model + ("-diblock" if diblock else "") + \
        ("" if method == "rqm4" else "-" + method)
    print(f"  {platform:8s} {tag:18s}: |dQ| = {q_err:.2e}, "
          f"mirror = {sym_err:.2e}, reflecting-vs-periodic = {eq_err:.2e}, "
          f"max|Im phi| = {im_max:.2e}")
    assert q_err < TOL, f"Q mismatch: {q_err}"
    assert sym_err < TOL, f"mirror symmetry broken: {sym_err}"
    assert eq_err < TOL, f"reflecting != mirrored periodic: {eq_err}"
    assert im_max > 1e-2, "field not genuinely complex — test is vacuous"


def main():
    avail = _core.PlatformSelector.avail_platforms()
    print("Available platforms:", avail)
    ran = 0
    for platform in avail:
        if platform.startswith("cpu"):
            for chain_model in ["discrete", "continuous"]:
                run_case(platform, chain_model)
                ran += 1
            run_case(platform, "continuous", method="rk2")
            run_case(platform, "discrete", diblock=True)   # half-bond junction
            ran += 2
        elif platform == "cuda":
            run_case(platform, "discrete")
            run_case(platform, "discrete", diblock=True)
            ran += 2
    assert ran > 0, "no platform available"
    print(f"ALL {ran} COMPLEX NON-PERIODIC MIRROR CASES PASSED")


if __name__ == "__main__":
    main()
