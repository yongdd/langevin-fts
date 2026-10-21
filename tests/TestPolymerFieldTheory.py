#!/usr/bin/env python3
"""
Coefficients of the multi-monomer field theory (SymmetricPolymerTheory).

For concentrations phi(r) on a grid (sum_i phi_i = 1 for the incompressible model),
the field part of the Hamiltonian made stationary in the auxiliary fields,
  V = ext_Omega [ h_const + <sum_k c2_k Omega_k^2 + c1_k Omega_k> + <w . phi> ],  w = A Omega,
is the Legendre transform that the Hubbard-Stratonovich transformation inverts:
  V = <(1/2) phi.chi.phi> (+ <(zeta/2)(sum phi - 1)^2> for the compressible model).
By the envelope theorem the derivative of the field part with respect to chi_ab at
the stationary fields is dV/dchi_ab = <phi_a phi_b>, which is what
compute_h_deriv_chin must return there.

Until 2026-10 the linear coefficients c1 and h_const were computed from the
eigenvectors before their sign normalization and sqrt(M) scaling (V wrong for
M >= 3 with unequal chi), and dH/dchiN differentiated the coefficients at fixed
Omega, which misses the rotation of A and blew up when an eigenvector flipped
sign between chi +- epsilon (chi N = 10/30/20: +-4e4 instead of ~0.1).
Both identities must hold to 1e-9 for 2, 3 and 4 monomer types, including a
degenerate chi N matrix (all pairs equal).
"""

import io
import sys
import contextlib

import numpy as np
from polymerfts.polymer_field_theory import SymmetricPolymerTheory

n_pass = 0
n_fail = 0


def check(cond, msg):
    global n_pass, n_fail
    if cond:
        n_pass += 1
        print(f"[PASS] {msg}")
    else:
        n_fail += 1
        print(f"[FAIL] {msg}")


def concentrations(M, n, incompressible, rng):
    phi = rng.uniform(0.05, 1.0, size=(M, n))
    if incompressible:
        phi /= np.sum(phi, axis=0)
    else:
        phi *= rng.uniform(0.8, 1.2, size=n) / np.sum(phi, axis=0)
    return phi


def main():
    rng = np.random.default_rng(1)
    cases = [
        ("AB", ["A", "B"], {"A,B": 15.0}),
        ("ABC equal", ["A", "B", "C"], {"A,B": 15.0, "A,C": 15.0, "B,C": 15.0}),
        ("ABC 9.5/16.6/9.5", ["A", "B", "C"], {"A,B": 9.5, "A,C": 16.625, "B,C": 9.5}),
        ("ABC 10/30/20", ["A", "B", "C"], {"A,B": 10.0, "A,C": 30.0, "B,C": 20.0}),
        ("ABCD", ["A", "B", "C", "D"], {"A,B": 12.0, "A,C": 18.0, "A,D": 25.0, "B,C": 8.0, "B,D": 14.0, "C,D": 20.0}),
    ]
    n = 200
    for name, types, chi_n in cases:
        M = len(types)
        X = np.zeros((M, M))
        for key, value in chi_n.items():
            i, j = [types.index(t) for t in key.split(",")]
            X[i, j] = X[j, i] = value
        for zeta_n in (None, 100.0):
            with contextlib.redirect_stdout(io.StringIO()):
                mpt = SymmetricPolymerTheory(types, chi_n, zeta_n)
            phi = concentrations(M, n, zeta_n is None, rng)
            A, c1, c2 = mpt.matrix_a, mpt.h_coef_mu1, mpt.h_coef_mu2

            # Stationary auxiliary fields: 2 c2_k Omega_k + c1_k + (A^T phi)_k = 0
            coupling = A.T @ phi
            omega = np.zeros((M, n))
            for k in range(M):
                if abs(c2[k]) > 0:
                    omega[k] = -(c1[k] + coupling[k]) / (2 * c2[k])
            w = A @ omega
            value = mpt.h_const + np.mean(np.sum(c2[:, None] * omega ** 2 + c1[:, None] * omega, axis=0)) \
                + np.mean(np.sum(w * phi, axis=0))
            exact = np.mean(0.5 * np.einsum("in,ij,jn->n", phi, X, phi))
            if zeta_n is not None:
                exact += np.mean(0.5 * zeta_n * (np.sum(phi, axis=0) - 1.0) ** 2)
            model = "incompressible" if zeta_n is None else "compressible"
            check(abs(value - exact) < 1e-9 * max(1.0, abs(exact)),
                  f"{name} {model}: stationary field part {value:.12f} = interaction energy {exact:.12f}")

            dH = mpt.compute_h_deriv_chin(chi_n, omega)
            errors = []
            for key in chi_n:
                a, b = [types.index(t) for t in key.split(",")]
                errors.append(abs(dH[key] - np.mean(phi[a] * phi[b])))
            check(max(errors) < 1e-9,
                  f"{name} {model}: dH/dchiN = <phi_a phi_b> (max error {max(errors):.1e})")

    print(f"\n{n_pass} passed, {n_fail} failed")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
