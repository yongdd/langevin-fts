#!/usr/bin/env python3
"""
Coefficients of the multi-monomer field theory (SymmetricPolymerTheory).

For concentrations phi(r) on a grid (sum_i phi_i = 1 for the incompressible model),
the field part of the Hamiltonian made stationary in the auxiliary fields,
  V = ext_Omega [ h_const + <sum_k c2_k Omega_k^2 + c1_k Omega_k> + <w . phi> ],  w = A Omega,
is the Legendre transform that the Hubbard-Stratonovich transformation inverts:
  V = <(1/2) phi.chi.phi> (+ <(zeta/2)(sum phi - 1)^2> for the compressible model).
By the envelope theorem the derivative with respect to chi_ab at the stationary
fields is dV/dchi_ab = <phi_a phi_b>, which is what compute_h_deriv_chin (taken at
fixed auxiliary fields, including the rotation of A through <phi . dA/dchi Omega>)
must return there.

The cases include nearly equal eigenvalues (chi N = 15/15/15.001 and 15.0001,
where eigenvectors turn by a finite angle within a chi step) and fields dropped at
a zero eigenvalue (B and C chemically identical; AB at chi N = 0), whose term in
dH/dchiN does not vanish although the field is absent.

The second check is a zero-dimensional model of the L-FTS approximation: three
monomer types, compressible, two real fields integrated by quadrature and the
imaginary field at its saddle point, with a concave model for -ln Q. The average
of compute_h_deriv_chin must equal the chi derivative of the free energy of that
model (central difference of the quadrature), to 1e-6, also with two nearly equal
real eigenvalues (chi N = 1/1/1.00001).

Until 2026-10 the linear coefficients c1 and h_const were computed from the
eigenvectors before their sign normalization and sqrt(M) scaling (V wrong for
M >= 3 with unequal chi), and dH/dchiN differentiated the coefficients at fixed
Omega without the change of the polymer part through A(chi), and blew up when an
eigenvector flipped sign between chi +- epsilon (chi N = 10/30/20: +-4e4 instead
of ~0.1), and it left out dropped fields (AB at chi N = 0, f = 0.3: 0.25 instead
of 0.21). A derivative by central differences in chi fails for nearly equal
eigenvalues (15/15/15.0001: 139 instead of ~0.1); the code differentiates
analytically. Both identities must hold to 1e-9 for 2, 3 and 4 monomer types,
including a degenerate chi N matrix (all pairs equal). dH/dchiN must not depend on
the rows of w_aux of dropped fields, which no simulation updates.
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


def test_dropped_fields_and_keys():
    """Dropped fields: dH/dchiN does not depend on their (unused) rows of w_aux, and the
    identity holds also where the dropped field has a linear coefficient (1^T chi^-1 1 = 0);
    pair keys in either order."""
    rng = np.random.default_rng(3)
    types = ["A", "B", "C"]
    for name, chi_n in (("B = C", {"A,B": 15.0, "A,C": 15.0, "B,C": 0.0}),
                        ("1^T chi^-1 1 = 0", {"A,B": 10.0, "A,C": 10.0, "B,C": 40.0})):
        with contextlib.redirect_stdout(io.StringIO()):
            mpt = SymmetricPolymerTheory(types, chi_n, None)
        phi = concentrations(3, 300, True, rng)
        A, c1, c2 = mpt.matrix_a, mpt.h_coef_mu1, mpt.h_coef_mu2
        omega = np.zeros((3, 300))
        for k in range(3):
            if abs(c2[k]) > 0:
                omega[k] = -(c1[k] + (A.T @ phi)[k]) / (2 * c2[k])
        dropped = [k for k in range(3) if np.isclose(mpt.eigenvalues[k], 0.0) and k != 2]
        noisy = omega.copy()
        noisy[dropped] = rng.normal(0.0, 2.0, (len(dropped), 300))
        phi_dict = {t: phi[i] for i, t in enumerate(types)}
        dH, dH_noisy = mpt.compute_h_deriv_chin(chi_n, omega, phi_dict), mpt.compute_h_deriv_chin(chi_n, noisy, phi_dict)
        error = max(abs(dH[key] - np.mean(phi[types.index(key[0])] * phi[types.index(key[2])])) for key in chi_n)
        change = max(abs(dH[key] - dH_noisy[key]) for key in chi_n)
        check(len(dropped) == 1 and error < 1e-12 and change == 0.0,
              f"{name}: dH/dchiN = <phi_a phi_b> ({error:.1e}), independent of the dropped row of w_aux ({change:.1e})")
    with contextlib.redirect_stdout(io.StringIO()):
        mpt = SymmetricPolymerTheory(["A", "B"], {"B,A": 15.0}, None)
    check(np.isclose(mpt.eigenvalues[0], -15.0) and mpt.chi_n == {"A,B": 15.0}, "chi N key 'B,A' is read as 'A,B'")


def test_saddle_point_model(chi0):
    """0-D L-FTS model: real fields by quadrature, the imaginary field at its saddle."""
    types = ["A", "B", "C"]
    f = np.array([0.3, 0.3, 0.4])
    cv, zeta_n = 3.0, 5.0
    n = 241

    def theory(chi):
        with contextlib.redirect_stdout(io.StringIO()):
            return SymmetricPolymerTheory(types, chi, zeta_n)

    def minus_ln_q(w):          # concave model of -ln Q with phi_j = d(-ln Q)/dw_j
        m = np.min(w, axis=0)
        return m - np.log(np.sum(f[:, None] * np.exp(-(w - m)), axis=0))

    def concentrations(w):
        e = f[:, None] * np.exp(-(w - np.min(w, axis=0)))
        return e / np.sum(e, axis=0)

    def hamiltonian(mpt, omega):
        return mpt.h_const + np.sum(mpt.h_coef_mu2[:, None] * omega ** 2 + mpt.h_coef_mu1[:, None] * omega, axis=0) \
            + minus_ln_q(mpt.matrix_a @ omega)

    def ensemble(chi, grid):
        mpt = theory(chi)
        real, (imag,) = mpt.aux_fields_real_idx, mpt.aux_fields_imag_idx
        X, Y = np.meshgrid(*grid, indexing="ij")
        omega = np.zeros((3, n * n))
        omega[real[0]], omega[real[1]] = X.ravel(), Y.ravel()
        a = mpt.matrix_a[:, imag]
        for _ in range(60):     # Newton iteration of the imaginary field to its saddle point
            phi = concentrations(mpt.matrix_a @ omega)
            g1 = 2 * mpt.h_coef_mu2[imag] * omega[imag] + mpt.h_coef_mu1[imag] + a @ phi
            cov = np.einsum("ij,in->ijn", np.eye(3), phi) - np.einsum("in,jn->ijn", phi, phi)
            omega[imag] -= g1 / (2 * mpt.h_coef_mu2[imag] - np.einsum("i,ijn,j->n", a, cov, a))
        h = hamiltonian(mpt, omega)
        weight = np.exp(-cv * (h - h.min()))
        free_energy = h.min() - np.log(np.sum(weight) * (grid[0][1] - grid[0][0]) * (grid[1][1] - grid[1][0])) / cv
        return mpt, omega, weight / np.sum(weight), free_energy

    # Grid around the distribution
    mpt, omega, weight, _ = ensemble(chi0, (np.linspace(-9, 9, n), np.linspace(-9, 9, n)))
    real = mpt.aux_fields_real_idx
    mean = [omega[r] @ weight for r in real]
    sd = [np.sqrt(((omega[r] - mu) ** 2) @ weight) for r, mu in zip(real, mean)]
    grid = tuple(np.linspace(mu - 12 * s, mu + 12 * s, n) for mu, s in zip(mean, sd))
    mpt, omega, weight, _ = ensemble(chi0, grid)
    phi = concentrations(mpt.matrix_a @ omega)
    dH = mpt.h_deriv_chin_from_moments(omega @ weight, (omega * weight) @ omega.T, (phi * weight) @ omega.T)
    errors = []
    for key in chi0:
        eps = 1e-4
        chi_p, chi_m = dict(chi0), dict(chi0)
        chi_p[key] += eps
        chi_m[key] -= eps
        d_free_energy = (ensemble(chi_p, grid)[3] - ensemble(chi_m, grid)[3]) / (2 * eps)
        errors.append(abs(dH[key] - d_free_energy))
    name = "/".join(f"{chi0[key]:g}" for key in ("A,B", "A,C", "B,C"))
    check(max(errors) < 1e-6, f"0-D saddle-point model, chi N = {name}: <dH/dchiN> = dF/dchiN (max error {max(errors):.1e})")


def main():
    rng = np.random.default_rng(1)
    cases = [
        ("AB", ["A", "B"], {"A,B": 15.0}),
        ("ABC equal", ["A", "B", "C"], {"A,B": 15.0, "A,C": 15.0, "B,C": 15.0}),
        ("ABC 9.5/16.6/9.5", ["A", "B", "C"], {"A,B": 9.5, "A,C": 16.625, "B,C": 9.5}),
        ("ABC 10/30/20", ["A", "B", "C"], {"A,B": 10.0, "A,C": 30.0, "B,C": 20.0}),
        ("ABCD", ["A", "B", "C", "D"], {"A,B": 12.0, "A,C": 18.0, "A,D": 25.0, "B,C": 8.0, "B,D": 14.0, "C,D": 20.0}),
        ("ABC 15/15/15.001", ["A", "B", "C"], {"A,B": 15.0, "A,C": 15.0, "B,C": 15.001}),
        ("ABC 15/15/15.0001", ["A", "B", "C"], {"A,B": 15.0, "A,C": 15.0, "B,C": 15.0001}),
        ("ABC with B = C", ["A", "B", "C"], {"A,B": 15.0, "A,C": 15.0, "B,C": 0.0}),
        ("AB chi N = 0", ["A", "B"], {"A,B": 0.0}),
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

            dH = mpt.compute_h_deriv_chin(chi_n, omega, {t: phi[i] for i, t in enumerate(types)})
            errors = []
            for key in chi_n:
                a, b = [types.index(t) for t in key.split(",")]
                errors.append(abs(dH[key] - np.mean(phi[a] * phi[b])))
            check(max(errors) < 1e-9,
                  f"{name} {model}: dH/dchiN = <phi_a phi_b> (max error {max(errors):.1e})")

    test_dropped_fields_and_keys()
    test_saddle_point_model({"A,B": 1.0, "A,C": 0.6, "B,C": 1.4})
    test_saddle_point_model({"A,B": 1.0, "A,C": 1.0, "B,C": 1.00001})
    print(f"\n{n_pass} passed, {n_fail} failed")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
