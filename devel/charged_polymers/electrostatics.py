"""Electrostatics for charged-polymer (polyelectrolyte) field-theoretic simulations.

Implements the smeared-charge, compressible model described in the
charged-polymer theory notes: every species i carries a Gaussian shape
function h_i of radius a_i (its smearing length / Born radius), all
interactions couple to the smeared densities phibar_i = h_i * phi_i, and the
Coulomb interaction is decoupled with an electrostatic potential field psi
that satisfies the (smeared) Poisson equation at its partial saddle point:

    -laplacian(psi) = E * c(r),      c = sum_i z_i (h_i * phi_i)

with the per-chain coupling E = 4 pi l_B rho_0 N R0^2 = sqrt(nbar) * E0,
E0 = 4 pi l_B N^2 / R0. psi is an imaginary-type field (stored as a real
array on the rotated contour) updated by a screening-preconditioned Newton
step per saddle iteration; it must NOT be added to the Anderson-mixing
compressor state.

All operations are diagonal in k-space; no real-space gradient of psi is
ever formed. The explicit Hamiltonian term is -(1/2V) int c psi dr (the
rotated -|grad psi|^2/2E, by the Poisson equation); the +c psi coupling is
carried inside -ln Q via the propagator inputs.

References: Wang, PRE 81, 021501 (2010); Riggleman, Kumar & Fredrickson,
JCP 136, 024903 (2012); Villet & Fredrickson, JCP 141, 224115 (2014).
"""

import numpy as np

from polymerfts.validation import ValidationError


class Electrostatics:
    """Per-species smearing + electrostatic potential for charged L-FTS.

    Parameters
    ----------
    nx, lx : list
        Grid points and box lengths (orthogonal box; R0 units).
    monomer_types : list of str
        All monomer type names.
    charges : dict
        Per-segment valence z_i per monomer type; None or 0 = neutral.
    radiuses : dict
        Smearing radius a_i (R0 units) per monomer type; None = no smearing
        (h_i = delta; acceptable only for testing/mean-field).
    e_coupling : float
        The per-chain electrostatic coupling E = 4 pi l_B rho_0 N R0^2.
    species_fractions : dict, optional
        Overall segment volume fraction per monomer type. Used to build the
        screening preconditioner for the psi update; if omitted, the update
        falls back to the undamped exact solve (only stable for weak
        coupling E).
    """

    def __init__(self, nx, lx, monomer_types, charges, radiuses, e_coupling,
                 species_fractions=None):
        self.nx = list(nx)
        self.lx = list(lx)
        self.dim = len(nx)
        self.n_grid = int(np.prod(nx))
        self.monomer_types = list(monomer_types)
        self.e_coupling = float(e_coupling)

        self.charges = {t: float(charges.get(t) or 0.0) for t in monomer_types}
        self.radiuses = {t: radiuses.get(t) for t in monomer_types}

        # k grids (full complex-FFT layout, same convention as Smearing)
        k_vectors = [2.0 * np.pi * np.fft.fftfreq(nx[i], d=lx[i] / nx[i])
                     for i in range(self.dim)]
        mesh = np.meshgrid(*k_vectors, indexing="ij")
        self.k_sq = sum(k * k for k in mesh)

        # Per-species Gaussian shape functions h_i(k) = exp(-a_i^2 k^2 / 2)
        self.h_hat = {}
        for t in monomer_types:
            a = self.radiuses.get(t)
            if a is None:
                self.h_hat[t] = None  # h = delta
            else:
                self.h_hat[t] = np.exp(-0.5 * float(a) ** 2 * self.k_sq)

        # Screening preconditioner for the psi fixed-point update.
        # A bare replacement psi = E c/k^2 inside the saddle loop has local
        # feedback gain g(k) ~ E sum_i z_i^2 phibar_i hhat_i^2 / k^2, which
        # exceeds 1 at long wavelengths for realistic E and diverges. The
        # Newton-preconditioned update
        #   psi_new = psi + (E c_hat - k^2 psi_hat) / (k^2 + E S_scr(k))
        # with the ideal (local) screening estimate
        #   S_scr(k) = sum_i z_i^2 phibar_i hhat_i^2
        # has the SAME fixed point (-lap psi = E c) but damps the unstable
        # modes; convergence is monitored via `residual`.
        scr = np.zeros_like(self.k_sq)
        if species_fractions is not None:
            for t in monomer_types:
                z = self.charges[t]
                if z != 0.0:
                    h = self.h_hat[t]
                    h2 = 1.0 if h is None else h * h
                    scr = scr + z * z * float(species_fractions.get(t, 0.0)) * h2
        self.jacobian = self.k_sq + self.e_coupling * scr
        self.jacobian[tuple([0] * self.dim)] = 1.0  # k=0 gauged, any nonzero

        # Electrostatic potential (real array, per-reference-chain units)
        self.psi = np.zeros(self.n_grid, dtype=np.float64)

    # ------------------------------------------------------------------ #
    def smear(self, monomer_type, field):
        """h_i * field for the given species (k-space multiplication).

        Accepts real or complex fields (complex is needed for CL-FTS);
        real input returns a real array, complex input stays complex.
        """
        h = self.h_hat.get(monomer_type)
        arr = np.asarray(field)
        is_complex = np.iscomplexobj(arr)
        dtype = np.complex128 if is_complex else np.float64
        if h is None:
            return np.ascontiguousarray(arr.astype(dtype).reshape(-1))
        f = np.reshape(arr.astype(dtype), self.nx)
        out = np.fft.ifftn(np.fft.fftn(f) * h)
        if not is_complex:
            out = out.real
        # .real is a strided view into the complex buffer; force a contiguous
        # copy so downstream C++ bindings read the correct data.
        return np.ascontiguousarray(out.reshape(-1))

    def smear_dict(self, fields):
        """Per-species smearing of a {monomer_type: field} dict.

        Types without an entry in `radiuses` (e.g. random-copolymer
        pseudo-types) pass through unsmeared.
        """
        out = {}
        for t, f in fields.items():
            if t in self.h_hat:
                out[t] = self.smear(t, f)
            else:
                out[t] = f
        return out

    # ------------------------------------------------------------------ #
    def charge_density(self, phi):
        """Smeared charge density c = sum_i z_i (h_i * phi_i).

        Real for real phi (L-FTS), complex for complex phi (CL-FTS).
        """
        any_complex = any(np.iscomplexobj(np.asarray(phi[t]))
                          for t in self.monomer_types)
        c = np.zeros(self.n_grid,
                     dtype=np.complex128 if any_complex else np.float64)
        for t in self.monomer_types:
            z = self.charges[t]
            if z != 0.0:
                c += z * self.smear(t, phi[t])
        return c

    def solve_psi(self, c):
        """One preconditioned Newton update of psi toward -lap(psi) = E c.

        psi_hat += (E c_hat - k^2 psi_hat) / (k^2 + E S_scr(k)); the k=0 mode
        stays gauged to zero. The fixed point is the exact Poisson solution;
        the screening term in the denominator keeps the outer saddle loop
        stable at strong coupling. Updates and returns self.psi.
        """
        c_hat = np.fft.fftn(np.reshape(c, self.nx))
        psi_hat = np.fft.fftn(np.reshape(self.psi, self.nx))
        rhs = self.e_coupling * c_hat - self.k_sq * psi_hat
        psi_hat = psi_hat + rhs / self.jacobian
        psi_hat[tuple([0] * self.dim)] = 0.0
        self.psi = np.ascontiguousarray(
            np.fft.ifftn(psi_hat).real.reshape(-1))
        return self.psi

    def residual(self, c):
        """Poisson residual  -lap(psi)/E - c  with the CURRENT psi.

        Zero immediately after solve_psi(c) for the same c; nonzero once phi
        (hence c) has moved in the outer saddle iteration. Feed its std into
        the saddle stopping criterion.
        """
        psi_hat = np.fft.fftn(np.reshape(self.psi, self.nx))
        lap = np.ascontiguousarray(
            np.fft.ifftn(-self.k_sq * psi_hat).real.reshape(-1))
        return -lap / self.e_coupling - c

    def hamiltonian_per_chain(self, c):
        """Explicit electrostatic Hamiltonian term: -(1/2V) int c psi dr.

        psi is an imaginary-type field stored as a real array (rotated
        contour, psi_phys = i psi). The explicit term |grad psi_phys|^2/(2E)
        therefore continues to -(grad psi)^2/(2E), which by the Poisson
        equation equals -(1/2) c psi per volume. (The +c psi coupling part
        lives inside -ln Q through W_i = h_i*(W_SPT + z_i psi), so it must
        NOT be added here.) This mirrors the negated quadratic coefficients
        used for the imaginary SPT fields in SymmetricPolymerTheory.
        Assumes self.psi is consistent with c.
        """
        return -0.5 * float(np.mean(c * self.psi))

    def charge_potential(self, monomer_type):
        """Species' electrostatic one-body term z_i * (h_i * psi).

        The full propagator input is W_i = h_i * (W_i^SPT + z_i psi); this
        helper returns only the z_i * psi part BEFORE the outer smearing —
        add it to W_i^SPT and smear the sum with `smear()`.
        """
        return self.charges[monomer_type] * self.psi

    # ------------------------------------------------------------------ #
    @staticmethod
    def check_electroneutrality(species_fractions, charges, tol=1e-10):
        """Validate sum_i z_i phibar_i = 0 (per-segment counting).

        Parameters
        ----------
        species_fractions : dict
            Overall segment volume fraction per monomer type.
        charges : dict
            Per-segment valences (None = 0).
        """
        total = 0.0
        for t, frac in species_fractions.items():
            z = charges.get(t)
            if z:
                total += float(z) * float(frac)
        if abs(total) > tol:
            raise ValidationError(
                f"System is not electroneutral: sum_i z_i * phi_i = {total:.6e}. "
                "The counter-ion volume fraction is not a free parameter - "
                "compute it from the polymer charge (see the charged-polymer "
                "theory notes).")
