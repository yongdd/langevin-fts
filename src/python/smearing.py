"""Smearing functions for finite-range interactions in polymer field theory.

This module provides the Smearing class for implementing finite-range
interactions by convolving fields with a smearing function in Fourier space.
"""

import numpy as np
import scipy.fft


class Smearing:
    """Smearing function for finite-range interactions.

    The smearing function Γ(r) defines finite-range interactions by convolving
    fields: w_Γ(r) = ∫ w(r')Γ(r-r')dr'. In Fourier space: w_Γ(k) = w(k)·Γ(k).

    Smearing helps stabilize field-theoretic simulations by damping high-frequency
    fluctuations that can cause numerical instabilities (hot spots).

    Parameters
    ----------
    nx : list of int
        Number of grid points in each dimension.
    lx : list of float
        Box dimensions in each direction.
    smearing_params : dict or None
        Smearing configuration. If None, smearing is disabled.

        For Gaussian smearing:

        - type : str
            "gaussian"
        - a_int : float or dict
            Smearing length scale in units of R₀, or one length per monomer
            type ({type: a_int}, CL-FTS only); fields and concentrations of a
            type are then smeared with its own Γ_i(k).

        For Sigmoidal smearing:

        - type : str
            "sigmoidal"
        - k_int : float
            Cutoff wavenumber.
        - dk_int : float, optional
            Transition width (default: 5.0).
    angles : list of float or None
        Cell angles in degrees, [alpha, beta, gamma] (or [gamma] in 2D).
        None means an orthogonal cell.
    bc : list of str or None
        Boundary conditions, two per axis as for the propagator solver
        ("periodic", "reflecting", "absorbing"). None means periodic. An axis
        with walls is smeared in the cosine (DCT-II) basis, i.e. with the mirror
        image of the field across the wall, instead of wrapping around the box.
        This holds for absorbing walls too: the propagators are odd across such
        a wall, so the fields and the concentrations (products of two
        propagators) are even.

    Attributes
    ----------
    enabled : bool
        Whether smearing is enabled.
    type : str or None
        Smearing type ("gaussian" or "sigmoidal").
    fourier : ndarray or None
        Smearing function in Fourier space Γ(k); None with per-type lengths.
    species_fourier : dict or None
        Γ_i(k) of each monomer type when a_int is given per type.
    params : dict or None
        Smearing parameters.
    k_mag : ndarray or None
        Magnitude of k vectors at each grid point.
    k_sq : ndarray or None
        Square of k magnitude at each grid point (with the cell metric).
    """

    def __init__(self, nx, lx, smearing_params, angles=None, bc=None):
        """Initialize smearing function.

        Parameters
        ----------
        nx : list of int
            Number of grid points in each dimension.
        lx : list of float
            Box dimensions in each direction.
        smearing_params : dict or None
            Smearing configuration dictionary or None to disable.
        angles : list of float or None
            Cell angles in degrees, [alpha, beta, gamma] (or [gamma] in 2D).
            None means an orthogonal cell.
        bc : list of str or None
            Boundary conditions, two per axis; None means periodic.
        """
        self.nx = list(nx)
        self.lx = list(lx)
        self.n_grid = int(np.prod(nx))
        self.dim = len(nx)

        # Boundary condition of each axis: periodic, or walls (any type: the fields and
        # concentrations are even across a wall, so a wall axis is smeared with the
        # mirror image, in the DCT-II basis)
        if bc is None:
            bc = ["periodic"] * (2 * self.dim)
        if len(bc) != 2 * self.dim:
            raise ValueError(f"bc needs two entries per axis, got {len(bc)} for {self.dim} axes.")
        self.axis_bc = ["periodic" if bc[2 * d].lower() == "periodic" else "wall" for d in range(self.dim)]
        self.periodic = all(b == "periodic" for b in self.axis_bc)

        self.species_fourier = None
        if smearing_params is None:
            self.enabled = False
            self.fourier = None
            self.type = None
            self.params = None
            self.per_species = False
            self.k_mag = None
            self.k_sq = None
            return

        self.enabled = True
        self.type = smearing_params.get("type", "gaussian")

        if self.type == "gaussian":
            if "a_int" not in smearing_params:
                raise ValueError("Gaussian smearing requires 'a_int' parameter.")
            a_int = smearing_params["a_int"]
            self.params = {"a_int": dict(a_int) if isinstance(a_int, dict) else a_int}
            print(f"Smearing: Gaussian, a_int = {self.params['a_int']}")
        elif self.type == "sigmoidal":
            if "k_int" not in smearing_params:
                raise ValueError("Sigmoidal smearing requires 'k_int' parameter.")
            self.params = {"k_int": smearing_params["k_int"], "dk_int": smearing_params.get("dk_int", 5.0)}
            print(f"Smearing: Sigmoidal, k_int = {self.params['k_int']}, dk_int = {self.params['dk_int']}")
        else:
            raise ValueError(f"Unknown smearing type: {self.type}. "
                           f"Supported types: 'gaussian', 'sigmoidal'")
        self.per_species = isinstance(self.params.get("a_int"), dict)

        self.set_box(lx, angles)

    def set_box(self, lx, angles=None):
        """Rebuild Γ(k) for new lattice parameters.

        Γ is a function of the physical wavenumber, so it must be rebuilt after
        every box change (SCFT box relaxation).

        Parameters
        ----------
        lx : list of float
            Box dimensions in each direction.
        angles : list of float or None
            Cell angles in degrees, [alpha, beta, gamma] (or [gamma] in 2D).
        """
        self.lx = list(lx)
        if angles is None:
            angles = [90.0, 90.0, 90.0]
        elif len(angles) == 1:
            angles = [90.0, 90.0, angles[0]]
        self.angles = [float(a) for a in angles]

        # Metric g_ab = a_a . a_b of the lattice vectors of the existing dimensions
        alpha, beta, gamma = np.deg2rad(self.angles)
        cos_full = np.array([[1.0, np.cos(gamma), np.cos(beta)],
                             [np.cos(gamma), 1.0, np.cos(alpha)],
                             [np.cos(beta), np.cos(alpha), 1.0]])
        sin_full = np.array([[0.0, np.sin(gamma), np.sin(beta)],
                             [np.sin(gamma), 0.0, np.sin(alpha)],
                             [np.sin(beta), np.sin(alpha), 0.0]])
        self._cos = cos_full[:self.dim, :self.dim]
        self._sin = sin_full[:self.dim, :self.dim]
        self.orthogonal = bool(np.allclose(self._cos, np.identity(self.dim), rtol=0.0, atol=1e-12))
        if not self.periodic and not self.orthogonal:
            raise ValueError("Smearing with reflecting or absorbing boundaries needs an orthogonal box.")
        if not self.enabled:
            return
        L = np.array(self.lx, dtype=np.float64)
        self._metric_inv = np.linalg.inv(np.outer(L, L) * self._cos)

        # Integer frequencies m (componentwise principal zone), shape (dim, n_grid)
        frequencies = [np.concatenate([np.arange((n + 1) // 2), np.arange(-(n // 2), 0)]) for n in self.nx]
        self._m = np.array(np.meshgrid(*frequencies, indexing="ij")).reshape(self.dim, -1)

        if self.orthogonal:
            # Separable wavenumbers of the transform of each axis: k = 2π·n/L (FFT; the
            # componentwise alias is minimal) or πn/L (DCT-II, wall axes)
            self._k_axes = []
            for i in range(self.dim):
                if self.axis_bc[i] == "periodic":
                    self._k_axes.append(2 * np.pi * np.fft.fftfreq(self.nx[i], d=self.lx[i] / self.nx[i]))
                else:
                    self._k_axes.append(np.pi * np.arange(self.nx[i]) / self.lx[i])
            k_grids = np.meshgrid(*self._k_axes, indexing="ij")
            self.k_sq = sum(k ** 2 for k in k_grids)
        else:
            # |k|^2 = 4π² mᵀ g⁻¹ m with the alias representative that minimizes the form
            # (first Brillouin zone), the same choice as the propagator solvers (Pseudo.cpp)
            k_sq = None
            for m in self._alias_representatives():
                k_sq_m = 4 * np.pi ** 2 * np.einsum("ij,in,jn->n", self._metric_inv, m, m)
                k_sq = k_sq_m if k_sq is None else np.minimum(k_sq, k_sq_m)
            self.k_sq = np.reshape(k_sq, self.nx)
        self.k_mag = np.sqrt(self.k_sq)

        if self.per_species:
            # Γ_i(k) = exp(-a_i²·k²/2) for each monomer type
            self.fourier = None
            self.species_fourier = {name: np.exp(-a ** 2 * self.k_sq / 2) for name, a in self.params["a_int"].items()}
        elif self.type == "gaussian":
            # Γ(k) = exp(-a_int²·k²/2)
            self.fourier = np.exp(-self.params["a_int"] ** 2 * self.k_sq / 2)
        else:
            # Γ(k) = C₀[1 - tanh((k - k_int)/Δk_int)], normalized so that Γ(0) = 1
            k_int, dk_int = self.params["k_int"], self.params["dk_int"]
            c0 = 1.0 / (1 - np.tanh(-k_int / dk_int))
            self.fourier = c0 * (1 - np.tanh((self.k_mag - k_int) / dk_int))

    def _alias_representatives(self):
        """Yield the alias representatives m + N·t, t in {-1, 0, 1} per axis with more than one point."""
        shifts = [(-1, 0, 1) if n > 1 else (0,) for n in self.nx]
        for t in np.array(np.meshgrid(*shifts, indexing="ij")).reshape(self.dim, -1).T:
            yield self._m + (t * np.array(self.nx))[:, None]

    def stress(self, fields, concentrations):
        """Lattice derivatives of H through the smearing kernel Γ(k).

        H contains -ln Q[Γ*w], so at fixed fields w it depends on the box also through
        Γ(k(θ)). The propagator solver's stress is the derivative at fixed smeared fields;
        the remaining part is Σ_i <φ_i · (∂Γ/∂θ * w_i)>, where < > is the grid mean.
        At alias ties (Brillouin-zone boundary) the derivative of k² is averaged over the
        tied representatives, which is the symmetric derivative of the minimum.

        Parameters
        ----------
        fields : sequence of ndarray
            Real monomer potential fields w_i (unsmeared), each of length n_grid.
        concentrations : sequence of ndarray
            Monomer concentrations φ_i in the same order.

        Returns
        -------
        ndarray
            Array of 6 in the get_stress() layout: 3D [L1, L2, L3, γ, β, α],
            2D [L1, L2, γ], 1D [L1]; angles per radian; unused entries are 0.
        """
        result = np.zeros(6)
        if not self.enabled:
            return result
        if self.per_species:
            raise NotImplementedError("The stress of per-type smearing lengths is not implemented.")

        # <φ (D*w)> = (1/n²) Σ_k conj(φ(k)) D(k) w(k) for a real, even multiplier D
        # (with the orthonormal transforms of wall axes: (1/n) Σ_k conj(φ(k)) D(k) w(k))
        correlation = 0.0
        for w, phi in zip(fields, concentrations):
            if self.periodic:
                w_k = np.fft.fftn(np.reshape(w, self.nx))
                phi_k = np.fft.fftn(np.reshape(phi, self.nx))
            else:
                w_k = self._forward(np.reshape(w, self.nx)) * np.sqrt(self.n_grid)
                phi_k = self._forward(np.reshape(phi, self.nx)) * np.sqrt(self.n_grid)
            correlation = correlation + np.real(np.conj(phi_k) * w_k)
        correlation = np.ravel(correlation) / self.n_grid ** 2

        # dΓ/d(k²)
        k_sq = np.ravel(self.k_sq)
        if self.type == "gaussian":
            dgamma_dk_sq = -0.5 * self.params["a_int"] ** 2 * np.ravel(self.fourier)
        else:
            k_int, dk_int = self.params["k_int"], self.params["dk_int"]
            c0 = 1.0 / (1 - np.tanh(-k_int / dk_int))
            k = np.sqrt(k_sq)
            dgamma_dk = -c0 / dk_int * (1 - np.tanh((k - k_int) / dk_int) ** 2)
            dgamma_dk_sq = np.divide(dgamma_dk, 2 * k, out=np.zeros_like(k), where=k > 0)

        weight = dgamma_dk_sq * correlation
        for slot, dk_sq in self._lattice_derivatives_k_sq().items():
            result[slot] = np.sum(weight * dk_sq)
        return result

    def _lattice_derivatives_k_sq(self):
        """∂(k²)/∂θ on the grid for each lattice parameter, keyed by its stress slot.

        k² = 4π² mᵀ g⁻¹ m with g_ab = L_a L_b cos θ_ab gives
        ∂k²/∂θ = -4π² uᵀ (∂g/∂θ) u, u = g⁻¹ m. At alias ties (Brillouin-zone
        boundary) the derivative is averaged over the tied representatives, which is
        the symmetric derivative of the minimum over the aliases.
        """
        angle_slots = {2: {(0, 1): 2}, 3: {(0, 1): 3, (0, 2): 4, (1, 2): 5}}.get(self.dim, {})
        L = np.array(self.lx, dtype=np.float64)

        if self.orthogonal:
            # k_a ∝ 1/L_a: ∂k²/∂L_a = -2 k_a²/L_a and ∂k²/∂θ_ab = 2 k_a k_b. The only
            # ties are the Nyquist planes (m_a = ∓N_a/2), where the two aliases give k_a of
            # opposite sign, so the angle derivatives average to zero there. A box with
            # walls keeps its angles at 90 degrees (no angle derivatives).
            k = np.array([np.ravel(k_grid) for k_grid in np.meshgrid(*self._k_axes, indexing="ij")])
            nyquist = np.array([(2 * self._m[a] == -self.nx[a]) & (self.nx[a] > 1) for a in range(self.dim)])
            derivatives = {a: -2 * k[a] ** 2 / L[a] for a in range(self.dim)}
            for (a, b), slot in angle_slots.items():
                if self.periodic:
                    derivatives[slot] = np.where(nyquist[a] | nyquist[b], 0.0, 2 * k[a] * k[b])
                else:
                    derivatives[slot] = np.zeros(self.n_grid)
            return derivatives

        k_sq = np.ravel(self.k_sq)
        derivatives = {slot: np.zeros(self.n_grid) for slot in list(range(self.dim)) + list(angle_slots.values())}
        n_tied = np.zeros(self.n_grid)
        for m in self._alias_representatives():
            u = self._metric_inv @ m
            tied = np.flatnonzero(np.isclose(4 * np.pi ** 2 * np.sum(m * u, axis=0), k_sq, rtol=1e-10, atol=1e-12))
            n_tied[tied] += 1
            u = u[:, tied]
            for a in range(self.dim):
                derivatives[a][tied] += -8 * np.pi ** 2 * u[a] * np.sum((self._cos[a] * L)[:, None] * u, axis=0)
            for (a, b), slot in angle_slots.items():
                derivatives[slot][tied] += 8 * np.pi ** 2 * u[a] * u[b] * L[a] * L[b] * self._sin[a, b]
        return {slot: d / n_tied for slot, d in derivatives.items()}

    def kernel(self, name=None):
        """Γ(k) on the grid, of monomer type `name` when the lengths are given per type."""
        if not self.per_species:
            return self.fourier
        if name not in self.species_fourier:
            raise ValueError(f"No smearing length for monomer type '{name}'.")
        return self.species_fourier[name]

    def apply(self, field, name=None):
        """Apply smearing to a single field array.

        Computes smeared field: w_Γ(r) = FFT⁻¹[FFT[w(r)]·Γ(k)]

        Parameters
        ----------
        field : numpy.ndarray
            Field array of length n_grid (can be complex).
        name : str, optional
            Monomer type of the field; required for per-type smearing lengths.

        Returns
        -------
        numpy.ndarray
            Smeared field array of same shape.
            Returns the original field unchanged if smearing is disabled.
        """
        if not self.enabled:
            return field
        gamma = self.kernel(name)

        # Reshape to grid dimensions
        field_grid = np.reshape(field, self.nx)

        # FFT -> multiply by Γ(k) -> IFFT (DCT-II along wall axes)
        if self.periodic:
            field_fourier = np.fft.fftn(field_grid)
            field_smeared_fourier = field_fourier * gamma
            field_smeared = np.fft.ifftn(field_smeared_fourier)
        else:
            field_smeared = self._backward(self._forward(field_grid) * gamma)

        # Reshape back to flat array
        # Return real part for real input, full complex for complex input
        result = np.reshape(field_smeared, self.n_grid)
        if np.isrealobj(field):
            return np.ascontiguousarray(np.real(result))
        return result

    def _forward(self, field):
        """Orthonormal transform of each axis: FFT (periodic) or DCT-II (walls)."""
        for axis, kind in enumerate(self.axis_bc):
            if kind == "periodic":
                field = scipy.fft.fft(field, axis=axis, norm="ortho")
            else:
                field = scipy.fft.dct(field, type=2, axis=axis, norm="ortho")
        return field

    def _backward(self, field):
        """Inverse of _forward."""
        for axis, kind in enumerate(self.axis_bc):
            if kind == "periodic":
                field = scipy.fft.ifft(field, axis=axis, norm="ortho")
            else:
                field = scipy.fft.idct(field, type=2, axis=axis, norm="ortho")
        return field

    def apply_to_dict(self, fields):
        """Apply smearing to a dictionary of fields.

        Parameters
        ----------
        fields : dict
            Dictionary of fields {name: field_array}; with per-type lengths the
            names are monomer types.

        Returns
        -------
        dict
            Dictionary of smeared fields with same structure.
            Returns the original dict unchanged if smearing is disabled.
        """
        if not self.enabled:
            return fields

        return {key: self.apply(field, key) for key, field in fields.items()}

    def __repr__(self):
        """Return string representation."""
        if not self.enabled:
            return "Smearing(enabled=False)"
        return f"Smearing(type='{self.type}', params={self.params})"
