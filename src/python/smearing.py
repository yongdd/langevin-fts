"""Smearing functions for finite-range interactions in polymer field theory.

This module provides the Smearing class for implementing finite-range
interactions by convolving fields with a smearing function in Fourier space.

References
----------
- Delaney & Fredrickson, J. Phys. Chem. B 120, 7615 (2016)
- Matsen et al., J. Chem. Phys. 164, 014905 (2026)
"""

import numpy as np


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
        - a_int : float
            Smearing length scale in units of R₀.

        For Sigmoidal smearing:

        - type : str
            "sigmoidal"
        - k_int : float
            Cutoff wavenumber.
        - dk_int : float, optional
            Transition width (default: 5.0).

    Attributes
    ----------
    enabled : bool
        Whether smearing is enabled.
    type : str or None
        Smearing type ("gaussian" or "sigmoidal").
    fourier : ndarray or None
        Smearing function in Fourier space Γ(k).
    params : dict or None
        Smearing parameters.
    k_mag : ndarray or None
        Magnitude of k vectors at each grid point.
    k_sq : ndarray or None
        Square of k magnitude at each grid point.

    Examples
    --------
    >>> # Create Gaussian smearing
    >>> smear = Smearing([32, 32, 32], [4.0, 4.0, 4.0],
    ...                  {"type": "gaussian", "a_int": 0.1})
    >>> w_smeared = smear.apply(w_field)

    >>> # Create sigmoidal smearing
    >>> smear = Smearing([32, 32, 32], [4.0, 4.0, 4.0],
    ...                  {"type": "sigmoidal", "k_int": 10.0})

    >>> # Disable smearing
    >>> smear = Smearing([32, 32, 32], [4.0, 4.0, 4.0], None)
    >>> assert not smear.enabled
    """

    def __init__(self, nx, lx, smearing_params):
        """Initialize smearing function.

        Parameters
        ----------
        nx : list of int
            Number of grid points in each dimension.
        lx : list of float
            Box dimensions in each direction.
        smearing_params : dict or None
            Smearing configuration dictionary or None to disable.
        """
        self.nx = list(nx)
        self.lx = list(lx)
        self.n_grid = int(np.prod(nx))
        self.dim = len(nx)

        if smearing_params is None:
            self.enabled = False
            self.fourier = None
            self.type = None
            self.params = None
            self.k_mag = None
            self.k_sq = None
            return

        self.enabled = True
        self.type = smearing_params.get("type", "gaussian")

        # Compute k vectors for each dimension
        # k = 2π·n/L where n is the frequency index from fftfreq
        k_vectors = []
        for i in range(self.dim):
            freq = np.fft.fftfreq(nx[i], d=lx[i] / nx[i])  # cycles per unit length
            k = 2 * np.pi * freq  # angular wavenumber
            k_vectors.append(k)

        # Create k² grid using meshgrid with proper indexing
        if self.dim == 1:
            self.k_sq = k_vectors[0] ** 2
            self.k_mag = np.abs(k_vectors[0])
        elif self.dim == 2:
            kx, ky = np.meshgrid(k_vectors[0], k_vectors[1], indexing='ij')
            self.k_sq = kx ** 2 + ky ** 2
            self.k_mag = np.sqrt(self.k_sq)
        else:  # 3D
            kx, ky, kz = np.meshgrid(k_vectors[0], k_vectors[1], k_vectors[2], indexing='ij')
            self.k_sq = kx ** 2 + ky ** 2 + kz ** 2
            self.k_mag = np.sqrt(self.k_sq)

        # Initialize smearing function based on type
        if self.type == "gaussian":
            self._init_gaussian(smearing_params)
        elif self.type == "sigmoidal":
            self._init_sigmoidal(smearing_params)
        else:
            raise ValueError(f"Unknown smearing type: {self.type}. "
                           f"Supported types: 'gaussian', 'sigmoidal'")

    def _init_gaussian(self, smearing_params):
        """Initialize Gaussian smearing: Γ(k) = exp(-a_int²·k²/2).

        Parameters
        ----------
        smearing_params : dict
            Must contain 'a_int' key.
        """
        if "a_int" not in smearing_params:
            raise ValueError("Gaussian smearing requires 'a_int' parameter.")

        a_int = smearing_params["a_int"]
        self.fourier = np.exp(-a_int ** 2 * self.k_sq / 2)
        self.params = {"a_int": a_int}
        print(f"Smearing: Gaussian, a_int = {a_int}")

    def _init_sigmoidal(self, smearing_params):
        """Initialize sigmoidal smearing: Γ(k) = C₀[1 - tanh((k - k_int)/Δk_int)].

        Parameters
        ----------
        smearing_params : dict
            Must contain 'k_int' key. Optional 'dk_int' (default: 5.0).
        """
        if "k_int" not in smearing_params:
            raise ValueError("Sigmoidal smearing requires 'k_int' parameter.")

        k_int = smearing_params["k_int"]
        dk_int = smearing_params.get("dk_int", 5.0)

        gamma_unnorm = 1 - np.tanh((self.k_mag - k_int) / dk_int)
        # Normalize so Γ(0) = 1
        c0 = 1.0 / (1 - np.tanh(-k_int / dk_int))
        self.fourier = c0 * gamma_unnorm
        self.params = {"k_int": k_int, "dk_int": dk_int}
        print(f"Smearing: Sigmoidal, k_int = {k_int}, dk_int = {dk_int}")

    def apply(self, field):
        """Apply smearing to a single field array.

        Computes smeared field: w_Γ(r) = FFT⁻¹[FFT[w(r)]·Γ(k)]

        Parameters
        ----------
        field : numpy.ndarray
            Field array of length n_grid (can be complex).

        Returns
        -------
        numpy.ndarray
            Smeared field array of same shape.
            Returns the original field unchanged if smearing is disabled.
        """
        if not self.enabled:
            return field

        # Reshape to grid dimensions
        field_grid = np.reshape(field, self.nx)

        # FFT -> multiply by Γ(k) -> IFFT
        field_fourier = np.fft.fftn(field_grid)
        field_smeared_fourier = field_fourier * self.fourier
        field_smeared = np.fft.ifftn(field_smeared_fourier)

        # Reshape back to flat array
        # Return real part for real input, full complex for complex input
        result = np.reshape(field_smeared, self.n_grid)
        if np.isrealobj(field):
            return np.real(result)
        return result

    def apply_to_dict(self, fields):
        """Apply smearing to a dictionary of fields.

        Parameters
        ----------
        fields : dict
            Dictionary of fields {name: field_array}.

        Returns
        -------
        dict
            Dictionary of smeared fields with same structure.
            Returns the original dict unchanged if smearing is disabled.
        """
        if not self.enabled:
            return fields

        return {key: self.apply(field) for key, field in fields.items()}

    def __repr__(self):
        """Return string representation."""
        if not self.enabled:
            return "Smearing(enabled=False)"
        return f"Smearing(type='{self.type}', params={self.params})"
