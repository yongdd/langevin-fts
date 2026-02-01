"""Well-Tempered Metadynamics (WTMD) for L-FTS simulations.

This module implements well-tempered metadynamics to enhance sampling
in Langevin field-theoretic simulations of block copolymers.

References
----------
T. M. Beardsley and M. W. Matsen, "Well-tempered metadynamics applied to
field-theoretic simulations of diblock copolymer melts",
J. Chem. Phys. 157, 114902 (2022).
"""

import numpy as np
from typing import List, Optional, Tuple


class WTMD:
    """Well-Tempered Metadynamics for enhanced sampling in L-FTS.

    Adds Gaussian hills to bias the order parameter, enabling efficient
    exploration of phase space and calculation of free energy landscapes.

    Parameters
    ----------
    nx : list of int
        Grid dimensions [nx, ny, nz].
    lx : list of float
        Box dimensions [lx, ly, lz].
    ell : int, optional
        Order parameter exponent. Default: 4.
    sigma_psi : float, optional
        Gaussian width for hills. Default: 40.0.
    delta_t : float, optional
        Well-tempering factor (ΔT). Default: 5.0.
    kc : float, optional
        Wavenumber cutoff for order parameter. Default: 6.02.
    psi_min : float, optional
        Minimum Ψ for bias histogram. Default: 0.0.
    psi_max : float, optional
        Maximum Ψ for bias histogram. Default: 500.0.
    n_bins : int, optional
        Number of bins for bias histogram. Default: 5000.
    update_freq : int, optional
        Hill deposition frequency (steps). Default: 1000.

    Attributes
    ----------
    u : ndarray
        Bias potential U(Ψ).
    up : ndarray
        Derivative of bias potential dU/dΨ.

    Examples
    --------
    >>> wtmd = WTMD(nx=[32, 32, 32], lx=[4.0, 4.0, 4.0])
    >>> # In L-FTS loop:
    >>> psi = wtmd.get_psi(w_minus)
    >>> bias_field = wtmd.get_bias_field(w_minus)
    >>> wtmd.update_bias(psi)
    """

    def __init__(
        self,
        nx: List[int],
        lx: List[float],
        ell: int = 4,
        sigma_psi: float = 40.0,
        delta_t: float = 5.0,
        kc: float = 6.02,
        psi_min: float = 0.0,
        psi_max: float = 500.0,
        n_bins: int = 5000,
        update_freq: int = 1000,
    ):
        self.nx = np.array(nx)
        self.lx = np.array(lx)
        self.dim = len(nx)
        self.n_grid = np.prod(nx)
        self.volume = np.prod(lx)

        # WTMD parameters
        self.ell = ell
        self.sigma_psi = sigma_psi
        self.delta_t = delta_t
        self.kc = kc
        self.update_freq = update_freq

        # Bias histogram
        self.psi_min = psi_min
        self.psi_max = psi_max
        self.n_bins = n_bins
        self.d_psi = (psi_max - psi_min) / n_bins
        self.psi_bins = np.linspace(psi_min, psi_max, n_bins)

        # Bias potential and derivative
        self.u = np.zeros(n_bins)      # U(Ψ)
        self.up = np.zeros(n_bins)     # dU/dΨ

        # Normalization integrals (for reweighting)
        self.I0 = np.zeros(n_bins)     # ∫ exp(-U/ΔT) dΨ
        self.I1 = np.zeros(n_bins)     # ∫ Ψ exp(-U/ΔT) dΨ

        # Step counter
        self.step = 0

        # Setup Fourier space arrays
        self._setup_fourier_space()

    def _setup_fourier_space(self):
        """Initialize wavenumber arrays and filters."""
        # Grid spacing
        dx = self.lx / self.nx

        # Wavenumber arrays for each dimension
        if self.dim == 3:
            kx = 2 * np.pi * np.fft.fftfreq(self.nx[0], dx[0])
            ky = 2 * np.pi * np.fft.fftfreq(self.nx[1], dx[1])
            kz = 2 * np.pi * np.fft.rfftfreq(self.nx[2], dx[2])

            KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
            self.K = np.sqrt(KX**2 + KY**2 + KZ**2)

            # Weighting for rfft (last dimension has half the modes)
            self.wt = 2.0 * np.ones_like(self.K)
            self.wt[:, :, 0] = 1.0
            if self.nx[2] % 2 == 0:
                self.wt[:, :, -1] = 1.0

        elif self.dim == 2:
            kx = 2 * np.pi * np.fft.fftfreq(self.nx[0], dx[0])
            ky = 2 * np.pi * np.fft.rfftfreq(self.nx[1], dx[1])

            KX, KY = np.meshgrid(kx, ky, indexing='ij')
            self.K = np.sqrt(KX**2 + KY**2)

            self.wt = 2.0 * np.ones_like(self.K)
            self.wt[:, 0] = 1.0
            if self.nx[1] % 2 == 0:
                self.wt[:, -1] = 1.0

        elif self.dim == 1:
            kx = 2 * np.pi * np.fft.rfftfreq(self.nx[0], dx[0])
            self.K = np.abs(kx)

            self.wt = 2.0 * np.ones_like(self.K)
            self.wt[0] = 1.0
            if self.nx[0] % 2 == 0:
                self.wt[-1] = 1.0

        # Low-pass filter
        self.fk = np.where(self.K < self.kc, 1.0, 0.0)

    def get_psi(self, w_minus: np.ndarray) -> float:
        """Calculate order parameter Ψ from exchange field.

        Parameters
        ----------
        w_minus : ndarray
            Exchange field w- in real space, shape (n_grid,).

        Returns
        -------
        psi : float
            Order parameter value.

        Notes
        -----
        The order parameter is defined as:

        .. math::
            \\Psi = \\left( \\sum_k |\\tilde{w}_k|^\\ell f_k w_t / M \\right)^{1/\\ell}

        where f_k is a low-pass filter and w_t accounts for rfft weighting.
        """
        # Reshape to grid
        w = w_minus.reshape(self.nx)

        # FFT to Fourier space
        w_k = np.fft.rfftn(w)

        # Order parameter: sum of |w_k|^ell with filter
        psi_sum = np.sum(np.abs(w_k)**self.ell * self.fk * self.wt)
        psi = psi_sum**(1.0 / self.ell) / self.n_grid

        return psi

    def get_bias_field(self, w_minus: np.ndarray) -> np.ndarray:
        """Calculate bias field contribution for Langevin dynamics.

        Parameters
        ----------
        w_minus : ndarray
            Exchange field w- in real space, shape (n_grid,).

        Returns
        -------
        bias_field : ndarray
            Bias field -dU/dw in real space, shape (n_grid,).

        Notes
        -----
        The bias field is computed as:

        .. math::
            -\\frac{dU}{dw} = -\\frac{dU}{d\\Psi} \\frac{d\\Psi}{dw}
        """
        # Reshape to grid
        w = w_minus.reshape(self.nx)

        # FFT to Fourier space
        w_k = np.fft.rfftn(w)

        # Calculate Ψ
        psi_sum = np.sum(np.abs(w_k)**self.ell * self.fk * self.wt)
        psi = psi_sum**(1.0 / self.ell) / self.n_grid

        # Get dU/dΨ from interpolation
        du_dpsi = self._interpolate_derivative(psi)

        # dΨ/dw_k (chain rule for |w_k|^ℓ derivative)
        with np.errstate(divide='ignore', invalid='ignore'):
            dpsi_dwk = (
                np.abs(w_k)**(self.ell - 2) *
                psi**(1.0 - self.ell) *
                w_k * self.fk
            )
            dpsi_dwk = np.nan_to_num(dpsi_dwk, nan=0.0, posinf=0.0, neginf=0.0)

        # IFFT to real space
        dpsi_dw = np.fft.irfftn(dpsi_dwk, s=self.nx) * self.n_grid**(2.0 - self.ell) / self.volume

        # Bias field: V * dU/dΨ * dΨ/dw (same as deep-langevin-fts)
        bias_field = self.volume * du_dpsi * dpsi_dw.flatten()

        return bias_field

    def _interpolate_derivative(self, psi: float) -> float:
        """Interpolate dU/dΨ at given Ψ value."""
        if psi <= self.psi_min:
            return self.up[0]
        elif psi >= self.psi_max:
            return self.up[-1]

        # Linear interpolation
        idx = (psi - self.psi_min) / self.d_psi
        i = int(idx)
        i = min(i, self.n_bins - 2)
        frac = idx - i

        return (1.0 - frac) * self.up[i] + frac * self.up[i + 1]

    def _interpolate_bias(self, psi: float) -> float:
        """Interpolate U(Ψ) at given Ψ value."""
        if psi <= self.psi_min:
            return self.u[0]
        elif psi >= self.psi_max:
            return self.u[-1]

        idx = (psi - self.psi_min) / self.d_psi
        i = int(idx)
        i = min(i, self.n_bins - 2)
        frac = idx - i

        return (1.0 - frac) * self.u[i] + frac * self.u[i + 1]

    def update_bias(self, psi: float):
        """Add Gaussian hill to bias potential (well-tempered).

        Parameters
        ----------
        psi : float
            Current order parameter value.

        Notes
        -----
        In well-tempered metadynamics, the hill height decreases as:

        .. math::
            h(\\Psi) = h_0 \\exp(-U(\\Psi) / \\Delta T)

        This ensures eventual convergence to the free energy surface.
        """
        self.step += 1

        if self.step % self.update_freq != 0:
            return

        # Well-tempered amplitude
        u_current = self._interpolate_bias(psi)

        # Normalization factor (prevent division by zero)
        cv = np.sum(np.exp(-self.u / self.delta_t)) * self.d_psi
        cv = max(cv, 1e-10)

        amplitude = np.exp(-u_current / self.delta_t) / cv

        # Add Gaussian to each bin
        for i in range(self.n_bins):
            psi_i = self.psi_bins[i]
            temp = (psi_i - psi) / self.sigma_psi
            gaussian = np.exp(-0.5 * temp**2)

            # Update bias potential
            self.u[i] += amplitude * gaussian

            # Update derivative (dU/dΨ)
            self.up[i] += amplitude * gaussian * (-temp / self.sigma_psi)

        # Update normalization integrals
        exp_factor = np.exp(-self.u / self.delta_t)
        self.I0 = np.cumsum(exp_factor) * self.d_psi
        self.I1 = np.cumsum(self.psi_bins * exp_factor) * self.d_psi

    def get_free_energy(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get free energy estimate from bias potential.

        Returns
        -------
        psi_bins : ndarray
            Order parameter values.
        free_energy : ndarray
            Free energy F(Ψ) = -U(Ψ) * (1 + ΔT) / ΔT.
        """
        # In well-tempered metadynamics:
        # F(Ψ) = -(1 + ΔT/T) * U(Ψ) = -(1 + ΔT) * U(Ψ) (with T=1)
        free_energy = -(1.0 + self.delta_t) / self.delta_t * self.u

        return self.psi_bins.copy(), free_energy

    def save(self, filename: str):
        """Save WTMD state to file.

        Parameters
        ----------
        filename : str
            Output filename (.npz format).
        """
        np.savez(
            filename,
            nx=self.nx,
            lx=self.lx,
            ell=self.ell,
            sigma_psi=self.sigma_psi,
            delta_t=self.delta_t,
            kc=self.kc,
            psi_min=self.psi_min,
            psi_max=self.psi_max,
            n_bins=self.n_bins,
            update_freq=self.update_freq,
            u=self.u,
            up=self.up,
            I0=self.I0,
            I1=self.I1,
            step=self.step,
        )

    @classmethod
    def load(cls, filename: str) -> 'WTMD':
        """Load WTMD state from file.

        Parameters
        ----------
        filename : str
            Input filename (.npz format).

        Returns
        -------
        wtmd : WTMD
            Restored WTMD instance.
        """
        data = np.load(filename)

        wtmd = cls(
            nx=data['nx'].tolist(),
            lx=data['lx'].tolist(),
            ell=int(data['ell']),
            sigma_psi=float(data['sigma_psi']),
            delta_t=float(data['delta_t']),
            kc=float(data['kc']),
            psi_min=float(data['psi_min']),
            psi_max=float(data['psi_max']),
            n_bins=int(data['n_bins']),
            update_freq=int(data['update_freq']),
        )

        wtmd.u = data['u']
        wtmd.up = data['up']
        wtmd.I0 = data['I0']
        wtmd.I1 = data['I1']
        wtmd.step = int(data['step'])

        return wtmd
