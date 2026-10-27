"""Well-Tempered Metadynamics (WTMD) for L-FTS simulations.

This module implements well-tempered metadynamics to enhance sampling
in Langevin field-theoretic simulations of block copolymers.

This implementation follows deep-langevin-fts, except for the half-spectrum weight
of the last kz plane when nz is odd (see WTMD.__init__).
"""

import numpy as np

from .utils import atomic_savemat
from .validation import ValidationError


class WTMD:
    """Well-Tempered Metadynamics for enhanced sampling in L-FTS.

    This implementation matches deep-langevin-fts up to the odd-nz weight (see the
    module docstring). With `fourier`, the PropagatorSolver.fourier of the simulation
    (real fields, periodic box), the order parameter and its derivative are transformed
    on the platform (on the GPU with CUDA) instead of with numpy.fft, in the same rfftn
    layout.
    """

    def __init__(self, nx, lx, nbar,
                 eigenvalues, real_fields_idx,
                 l=4,
                 kc=6.02,
                 dT=5.0,
                 sigma_psi=0.16,
                 psi_min=0.0,
                 psi_max=10.0,
                 dpsi=1e-3,
                 update_freq=1000,
                 recording_period=100000,
                 u=None, up=None, I0=None, I1=None, angles=None, fourier=None):

        self.l = l
        self.kc = kc
        self.sigma_psi = sigma_psi
        self.dT = dT
        self.psi_min = psi_min
        self.psi_max = psi_max
        self.dpsi = dpsi
        self.bins = int(np.round((psi_max - psi_min) / dpsi))
        self.update_freq = update_freq
        self.recording_period = recording_period

        if len(nx) != 3:
            raise ValidationError("WTMD requires a 3D grid (nx with three entries)")
        self.nx = nx
        self.lx = lx
        self._fourier = fourier
        self.M = nx[0] * nx[1] * nx[2]
        self.Mk = nx[0] * nx[1] * (nx[2] // 2 + 1)

        # Cell volume and metric (angles [alpha, beta, gamma] in degrees; None: orthogonal)
        cos_a, cos_b, cos_g = np.cos(np.deg2rad(angles)) if angles is not None else (0.0, 0.0, 0.0)
        self.orthogonal = bool(np.allclose([cos_a, cos_b, cos_g], 0.0, rtol=0.0, atol=1e-12))
        if self.orthogonal:
            self.V = lx[0] * lx[1] * lx[2]
        else:
            self.V = lx[0] * lx[1] * lx[2] * np.sqrt(1.0 - cos_a**2 - cos_b**2 - cos_g**2 + 2.0 * cos_a * cos_b * cos_g)
            L = np.array(lx, dtype=np.float64)
            self._metric_inv = np.linalg.inv(np.outer(L, L) * np.array([[1.0, cos_g, cos_b],
                                                                         [cos_g, 1.0, cos_a],
                                                                         [cos_b, cos_a, 1.0]]))
        self.CV = np.sqrt(nbar) * self.V

        # Choose one index of w_aux to use order parameter
        self.eigenvalues = eigenvalues
        self.real_fields_idx = real_fields_idx
        eigen_value_min = 0.0
        self.exchange_idx = None
        for count, i in enumerate(self.real_fields_idx):
            if self.eigenvalues[i] < eigen_value_min:
                eigen_value_min = self.eigenvalues[i]
                self.exchange_idx = i
                self.langevin_idx = count
        if self.exchange_idx is None:
            raise ValidationError("WTMD needs a real auxiliary (exchange) field, i.e. a negative value among the "
                                  "printed 'Eigenvalues' of the auxiliary fields (for two monomer types, a positive chi_n)")

        self.u = np.zeros(self.bins)
        self.up = np.zeros(self.bins)
        self.I0 = np.zeros(self.bins)
        self.I1 = {}

        # Copy data and normalize them by sqrt(nbar)*V
        if u is not None:
            self.u = u.copy() / self.CV
        if up is not None:
            self.up = up.copy() / self.CV
        if I0 is not None:
            self.I0 = I0.copy()
        if I1 is not None:
            self.I1 = {key: np.array(value, dtype=np.float64) / self.CV for key, value in I1.items()}

        self.psi_range = np.linspace(self.psi_min, self.psi_max, num=self.bins, endpoint=False)
        self.psi_range_hist = np.linspace(self.psi_min - self.dpsi / 2, self.psi_max - self.dpsi / 2, num=self.bins + 1, endpoint=True)

        # Store order parameters for updating U and U_hat
        self.order_parameter_history = []
        # Store dH for updating I1
        self.dH_history = {}

        # Initialize arrays in Fourier spaces
        # Half-spectrum weights: the planes kz = 0 and, for even nz, kz = nz/2 are their own
        # conjugates; every other plane stands for itself and its conjugate
        self.wt = 2 * np.ones([nx[0], nx[1], nx[2] // 2 + 1])
        self.wt[:, :, 0] = 1.0
        if nx[2] % 2 == 0:
            self.wt[:, :, nx[2] // 2] = 1.0

        if self.orthogonal:
            space_kx, space_ky, space_kz = np.meshgrid(
                2 * np.pi / lx[0] * np.concatenate([np.arange((nx[0] + 1) // 2), nx[0] // 2 - np.arange(nx[0] // 2)]),
                2 * np.pi / lx[1] * np.concatenate([np.arange((nx[1] + 1) // 2), nx[1] // 2 - np.arange(nx[1] // 2)]),
                2 * np.pi / lx[2] * np.arange(nx[2] // 2 + 1), indexing='ij')
            mag_k = np.sqrt(space_kx**2 + space_ky**2 + space_kz**2)
        else:
            # |k|^2 = 4π² mᵀ g⁻¹ m with the alias representative that minimizes the form,
            # as the propagator solvers choose it (r2c layout: last axis m3 = 0 ... N3/2)
            m = np.array(np.meshgrid(np.fft.fftfreq(nx[0]) * nx[0], np.fft.fftfreq(nx[1]) * nx[1],
                                     np.arange(nx[2] // 2 + 1), indexing="ij")).reshape(3, -1)
            k_sq = None
            for t in np.array(np.meshgrid(*[(-1, 0, 1)] * 3, indexing="ij")).reshape(3, -1).T:
                mt = m + (t * np.array(nx))[:, None]
                k_sq_t = 4 * np.pi**2 * np.einsum("ij,in,jn->n", self._metric_inv, mt, mt)
                k_sq = k_sq_t if k_sq is None else np.minimum(k_sq, k_sq_t)
            mag_k = np.reshape(np.sqrt(k_sq), [nx[0], nx[1], nx[2] // 2 + 1])
        self.fk = 1.0 / (1.0 + np.exp(12.0 * (mag_k - kc) / kc))

        # Compute fourier transform of gaussian functions
        X = self.dpsi * np.concatenate([np.arange((self.bins + 1) // 2), np.arange(self.bins // 2) - self.bins // 2]) / self.sigma_psi
        self.u_kernel = np.fft.rfft(np.exp(-0.5 * X**2))
        self.up_kernel = np.fft.rfft(-X / self.sigma_psi * np.exp(-0.5 * X**2))

    # Compute order parameter Ψ
    def compute_order_parameter(self, langevin_step, w_aux):
        if self._fourier is not None:
            self.w_aux_k = self._fourier.forward(np.reshape(w_aux[self.exchange_idx], -1))
        else:
            self.w_aux_k = np.fft.rfftn(np.reshape(w_aux[self.exchange_idx], self.nx))
        psi = np.sum(np.power(np.absolute(self.w_aux_k), self.l) * self.fk * self.wt)
        psi = np.power(psi, 1.0 / self.l) / self.M
        return psi

    def store_order_parameter(self, psi, dH):
        self.order_parameter_history.append(psi)
        for key in dH:
            if key not in self.dH_history:
                self.dH_history[key] = []
            self.dH_history[key].append(dH[key])

    # Compute bias from psi and w_aux_k, and add it to the DH/DW
    def add_bias_to_langevin(self, psi, langevin):

        # Calculate current value of U'(Ψ) using linear interpolation
        up_hat = np.interp(psi, self.psi_range, self.up)
        if psi == 0.0:
            # Ψ = 0 only for a zero exchange field, where dΨ/dw (0/0) has no direction: no bias
            dpsi_dwr = np.zeros(self.M)
        else:
            # Calculate derivative of order parameter with respect to w_aux_k
            dpsi_dwk = np.power(np.absolute(self.w_aux_k), self.l - 2.0) * np.power(psi, 1.0 - self.l) * self.w_aux_k * self.fk
            # Calculate derivative of order parameter with respect to w
            if self._fourier is not None:
                dpsi_dwr = self._fourier.backward(dpsi_dwk) * np.power(self.M, 2.0 - self.l) / self.V
            else:
                dpsi_dwr = np.fft.irfftn(dpsi_dwk, self.nx, axes=tuple(range(len(self.nx)))) * np.power(self.M, 2.0 - self.l) / self.V

        # Add bias
        bias = np.reshape(self.V * up_hat * dpsi_dwr, self.M)
        langevin[self.langevin_idx] += bias

        print("\t[WTMD] Ψ:%8.5f, np.std(dΨ_dwr):%8.5e, np.std(bias):%8.5e" % (psi, np.std(dpsi_dwr), np.std(bias)))

    def update_statistics(self):

        # Compute histogram
        hist, bin_edges = np.histogram(self.order_parameter_history, bins=self.psi_range_hist, density=True)
        hist_k = np.fft.rfft(hist)
        bin_mids = bin_edges[1:] - self.dpsi / 2

        dI1 = {}
        for key in self.dH_history:
            hist_dH_, _ = np.histogram(self.order_parameter_history,
                                       weights=self.dH_history[key],
                                       bins=self.psi_range_hist, density=False)
            hist_dH_ /= len(self.order_parameter_history)
            hist_dH_k = np.fft.rfft(hist_dH_)
            dI1[key] = np.fft.irfft(hist_dH_k * self.u_kernel, self.bins)

        # Compute dU(Ψ), dU'(Ψ)
        amplitude = np.exp(-self.CV * self.u / self.dT) / self.CV
        gaussian = np.fft.irfft(hist_k * self.u_kernel, self.bins) * self.dpsi
        du = amplitude * gaussian
        dup = amplitude * np.fft.irfft(hist_k * self.up_kernel, self.bins) * self.dpsi - self.CV * self.up / self.dT * du

        print("np.max(np.abs(amplitude)):%8.5e" % (np.max(np.abs(amplitude))))
        print("np.max(np.abs(gaussian)):%8.5e" % (np.max(np.abs(gaussian))))
        print("np.max(np.abs(du)):%8.5e" % (np.max(np.abs(du))))
        print("np.max(np.abs(dup)):%8.5e" % (np.max(np.abs(dup))))
        for key in dI1:
            print("np.max(np.abs(dI1[%s])):%8.5e" % (key, np.max(np.abs(dI1[key]))))

        # Update u, up, I0, I1
        self.u += du
        self.up += dup
        self.I0 += gaussian
        for key in dI1:
            if key not in self.I1:
                self.I1[key] = np.zeros(self.bins)
            self.I1[key] += dI1[key]

        # Reset lists
        self.order_parameter_history = []
        self.dH_history = {}

    def write_data(self, file_name):
        mdic = {
            "l": self.l,
            "kc": self.kc,
            "sigma_psi": self.sigma_psi,
            "dT": self.dT,
            "psi_min": self.psi_min,
            "psi_max": self.psi_max,
            "dpsi": self.dpsi,
            "bins": self.bins,
            "psi_range": self.psi_range,
            "update_freq": self.update_freq,
            "nx": self.nx,
            "lx": self.lx,
            "volume": self.V,
            "nbar": (self.CV / self.V)**2,
            "eigenvalues": self.eigenvalues,
            "real_fields_idx": self.real_fields_idx,
            "exchange_idx": self.exchange_idx,
            "langevin_idx": self.langevin_idx,
            "u": self.u * self.CV,
            "up": self.up * self.CV,
            "I0": self.I0
        }

        # Add I0 and dH_Ψ to the dictionary
        for key in self.I1:
            I1 = self.I1[key] * self.CV
            dH_psi = I1.copy()
            dH_psi[np.abs(self.I0) > 0] /= self.I0[np.abs(self.I0) > 0]
            monomer_pair = sorted(key.split(","))
            mdic["I1_" + monomer_pair[0] + "_" + monomer_pair[1]] = I1
            mdic["dH_psi_" + monomer_pair[0] + "_" + monomer_pair[1]] = dH_psi

        atomic_savemat(file_name, mdic, long_field_names=True, do_compression=True)
