"""Well-Tempered Metadynamics (WTMD) for L-FTS simulations.

This module implements well-tempered metadynamics to enhance sampling
in Langevin field-theoretic simulations of block copolymers.

This implementation follows deep-langevin-fts exactly.

References
----------
T. M. Beardsley and M. W. Matsen, "Well-tempered metadynamics applied to
field-theoretic simulations of diblock copolymer melts",
J. Chem. Phys. 157, 114902 (2022).
"""

import os
import numpy as np
from scipy.io import savemat, loadmat


class WTMD:
    """Well-Tempered Metadynamics for enhanced sampling in L-FTS.

    This implementation matches deep-langevin-fts exactly.
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
                 u=None, up=None, I0=None, I1=None):

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

        self.nx = nx
        self.lx = lx
        self.M = nx[0] * nx[1] * nx[2]
        self.Mk = nx[0] * nx[1] * (nx[2] // 2 + 1)

        self.V = lx[0] * lx[1] * lx[2]
        self.CV = np.sqrt(nbar) * self.V

        # Choose one index of w_aux to use order parameter
        self.eigenvalues = eigenvalues
        self.real_fields_idx = real_fields_idx
        eigen_value_min = 0.0
        for count, i in enumerate(self.real_fields_idx):
            if self.eigenvalues[i] < eigen_value_min:
                eigen_value_min = self.eigenvalues[i]
                self.exchange_idx = i
                self.langevin_idx = count

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
            self.I1 = I1.copy()
            for key in self.I1:
                self.I1[key] /= self.CV

        self.psi_range = np.linspace(self.psi_min, self.psi_max, num=self.bins, endpoint=False)
        self.psi_range_hist = np.linspace(self.psi_min - self.dpsi / 2, self.psi_max - self.dpsi / 2, num=self.bins + 1, endpoint=True)

        # Store order parameters for updating U and U_hat
        self.order_parameter_history = []
        # Store dH for updating I1
        self.dH_history = {}

        # Initialize arrays in Fourier spaces
        self.wt = 2 * np.ones([nx[0], nx[1], nx[2] // 2 + 1])
        self.wt[:, :, [0, nx[2] // 2]] = 1.0

        space_kx, space_ky, space_kz = np.meshgrid(
            2 * np.pi / lx[0] * np.concatenate([np.arange((nx[0] + 1) // 2), nx[0] // 2 - np.arange(nx[0] // 2)]),
            2 * np.pi / lx[1] * np.concatenate([np.arange((nx[1] + 1) // 2), nx[1] // 2 - np.arange(nx[1] // 2)]),
            2 * np.pi / lx[2] * np.arange(nx[2] // 2 + 1), indexing='ij')
        mag_k = np.sqrt(space_kx**2 + space_ky**2 + space_kz**2)
        self.fk = 1.0 / (1.0 + np.exp(12.0 * (mag_k - kc) / kc))

        # Compute fourier transform of gaussian functions
        X = self.dpsi * np.concatenate([np.arange((self.bins + 1) // 2), np.arange(self.bins // 2) - self.bins // 2]) / self.sigma_psi
        self.u_kernel = np.fft.rfft(np.exp(-0.5 * X**2))
        self.up_kernel = np.fft.rfft(-X / self.sigma_psi * np.exp(-0.5 * X**2))

    # Compute order parameter Ψ
    def compute_order_parameter(self, langevin_step, w_aux):
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
        # Calculate derivative of order parameter with respect to w_aux_k
        dpsi_dwk = np.power(np.absolute(self.w_aux_k), self.l - 2.0) * np.power(psi, 1.0 - self.l) * self.w_aux_k * self.fk
        # Calculate derivative of order parameter with respect to w
        dpsi_dwr = np.fft.irfftn(dpsi_dwk, self.nx) * np.power(self.M, 2.0 - self.l) / self.V

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

        savemat(file_name, mdic, long_field_names=True, do_compression=True)
