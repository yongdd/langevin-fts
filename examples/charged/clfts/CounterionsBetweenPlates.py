"""CL-FTS of counterions between two charged plates at weak coupling: the loop expansion of
Netz, Eur. Phys. J. E 5, 557 (2001), Section 3.

Point counterions (valence q, Gaussian smearing a) neutralize two plates of fixed surface charge
-sigma in the first and the last cell of a closed box (walls on every side). The coupling parameter
Xi = 2 pi q^3 l_B^2 sigma = q^2 l_B/mu (Netz Eq. 4) controls the loop expansion: the density is the
Poisson-Boltzmann profile plus a one-loop correction proportional to Xi (Netz Eqs. 39-47), and both
are exact as Xi -> 0. Netz evaluates the correction with the Green function of a uniform dielectric
(Eq. 56). In the code the walls let no field through (psi is a cosine series), which is a wall of
vanishing dielectric constant outside: every ion is repelled by its image. The one-loop formulas of
Netz Section 3.1 hold for any Green function, so the script evaluates them with the Green function of
the simulated lattice model (cosine modes of the box, Gamma(k) of the ions, the canonical constraint)
and compares the laterally averaged CL-FTS profile with PB and with PB + one loop.

Here d = 4 mu, a = mu/4, a lateral box of 4 mu and 50 counterions: Xi = 0.10, l_B/a = 0.41.
The images deplete the counterions next to the plates by 5 %, against an enrichment in Netz's model
without images (his Fig. 2). Code units: lengths in R0, c per segment of rho0, N = 1 (the neutral
solvent S of one segment is the reference chain; the ions do not feel it), l_B = E/(4 pi N^2 C),
mu = 2/(q (E/N) sigma), C = Nbar^(1/2).

Result (60000 steps, about 6 min on one CPU core): the CL-FTS profile agrees with an exact Monte Carlo
simulation of the same lattice model (ions on the grid sites, smeared Coulomb pair potential of the
cosine series) within its statistical errors, chi^2 = 43/32, and with PB + one loop to about 2 % of
the fluctuation correction (the next order, Xi^2). At stronger coupling CL-FTS needs smaller time
steps (dt <= 0.25 beyond Xi = 0.4), and at Xi >= 0.85 it departs from the exact Monte Carlo result even
as dt -> 0 (examples/README.md, Validation). Writes counterions_between_plates.json and .png.
"""

import json
import os

import numpy as np
from scipy.fft import dct, idct
from scipy.optimize import root

# OpenMP environment variables (set before importing polymerfts)
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from polymerfts import CLFTS

N = 1.0                       # segments of the reference chain (the solvent molecule)
E = 100.0                     # coupling E = 4 pi l_B rho0 N R0^2
q, sigma = 1.0, 0.1           # counterion valence and plate charge (charges per segment of rho0 times R0)
a = 0.05                      # Gaussian smearing length of the counterions
nx, lx = [16, 16, 32], [0.8, 0.8, 0.8]
n_ions = 50                   # counterions in the box: 2 sigma C L_x L_y / q
C = n_ions * q / (2 * sigma * lx[0] * lx[1])
mu = 2 / (q * (E / N) * sigma)
xi = q ** 3 * E ** 2 * sigma / (8 * np.pi * N ** 3 * C)
steps, block = 60000, 5000

nz, lz = nx[2], lx[2]
fixed = np.zeros(nx)
fixed[..., 0] = fixed[..., -1] = -sigma / (lz / nz)

params = {
    "platform": "cpu-mkl",
    "nx": nx, "lx": lx, "bc": ["reflecting"] * 6,        # closed box
    "chain_model": "discrete", "ds": 1.0,
    "segment_lengths": {"S": 1.0}, "chi_n": {}, "zeta_n": 1.0,
    "polymers": [{"volume_fraction": 1.0, "blocks": [{"type": "S", "length": 1.0}]}],
    "smearing": {"type": "gaussian", "a_int": {"S": a}},
    "electrostatics": {"charges": {"S": 0.0}, "coupling": E, "n_segments": N, "fixed_charge": fixed.ravel(),
                       "ions": {"Ci": {"valence": q, "concentration": 0.0, "smearing_length": a}}},
    "langevin": {"max_step": steps, "dt": 1.0, "nbar": C ** 2},
    "recording": {"dir": "data_counterions", "checkpoint_interval": steps, "sample_interval": 100,
                  "statistics_save_interval": steps},
    "verbose_level": 0,
}


class Profile:
    """Block means of the laterally averaged counterion density Re c(z)."""

    def __init__(self):
        self.blocks, self.total, self.count, self.sim = [], np.zeros(nz), 0, None

    def __call__(self, sample):
        (density,), _ = self.sim.electrostatics.ion_densities()
        self.total += np.real(np.reshape(density, nx)).mean(axis=(0, 1))
        self.count += 1
        if self.count == block:
            self.blocks.append(self.total / block)
            self.total, self.count = np.zeros(nz), 0


profile = Profile()
simulation = CLFTS(params=params, random_seed=2001, callbacks={"profile": profile})
profile.sim = simulation
simulation.run(initial_fields={"S": np.zeros(int(np.prod(nx)))})
kept = np.array(profile.blocks[1:])                 # the first block is equilibration
cl, cl_err = kept.mean(axis=0), kept.std(axis=0, ddof=1) / np.sqrt(len(kept))

# ---- Exact limits of the lattice model: PB (zero loop) and the one-loop correction ----
cbar = 2 * sigma / (q * lz)
k = np.pi * np.arange(nz) / lz
g = np.exp(-a * a * k * k / 2)


def counterions(modes):
    u = q * idct(g * np.r_[0.0, modes], norm="ortho") / N
    w = np.exp(-(u - u.min()))
    return cbar * w / w.mean()


def poisson_residual(modes):
    charge = fixed[0, 0] + q * idct(g * dct(counterions(modes), norm="ortho"), norm="ortho")
    return (dct(charge, norm="ortho") - k * k * np.r_[0.0, modes] / E)[1:]


n_pb = counterions(root(poisson_residual, np.zeros(nz - 1), method="hybr", tol=1e-13).x)

# Hessian of the action in the imaginary direction psi = psi_PB + i chi, block diagonal in the lateral
# cosine modes: A = C dV [(k_perp^2 + k_z^2)/E + (q^2/N) Gamma (diag(n) - P) Gamma], P = n n^T/(cbar nz)
# only for k_perp = 0 (canonical ions), whose uniform mode is not integrated.
U = np.sqrt(2.0 / nz) * np.cos(np.pi * np.arange(nz)[:, None] * (np.arange(nz)[None, :] + 0.5) / nz)
U[0] /= np.sqrt(2.0)
V, M = np.prod(lx), np.prod(nx)
response = np.outer(g, g) * (U @ np.diag(n_pb) @ U.T)
un = g * (U @ n_pb)
mxy = np.meshgrid(np.arange(nx[0]), np.arange(nx[1]), indexing="ij")
kp2s, counts = np.unique(np.round(((np.pi * mxy[0] / lx[0]) ** 2 + (np.pi * mxy[1] / lx[1]) ** 2).ravel(), 12),
                         return_counts=True)
variance = np.zeros(nz)
for kp2, count in zip(kp2s, counts):
    A = C * V / M * (np.diag((kp2 + k * k) / E) + (q * q / N) * np.exp(-a * a * kp2) * response
                     - (kp2 == 0) * (q * q / N) * np.outer(un, un) / (cbar * nz))
    G = np.zeros((nz, nz))
    if kp2 == 0:
        G[1:, 1:] = np.linalg.inv(A[1:, 1:])
        G00 = G
    else:
        G = np.linalg.inv(A)
    variance += count * np.exp(-a * a * kp2) * np.diag(U.T @ (g[:, None] * G * g[None, :]) @ U)
ny2 = nx[0] * nx[1]
vbar = (q / N) ** 2 * variance / ny2                                     # <y(z)^2>, y = q Gamma chi/N
gn = U.T @ (g * (G00 @ (g * (U @ n_pb))))
cov = (q / N) ** 2 * gn / (cbar * nz * ny2)                              # <y(z) ybar>
var = (q / N) ** 2 * (n_pb @ gn) / (cbar ** 2 * nz ** 2 * ny2)           # <ybar^2>
fluct = n_pb * (-vbar / 2 + np.sum(n_pb * vbar) / (2 * cbar * nz) - var + cov)
s_bar = vbar - 2 * cov + var                                             # cubic vertex (tadpole) source
gs = n_pb * s_bar / (cbar * nz)
source = N * cbar * (q / N) / np.sqrt(ny2) * (U.T @ (g * (U @ (gs - gs.sum() * n_pb / (cbar * nz)))))
shift = U.T @ (g * (U @ (0.5 * C * V * (U.T @ (G00 @ (U @ source))))))    # Gamma <chi>/i, lateral mode 0
tadpole = n_pb * (q / N) / np.sqrt(ny2) * (shift - np.sum(n_pb * shift) / (cbar * nz))
n_1l = n_pb + fluct + tadpole

z = (np.arange(nz) + 0.5) * lz / nz
print(f"Xi = {xi:.4f}, l_B/a = {E / (4 * np.pi * N ** 2 * C) / a:.3f}, d/mu = {lz / mu:g}, {len(kept)} blocks")
print("   z/mu    PB       PB + 1 loop   CL-FTS          (densities / cbar)")
for i in range(nz // 2):
    print(f"  {z[i] / mu:5.3f}  {n_pb[i] / cbar:7.4f}  {n_1l[i] / cbar:10.4f}  {cl[i] / cbar:8.4f} +- {cl_err[i] / cbar:.4f}")
corr = np.max(np.abs(cl - n_pb))
print(f"max |CL-FTS - (PB + 1 loop)| / max |CL-FTS - PB| = {np.max(np.abs(cl - n_1l)) / corr:.3f}")
json.dump({"z_over_mu": (z / mu).tolist(), "pb": n_pb.tolist(), "one_loop": n_1l.tolist(), "clfts": cl.tolist(),
           "clfts_err": cl_err.tolist(), "Xi": xi}, open("counterions_between_plates.json", "w"))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 3.6))
    ax.axhline(0.0, color="#8a8984", lw=1, label="PB (zero loop)")
    ax.plot(z / mu, (n_1l - n_pb) / cbar, "-", color="#2a78d6", lw=1.5, label="one-loop correction")
    ax.errorbar(z / mu, (cl - n_pb) / cbar, yerr=cl_err / cbar, fmt="o", ms=4, color="#eb6834", label="CL-FTS - PB")
    ax.set_xlabel("z / mu")
    ax.set_ylabel("(c(z) - c_PB(z)) / mean")
    ax.set_title(f"Counterions between plates, Xi = {xi:.2f}", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("counterions_between_plates.png", dpi=150)
except ImportError:
    pass
