"""CL-FTS of the salt-free polycation/polyanion solution of Lee, Popov & Fredrickson,
J. Chem. Phys. 128, 224908 (2008): the structure factors of their Figs. 4 and 5, and the chemical
potential and pressures at the same state point against the one-loop theory. The phase diagram of
their Fig. 6 is not reproduced (docs/internals/ExampleValidation.md).

Lee's model: n polycations and n polyanions, discrete Gaussian chains of N_L bonds (N_L + 1 beads) with the
charge +-z per bead, in an implicit good solvent with the Edwards excluded volume (u0/2) int rho^2 and the
Coulomb interaction l_B/r. Lengths in Rg = b (N_L/6)^(1/2); reduced parameters
    C = 2 n Rg^3 / V,   B = u0 N_L^2 / Rg^3,   E = 4 pi l_B z^2 N_L^2 / Rg.
Lee's box: V = (4 Rg)^3 on 32^3 grid points, without smearing (the grid is the only cutoff).

Mapping to polymerfts (lengths in R0, fields per chain, C_ours = rho0 R0^3 / N = nbar^(1/2)):
  * polymerfts discrete chains have N beads and N - 1 bonds of variance R0^2/N, so N = N_L + 1 and
    R0 = N^(1/2) b, i.e. R0/Rg = (6 N / N_L)^(1/2).
  * The compressible model (zeta/2 rho0) int (rho - rho0)^2 of a canonical box at the reference density rho0
    is the Edwards model with u0 = zeta/rho0: the linear term -zeta int rho is fixed by the amount. Hence
    zeta N = u0 rho0 N = B C (N/N_L)^2, and rho0 is the bead density of the state, so a concentration scan
    is a scan of nbar:  C_ours = C (R0/Rg)^3.
  * Coupling with charges +-1: E_ours = 4 pi l_B z^2 rho0 N R0^2 = E C (N/N_L)^2 (R0/Rg)^2.
  * Charged types need a positive smearing length; a = 1e-3 R0 leaves Lee's lattice model unchanged
    (Gamma^2 > 0.99 at every grid wavevector).
  * Observables: u0 S_nn = zeta N (S_PP + S_MM + 2 S_PM), 4 pi l_B Rg^2 S_cc = E_ours (Rg/R0)^2 S_zz,
    beta mu_Edwards = beta mu_ours + ln C + zeta N + const, and the osmotic pressure
    p Rg^3 = C (pi_ours + zeta N / 2), with pi_ours = beta Pi R0^3 / C_ours the recorded operator
    (Edwards: (u0/2) rho^2 = (zeta/2 rho0)(rho - rho0)^2 + zeta rho - zeta rho0/2).

Pressure caveat: the recorded pressure dilates the box at a fixed number of grid points. Without
resolved smearing (as here) the free energy depends on the grid spacing, and this differs from the
pressure at fixed grid spacing that sets the coexistence of two phases on one lattice. The script
therefore also prints the one-loop values of both pressures on this lattice; the chemical potential
(fixed box and grid) needs no such care. With a smearing length of at least half the grid spacing
the two pressures agree.

The script compares the CL-FTS averages (block errors over the statistics files) with the one-loop
(RPA) theory of exactly this discretized model, and with Lee's FTS values read from their figures.
"""

import os
import glob
import numpy as np
from scipy.io import loadmat

os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
os.environ["OMP_NUM_THREADS"] = "4"

from polymerfts import CLFTS

# ------------------------------------------------------------------ Lee's state point (Figs. 4 and 5)
B, C_lee, E_lee = 1.0, 12.0, 14400.0
N_L = 1024                      # bonds; polymerfts N = N_L + 1 beads
nx = 32                         # Lee: 32^3 grid on (4 Rg)^3
a_smear = 1.0e-3                # in R0; effectively Lee's unsmeared lattice model
max_step, save = 12000, 2000    # each statistics file (save steps) is one block; the first is equilibration

N = N_L + 1
r0_rg = np.sqrt(6.0 * N / N_L)                 # R0/Rg
c_ours = C_lee * r0_rg**3                      # nbar^(1/2)
zeta_n = B * C_lee * (N / N_L) ** 2
coupling = E_lee * C_lee * (N / N_L) ** 2 * r0_rg**2
lx = 4.0 / r0_rg                               # 4 Rg in R0

params = {
    "nx": [nx] * 3, "lx": [lx] * 3,
    "chain_model": "discrete", "ds": 1.0 / N,
    "segment_lengths": {"P": 1.0, "M": 1.0},
    "chi_n": {},                                # no chi: only excluded volume and charges
    "polymers": [{"volume_fraction": 0.5, "blocks": [{"type": "P", "length": 1.0}]},    # polycation
                 {"volume_fraction": 0.5, "blocks": [{"type": "M", "length": 1.0}]}],   # polyanion
    "langevin": {"max_step": max_step, "dt": 1.0, "nbar": c_ours**2, "integrator": "etd"},
    "recording": {"dir": "data_coacervate", "checkpoint_interval": max_step, "sample_interval": 10,
                  "statistics_save_interval": save, "pressure": True},
    "zeta_n": zeta_n,
    "smearing": {"type": "gaussian", "a_int": a_smear},
    "electrostatics": {"charges": {"P": 1.0, "M": -1.0}, "coupling": coupling},
    "platform": "auto",        # "auto" (CUDA for 2D/3D if available), "cuda", "cpu-mkl" or "cpu-fftw"
    "verbose_level": 0,
}
print("polymerfts parameters: C = nbar^1/2 = %.3f, zeta_N = %.4f, E = %.6g, L = %.5f R0, N = %d" %
      (c_ours, zeta_n, coupling, lx, N))

simulation = CLFTS(params=params, random_seed=2008)
n_grid = nx**3
simulation.run(initial_fields={"P": np.zeros(n_grid), "M": np.zeros(n_grid)})


# ------------------------------------------------------------------ one-loop theory of this lattice model
def debye(k2, n):
    x = np.exp(-k2 / (6.0 * n))
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (n * (1 + x) / (1 - x) - 2 * x * (1 - x**n) / (1 - x) ** 2) / n**2
    return np.where(k2 < 1e-12, 1.0, s)


def k2_grid(L):
    k = 2 * np.pi * np.fft.fftfreq(nx, d=L / nx)
    K = np.meshgrid(k, k, k, indexing="ij")
    return K[0] ** 2 + K[1] ** 2 + K[2] ** 2


def lnz1(L, s):
    k2 = k2_grid(L).ravel()
    k2 = k2[k2 > 0]
    rg2 = np.exp(-a_smear**2 * k2) * debye(k2, N) * s
    return -0.5 * np.sum(np.log1p(zeta_n * rg2) + np.log1p(coupling * rg2 / k2))


h = 1e-4
V = lx**3
n_ref = c_ours * V
mu_1 = -(lnz1(lx, 1 + h) - lnz1(lx, 1 - h)) / (2 * h) / n_ref
pi_dil_1 = (lnz1(lx * (1 + h) ** (1 / 3), 1 / (1 + h)) - lnz1(lx * (1 - h) ** (1 / 3), 1 / (1 - h))) / (2 * h * V) / c_ours
pi_lat_1 = mu_1 + lnz1(lx, 1.0) / n_ref

# ------------------------------------------------------------------ CL-FTS averages (block errors)
def mean_err(x):
    x = np.asarray(x)
    return np.mean(x, axis=0), np.std(x, axis=0, ddof=1) / np.sqrt(len(x))


rec = params["recording"]["dir"]
dh = sorted(glob.glob(os.path.join(rec, "dH_*.mat")))[1:]
mu_b = [np.mean(loadmat(f)["chemical_potential_history_real"]) for f in dh]
pi_b = [np.mean(loadmat(f)["pressure_history_real"]) for f in dh]
(mu, mu_e), (pi, pi_e) = mean_err(mu_b), mean_err(pi_b)
print("\nThermodynamics, %d blocks (mean-field values: mu = ln(1/2), pi = 1)" % len(dh))
print("  excess chemical potential per chain: CL-FTS %.4f +- %.4f, one loop %.4f" % (mu - np.log(0.5), mu_e, mu_1))
print("  pressure operator (dilation) pi - 1: CL-FTS %.4f +- %.4f, one loop %.4f" % (pi - 1, pi_e, pi_dil_1))
print("  Lee units p Rg^3 = C (pi + zeta_N/2): CL-FTS (dilation) %.2f +- %.2f; one loop: dilation %.2f,"
      " fixed lattice %.2f; Lee Eq. 35 (continuum) %.2f" % (
          C_lee * (pi + zeta_n / 2), C_lee * pi_e, C_lee * (1 + pi_dil_1 + zeta_n / 2),
          C_lee * (1 + pi_lat_1 + zeta_n / 2),
          C_lee + B * C_lee**2 / 2 - ((2 * B * C_lee) ** 1.5 + (2 * E_lee * C_lee) ** 0.75 / np.sqrt(2)) / (24 * np.pi)))

# ------------------------------------------------------------------ structure factors in Lee's units
k = np.sqrt(k2_grid(lx))
shells, inv = np.unique(np.round(k.ravel(), 8), return_inverse=True)
count = np.bincount(inv)
sf_files = sorted(glob.glob(os.path.join(rec, "structure_function_*.mat")))[1:]
nn_b, cc_b = [], []
gamma2 = np.exp(-a_smear**2 * k**2)
for f in sf_files:
    d = loadmat(f)
    spp, smm, spm = (d["structure_function_%s_real" % key] for key in ("P_P", "M_M", "M_P"))
    nn_b.append(np.bincount(inv, weights=(zeta_n * (spp + smm + 2 * spm)).ravel()) / count)
    s_zz = d["charge_structure_function_real"] / gamma2          # recorded for the smeared charge density
    cc_b.append(np.bincount(inv, weights=(coupling / r0_rg**2 * s_zz).ravel()) / count)
(nn, nn_e), (cc, cc_e) = mean_err(nn_b), mean_err(cc_b)
g = debye(shells**2, N)
gm2 = np.exp(-a_smear**2 * shells**2)
rpa_nn = zeta_n * g / (1 + zeta_n * gm2 * g)
rpa_cc = coupling / r0_rg**2 * g * shells**2 / np.where(shells > 0, shells**2 + coupling * gm2 * g, 1.0)
krg = shells / r0_rg

lee_lowk = {900: 0.95, 3600: 1.11, 14400: 1.92, 19600: 2.48}     # Lee Fig. 4 at k Rg = pi/2 (digitized)
print("\nu0 S_nn(k) at the lowest shells (RPA independent of E):")
for i in range(1, 5):
    print("  k Rg = %.3f: CL-FTS %.3f +- %.3f, RPA %.3f" % (krg[i], nn[i], nn_e[i], rpa_nn[i]))
print("  Lee's FTS at k Rg = pi/2: %s" % lee_lowk.get(int(E_lee), "-"))
smooth = np.convolve(cc, np.ones(9) / 9, mode="same")
i = int(np.argmax(smooth[5:-5])) + 5
print("4 pi l_B Rg^2 S_cc peak: CL-FTS %.1f at k Rg %.1f; RPA %.1f at k Rg %.1f" %
      (smooth[i], krg[i], np.max(rpa_cc), krg[np.argmax(rpa_cc)]))

np.savez("complex_coacervate.npz", k_rg=krg[1:], count=count[1:], u0_snn=nn[1:], u0_snn_err=nn_e[1:],
         rpa_nn=rpa_nn[1:], s_cc=cc[1:], s_cc_err=cc_e[1:], rpa_cc=rpa_cc[1:],
         mu_excess=mu - np.log(0.5), mu_excess_err=mu_e, mu_1loop=mu_1, pi=pi, pi_err=pi_e,
         pi_dil_1loop=pi_dil_1, pi_lat_1loop=pi_lat_1)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].errorbar(krg[1:], nn[1:], yerr=nn_e[1:], fmt="o", ms=2, label="CL-FTS")
ax[0].plot(krg[1:], rpa_nn[1:], "k-", label="RPA (this lattice)")
if int(E_lee) in lee_lowk:
    ax[0].plot([np.pi / 2], [lee_lowk[int(E_lee)]], "rx", ms=9, label="Lee et al. FTS")
ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("k Rg"); ax[0].set_ylabel("u0 S_nn")
ax[0].legend()
ax[1].errorbar(krg[1:], cc[1:], yerr=cc_e[1:], fmt="o", ms=2, label="CL-FTS")
ax[1].plot(krg[1:], rpa_cc[1:], "k-", label="RPA (this lattice)")
ax[1].set_xlabel("k Rg"); ax[1].set_ylabel("4 pi l_B Rg^2 S_cc"); ax[1].legend()
fig.suptitle("B = %g, C = %g, E = %g, N = %d (Lee et al. 2008)" % (B, C_lee, E_lee, N_L))
fig.tight_layout()
fig.savefig("complex_coacervate.png", dpi=150)
