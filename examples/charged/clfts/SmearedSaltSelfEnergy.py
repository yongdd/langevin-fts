"""CL-FTS of a bulk salt of Gaussian-smeared point ions: Born self-energy and Debye-Hueckel screening.

Wang, Phys. Rev. E 81, 021501 (2010), Section III, gives the excess chemical potential of ions with a
Gaussian charge spread in a uniform salt at the Gaussian (one-loop) level as the Born energy minus
the screening term, Eq. (3.3): mu_ex = z^2 l_B/(2 a_W) - z^2 l_B kappa/2 to O(a^0), and Agrawal &
Wang, Phys. Rev. Lett. 129, 228001 (2022), Eq. (6), the closed form at finite a_W,
    mu_ex = (z^2 l_B/2) [1/a_W - kappa exp(a_W^2 kappa^2/pi) erfc(a_W kappa/sqrt(pi))],
for screening by point charges. Their Gaussian h(r) ~ exp(-pi r^2/(2 a_W^2)) is the smearing
Gamma(k) = exp(-a^2 k^2/2) of the code with a_W = sqrt(pi) a. The one-loop theory is the exact
weak-coupling limit of the model; for the model itself (the screening ions are smeared too, the box
is periodic and finite and the field lives on the grid) it is, per ion of valence z_s,
    mu_ex,s = (z_s^2/(2 N^2)) (1/(C V)) sum_{k != 0} Gamma_s(k)^2 / (k^2/E + sum_t z_t^2 c_t Gamma_t(k)^2/N).
CL-FTS records beta mu_s = ln(N c_s) - ln Q_s of every point ion (chemical_potential_history in the
dH files), so mu_ex = -<ln Q_s>.

The neutral solvent S of one segment is the reference chain (N = 1); the ions do not feel it and
take no volume. With zeta_n = 0 the solvent has no field (docs/theory/ComplexLangevinSimulation.md,
Section 6). Code units: lengths in R0, c per segment of rho0, l_B = E/(4 pi N^2 Nbar^(1/2)),
kappa^2 = E sum_s z_s^2 c_s/N. Here a = dx = 0.05, kappa = 5 (a kappa = 0.25), l_B/a = 1.07 and
l_B kappa = 0.27 with 32 Na+ and 32 Cl- in a periodic box of 24^3 cells, (1.2 R0)^3.

Result (30000 steps, ~8 min on one CPU core; longer runs and other couplings in docs/internals/ExampleValidation.md):
CL-FTS mu_ex = 0.1999 +- 0.0003 against 0.1998 of the one-loop theory on this grid; the
continuum one-loop of the smeared model is 0.2073 (a 96^3 box gives 0.2071) and Agrawal Eq. (6)
0.1996; Wang Eq. (3.3), the a -> 0 form, 0.169. Writes salt_self_energy.json.
"""

import glob
import json
import os

import numpy as np
from scipy.integrate import quad
from scipy.io import loadmat
from scipy.special import erfcx

# OpenMP environment variables (set before importing polymerfts)
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from polymerfts import CLFTS

n_segments = 1           # The solvent molecule is the reference chain
coupling = 250.0         # E = 4 pi l_B rho0 N R0^2
c_salt = 0.05            # Concentration of each ion, per segment of rho0
a = 0.05                 # Gaussian smearing length of the ions (R0); Wang's Born radius is sqrt(pi) a
nx, lx = [24, 24, 24], [1.2, 1.2, 1.2]
n_ions = 32              # Ions of each sign in the box: N c C V = n_ions fixes C = Nbar^(1/2)
C = n_ions / (n_segments * c_salt * np.prod(lx))

params = {
    "platform": "auto",        # "auto" (CUDA for 2D/3D if available), "cuda", "cpu-mkl" or "cpu-fftw"
    "nx": nx, "lx": lx,
    "chain_model": "discrete", "ds": 1.0 / n_segments,
    "segment_lengths": {"S": 1.0}, "chi_n": {}, "zeta_n": 0.0,   # the solvent has no field
    "polymers": [{"volume_fraction": 1.0, "blocks": [{"type": "S", "length": 1.0}]}],
    "smearing": {"type": "gaussian", "a_int": {"S": a}},
    "electrostatics": {"charges": {"S": 0.0}, "coupling": coupling, "n_segments": n_segments,
                       "ions": {"Na": {"valence": 1.0, "concentration": c_salt, "smearing_length": a},
                                "Cl": {"valence": -1.0, "concentration": c_salt, "smearing_length": a}}},
    "langevin": {"max_step": 30000, "dt": 1.0, "nbar": C ** 2},
    "recording": {"dir": "data_salt_self_energy", "checkpoint_interval": 30000, "sample_interval": 2,
                  "statistics_save_interval": 3000},
    "verbose_level": 0,
}

simulation = CLFTS(params=params, random_seed=2010)
simulation.run(initial_fields={"S": np.zeros(int(np.prod(nx)))})

# Block means of mu_ex = beta mu - ln(N c) of Na and Cl; the first file is equilibration
files = sorted(glob.glob(os.path.join(params["recording"]["dir"], "dH_*.mat")))[1:]
blocks = np.array([np.mean(loadmat(f)["chemical_potential_history_real"][:, 1:3], axis=0) for f in files])
mu_ex_blocks = blocks.mean(axis=1) - np.log(n_segments * c_salt)
mu_ex, mu_err = mu_ex_blocks.mean(), mu_ex_blocks.std(ddof=1) / np.sqrt(len(mu_ex_blocks))

# One-loop theory on the same grid
m = [np.fft.fftfreq(n, 1.0 / n) for n in nx]
k2 = sum(np.meshgrid(*[(2 * np.pi * mm / l) ** 2 for mm, l in zip(m, lx)], indexing="ij"))
gamma2 = np.exp(-a * a * k2)
operator = k2 / coupling + 2 * c_salt * gamma2 / n_segments
mask = k2 > 0
one_loop = np.sum(gamma2[mask] / operator[mask]) / (2 * n_segments ** 2 * C * np.prod(lx))

# Continuum: exact one-loop of the smeared model, Agrawal & Wang Eq. (6), Wang Eq. (3.3)
lb = coupling / (4 * np.pi * n_segments ** 2 * C)
kappa2 = 2 * coupling * c_salt / n_segments
kappa = np.sqrt(kappa2)
smeared = lb / np.pi * quad(lambda k: k * k * np.exp(-a * a * k * k) / (k * k + kappa2 * np.exp(-a * a * k * k)),
                            0, np.inf, limit=500)[0]
agrawal = lb / 2 * (1 / (np.sqrt(np.pi) * a) - kappa * erfcx(a * kappa))
wang = lb / (2 * np.sqrt(np.pi) * a) - lb * kappa / 2

print(f"l_B/a = {lb / a:.3f}, l_B kappa = {lb * kappa:.3f}, a kappa = {a * kappa:.3f}, {len(files)} blocks")
print(f"CL-FTS mu_ex                         {mu_ex:.5f} +- {mu_err:.5f}")
print(f"one loop on this grid                {one_loop:.5f}   (CL-FTS/one loop = {mu_ex / one_loop:.4f} +- {mu_err / one_loop:.4f})")
print(f"one loop, continuum (smeared ions)   {smeared:.5f}")
print(f"Agrawal & Wang Eq. (6)               {agrawal:.5f}")
print(f"Wang Eq. (3.3), O(a^0)               {wang:.5f}   (Born energy alone {lb / (2 * np.sqrt(np.pi) * a):.5f})")
json.dump({"mu_ex_clfts": mu_ex, "mu_ex_err": mu_err, "one_loop_grid": one_loop, "one_loop_continuum": smeared,
           "agrawal_eq6": agrawal, "wang_eq3_3": wang, "lB": lb, "kappa": kappa, "a": a},
          open("salt_self_energy.json", "w"), indent=1)
