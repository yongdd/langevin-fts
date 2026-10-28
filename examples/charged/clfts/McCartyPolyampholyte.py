"""CL-FTS of the E/K polyampholyte sv30 of McCarty et al., J. Phys. Chem. Lett. 10, 1644 (2019).

The paper's model: n chains of N = 50 beads (25 glutamate E, charge -1, then 25 lysine K, +1)
with harmonic bonds of length b, an implicit solvent, an Edwards excluded volume (v/2) int rho^2
with v = 0.0068 b^3, the Coulomb interaction l_B / r, and Gaussian smearing of mass and charge with
a = b/sqrt(6). Lengths below are in b and densities in beads per b^3.

polymerfts has no implicit-solvent excluded volume, but its compressible model with the reference
density rho0 equal to the bead density rho of the box is the same model:
(zeta/2 rho0) int (rho_bar - rho0)^2 = (v/2) int rho_bar^2 - v rho0 n N + v rho0^2 V / 2 with
zeta = v rho0. The last two terms are linear in n and V, so they shift the chemical potential and
the pressure by known amounts:
    beta mu  = beta mu_pfts + ln(rho b^3 / N) + N v rho             (beta mu_pfts = -ln Q)
    beta Pi  = (beta Pi R0^3 / C) rho / N + v rho^2 / 2
The parameters follow from R0 = b N^(1/2): zeta N = N v rho, C = rho R0^3 / N (nbar = C^2), the
coupling E = 4 pi l_B rho0 N R0^2, the smearing length a / R0 and the box L = 34 a of the paper.
McCarty et al. plot mu with rho in units of R0_paper^-3 = (b/sqrt 6)^-3, which subtracts
ln(6^1.5) = 2.688 from the mu above; this script prints both.

Two state points (first command-line argument, "dilute" by default):
  "dilute": rho b^3 = 7.4e-4, l_B = 0.195 b. Fig. 2D of the paper (labelled l_B = 0.33 b; its RPA
            curve and its data match l_B = 0.195 b, E = 4 pi l_B / R0_paper = 6) gives
            beta mu = -1.85 in the paper's convention, i.e. beta mu_ex = 11.96, 20 kT below the RPA
            (32.0) because the diblock collapses. A single-chain Monte Carlo of the same model
            (isolated chain, no periodic images) gives beta mu_ex = 11.83; the CL-FTS value depends
            on the box (11.875 in a box 1.5 times larger), and extrapolated linearly in 1/V the two
            boxes give 11.84 +- 0.02.
  "dense":  rho b^3 = 1.47, l_B = 0.033 b (Fig. S3): the density structure factor at the smallest
            wavevector k = 2 pi / L is 34.5 in the paper's CL-FTS, 23.6 in the RPA.
"""

import os
import sys
import glob
import numpy as np
from scipy.io import loadmat

# OpenMP environment variables (set before importing polymerfts)
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
os.environ["OMP_NUM_THREADS"] = "2"

from polymerfts import CLFTS

STATE = sys.argv[1] if len(sys.argv) > 1 else "dilute"   # "dilute" or "dense"
rho, l_b = {"dilute": (7.4e-4, 0.195), "dense": (1.47, 0.033)}[STATE]
v = 0.0068                           # Excluded volume v / b^3
N = 50                               # Beads per chain
sequence = "E" * 25 + "K" * 25       # sv30

# Units of polymerfts: R0 = b N^(1/2)
r0 = np.sqrt(N)
C = rho * r0 ** 3 / N
box = 34.0 / np.sqrt(6.0)            # L = 34 a, in units of b

# Blocks of the discrete chain: runs of equal residues, each bead of length ds = 1/N
blocks, run = [], 1
for i in range(1, N + 1):
    if i == N or sequence[i] != sequence[i - 1]:
        blocks.append({"type": sequence[i - 1], "length": run / N})
        run = 1
    else:
        run += 1

params = {
    #---------------- Simulation parameters -----------------------------
    "nx":[34, 34, 34],          # Grid spacing a, as in the paper
    "lx":[box / r0] * 3,        # Box size in units of R0 = b N^(1/2)

    "chain_model":"discrete",   # Bead-spring chains: N = 1/ds beads and N - 1 bonds
    "ds":1 / N,

    "segment_lengths":{"E":1.0, "K":1.0},
    "chi_n":{},                 # No contact interaction between E and K

    "polymers":[{"volume_fraction":1.0, "blocks":blocks}],   # rho0 = rho: no solvent species

    "langevin":{
        "max_step":12000,       # Langevin steps
        "dt":1.0,               # delta tau N; dt = 0.5 gives the same averages
        "nbar":C ** 2,          # C = rho R0^3 / N
        "integrator":"etd",     # Exact linear response of the homogeneous state
    },

    "recording":{
        "dir":"data_mccarty_" + STATE,
        "checkpoint_interval":1000,
        "sample_interval":10,
        "statistics_save_interval":1000,
        "pressure":True,        # Osmotic pressure at every sample
    },

    # Excluded volume: zeta N = N v rho0
    "zeta_n":N * v * rho,

    # Smearing a = b / sqrt(6) of mass and charge
    "smearing":{"type":"gaussian", "a_int":{"E":1 / np.sqrt(6 * N), "K":1 / np.sqrt(6 * N)}},

    # Coulomb interaction: E = 4 pi l_B rho0 N R0^2 = 4 pi (l_B/b) (rho b^3) N^2
    "electrostatics":{"charges":{"E":-1.0, "K":1.0}, "coupling":4 * np.pi * l_b * rho * N ** 2},

    "platform":"cuda",
    "verbose_level":0,
}

simulation = CLFTS(params=params, random_seed=12345)
n_grid = np.prod(params["nx"])
simulation.run(initial_fields={"E": np.zeros(n_grid), "K": np.zeros(n_grid)})

# Block averages: the first two files are left out as equilibration, each later file is one block
directory = params["recording"]["dir"]
files = sorted(glob.glob(os.path.join(directory, "dH_*.mat")))[2:]
mu_blocks = np.array([np.mean(loadmat(f)["chemical_potential_history_real"][:, 0]) for f in files])
pi_blocks = np.array([np.mean(loadmat(f)["pressure_history_real"]) for f in files])

def mean_error(x):
    return np.mean(x), np.std(x, ddof=1) / np.sqrt(len(x))

mu_pfts, mu_err = mean_error(mu_blocks)
pi_pfts, pi_err = mean_error(pi_blocks)
mu = mu_pfts + np.log(rho / N) + N * v * rho
pi_excess = pi_pfts * rho / N + v * rho ** 2 / 2 - rho / N
print("rho b^3 = %g, l_B = %g b, %d blocks" % (rho, l_b, len(files)))
print("  beta mu            = %8.3f +- %.3f   (paper convention: %8.3f)" % (mu, mu_err, mu - 1.5 * np.log(6)))
print("  beta mu_ex         = %8.3f +- %.3f" % (mu_pfts + N * v * rho, mu_err))
print("  beta (Pi - Pi_id) b^3 = %9.5f +- %.5f" % (pi_excess, pi_err * rho / N))
if STATE == "dilute":
    print("  paper (Fig. 2D): beta mu_ex = 11.96; single-chain Monte Carlo (isolated chain): 11.83; RPA: 32.0")
else:
    sf = sorted(glob.glob(os.path.join(directory, "structure_function_*.mat")))[2:]
    m = params["nx"][0]
    q = 2 * np.pi * np.fft.fftfreq(m, d=1.0 / m) / box
    k = np.sqrt(q[:, None, None] ** 2 + q[None, :, None] ** 2 + q[None, None, :] ** 2)
    shell = np.isclose(k, 2 * np.pi / box)
    s_blocks = []
    for f in sf:
        d = loadmat(f)
        s = N * (d["structure_function_E_E_real"] + 2 * d["structure_function_E_K_real"]
                 + d["structure_function_K_K_real"])
        s_blocks.append(np.mean(s[shell]))
    s1, s1_err = mean_error(np.array(s_blocks))
    print("  S_rho(k = 2 pi/L = %.3f / b) = %.2f +- %.2f   (paper CL-FTS 34.5, RPA 23.6)"
          % (2 * np.pi / box, s1, s1_err))
