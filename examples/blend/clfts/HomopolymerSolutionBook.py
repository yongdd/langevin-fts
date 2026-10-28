"""CL-FTS of a homopolymer in implicit good solvent: excess chemical potential and pressure.

Reproduces the benchmark of Fredrickson & Delaney, Field-Theoretic Simulations in Soft Matter and
Quantum Fluids (Oxford, 2023), Figs. 5.6 and 7.14 (from Villet & Fredrickson, J. Chem. Phys. 141,
224115 (2014) and Villet, PhD thesis (2012)): continuous Gaussian chains with a Gaussian-smeared
excluded-volume repulsion (beta u0/2) int rhobar^2, at the reduced chain concentration
C = n Rg^3/V = 1, excluded-volume strength B = beta u0 N^2/Rg^3 = 1, smearing length a = 0.1 Rg,
grid spacing Delta x = 0.1 Rg, RQM4 with ds = 0.01, in a cubic cell of side L = 3.2 Rg.

Mapping to polymerfts (one monomer type, canonical, no chi): the compressible penalty
(zeta/2 rho0) int (rho - rho0)^2 equals the Edwards term up to -zeta int rho, a constant at fixed
numbers of chains, so zeta_n = B C = 1, nbar = (6^1.5 C)^2 = 216 and lengths/R0 = (lengths/Rg)/sqrt(6).
The book's observables follow exactly from ours:
    beta mu_ex          = -ln Q + zeta_n                 (recorded chemical potential + zeta_n)
    beta P_ex Rg^3      = C (Pi - 1) + B C^2/2           (Pi = beta Pi R0^3/C as recorded)

The recorded pressure is the virial form of the operator (CLFTS.pressure_virial), the scaled-field
operator of the book (its eq. B.26). The dilation of the box at fixed grid values of the fields
(CLFTS.pressure) has the same average, but its field term and the Gaussian normalization
n_g/(2 C V/R0^3) fluctuate by about sqrt(n_g/2)/(C V/R0^3) per sample (about 4 here); the virial
form adds the Schwinger-Dyson zero (1/2)[mean(Omega dH/dOmega) - n_g/(C V/R0^3)], which removes
that fluctuation: its standard deviation is several hundred times smaller here.

Book values at L = 3.2 Rg (read from Fig. 7.14; Fig. 5.6 gives 0.4769 for ETD1 with 32^3 points):
beta mu_ex = 1.330 +- 0.002 and beta P_ex Rg^3 = 0.4773 +- 0.001; the mean-field values are 1 and 0.5.
"""

import os
import glob
import numpy as np
from scipy.io import loadmat

# OpenMP environment variables (set before importing polymerfts)
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"  # 0, 1
os.environ["OMP_NUM_THREADS"] = "4"  # 1 ~ 4

from polymerfts import CLFTS

C_book = 1.0       # n Rg^3 / V
B_book = 1.0       # beta u0 N^2 / Rg^3
L_rg = 3.2         # cell side in units of Rg
n_grid = 32        # grid points per side, Delta x = 0.1 Rg = a
rg = 1.0 / np.sqrt(6.0)   # Rg in units of R0

params = {
    #---------------- Simulation parameters -----------------------------
    "nx":[n_grid]*3,                # Simulation grid numbers
    "lx":[L_rg*rg]*3,               # Simulation box size as a_Ref * N_Ref^(1/2) unit

    "chain_model":"continuous",     # Continuous Gaussian chains
    "ds":0.01,                      # Contour step interval (Ms = 100 as in the book)
    "numerical_method":"rqm4",      # RQM4, as in the book

    "segment_lengths":{"A":1.0},    # One monomer type
    "chi_n":{},                     # No chi: the solvent is implicit

    "polymers":[{"volume_fraction":1.0, "blocks":[{"type":"A", "length":1.0}]}],

    "langevin":{                    # Langevin dynamics
        "max_step":20000,           # Langevin steps for simulation
        "dt":1.0,                   # Langevin step interval, delta tau*N_Ref
        "nbar":(6**1.5*C_book)**2,  # Invariant polymerization index: sqrt(nbar) = 6^1.5 C_book
        "integrator":"etd",         # Exponential time differencing (exact for the linear response)
    },

    "recording":{                       # Recording simulation data
        "dir":"data_homopolymer",       # Directory name
        "checkpoint_interval":5000,     # Steps between full-state checkpoints
        "sample_interval":10,           # Steps between observable samples
        "statistics_save_interval":1000,   # Steps between statistics saves
        "pressure":True,                # Record the osmotic pressure operator
    },

    "zeta_n":B_book*C_book,         # Excluded volume: zeta N = B C

    # Gaussian smearing a = 0.1 Rg (the range of the excluded-volume repulsion)
    "smearing":{"type":"gaussian", "a_int":0.1*rg},

    "platform":"cuda",
    "verbose_level":0,
}

# Set random seed
random_seed = 12345

# Initialize simulation
simulation = CLFTS(params=params, random_seed=random_seed)

# The homogeneous saddle point of the compressible model is Omega = 0
simulation.run(initial_fields={"A": np.zeros(n_grid**3)})

# Samples of all statistics files; the first tenth is left out as equilibration
files = sorted(glob.glob(os.path.join(params["recording"]["dir"], "dH_*.mat")))
mu = np.concatenate([loadmat(name)["chemical_potential_history_real"][:, 0] for name in files])
pi = np.concatenate([loadmat(name)["pressure_history_real"].ravel() for name in files])
start = len(mu) // 10

def block_average(x, n_blocks=20):
    x = x[len(x) % n_blocks:].reshape(n_blocks, -1).mean(axis=1)
    return x.mean(), x.std(ddof=1) / np.sqrt(n_blocks)

mu_ex = block_average(mu[start:] + params["zeta_n"])
p_ex = block_average(C_book * (pi[start:] - 1.0) + B_book * C_book**2 / 2)

print("L = %.1f Rg, %d^3 grid, %d samples (block averages of 20 blocks):" % (L_rg, n_grid, len(mu) - start))
print("  beta mu_ex       CL-FTS %.4f +- %.4f   book 1.330 +- 0.002    mean field 1" % mu_ex)
print("  beta P_ex Rg^3   CL-FTS %.4f +- %.4f   book 0.4773 +- 0.001   mean field 0.5" % p_ex)

# # Continue simulation with recorded field configurations and random states.
# simulation.continue_run(file_name="data_homopolymer/fields_020000.mat")
