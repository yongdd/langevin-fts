"""CL-FTS pressure and chemical potential of a soft-bead diblock melt, compared with Lequieu (2024).

J. Lequieu, "Quantitative Equivalence and Performance Comparison of Particle and Field-Theoretic
Simulations", Macromolecules 57, 10870 (2024), Fig. 2: symmetric diblock copolymers of N discrete
Gaussian beads that interact through Gaussian-smeared pair potentials,
    beta u_ij(r) = (beta u0 + chi_ij/rho0) exp(-r^2/4a^2) / (8 pi^1.5 a^3),
with the reduced excluded volume B = beta u0 N^2/Rg^3 = 20, the chain density C = rho0 Rg^3/N = 5,
a = 0.15 Rg, Rg = b ((N-1)/6)^(1/2), in a periodic cube L = 6.4 Rg on 48^3 grid points. Molecular
dynamics (HOOMD) and CL-FTS (OpenFTS) gave the same pressure and chemical potential.

The model is the compressible model of polymerfts with Gaussian smearing (Gamma(k) = exp(-a^2 k^2/2)
on fields and concentrations, so that the pair kernel is Gamma^2) and discrete chains of N segments
and N-1 bonds. With r = R0/Rg = (6N/(N-1))^(1/2) for R0 = b N^(1/2):
    lx = L/r, a_int = a/r, nbar^(1/2) = C r^3, zeta_n = B C, chi_n = chi N.
The paper's excluded volume (beta u0/2) int (Gamma*rho)^2 and our (zeta N/2) <(phi - 1)^2> differ by a
term linear in the (fixed) number of beads and a constant times V, so in a canonical box
    beta P Rg^3 = C (beta Pi R0^3/C)_polymerfts + B C^2/2,
    beta(mu_ex - mu_s) = -<ln Q>_paper = -<Re ln Q>_polymerfts + B C,
where beta(mu_ex - mu_s) is the quantity of Fig. 2b (mu_s removes the self-interaction of the beads,
which the field theory contains). polymerfts records beta mu = ln(phi/alpha) - ln Q = -ln Q here.

This script runs the disordered state chi N = 2 with N = 50, for which Fig. 2 gives (FTS, read from the
vector data of the figure) beta P Rg^3 = 254.254 and beta(mu_ex - mu_s) = 102.109; the mean-field values
are 257.5 and 101.0. The ETD integrator samples the linear (Gaussian) part exactly for any time step;
the remaining time-step error is of second order (run dt = 1.0 and 0.5 to check). Errors are standard
errors of block averages over the statistics files after the first fifth of the run.

Validation runs with these parameters (cpu-mkl, about 33000 steps each, first fifth discarded):
    dt = 0.5: beta P Rg^3 = 254.33 +- 0.08, beta(mu_ex - mu_s) = 102.106 +- 0.001
    dt = 1.0: beta P Rg^3 = 254.23 +- 0.05, beta(mu_ex - mu_s) = 102.108 +- 0.001
    dt -> 0 (second order): 254.37 +- 0.11 and 102.106 +- 0.001
against 254.254 and 102.109 of the paper, whose fluctuation corrections are -3.25 and +1.11.
"""

import os
import glob
import numpy as np
from scipy.io import loadmat

# OpenMP environment variables (set before importing polymerfts)
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"  # 0, 1
os.environ["OMP_NUM_THREADS"] = "2"  # 1 ~ 4

from polymerfts import CLFTS

# Paper parameters (lengths in Rg)
N = 50           # beads per chain (reference chain)
B = 20.0         # reduced excluded volume, beta u0 N^2/Rg^3
C = 5.0          # chain density, rho0 Rg^3/N
a = 0.15         # bead size (smearing length)
L = 6.4          # box length
chi_n = 2.0      # chi N
paper = {"P": 254.254, "mu": 102.109}   # Fig. 2, FTS

# Mapping onto polymerfts (lengths in R0 = b N^(1/2))
r = np.sqrt(6.0 * N / (N - 1.0))         # R0/Rg

params = {
    #---------------- Simulation parameters -----------------------------
    "nx":[48, 48, 48],          # Simulation grid numbers (grid spacing below a, as in the paper)
    "lx":[L/r, L/r, L/r],       # Simulation box size as a_Ref * N_Ref^(1/2) unit

    "chain_model":"discrete",   # N beads and N-1 Gaussian bonds, every bead in the field
    "ds":1/N,                   # Contour step interval, which is equal to 1/N_Ref.

    "segment_lengths":{         # Relative statistical segment lengths
        "A":1.0,
        "B":1.0, },

    "chi_n": {"A,B":chi_n},     # Flory-Huggins parameter * N_Ref

    "polymers":[{               # Symmetric diblock melt
        "volume_fraction":1.0,
        "blocks":[
            {"type":"A", "length":0.5},
            {"type":"B", "length":0.5}, ], }, ],

    "langevin":{                # Langevin Dynamics
        "max_step":40000,       # Langevin steps for simulation
        "dt":0.5,               # Langevin step interval, delta tau*N_Ref
        "nbar":(C * r**3)**2,   # Invariant polymerization index, (rho0 R0^3/N)^2
        "integrator":"etd",     # Exact linear response of the homogeneous state
    },

    "recording":{                       # Recording Simulation Data
        "dir":"data_soft_bead_melt",           # Directory name
        "checkpoint_interval":2000,     # Steps between full-state checkpoints
        "sample_interval":10,           # Steps between observable samples
        "statistics_save_interval":2000,   # Steps between statistics saves
        "pressure":True,                # Record the osmotic pressure operator
    },

    # Compressibility parameter: zeta N = beta u0 rho0 N = B C
    "zeta_n": B * C,

    # Gaussian smearing with the bead size of the paper
    "smearing": {"type": "gaussian", "a_int": a / r},

    "platform":"cuda",
    "verbose_level":1,      # 1 : Print at each langevin step.
}

# Set random seed
# If you want to obtain different results for each execution, set random_seed=None
random_seed = 12345

# Initialize simulation
simulation = CLFTS(params=params, random_seed=random_seed)

# Start from the homogeneous saddle point, w_A = w_B = chi N/2: the uniform mode of the pressure
# field relaxes slowly (rate 2/(chi N + 2 zeta N)) and shifts the chemical potential and the pressure
n_grid = np.prod(params["nx"])
simulation.run(initial_fields={"A": np.full(n_grid, chi_n / 2), "B": np.full(n_grid, chi_n / 2)})

# Each statistics file after the first fifth gives one block average
files = sorted(glob.glob(os.path.join(params["recording"]["dir"], "dH_*.mat")))
files = files[len(files) // 5:]
pressure = np.array([np.mean(loadmat(name)["pressure_history_real"]) for name in files])
mu = np.array([np.mean(loadmat(name)["chemical_potential_history_real"][:, 0]) for name in files])

# Paper units
pressure = C * pressure + 0.5 * B * C**2
mu = mu + B * C
print("Averages over %d blocks (chi N = %.1f, N = %d):" % (len(files), chi_n, N))
for name, values, mean_field in (("beta P Rg^3", pressure, C + 0.5 * B * C**2 + C * chi_n / 4),
                                 ("beta(mu_ex - mu_s)", mu, B * C + chi_n / 2)):
    key = "P" if name.startswith("beta P") else "mu"
    error = np.std(values, ddof=1) / np.sqrt(len(values))
    print("  %-19s CL-FTS %.3f +- %.3f, Lequieu (2024) %.3f, mean field %.3f"
          % (name, np.mean(values), error, paper[key], mean_field))

# # Continue simulation with recorded field configurations and random states.
# simulation.continue_run(file_name="data_soft_bead_melt/fields_040000.mat")
