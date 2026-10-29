"""CL-FTS of a charged diblock melt in contact with a salt reservoir.

The A blocks carry the valence z_A per segment. Na+ and Cl- are point ions without volume and
without chi (electrostatics.ions) at the chemical potentials of a reservoir with the salt
concentration c0 per segment: the numbers of ions fluctuate, and the uniform mode of psi evolves as
the Donnan potential that keeps the melt neutral on average (docs/theory/PolymerFieldTheory.md,
Section 6). The melt needs no counterions in the input: it takes up more Cl- than Na+ from the
reservoir. In mean-field theory it holds c_Na = c0 x and c_Cl = c0/x with
z_A phi_A + c0 (x - 1/x) = 0, which excludes the co-ions (Donnan exclusion). CL-FTS records the
numbers of ions per reference chain at every sample (number_density_history in dH_*.mat); the
script compares their averages with the Donnan values. The fluctuations of psi (the self-energy of
the smeared ions and the screening by the other charges) lower both by the same amount, about
4e-5 per segment at this weak coupling (0.13 % for Na+, 0.05 % for Cl-), as the one-loop theory of
psi predicts (docs/internals/ExampleValidation.md).
"""

import os
import glob
import numpy as np
from scipy.io import loadmat

# OpenMP environment variables (set before importing polymerfts)
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"  # 0, 1
os.environ["OMP_NUM_THREADS"] = "2"  # 1 ~ 4

from polymerfts import CLFTS

f = 0.5          # A-fraction of the diblock
z_a = 0.1        # Valence of an A segment
c0 = 0.05        # Salt concentration of the reservoir, ions per segment
n_segments = 90  # Segments of the diblock, 1/ds

# A reservoir at psi = 0 has the concentration e^(beta mu)/N of each ion
mu_salt = float(np.log(n_segments * c0))

params = {
    #---------------- Simulation parameters -----------------------------
    "nx":[32, 32, 32],          # Simulation grid numbers
    "lx":[4.21, 4.21, 4.21],    # Simulation box size as a_Ref * N_Ref^(1/2) unit

    "chain_model":"discrete",   # Discrete chains: N = 1/ds is the number of segments
    "ds":1/n_segments,          # Contour step interval, which is equal to 1/N_Ref.

    "segment_lengths":{         # Relative statistical segment lengths
        "A":1.0,
        "B":1.0, },

    "chi_n": {"A,B":8.0},       # Bare interaction parameter, Flory-Huggins params * N_Ref (disordered melt)

    "polymers":[{               # Charged diblock; the point ions take no volume
        "volume_fraction":1.0,
        "blocks":[
            {"type":"A", "length":f},
            {"type":"B", "length":1-f}, ], }, ],

    "langevin":{                # Langevin Dynamics
        "max_step":20000,       # Langevin steps for simulation
        "dt":1.0,               # Langevin step interval, delta tau*N_Ref
        "nbar":10000,           # Invariant polymerization index, nbar of N_Ref
    },

    "recording":{                       # Recording Simulation Data
        "dir":"data_salt",              # Directory name
        "checkpoint_interval":2000,     # Steps between full-state checkpoints
        "sample_interval":10,           # Steps between observable samples
        "statistics_save_interval":2000,   # Steps between statistics saves
    },

    # Compressibility parameter (zeta_n = kappa * N_Ref)
    "zeta_n": 100.0,

    # Dynamical stabilization of the exchange fields (Willis & Matsen, J. Chem. Phys. 161, 244903 (2024))
    "dynamic_stabilization": 0.01,

    # Smearing: one Gaussian length per type (Born radii, in units of R0); required with charges
    "smearing": {"type": "gaussian", "a_int": {"A": 0.1, "B": 0.1}},

    # Smeared Coulomb interactions: valences per segment, the Bjerrum length l_B/R0 (coupling
    # E = 4 pi (l_B/R0) N^2 Nbar^(1/2), about 5.1e3 here) and the point ions. With
    # chemical_potential in place of concentration an ion species is grand canonical.
    "electrostatics": {
        "charges": {"A": z_a},
        "bjerrum_length": 0.0005,
        "ions": {"Na": {"valence": 1.0, "chemical_potential": mu_salt, "smearing_length": 0.1},
                 "Cl": {"valence": -1.0, "chemical_potential": mu_salt, "smearing_length": 0.1}},
        # Mobility of psi relative to the auxiliary fields; its uniform mode (the Donnan
        # potential) relaxes only through the uptake of ions, which this speeds up
        "psi_dt_scaling": 10.0,
    },

    "platform": "auto",        # "auto" (CUDA for 2D/3D if available), "cuda", "cpu-mkl" or "cpu-fftw"
    "verbose_level":1,      # 1 : Print at each langevin step.
}

# Set random seed
# If you want to obtain different results for each execution, set random_seed=None
random_seed = 12345

# Initialize simulation
simulation = CLFTS(params=params, random_seed=random_seed)

# Start from the homogeneous melt; the Donnan potential starts at zero
n_grid = np.prod(params["nx"])
simulation.run(initial_fields={"A": np.zeros(n_grid), "B": np.zeros(n_grid)})

# Numbers of chains and ions per reference chain: one column per species (the diblock, Na, Cl).
# The first file is left out as equilibration; each later file gives one block average.
files = sorted(glob.glob(os.path.join(params["recording"]["dir"], "dH_*.mat")))[1:]
blocks = np.array([np.mean(loadmat(name)["number_density_history_real"], axis=0) for name in files])
mean = np.mean(blocks, axis=0)
error = np.std(blocks, axis=0, ddof=1) / np.sqrt(len(blocks))

# Donnan equilibrium of the mean-field theory
charge = z_a * f
x = (-charge / c0 + np.sqrt((charge / c0) ** 2 + 4.0)) / 2.0
donnan = {"Na": c0 * x, "Cl": c0 / x}

print("Ion concentrations per segment, averaged over %d blocks:" % len(blocks))
for column, name in ((1, "Na"), (2, "Cl")):
    print("  %s: CL-FTS %.6f +- %.1e, Donnan %.6f, reservoir %.6f"
          % (name, mean[column] / n_segments, error[column] / n_segments, donnan[name], c0))

# # Continue simulation with recorded field configurations, psi and random states.
# simulation.continue_run(file_name="data_salt/fields_020000.mat")
