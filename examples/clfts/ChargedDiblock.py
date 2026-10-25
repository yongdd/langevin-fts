"""CL-FTS of a charged AB diblock melt with counterions.

The A segments carry the valence z_A (a fraction of a charge per segment) and are
neutralized by monovalent counterions C, molecules of one segment. The charges are
smeared with one Gaussian length per type, and the Coulomb interaction is decoupled
with a fluctuating electrostatic potential psi; see Section 5.6 of
docs/theory/LangevinFieldTheoreticSimulation.md and the 'electrostatics' parameters
in docs/references/Parameters.md.
"""

import os
import time
import numpy as np

# OpenMP environment variables (set before importing polymerfts)
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"  # 0, 1
os.environ["OMP_NUM_THREADS"] = "2"  # 1 ~ 4

from polymerfts import CLFTS

f = 0.5          # A-fraction of the diblock
z_a = 0.1        # valence of an A segment
z_c = -1.0       # valence of a counterion
n_segments = 90  # segments of the diblock, 1/ds

# Electroneutrality, z_A phi_A + z_C phi_C = 0 with phi_A = f phi_diblock and
# phi_diblock + phi_C = 1: the counterion fraction follows from the polymer charge.
phi_c = z_a * f / (abs(z_c) + z_a * f)
phi_diblock = 1.0 - phi_c

params = {
    #---------------- Simulation parameters -----------------------------
    "nx":[32, 32, 32],          # Simulation grid numbers
    "lx":[4.21, 4.21, 4.21],    # Simulation box size as a_Ref * N_Ref^(1/2) unit

    "chain_model":"discrete",   # Small ions are molecules of one discrete segment
    "ds":1/n_segments,          # Contour step interval, which is equal to 1/N_Ref.

    "segment_lengths":{         # Relative statistical segment lengths
        "A":1.0,
        "B":1.0,
        "C":1.0, },

    "chi_n": {"A,B":12.0},      # Bare interaction parameter, Flory-Huggins params * N_Ref

    "polymers":[
        {   # Charged diblock
            "volume_fraction":phi_diblock,
            "blocks":[
                {"type":"A", "length":f},
                {"type":"B", "length":1-f}, ], },
        {   # Counterion (one segment)
            "volume_fraction":phi_c,
            "blocks":[
                {"type":"C", "length":1/n_segments}, ], }, ],

    "langevin":{                # Langevin Dynamics
        "max_step":100000,      # Langevin steps for simulation
        "dt":1.0,               # Langevin step interval, delta tau*N_Ref
        "nbar":10000,           # Invariant polymerization index, nbar of N_Ref
    },

    "recording":{                       # Recording Simulation Data
        "dir":"data_simulation",        # Directory name
        "checkpoint_interval":1000,     # Steps between full-state checkpoints
        "sample_interval":10,           # Steps between observable samples
        "statistics_save_interval":10000,  # Steps between statistics saves
    },

    # Compressibility parameter (zeta_n = kappa * N_Ref); the counterions take volume too
    "zeta_n": 100.0,

    # Dynamical stabilization of the exchange fields (Willis & Matsen, J. Chem. Phys. 161, 244903 (2024))
    "dynamic_stabilization": 0.01,

    # Smearing: one Gaussian length per type (Born radii, in units of R0); required with charges
    "smearing": {"type": "gaussian", "a_int": {"A": 0.1, "B": 0.1, "C": 0.1}},

    # Smeared Coulomb interactions: valences per segment and the Bjerrum length l_B/R0,
    # giving the coupling E = 4 pi (l_B/R0) N^2 Nbar^(1/2) (here a weak E = 1.0e3 for the
    # demonstration). "coupling": E gives E directly.
    "electrostatics": {
        "charges": {"A": z_a, "C": z_c},
        "bjerrum_length": 0.0001,
        "psi_dt_scaling": 1.0,  # mobility of psi relative to the auxiliary fields
    },

    "platform":"cuda",
    "verbose_level":1,      # 1 : Print at each langevin step.
}

# Set random seed
# If you want to obtain different results for each execution, set random_seed=None
random_seed = 12345

# Initialize simulation (checks electroneutrality)
simulation = CLFTS(params=params, random_seed=random_seed)

# Initial fields with a lamellar pattern along x (2 periods); the counterions start uniform
nx = params["nx"]
lx = params["lx"]
x = np.linspace(0, lx[0], nx[0], endpoint=False)
X = np.broadcast_to(x[:, None, None], nx)
w_A = (2.0 * np.cos(2 * np.pi * X / (lx[0] / 2))).flatten()
w_B = -w_A.copy()
w_C = np.zeros(np.prod(nx))

# Set a timer
time_start = time.time()

# Run simulation
simulation.run(initial_fields={"A": w_A, "B": w_B, "C": w_C})

# Print elapsed time
print(f"Total elapsed time: {time.time() - time_start:.2f} seconds")

# # Continue simulation with recorded field configurations, psi and random states.
# simulation.continue_run(file_name="data_simulation/fields_010000.mat")
