import os
import time
import numpy as np
from scipy.io import savemat, loadmat
from polymerfts import CLFTS

# OpenMP environment variables
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"  # 0, 1
os.environ["OMP_NUM_THREADS"] = "2"  # 1 ~ 4

f = 0.5        # A-fraction of major BCP chain, f
eps = 1.0      # a_A/a_B, conformational asymmetry

params = {
    #---------------- Simulation parameters -----------------------------
    "nx":[32, 32, 32],          # Simulation grid numbers
    "lx":[4.21, 4.21, 4.21],    # Simulation box size as a_Ref * N_Ref^(1/2) unit,
                                # where "a_Ref" is reference statistical segment length
                                # and "N_Ref" is the number of segments of reference linear homopolymer chain.

    "chain_model":"discrete",   # "discrete" or "continuous" chain model
    "ds":1/90,                  # Contour step interval, which is equal to 1/N_Ref.

    "segment_lengths":{         # Relative statistical segment length compared to "a_Ref.
        "A":1.0,
        "B":1.0, },

    "chi_n": {"A,B":12.0},      # Bare interaction parameter, Flory-Huggins params * N_Ref

    "distinct_polymers":[{      # Distinct Polymers
        "volume_fraction":1.0,  # Volume fraction of polymer chain
        "blocks":[              # AB diBlock Copolymer
            {"type":"A", "length":f, }, # A-block
            {"type":"B", "length":1-f}, # B-block
        ],},],

    "langevin":{                # Langevin Dynamics
        "max_step":100000,      # Langevin steps for simulation
        "dt":1.0,               # Langevin step interval, delta tau*N_Ref
        "nbar":10000,           # Invariant polymerization index, nbar of N_Ref
    },

    "recording":{                       # Recording Simulation Data
        "dir":"data_simulation",        # Directory name
        "recording_period":1000,        # Period for recording concentrations and fields
        "sf_computing_period":10,       # Period for computing structure function
        "sf_recording_period":10000,    # Period for recording structure function
    },

    # Compressibility parameter (zeta_n = kappa * N_Ref)
    # Lower values make the system more compressible
    "zeta_n": 100.0,

    # Dynamical stabilization constant for complex Langevin
    # Set to 0 for standard CL dynamics
    # Set to small positive value (e.g., 0.01-0.1) if fields develop large imaginary parts
    "dynamic_stabilization": 0.0,

    "verbose_level":1,      # 1 : Print at each langevin step.
                            # 2 : Print at each iteration (not used in CL-FTS).
}

# Set random seed
# If you want to obtain different results for each execution, set random_seed=None
random_seed = 12345

# Initialize simulation
simulation = CLFTS(params=params, random_seed=random_seed)

# Generate initial fields with lamellar pattern
print("Generating initial fields with lamellar pattern...")
nx = params["nx"]
lx = params["lx"]
n_grid = np.prod(nx)

# Create coordinate grid
x = np.linspace(0, lx[0], nx[0], endpoint=False)
y = np.linspace(0, lx[1], nx[1], endpoint=False)
z = np.linspace(0, lx[2], nx[2], endpoint=False)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Lamellar pattern along x-direction (2 periods)
period = lx[0] / 2
w_A = 2.0 * np.cos(2 * np.pi * X / period).flatten()
w_B = -w_A.copy()

# Set a timer
time_start = time.time()

# Run simulation
simulation.run(initial_fields={"A": w_A, "B": w_B})

# Print elapsed time
print(f"Total elapsed time: {time.time() - time_start:.2f} seconds")

# # Continue simulation with recorded field configurations and random state.
# simulation.continue_run(file_name="data_simulation/fields_010000.mat")

# Recording first a few iteration results for reference (with random_seed=12345):
# ---------- Run ----------
# step    mass_error                     partition_function                    Hamiltonian                           error_levels
#        1 -5.607E-02-3.487E-05j  [ 1.159075E+00+2.318895E-17j ] +2.797237765E+00+3.489285278E-03j  [ 3.312E-02 6.970E-03 ]
#        2 -5.554E-02-1.464E-04j  [ 1.077628E+00+3.969140E-03j ] +2.825749336E+00+1.097152071E-02j  [ 4.751E-02 8.743E-02 ]
#        3 -5.502E-02-2.245E-04j  [ 9.700087E-01+1.467895E-02j ] +2.886240260E+00+7.348049641E-03j  [ 5.811E-02 1.630E-01 ]
#        4 -5.450E-02-2.013E-04j  [ 8.884174E-01+2.135688E-02j ] +2.929704074E+00-3.856581906E-03j  [ 6.514E-02 1.926E-01 ]
#        5 -5.398E-02-1.703E-04j  [ 8.207961E-01+1.784909E-02j ] +2.965189810E+00-4.652779305E-03j  [ 7.149E-02 2.128E-01 ]
