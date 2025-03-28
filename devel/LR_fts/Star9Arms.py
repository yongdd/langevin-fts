import os
import time
import numpy as np
from scipy.io import savemat, loadmat
from scipy.ndimage import gaussian_filter
import lfts

# OpenMP environment variables
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"  # 0, 1
os.environ["OMP_NUM_THREADS"] = "2"  # 1 ~ 4

f = 0.4        # A-fraction of major BCP chain, f
eps = 1.0       # a_A/a_B, conformational asymmetry

params = {
    #---------------- Simulation parameters -----------------------------

    # The neural network is trained in 64^3 grids.
    # To use the trained NN in different simulation box size, we change "nx" as well as "lx", fixing grid interval "dx".

    "nx":[64,64,64],        # Simulation grid numbers
    "lx":[7.37, 7.37, 7.37],
                            # Simulation box size as a_Ref * N_Ref^(1/2) unit,
                            # where "a_Ref" is reference statistical segment length
                            # and "N_Ref" is the number of segments of reference linear homopolymer chain.

    "chain_model":"discrete",      # "discrete" or "continuous" chain model
    "ds":1/100,                    # Contour step interval, which is equal to 1/N_Ref.
    
    "segment_lengths":{            # Relative statistical segment length compared to "a_Ref.
        "A":np.sqrt(eps*eps/(eps*eps*f + (1-f))), 
        "B":np.sqrt(    1.0/(eps*eps*f + (1-f))), },

    "chi_n": {"A,B": 12.5},     # Bare interaction parameter, Flory-Huggins params * N_Ref

    "distinct_polymers":[{      # Distinct Polymers
        "volume_fraction":1.0,  # Volume fraction of polymer chain
        "blocks":[              # 9-arm star-shaped AB Copolymer
            {"type":"A", "length":f,   "v":0, "u":1},    # A-block
            {"type":"A", "length":f,   "v":0, "u":2},    # A-block
            {"type":"A", "length":f,   "v":0, "u":3},    # A-block
            {"type":"A", "length":f,   "v":0, "u":4},    # A-block
            {"type":"A", "length":f,   "v":0, "u":5},    # A-block
            {"type":"A", "length":f,   "v":0, "u":6},    # A-block
            {"type":"A", "length":f,   "v":0, "u":7},    # A-block
            {"type":"A", "length":f,   "v":0, "u":8},    # A-block
            {"type":"A", "length":f,   "v":0, "u":9},    # A-block
            {"type":"B", "length":1-f, "v":1, "u":10},    # B-block
            {"type":"B", "length":1-f, "v":2, "u":11},    # B-block
            {"type":"B", "length":1-f, "v":3, "u":12},    # B-block
            {"type":"B", "length":1-f, "v":4, "u":13},    # B-block
            {"type":"B", "length":1-f, "v":5, "u":14},    # B-block
            {"type":"B", "length":1-f, "v":6, "u":15},    # B-block
            {"type":"B", "length":1-f, "v":7, "u":16},    # B-block
            {"type":"B", "length":1-f, "v":8, "u":17},    # B-block
            {"type":"B", "length":1-f, "v":9, "u":18},    # B-block
        ],},],
    
    "langevin":{                # Langevin Dynamics
        "max_step":200000,      # Langevin steps for simulation
        "dt":8.0,               # Langevin step interval, delta tau*N_Ref
        "nbar":10000,           # Invariant polymerization index, nbar of N_Ref
    },
    
    "recording":{                       # Recording Simulation Data
        "dir":"data_simulation",        # Directory name
        "recording_period":1000,        # Period for recording concentrations and fields
        "sf_computing_period":10,       # Period for computing structure function
        "sf_recording_period":10000,    # Period for recording structure function
    },

    "saddle":{                # Iteration for the pressure field 
        "max_iter" :200,      # Maximum the number of iterations
        "tolerance":1e-4,     # Tolerance of incompressibility 
    },

    "optimizer":{
        # "name":"am",                # Anderson Mixing
        # "name":"lr",                # Linear Response
        "name":"lram",              # Linear Response + Anderson Mixing
        "max_hist":20,              # Maximum number of history
        "start_error":5e-1,         # When switch to AM from simple mixing
        "mix_min":0.01,             # Minimum mixing rate of simple mixing
        "mix_init":0.01,            # Initial mixing rate of simple mixing
    },

    "verbose_level":1,      # 1 : Print at each Langevin step.
                            # 2 : Print at each saddle point iteration.
}
# Set random seed
# If you want to obtain different results for each execution, set random_seed=None
random_seed = 12345

# Set initial fields
print("w_A and w_B are initialized to random Gaussian.")
w_A = np.random.normal(0.0, 1.0, params["nx"])
w_B = np.random.normal(0.0, 1.0, params["nx"])

# Initialize calculation
simulation = lfts.LFTS(params=params, random_seed=random_seed)

# Set a timer
time_start = time.time()

# # Continue simulation with recorded field configurations and random state.
# simulation.continue_run(file_name="fields_010000.mat")

# Run
simulation.run(initial_fields={"A": w_A, "B": w_B})