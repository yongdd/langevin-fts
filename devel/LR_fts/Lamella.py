import os
import time
import numpy as np
from scipy.io import savemat, loadmat
from scipy.ndimage import gaussian_filter
import lfts

# OpenMP environment variables
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"  # 0, 1
os.environ["OMP_NUM_THREADS"] = "2"  # 1 ~ 4

f = 0.5         # A-fraction of major BCP chain, f
eps = 1.0       # a_A/a_B, conformational asymmetry

params = {
    # "platform":"cuda",           # choose platform among [cuda, cpu-mkl]

    "nx":[40, 40, 40],             # Simulation grid numbers
    "lx":[4.36, 4.36, 4.36],       # Simulation box size as a_Ref * N_Ref^(1/2) unit,
                                   # where "a_Ref" is reference statistical segment length
                                   # and "N_Ref" is the number of segments of reference linear homopolymer chain.

    "chain_model":"continuous",  # "discrete" or "continuous" chain model
    "ds":1/100,                  # Contour step interval, which is equal to 1/N_Ref.

    "segment_lengths":{          # Relative statistical segment length compared to "a_Ref.
        "A":1.0, 
        "B":1.0, },

    "chi_n": {"A,B": 17.0},       # Bare interaction parameter, Flory-Huggins params * N_Ref

    "distinct_polymers":[{        # Distinct Polymers
        "volume_fraction":1.0,    # Volume fraction of polymer chain
        "blocks":[                # AB diBlock Copolymer
            {"type":"A", "length":f, }, # A-block
            {"type":"B", "length":1-f}, # B-block
        ],},],
        
    "langevin":{                # Langevin Dynamics
        "max_step":200,       # Langevin steps for simulation
        "dt":5.0,               # Langevin step interval, delta tau*N_Ref
        "nbar":10000,           # Invariant polymerization index, nbar of N_Ref
    },
    
    "recording":{                       # Recording Simulation Data
        "dir":"data_simulation",        # Directory name
        "recording_period":1000,        # Period for recording concentrations and fields
        "sf_computing_period":10,       # Period for computing structure function
        "sf_recording_period":10000,    # Period for recording structure function
    },

    "saddle":{                # Iteration for the pressure field 
        "max_iter" :100,      # Maximum number of iterations
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
np.random.seed(random_seed)

# Set initial fields
print("w_A and w_B are initialized to random Gaussian.")
# w_A = np.random.normal(0.0, 0.1, params["nx"])
# w_B = np.random.normal(0.0, 0.1, params["nx"])

# Initialize calculation
simulation = lfts.LFTS(params=params, random_seed=random_seed)

# Run
input_data = loadmat("lamella_equil_chin17.0.mat", squeeze_me=True)
w_A = input_data["w_A"]
w_B = input_data["w_B"]
simulation.run(initial_fields={"A": w_A, "B": w_B})

# # Continue simulation with recorded field configurations and random state.
# simulation.continue_run(file_name="fields_010000.mat")

# Recording first a few iteration results for debugging and refactoring
