import os
import time
import numpy as np
from scipy.io import savemat, loadmat
from scipy.ndimage import gaussian_filter
import clfts

# OpenMP environment variables
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"  # 0, 1
os.environ["OMP_NUM_THREADS"] = "2"  # 1 ~ 4

f = 0.5        # A-fraction of major BCP chain, f
eps = 1.0       # a_A/a_B, conformational asymmetry

params = {
    #---------------- Simulation parameters -----------------------------
    "nx":[32, 32],          # Simulation grid numbers
    "lx":[4.0, 4.0],    # Simulation box size as a_Ref * N_Ref^(1/2) unit,
                             # where "a_Ref" is reference statistical segment length
                             # and "N_Ref" is the number of segments of reference linear homopolymer chain.

    "chain_model":"discrete",   # "discrete" or "continuous" chain model
    "ds":1/100,                 # Contour step interval, which is equal to 1/N_Ref.

    "segment_lengths":{         # Relative statistical segment length compared to "a_Ref.
        "A":1.0, 
        "B":1.0, },

    "chi_n": {"A,B":20.0},     # Bare interaction parameter, Flory-Huggins params * N_Ref

    "distinct_polymers":[{      # Distinct Polymers
        "volume_fraction":1.0,  # Volume fraction of polymer chain
        "blocks":[              # AB diBlock Copolymer
            {"type":"A", "length":f, }, # A-block
            {"type":"B", "length":1-f}, # B-block
        ],},],
        
    "langevin":{               # Langevin Dynamics
        "max_step":50000,      # Langevin steps for simulation
        "dt":1.25,                # Langevin step interval, delta tau*N_Ref
        "nbar":10000,         # Invariant polymerization index, nbar of N_Ref
    },
    
    "recording":{                       # Recording Simulation Data
        "dir":"data_simulation_ds",        # Directory name
        "recording_period":1,        # Period for recording concentrations and fields
        "sf_computing_period":10,       # Period for computing structure function
        "sf_recording_period":10000,    # Period for recording structure function
    },

    "verbose_level":1,      # 1 : Print at each langevin step.
                            # 2 : Print at each saddle point iteration.
}
# Set random seed
# If you want to obtain different results for each execution, set random_seed=None
random_seed = 12345
np.random.seed(random_seed)

# # Set initial fields
# input_data = loadmat("LamellaInput.mat", squeeze_me=True)
# w_A = input_data["w_A"]
# w_B = input_data["w_B"]

# Set initial fields
print("w_A and w_B are initialized to random Gaussian.")
w_A = np.random.normal(0.0, 1.0, params["nx"])
w_B = np.random.normal(0.0, 1.0, params["nx"])

# Initialize calculation
simulation = clfts.CLFTS(params=params, random_seed=random_seed)

# Set a timer
time_start = time.time()

# Continue simulation with recorded field configurations and random state.
simulation.continue_run(file_name="fields_009274.mat")

# # Run
# simulation.run(initial_fields={"A": w_A, "B": w_B})