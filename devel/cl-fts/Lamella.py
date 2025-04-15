import os
import time
import numpy as np
from scipy.io import savemat, loadmat
from scipy.ndimage import gaussian_filter
import clfts

# OpenMP environment variables
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"  # 0, 1
os.environ["OMP_NUM_THREADS"] = "2"  # 1 ~ 4

f = 0.4        # A-fraction of major BCP chain, f
eps = 1.0       # a_A/a_B, conformational asymmetry

params = {
    #---------------- Simulation parameters -----------------------------
    "nx":[40, 40, 40],          # Simulation grid numbers
    "lx":[4.36, 4.36, 4.36],    # Simulation box size as a_Ref * N_Ref^(1/2) unit,
                                # where "a_Ref" is reference statistical segment length
                                # and "N_Ref" is the number of segments of reference linear homopolymer chain.

    "chain_model":"discrete",   # "discrete" or "continuous" chain model
    "ds":1/90,                  # Contour step interval, which is equal to 1/N_Ref.

    "segment_lengths":{         # Relative statistical segment length compared to "a_Ref.
        "A":np.sqrt(eps*eps/(eps*eps*f + (1-f))), 
        "B":np.sqrt(    1.0/(eps*eps*f + (1-f))), },

    "chi_n": {"A,B":17.0},     # Bare interaction parameter, Flory-Huggins params * N_Ref

    "distinct_polymers":[{      # Distinct Polymers
        "volume_fraction":1.0,  # Volume fraction of polymer chain
        "blocks":[              # AB diBlock Copolymer
            {"type":"A", "length":f, }, # A-block
            {"type":"B", "length":1-f}, # B-block
        ],},],
        
    "langevin":{                # Langevin Dynamics
        "max_step":500000,      # Langevin steps for simulation
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
        "max_iter" :100,      # Maximum number of iterations
        "tolerance":1e-4,     # Tolerance of incompressibility 
    },

    "compressor":{
        # "name":"am",                # Anderson Mixing
        # "name":"lr",                # Linear Response
        "name":"lram",              # Linear Response + Anderson Mixing
        "max_hist":20,              # Maximum number of history
        "start_error":5e-1,         # When switch to AM from simple mixing
        "mix_min":0.01,             # Minimum mixing rate of simple mixing
        "mix_init":0.01,            # Initial mixing rate of simple mixing
    },

    "verbose_level":1,      # 1 : Print at each langevin step.
                            # 2 : Print at each saddle point iteration.
}
# Set random seed
# If you want to obtain different results for each execution, set random_seed=None
random_seed = 12345
np.random.seed(random_seed)

# Set initial fields
input_data = loadmat("LamellaInput.mat", squeeze_me=True)
w_A = input_data["w_A"]
w_B = input_data["w_B"]

# Initialize calculation
simulation = clfts.CLFTS(params=params, random_seed=random_seed)

# Set a timer
time_start = time.time()

# # Continue simulation with recorded field configurations and random state.
# simulation.continue_run(file_name="fields_010000.mat")

# Run
simulation.run(initial_fields={"A": w_A, "B": w_B})

# # Recording first a few iteration results for debugging and refactoring
# ---------- Run  ----------
# iterations, mass error, total partitions, Hamiltonian, incompressibility error (or saddle point error)
#        6   -1.266E-16  [ 3.3997238E+00  ]     3.945717170   [6.8486259E-05 ]
# Langevin step:  1
#        8   -7.488E-16  [ 8.8496743E+00  ]     5.302695582   [3.7359427E-05 ]
# Langevin step:  2
#        8    2.424E-16  [ 1.5278556E+01  ]     7.448860242   [9.9035995E-05 ]
# Langevin step:  3
#        9    5.630E-17  [ 1.4894703E+01  ]     7.507150210   [6.7247205E-05 ]
# Langevin step:  4
#        9    6.863E-16  [ 1.4340880E+01  ]     7.550531372   [5.1597189E-05 ]
# Langevin step:  5
#        9    5.117E-16  [ 1.3670551E+01  ]     7.553096384   [4.2796455E-05 ]