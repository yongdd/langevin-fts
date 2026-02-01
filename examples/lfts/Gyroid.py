import os
import time
import numpy as np
from scipy.io import savemat, loadmat
from scipy.ndimage import gaussian_filter
from polymerfts import lfts

# OpenMP environment variables
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"  # 0, 1
os.environ["OMP_NUM_THREADS"] = "2"  # 1 ~ 4

f = 0.4        # A-fraction of major BCP chain, f
eps = 1.0       # a_A/a_B, conformational asymmetry

params = {
    "nx":[64, 64, 64],          # Simulation grid numbers
    "lx":[7.31, 7.31, 7.31],    # Simulation box size as a_Ref * N_Ref^(1/2) unit,
                                # where "a_Ref" is reference statistical segment length
                                # and "N_Ref" is the number of segments of reference linear homopolymer chain.

    "chain_model":"discrete",   # "discrete" or "continuous" chain model
    "ds":1/90,                  # Contour step interval, which is equal to 1/N_Ref.

    "segment_lengths":{         # Relative statistical segment length compared to "a_Ref.
        "A":1.0, 
        "B":1.0, },

    "chi_n": {"A,B": 18.35},     # Bare interaction parameter, Flory-Huggins params * N_Ref

    "distinct_polymers":[{      # Distinct Polymers
        "volume_fraction":1.0,  # Volume fraction of polymer chain
        "blocks":[              # AB diBlock Copolymer
            {"type":"A", "length":f, }, # A-block
            {"type":"B", "length":1-f}, # B-block
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

    "verbose_level":1,      # 1 : Print at each Langevin step.
                            # 2 : Print at each saddle point iteration.
}
# Set random seed
# If you want to obtain different results for each execution, set random_seed=None
random_seed = 12345

# Set initial fields
print("w_A and w_B are initialized to gyroid phase.")
input_data = loadmat("GyroidInput.mat", squeeze_me=True)
w_A = input_data["w_A"]
w_B = input_data["w_B"]

# Initialize simulation
simulation = lfts.LFTS(params=params, random_seed=random_seed)

# Set a timer
time_start = time.time()

# # Continue simulation with recorded field configurations and random state.
# simulation.continue_run(file_name="fields_010000.mat")

# Run
simulation.run(initial_fields={"A": w_A, "B": w_B})

# Recording first a few iteration results for debugging and refactoring
# (Anderson Mixing)
# ---------- Run  ----------
# iterations, mass error, total partitions, Hamiltonian, incompressibility error (or saddle point error)
#        1    2.295E-18  [ 1.6835907E+01  ]     7.272741449   [9.3906464E-05 ]
# Langevin step:  1
#       38    5.704E-16  [ 1.3958888E+01  ]     5.401031083   [8.9368844E-05 ]
# Langevin step:  2
#       36   -1.176E-16  [ 2.2166101E+01  ]     7.118772535   [9.4230077E-05 ]
# Langevin step:  3
#       39    2.325E-16  [ 2.2177780E+01  ]     7.217326145   [6.6307238E-05 ]
# Langevin step:  4
#       38   -8.821E-16  [ 2.1590641E+01  ]     7.240011784   [8.6466100E-05 ]
# Langevin step:  5
#       38    7.677E-18  [ 2.1011830E+01  ]     7.239727806   [8.7919974E-05 ]

# (LRAM)
# ---------- Run  ----------
# iterations, mass error, total partitions, Hamiltonian, incompressibility error (or saddle point error)
#        1    2.295E-18  [ 1.6835907E+01  ]     7.272741449   [9.3906464E-05 ]
# Langevin step:  1
#        9   -1.613E-17  [ 1.3958879E+01  ]     5.401031270   [4.0786240E-05 ]
# Langevin step:  2
#        9    6.963E-16  [ 2.2166390E+01  ]     7.118768389   [5.1305878E-05 ]
# Langevin step:  3
#        9   -3.622E-16  [ 2.2178084E+01  ]     7.217319886   [4.6183956E-05 ]
# Langevin step:  4
#        9   -4.441E-16  [ 2.1590972E+01  ]     7.239990296   [6.3314222E-05 ]
# Langevin step:  5
#        9   -4.787E-16  [ 2.1012985E+01  ]     7.239751839   [4.0090717E-05 ]
