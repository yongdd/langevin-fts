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
    # "platform":"cuda",           # choose platform among [cuda, cpu-mkl]

    "nx":[64, 64, 64],          # Simulation grid numbers
    "lx":[7.31, 7.31, 7.31],    # Simulation box size as a_Ref * N_Ref^(1/2) unit,
                                # where "a_Ref" is reference statistical segment length
                                # and "N_Ref" is the number of segments of reference linear homopolymer chain.

    "chain_model":"discrete",   # "discrete" or "continuous" chain model
    "ds":1/90,                  # Contour step interval, which is equal to 1/N_Ref.

    "segment_lengths":{         # Relative statistical segment length compared to "a_Ref.
        "A":np.sqrt(eps*eps/(eps*eps*f + (1-f))), 
        "B":np.sqrt(    1.0/(eps*eps*f + (1-f))), },

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

    "am":{
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

# Initialize calculation
simulation = lfts.LFTS(params=params, random_seed=random_seed)

# Set a timer
time_start = time.time()

# # Continue simulation with recorded field configurations and random state.
# simulation.continue_run(file_name="fields_010000.mat")

# Run
simulation.run(initial_fields={"A": w_A, "B": w_B})

# Estimate execution time
time_duration = time.time() - time_start
print("total time: %f, time per step: %f" %
    (time_duration, time_duration/params["langevin"]["max_step"]) )

# Recording first a few iteration results for debugging and refactoring

#        1   -1.379E-16  [ 1.6835907E+01  ]     7.272741449   9.3906464E-05 
# iteration, mass error, total partitions, total energy, incompressibility error
# ---------- Run  ----------
# Langevin step:  1
#       35   -4.947E-16  [ 4.4022050E+00  ]     6.436670785   [9.5108379E-05 ]
# Langevin step:  2
#       36    6.630E-16  [ 4.1851269E+00  ]     6.716426574   [8.7159749E-05 ]
# Langevin step:  3
#       36   -1.719E-17  [ 3.9864870E+00  ]     6.898253870   [9.3443534E-05 ]
# Langevin step:  4
#       36    2.451E-16  [ 3.8347445E+00  ]     7.021231100   [9.7467063E-05 ]
# Langevin step:  5
#       36    4.186E-17  [ 3.7154179E+00  ]     7.099604680   [8.9711866E-05 ]