import os
import time
import numpy as np
from scipy.io import savemat, loadmat
from scipy.ndimage.filters import gaussian_filter
import lfts

f = 0.5         # A-fraction of major BCP chain, f
eps = 1.0       # a_A/a_B, conformational asymmetry

params = {
    # "platform":"cuda",           # choose platform among [cuda, cpu-mkl]

    "nx":[32, 32, 32],          # Simulation grid numbers
    "lx":[8.0, 8.0, 8.0],       # Simulation box size as a_Ref * N_Ref^(1/2) unit,
                                # where "a_Ref" is reference statistical segment length
                                # and "N_Ref" is the number of segments of reference linear homopolymer chain.

    "chain_model":"continuous", # "discrete" or "continuous" chain model
    "ds":1/16,                  # Contour step interval, which is equal to 1/N_Ref.

    "segment_lengths":{         # Relative statistical segment length compared to "a_Ref.
        "A":np.sqrt(eps*eps/(eps*eps*f + (1-f))), 
        "B":np.sqrt(    1.0/(eps*eps*f + (1-f))), },

    "chi_n": [["A","B", 25],     # Bare interaction parameter, Flory-Huggins params * N_Ref
             ],

    "distinct_polymers":[{      # Distinct Polymers
        "volume_fraction":0.7,  # Volume fraction of polymer chain
        "blocks":[              # AB homopolymer
            {"type":"A", "length":f, },     # A-block
            {"type":"B", "length":1.0-f, }, # B-block
        ],},
        {
        "volume_fraction":0.3,  
        "blocks":[              # Random Copolymer. Currently, Only single block random copolymer is supported.
            {"type":"R", "length":0.5, "fraction":{"A":0.5, "B":0.5},},
        ],}],

    "langevin":{                # Langevin Dynamics
        "max_step":200000,      # Langevin steps for simulation
        "dt":8.0,               # Langevin step interval, delta tau*N_ref
        "nbar":10000,           # Invariant polymerization index, nbar of N_Ref
    },
    
    "recording":{                       # Recording Simulation Data
        "dir":"data_simulation",        # Directory name
        "recording_period":1000,        # Period for recording concentrations and fields
        "sf_computing_period":10,       # Period for computing structure function
        "sf_recording_period":10000,    # Period for recording structure function
    },

    "saddle":{                # Iteration for the pressure field 
        "max_iter" :100,      # Maximum the number of iterations
        "tolerance":1e-4,     # Tolerance of incompressibility 
    },

    "am":{
        "max_hist":20,              # Maximum number of history
        "start_error":5e-1,         # When switch to AM from simple mixing
        "mix_min":0.01,              # Minimum mixing rate of simple mixing
        "mix_init":0.01,             # Initial mixing rate of simple mixing
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
w_A  = np.random.normal(0.0, 1.0, params["nx"])
w_B = np.random.normal(0.0, 1.0, params["nx"])

# Initialize calculation
simulation = lfts.LFTS(params=params, random_seed=random_seed)

# Set a timer
time_start = time.time()

# Run
simulation.run(initial_fields={"A": w_A, "B": w_B})

# Estimate execution time
time_duration = time.time() - time_start
print("total time: %f, time per step: %f" %
    (time_duration, time_duration/params["langevin"]["max_step"]) )

# Recording first a few iteration results for debugging and refactoring

#       20   -6.668E-17  [ 1.0217332E+00  1.0004506E+00  ]     6.255662440   6.8265095E-05 
# iteration, mass error, total partitions, total energy, incompressibility error
# ---------- Run  ----------
# Langevin step:  1
#       20    2.249E-16  [ 1.0473789E+00  1.0003198E+00  ]     6.261652588   [7.8185257E-05 ]
# Langevin step:  2
#       21    6.946E-19  [ 1.1468501E+00  9.9725365E-01  ]     6.287053766   [8.2566627E-05 ]
# Langevin step:  3
#       21   -2.104E-16  [ 1.2467280E+00  9.9391438E-01  ]     6.307844514   [8.3781278E-05 ]
# Langevin step:  4
#       21   -3.212E-16  [ 1.3546607E+00  9.9351465E-01  ]     6.325489881   [8.4406120E-05 ]
# Langevin step:  5
#       21   -2.289E-16  [ 1.4740321E+00  9.9596718E-01  ]     6.340174707   [8.3568662E-05 ]