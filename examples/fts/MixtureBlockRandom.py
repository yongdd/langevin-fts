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
    "chi_n":25,                 # Bare interaction parameter, Flory-Huggins params*N_ref

    "segment_lengths":{         # Relative statistical segment length compared to "a_Ref.
        "A":np.sqrt(eps*eps/(eps*eps*f + (1-f))), 
        "B":np.sqrt(    1.0/(eps*eps*f + (1-f))), },

    "distinct_polymers":[{      # Distinct Polymers
        "volume_fraction":0.7,  # Volume fraction of polymer chain
        "blocks":[              # AB homopolymer
            {"type":"A", "length":f, },     # A-block
            {"type":"B", "length":1.0-f, }, # B-block
        ],},
        {
        "volume_fraction":0.3,  
        "blocks":[              # Random Copolymer. Currently, Only single block random copolymer is supported.
            {"type":"random", "length":0.5, "fraction":{"A":0.5, "B":0.5},},
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
        "start_error":8e-1,         # When switch to AM from simple mixing
        "mix_min":0.1,              # Minimum mixing rate of simple mixing
        "mix_init":0.1,             # Initial mixing rate of simple mixing
    },

    "verbose_level":1,      # 1 : Print at each Langevin step.
                            # 2 : Print at each saddle point iteration.
}

## random seed for MT19937
np.random.seed(5489)

# Set initial fields
print("w_minus and w_plus are initialized to random")
w_plus  = np.random.normal(0.0, 1.0, params["nx"])
w_minus = np.random.normal(0.0, 1.0, params["nx"])

# Initialize calculation
simulation = lfts.LFTS(params=params)

# Set a timer
time_start = time.time()

# Run
simulation.run(w_minus=w_minus, w_plus=w_plus)

# Estimate execution time
time_duration = time.time() - time_start
print("total time: %f, time per step: %f" %
    (time_duration, time_duration/params["langevin"]["max_step"]) )

# Recording first a few iteration results for debugging and refactoring

#       21    4.441E-16  [ 1.0359367E+00  9.9716427E-01  ]     6.261376974   7.4857853E-05 
# ---------- Run  ----------
# Langevin step:  1
#       16    2.665E-15  [ 1.1504483E+00  1.0010009E+00  ]     6.282266480   9.0886276E-05 
# Langevin step:  2
#       19   -6.661E-16  [ 1.6535901E+00  1.0113321E+00  ]     6.361301332   8.9827704E-05 
# Langevin step:  3
#       20    2.665E-15  [ 2.2020159E+00  1.0256580E+00  ]     6.386744506   7.6052619E-05 
# Langevin step:  4
#       20   -1.776E-15  [ 2.8239625E+00  1.0405886E+00  ]     6.381340056   9.3209010E-05 
# Langevin step:  5
#       21    3.997E-15  [ 3.5696714E+00  1.0569576E+00  ]     6.363888540   7.1880345E-05 
