import os
import time
import numpy as np
from scipy.io import savemat, loadmat
from scipy.ndimage.filters import gaussian_filter
import lfts

# # Major Simulation params
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
    "chi_n":25,                 # Interaction parameter, Flory-Huggins params * N

    "segment_lengths":{         # Relative statistical segment length compared to "a_Ref.
        "A":np.sqrt(eps*eps/(eps*eps*f + (1-f))), 
        "B":np.sqrt(    1.0/(eps*eps*f + (1-f))), },

    "distinct_polymers":[{      # Distinct Polymers
        "volume_fraction":0.7,  # volume fraction of polymer chain
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
        "max_step":200,         # Langevin steps for simulation
        "dt":0.8,               # Langevin step interval, delta tau*N
        "nbar":10000,           # invariant polymerization index, nbar
    },
    
    "recording":{                       # Recording Simulation Data
        "dir":"data_simulation",        # Directory name
        "recording_period":100,         # period for recording concentrations and fields
        "sf_computing_period":10,       # period for computing structure function
        "sf_recording_period":100,      # period for recording structure function
    },

    "saddle":{                # Iteration for the pressure field 
        "max_iter" :100,      # maximum the number of iterations
        "tolerance":1e-4,     # tolerance of incompressibility 
    },

    "am":{
        "max_hist":20,              # Maximum number of history
        "start_error":8e-1,         # When switch to AM from simple mixing
        "mix_min":0.1,              # Minimum mixing rate of simple mixing
        "mix_init":0.1,             # Initial mixing rate of simple mixing
    },

    "verbose_level":1,      # 1 : Print at each langevin step.
                            # 2 : Print at each saddle point iteration.
}

## random seed for MT19937
np.random.seed(5489)

# standard deviation of normal noise
langevin_sigma = lfts.calculate_sigma(params["langevin"]["nbar"], params["langevin"]["dt"], np.prod(params["nx"]), np.prod(params["lx"]))

# Set initial fields
print("w_minus and w_plus are initialized to random")
w_plus  = np.random.normal(0.0, langevin_sigma, params["nx"])
w_minus = np.random.normal(0.0, langevin_sigma, params["nx"])

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

#       21   -2.220E-16  [ 5.3403777E+02  5.1204688E+02  ]     6.261653163   7.6180036E-05
# iteration, mass error, total_partitions, energy_total, error_level
# ---------- Run  ----------
# Langevin step:  1
#       14   -1.998E-15  [ 5.5627301E+02  5.1217781E+02  ]     6.272363707   9.3104738E-05
#        4   -3.331E-16  [ 5.5608440E+02  5.1217731E+02  ]     6.271847736   8.3069653E-05
# Langevin step:  2
#       15   -1.554E-15  [ 5.7911662E+02  5.1239117E+02  ]     6.282408942   8.1379948E-05
#        4    0.000E+00  [ 5.7890701E+02  5.1239004E+02  ]     6.281888837   9.1595030E-05
# Langevin step:  3
#       15    1.554E-15  [ 6.0136857E+02  5.1266494E+02  ]     6.291607328   9.5658255E-05
#        5    2.220E-16  [ 6.0114725E+02  5.1266287E+02  ]     6.291103931   7.0976167E-05