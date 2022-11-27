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
        "max_step":200000,      # Langevin steps for simulation
        "dt":0.8,               # Langevin step interval, delta tau*N
        "nbar":10000,           # invariant polymerization index, nbar
    },
    
    "recording":{                       # Recording Simulation Data
        "dir":"data_simulation",        # Directory name
        "recording_period":1000,        # period for recording concentrations and fields
        "sf_computing_period":10,       # period for computing structure function
        "sf_recording_period":10000,    # period for recording structure function
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

#       21    4.441E-16  [ 5.3351383E+02  5.1204474E+02  ]     6.261376974   7.4857853E-05 
# iteration, mass error, total_partitions, energy_total, error_level
# ---------- Run  ----------
# Langevin step:  1
#       14   -4.885E-15  [ 5.5574282E+02  5.1217385E+02  ]     6.272106803   9.2404621E-05 
#        4   -3.886E-15  [ 5.5555454E+02  5.1217336E+02  ]     6.271590552   8.2536230E-05 
# Langevin step:  2
#       15    8.882E-16  [ 5.7858106E+02  5.1238556E+02  ]     6.282165497   8.1052449E-05 
#        4   -1.332E-15  [ 5.7837184E+02  5.1238444E+02  ]     6.281645188   9.1256208E-05 
# Langevin step:  3
#       15   -4.441E-16  [ 6.0083026E+02  5.1265787E+02  ]     6.291378576   9.5364084E-05 
#        5    1.998E-15  [ 6.0060929E+02  5.1265582E+02  ]     6.290874944   7.0780348E-05 
# Langevin step:  4
#       16    2.220E-15  [ 6.2419634E+02  5.1299139E+02  ]     6.299596453   7.3069145E-05 
#        5    5.107E-15  [ 6.2397834E+02  5.1298871E+02  ]     6.299088761   7.1999445E-05 
# Langevin step:  5
#       16    3.553E-15  [ 6.4789656E+02  5.1336815E+02  ]     6.307651731   8.0408833E-05 
#        5    6.661E-16  [ 6.4764190E+02  5.1336458E+02  ]     6.307160625   7.7852156E-05 
