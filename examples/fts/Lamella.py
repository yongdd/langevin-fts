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
    "chi_n":20,                 # Interaction parameter, Flory-Huggins params * N

    "segment_lengths":{         # Relative statistical segment length compared to "a_Ref.
        "A":np.sqrt(eps*eps/(eps*eps*f + (1-f))), 
        "B":np.sqrt(    1.0/(eps*eps*f + (1-f))), },

    "distinct_polymers":[{      # Distinct Polymers
        "volume_fraction":1.0,  # volume fraction of polymer chain
        "blocks":[              # AB diBlock Copolymer
            {"type":"A", "length":f, }, # A-block
            {"type":"B", "length":1-f}, # B-block
        ],},],
        
    "langevin":{                # Langevin Dynamics
        "max_step":200000,      # Langevin steps for simulation
        "dt":0.8,               # Langevin step interval, delta tau*N
        "nbar":1024,            # invariant polymerization index, nbar
    },
    
    "recording":{                       # Recording Simulation Data
        "dir":"data_simulation",        # Directory name
        "recording_period":1000,        # period for recording concentrations and fields
        "sf_computing_period":10,       # period for computing structure function
        "sf_recording_period":1000,     # period for recording structure function
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

# w_minus and w_plus are initialized to random
#       21    1.776E-15  [ 5.3349931E+02  ]     5.009168908   9.3083443E-05 
# iteration, mass error, total_partitions, energy_total, error_level
# ---------- Run  ----------
# Langevin step:  1
#       18   -1.221E-15  [ 6.0606825E+02  ]     5.040091197   8.1538804E-05 
#        7    1.332E-15  [ 6.0574572E+02  ]     5.037712444   8.9411454E-05 
# Langevin step:  2
#       19   -3.553E-15  [ 6.8625017E+02  ]     5.070141937   7.9617551E-05 
#        8   -8.882E-16  [ 6.8551820E+02  ]     5.067732088   8.0144714E-05 
# Langevin step:  3
#       19   -3.109E-15  [ 7.6914055E+02  ]     5.099658828   9.5416355E-05 
#        8    4.441E-16  [ 7.6807098E+02  ]     5.097283111   9.5933782E-05 
# Langevin step:  4
#       20   -3.220E-15  [ 8.5964523E+02  ]     5.127049567   7.4926202E-05 
#        9   -3.442E-15  [ 8.5832988E+02  ]     5.124597150   7.2687167E-05 
# Langevin step:  5
#       20   -2.442E-15  [ 9.5610439E+02  ]     5.154916422   8.3472264E-05 
#        9   -3.553E-15  [ 9.5435437E+02  ]     5.152498970   7.9975233E-05 