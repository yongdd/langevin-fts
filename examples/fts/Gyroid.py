import os
import time
import numpy as np
from scipy.io import savemat, loadmat
from scipy.ndimage.filters import gaussian_filter
import lfts

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
    "chi_n":18.35,              # Interaction parameter, Flory-Huggins params * N

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
        "nbar":10000,           # invariant polymerization index, nbar
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
print("w_minus and w_plus are initialized to gyroid")
input_data = loadmat("GyroidInput.mat", squeeze_me=True)
w_minus = input_data["w_minus"]
w_plus = input_data["w_plus"]

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

#        1    3.109E-15  [ 6.5764065E+03  ]     7.272741449   9.3906464E-05
# iteration, mass error, total_partitions, energy_total, error_level
# ---------- Run  ----------
# Langevin step:  1
#       33    5.329E-15  [ 7.2863238E+03  ]     7.744996235   9.0632164E-05
#       17    4.441E-15  [ 7.2298612E+03  ]     7.709080595   9.1220098E-05
# Langevin step:  2
#       33   -9.215E-15  [ 7.8703656E+03  ]     8.121646393   9.9045819E-05
#       17   -7.105E-15  [ 7.8148915E+03  ]     8.088031001   9.0288780E-05
# Langevin step:  3
#       33    6.883E-15  [ 8.3426268E+03  ]     8.429947008   9.6396998E-05
#       17   -8.549E-15  [ 8.2913023E+03  ]     8.399015117   9.1699316E-05
