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
        "max_step":2000,        # Langevin steps for simulation
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

# w_minus and w_plus are initialized to random
#       23   -2.220E-16   5.8293236E+02     5.031220086   9.5822227E-05
# ---------- Run ----------
# iteration, mass error, total_partition, energy_total, error_level
# langevin step:  1
#       19   -3.553E-15   6.5966016E+02     5.062036019   7.6462958E-05
#        8   -2.998E-15   6.5907274E+02     5.059683923   7.3716113E-05
# langevin step:  2
#       19   -2.220E-16   7.4356322E+02     5.092392378   8.9084460E-05
#        8    4.441E-16   7.4257836E+02     5.089977265   9.0169219E-05
# langevin step:  3
#       20    2.665E-15   8.2950944E+02     5.121605195   7.2575434E-05
#        9   -3.553E-15   8.2821693E+02     5.119221963   7.0321982E-05
