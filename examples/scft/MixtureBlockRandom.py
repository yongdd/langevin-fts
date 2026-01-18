import os
import time
import numpy as np
from scipy.io import savemat, loadmat
from polymerfts import scft

# OpenMP environment variables
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"  # 0, 1
os.environ["OMP_NUM_THREADS"] = "2"  # 1 ~ 4

# Major Simulation params
f = 0.5             # A-fraction of major BCP chain, f
eps = 2.0           # a_A/a_B, conformational asymmetry
RCP_A_frac = 0.7    # fraction of A monomer in random copolymer

params = {
    # "platform":"cuda",           # choose platform among [cuda, cpu-mkl]
    
    "nx":[32,32,32],          # Simulation grid numbers
    "lx":[4.36,4.36,4.36],    # Simulation box size as a_Ref * N_Ref^(1/2) unit,
                              # where "a_Ref" is reference statistical segment length
                              # and "N_Ref" is the number of segments of reference linear homopolymer chain.

    "reduce_memory":False,     # Reduce memory usage by storing only check points.
    "box_is_altering":True,      # Find box size that minimizes the free energy during saddle point iteration.
    "chain_model":"continuous",  # "discrete" or "continuous" chain model
    "ds":1/90,                   # Contour step interval, which is equal to 1/N_Ref.

    "segment_lengths":{         # Relative statistical segment length compared to "a_Ref.
        "A":np.sqrt(eps*eps/(eps*eps*f + (1-f))), 
        "B":np.sqrt(    1.0/(eps*eps*f + (1-f))), },

    "chi_n": {"A,B": 16},       # Interaction parameter, Flory-Huggins params * N_Ref

    "distinct_polymers":[{      # Distinct Polymers
        "volume_fraction":0.7,  # volume fraction of polymer chain
        "blocks":[              # AB homopolymer
            {"type":"A", "length":f, },     # A-block
            {"type":"B", "length":1.0-f, }, # B-block
        ],},
        {
        "volume_fraction":0.3,  
        "blocks":[              # Random Copolymer. Only single block random copolymer is supported.
            {"type":"R", "length":7.0/9.0, "fraction":{"A":RCP_A_frac, "B":1-RCP_A_frac},},
        ],}],

    "optimizer":{
        "name":"am",            # Anderson Mixing
        "max_hist":20,          # Maximum number of history
        "start_error":1e-2,     # When switch to AM from simple mixing
        "mix_min":0.1,          # Minimum mixing rate of simple mixing
        "mix_init":0.1,         # Initial mixing rate of simple mixing
    },

    "max_iter":2000,     # The maximum relaxation iterations
    "tolerance":1e-8     # Terminate iteration if the self-consistency error is less than tolerance
}

# Set initial fields
w_A = np.zeros(list(params["nx"]), dtype=np.float64)
w_B = np.zeros(list(params["nx"]), dtype=np.float64)
print("w_A and w_B are initialized to lamellar phase.")
for i in range(0,params["nx"][2]):
    w_A[:,:,i] =  np.cos(3*2*np.pi*i/params["nx"][2])
    w_B[:,:,i] = -np.cos(3*2*np.pi*i/params["nx"][2])

# Initialize calculation
calculation = scft.SCFT(params=params)

# Set a timer
time_start = time.time()

# Run
calculation.run(initial_fields={"A": w_A, "B": w_B})

# Estimate execution time
time_duration = time.time() - time_start
print("total time: %f " % time_duration)

# Save final results (.mat, .json or .yaml format)
calculation.save_results("fields.mat")

# Recording first a few iteration results for debugging and refactoring
    #    1   -2.010E-14  [ 1.0498229E+00  1.0110509E+00  ]    -0.007024164   2.4661560E-01  [  4.3600000, 4.3600000, 4.3600000 ]
    #    2    9.326E-15  [ 1.0517443E+00  1.0120736E+00  ]    -0.007281942   2.3431712E-01  [  4.3600000, 4.3600000, 4.3598315 ]
    #    3   -1.354E-14  [ 1.0538978E+00  1.0129840E+00  ]    -0.007578146   2.2787688E-01  [  4.3600000, 4.3600000, 4.3596648 ]
    #    4    6.239E-14  [ 1.0562507E+00  1.0138247E+00  ]    -0.007901204   2.2427552E-01  [  4.3600000, 4.3600000, 4.3594972 ]
    #    5    5.729E-14  [ 1.0587865E+00  1.0146281E+00  ]    -0.008246396   2.2205958E-01  [  4.3600000, 4.3600000, 4.3593262 ]
    