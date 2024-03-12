import os
import time
import numpy as np
from scipy.io import savemat
import scft

# OpenMP environment variables
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"  # 0, 1
os.environ["OMP_NUM_THREADS"] = "2"  # 1 ~ 4

# Major Simulation params
f = 0.5         # A-fraction of major BCP chain, f

params = {
    # "platform":"cpu-mkl",           # choose platform among [cuda, cpu-mkl]
    
    "nx":[256],         # Simulation grid numbers
    "lx":[1.5],         # Simulation box size as a_Ref * N_Ref^(1/2) unit,
                        # where "a_Ref" is reference statistical segment length
                        # and "N_Ref" is the number of segments of reference linear homopolymer chain.

    "box_is_altering":True,       # Find box size that minimizes the free energy during saddle point iteration.
    "chain_model":"continuous",   # "discrete" or "continuous" chain model
    "ds":1/200,                   # Contour step interval, which is equal to 1/N_Ref.

    "segment_lengths":{         # Relative statistical segment length compared to "a_Ref.
        "A":1.0, 
        "B":1.0, },
    
    "chi_n": {"A,B": 9.5},      # Interaction parameter, Flory-Huggins params * N_Ref

    "distinct_polymers":[{      # Distinct Polymers
        "volume_fraction":1.0,  # volume fraction of polymer chain
        "blocks":[              # 3-arm star-shaped AB Copolymer
            {"type":"A", "length":f,   "v":0, "u":1},    # A-block
            {"type":"A", "length":f,   "v":0, "u":2},    # A-block
            {"type":"A", "length":f,   "v":0, "u":3},    # A-block
            {"type":"B", "length":1-f, "v":1, "u":4},    # B-block
            {"type":"B", "length":1-f, "v":2, "u":5},    # B-block
            {"type":"B", "length":1-f, "v":3, "u":6},    # B-block
        ],},],

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
for i in range(0,params["nx"][0]):
    w_A[i] =  np.cos(2*np.pi*i/params["nx"][0])
    w_B[i] = -np.cos(2*np.pi*i/params["nx"][0])

# Initialize calculation
calculation = scft.SCFT(params=params)

# Set a timer
time_start = time.time()

# Run
calculation.run(initial_fields={"A": w_A, "B": w_B})

# Estimate execution time
time_duration = time.time() - time_start
print("total time: %f " % time_duration)

# Save final results
calculation.save_results("fields.mat")

# Recording first a few iteration results for debugging and refactoring
    #    1   -2.109E-15  [ 1.2195802E+00  ]    -0.013537312   4.8603177E-01  [  1.5000000 ]
    #    2   -2.554E-15  [ 1.2204438E+00  ]    -0.011167461   3.6789421E-01  [  1.4996449 ]
    #    3    3.775E-15  [ 1.2253469E+00  ]    -0.010065926   2.8544670E-01  [  1.4988889 ]
    #    4   -3.220E-15  [ 1.2322967E+00  ]    -0.009631768   2.2965879E-01  [  1.4978956 ]
    #    5    2.220E-15  [ 1.2403078E+00  ]    -0.009546745   1.9308599E-01  [  1.4967546 ]
