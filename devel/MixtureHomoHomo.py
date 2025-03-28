import os
import time
import numpy as np
from scipy.io import savemat
import scft

# OpenMP environment variables
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"  # 0, 1
os.environ["OMP_NUM_THREADS"] = "2"  # 1 ~ 4

# Major Simulation params
eps = 3.0       # a_A/a_B, conformational asymmetry

params = {
    # "platform":"cuda",           # choose platform among [cuda, cpu-mkl]
    
    "nx":[32,32,32],          # Simulation grid numbers
    "lx":[4.36,4.36,4.36],    # Simulation box size as a_Ref * N_Ref^(1/2) unit,
                              # where "a_Ref" is reference statistical segment length
                              # and "N_Ref" is the number of segments of reference linear homopolymer chain.

    "reduce_gpu_memory_usage":False, # Reduce gpu memory usage by storing propagators in main memory instead of gpu memory.
    "box_is_altering":False,      # Find box size that minimizes the free energy during saddle point iteration.
    "chain_model":"continuous",   # "discrete" or "continuous" chain model
    "ds":1/50,                    # Contour step interval, which is equal to 1/N_Ref.

    "segment_lengths":{         # Relative statistical segment length compared to "a_Ref.
        "A":eps, 
        "B":1.0, },

    "chi_n": {"A,B": 5},        # Interaction parameter, Flory-Huggins params * N_Ref

    "distinct_polymers":[{      # Distinct Polymers
        "volume_fraction":0.5,  # volume fraction of polymer chain
        "blocks":[              # A homopolymer
            {"type":"A", "length":1.0, }, # A-block
        ],},
        {
        "volume_fraction":0.5,
        "blocks":[              # B homopolymer
            {"type":"B", "length":1.0, }, # B-block
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
    w_A[:,:,i] =  np.cos(2*np.pi*i/params["nx"][2])
    w_B[:,:,i] = -np.cos(2*np.pi*i/params["nx"][2])

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
    #    1    3.331E-14  [ 1.1153044E+00  1.2364116E+00  ]    -0.060670351   7.7365818E-01 
    #    2   -3.209E-14  [ 1.1404315E+00  1.2391044E+00  ]    -0.061562162   6.9931040E-01 
    #    3   -1.521E-14  [ 1.1671609E+00  1.2436602E+00  ]    -0.063007193   6.3517639E-01 
    #    4    2.820E-14  [ 1.1953798E+00  1.2498333E+00  ]    -0.064857197   5.7961311E-01 
    #    5    6.484E-14  [ 1.2249933E+00  1.2574211E+00  ]    -0.066996910   5.3111122E-01 
    