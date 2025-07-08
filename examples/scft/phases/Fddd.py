import os
import time
import numpy as np
import scipy.io
from scipy.ndimage import gaussian_filter
from polymerfts import scft

# OpenMP environment variables
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"  # 0, 1
os.environ["OMP_NUM_THREADS"] = "2"  # 1 ~ 4

# Major Simulation params
f = 0.43        # A-fraction of major BCP chain, f

params = {
    # "platform":"cuda",           # choose platform among [cuda, cpu-mkl]
    
    "nx":[84,48,24],            # Simulation grid numbers
    "lx":[5.58,3.17,1.59],      # Simulation box size as a_Ref * N_Ref^(1/2) unit,
                                # where "a_Ref" is reference statistical segment length
                                # and "N_Ref" is the number of segments of reference linear homopolymer chain.

    "reduce_gpu_memory_usage":False, # Reduce gpu memory usage by storing propagators in main memory instead of gpu memory.
    "box_is_altering":True,          # Find box size that minimizes the free energy during saddle point iteration.
    "chain_model":"continuous",      # "discrete" or "continuous" chain model
    "ds":1/100,                      # Contour step interval, which is equal to 1/N_Ref.

    "segment_lengths":{         # Relative statistical segment length compared to "a_Ref.
        "A":1.0, 
        "B":1.0, },

    "chi_n": {"A,B": 14.0},     # Interaction parameter, Flory-Huggins params * N_Ref

    "distinct_polymers":[{      # Distinct Polymers
        "volume_fraction":1.0,  # volume fraction of polymer chain
        "blocks":[              # AB diBlock Copolymer
            {"type":"A", "length":f, }, # A-block
            {"type":"B", "length":1-f}, # B-block
        ],},],

    "space_group" :{
        "symbol":"Fddd",   # IT symbol of the space group
        "number": 336,     # (optional) Hall number of the space group
    },

    "optimizer":{       
        # "name":"adam",     # ADAM optimizer
        # "lr":1e-1,         # initial learning rate,
        # "gamma":0.9993,    # learning rate at Tth iteration is lr*γ^(T-1)
        
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
print("w_A and w_B are initialized to Fddd phase.")
input_data = scipy.io.loadmat("FdddInput.mat", squeeze_me=True)
w_A = input_data["w_A"]
w_B = input_data["w_B"]

# Interpolate input data on params["nx"], if necessary
w_A = scipy.ndimage.zoom(np.reshape(w_A, input_data["nx"]), params["nx"]/input_data["nx"])
w_B = scipy.ndimage.zoom(np.reshape(w_B, input_data["nx"]), params["nx"]/input_data["nx"])

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
calculation.save_results("Fddd.json")

# Recording first a few iteration results for debugging and refactoring
#    1   -4.716E-17  [ 1.7113060E+00  ]    -0.099581166   1.6450075E-01  [  5.5800000, 3.1700000, 1.5900000 ]
#    2    3.283E-13  [ 1.7411991E+00  ]    -0.102040548   1.6251983E-01  [  5.5799986, 3.1699924, 1.5899349 ]
#    3   -1.650E-12  [ 1.7724644E+00  ]    -0.104533095   1.6033448E-01  [  5.5799966, 3.1699828, 1.5898623 ]
#    4    9.538E-13  [ 1.8050994E+00  ]    -0.107044909   1.5791765E-01  [  5.5799944, 3.1699729, 1.5897906 ]
#    5    1.732E-12  [ 1.8390979E+00  ]    -0.109563065   1.5525652E-01  [  5.5799926, 3.1699643, 1.5897281 ]