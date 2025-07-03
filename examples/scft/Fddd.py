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
    "box_is_altering":True,     # Find box size that minimizes the free energy during saddle point iteration.
    "chain_model":"continuous", # "discrete" or "continuous" chain model
    "ds":1/100,                  # Contour step interval, which is equal to 1/N_Ref.

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
calculation.save_results("fields.mat")

# Recording first a few iteration results for debugging and refactoring
# (without space group)
#    1   -9.816E-16  [ 1.7113060E+00  ]    -0.026635221   1.1182156E-03  [  5.5800000, 3.1700000, 1.5900000 ]
#    2    3.273E-13  [ 1.7113039E+00  ]    -0.026635248   1.0560161E-03  [  5.5799986, 3.1699924, 1.5899349 ]
#    3   -1.255E-12  [ 1.7114120E+00  ]    -0.026635755   6.1469774E-04  [  5.5799774, 3.1698765, 1.5889437 ]
#    4   -9.190E-13  [ 1.7115022E+00  ]    -0.026635886   5.6580261E-04  [  5.5799711, 3.1698435, 1.5886610 ]
#    5    1.010E-12  [ 1.7116056E+00  ]    -0.026636007   4.1350304E-04  [  5.5799640, 3.1698058, 1.5883267 ]

# (with space group)
#    1   -9.816E-16  [ 1.7113060E+00  ]    -0.026635221   1.1182156E-03  [  5.5800000, 3.1700000, 1.5900000 ]
#    2    1.970E-12  [ 1.7113044E+00  ]    -0.026635590   8.4505486E-04  [  5.5799860, 3.1699237, 1.5893487 ]
#    3    1.234E-12  [ 1.7113566E+00  ]    -0.026636098   5.9584334E-04  [  5.5799089, 3.1695182, 1.5856175 ]
#    4   -1.557E-12  [ 1.7113659E+00  ]    -0.026636183   3.8796429E-04  [  5.5799196, 3.1695916, 1.5860956 ]
#    5   -1.332E-12  [ 1.7113697E+00  ]    -0.026636219   2.0767394E-04  [  5.5799293, 3.1696664, 1.5864523 ]