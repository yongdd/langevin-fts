import os
import sys
import time
import numpy as np
import json
import scipy.io
from scipy.ndimage import gaussian_filter
from polymerfts import scft

# OpenMP environment variables
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"  # 0, 1
os.environ["OMP_NUM_THREADS"] = "2"  # 1 ~ 4

# Major Simulation params
f = 0.4       # A-fraction of major BCP chain, f

# Open and read the JSON file
input_data = scipy.io.loadmat("PL.mat", squeeze_me=True)
print(input_data["nx"])
print(input_data["lx"])

params = {
    "nx":[64,32,32],            # Simulation grid numbers
    "lx":input_data["lx"],      # Simulation box size as a_Ref * N_Ref^(1/2) unit,
                                # where "a_Ref" is reference statistical segment length
                                # and "N_Ref" is the number of segments of reference linear homopolymer chain.

    "reduce_gpu_memory_usage":False, # Reduce gpu memory usage by storing propagators in main memory instead of gpu memory.
    "box_is_altering":True,          # Find box size that minimizes the free energy during saddle point iteration.
    "chain_model":"continuous",      # "discrete" or "continuous" chain model
    "ds":1/100,                      # Contour step interval, which is equal to 1/N_Ref.

    "scale_stress": 1.0,

    "segment_lengths":{         # Relative statistical segment length compared to "a_Ref.
        "A":1.0, 
        "B":1.0, },

    "chi_n": {"A,B": 15},     # Interaction parameter, Flory-Huggins params * N_Ref

    "distinct_polymers":[{      # Distinct Polymers
        "volume_fraction":1.0,  # volume fraction of polymer chain
        "blocks":[              # AB diBlock Copolymer
            {"type":"A", "length":f, }, # A-block
            {"type":"B", "length":1-f}, # B-block
        ],},],
    
    # "space_group" :{
    #     "symbol":"P6_3/mmc",  # IT symbol of the space group
    # },
    
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
calculation.save_results("PL.json")

# Recording first a few iteration results for debugging and refactoring
#    1    1.688E-16  [ 3.3282272E+04  ]     8.543126219   2.3374074E+00  [  1.7547415, 1.2021598, 1.2021598 ]
#    2    1.254E-12  [ 1.6403619E+04  ]     7.050990161   1.8765011E+00  [  1.9115585, 1.2570578, 1.2570578 ]
#    3    2.677E-14  [ 7.9971332E+03  ]     5.871700645   1.5899486E+00  [  2.0287933, 1.3007231, 1.3007231 ]
#    4   -1.201E-12  [ 4.0556552E+03  ]     4.901541934   1.3949141E+00  [  2.1224039, 1.3371794, 1.3371794 ]
#    5   -1.217E-13  [ 2.1738442E+03  ]     4.084484179   1.2557275E+00  [  2.2005246, 1.3687938, 1.3687938 ]