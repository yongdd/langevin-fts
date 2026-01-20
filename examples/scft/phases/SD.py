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
input_data = scipy.io.loadmat("SD.mat", squeeze_me=True)
print(input_data["nx"])
print(input_data["lx"])

params = {
    "nx":[48,48,48],            # Simulation grid numbers
    "lx":input_data["lx"],      # Simulation box size as a_Ref * N_Ref^(1/2) unit,
                                # where "a_Ref" is reference statistical segment length
                                # and "N_Ref" is the number of segments of reference linear homopolymer chain.

    "reduce_memory":False,     # Reduce memory usage by storing only check points.
    "box_is_altering":True,
    "stress_interval":1,     # Compute stress every iteration (for reproducibility)          # Find box size that minimizes the free energy during saddle point iteration.
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
    
    "space_group" :{
        "symbol":"Fd-3m",  # IT symbol of the space group
        "number": 526,     # (optional) Hall number of the space group
    },
    
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
calculation.save_results("SD.json")

# Recording first a few iteration results for debugging and refactoring
#    1   -1.466E-15  [ 4.6412009E+00  ]    -0.192474745   1.1913250E-02  [  2.5748822, 2.5748822, 2.5748822 ]
#    2    2.023E-12  [ 4.5620019E+00  ]    -0.190673407   3.7367008E-02  [  2.5748723, 2.5748723, 2.5748723 ]
#    3    2.820E-14  [ 4.5632945E+00  ]    -0.190930724   3.4331178E-02  [  2.5748511, 2.5748511, 2.5748511 ]
#    4   -1.265E-12  [ 4.5648601E+00  ]    -0.191155912   3.1556080E-02  [  2.5748298, 2.5748298, 2.5748298 ]
#    5   -9.358E-13  [ 4.5666724E+00  ]    -0.191353784   2.9032107E-02  [  2.5748082, 2.5748082, 2.5748082 ]