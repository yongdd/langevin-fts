import os
import sys
import numpy as np
import json
import scipy.io
from scipy.ndimage import gaussian_filter
from polymerfts import scft

# OpenMP environment variables
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"  # 0, 1
os.environ["OMP_NUM_THREADS"] = "2"  # 1 ~ 4

# Major Simulation params
f = 0.40       # A-fraction of major BCP chain, f

# Open and read the MAT file
input_data = scipy.io.loadmat("DP.mat", squeeze_me=True)
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
        "symbol":"Im-3m",  # IT symbol of the space group
        "number": 529,     # (optional) Hall number of the space group
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

# Run
calculation.run(initial_fields={"A": w_A, "B": w_B})

# Estimate execution time

# Save final results (.mat, .json or .yaml format)
calculation.save_results("DP.json")

# Recording first a few iteration results for debugging and refactoring
#        1   -2.378E-15  [ 3.5012541E+00  ]    -0.146832056   2.2156985E-02  lx=[  2.3750164, 2.3750164, 2.3750164 ]
#        2   -1.952E-15  [ 3.5104288E+00  ]    -0.146882023   2.0679812E-02  lx=[  2.3746860, 2.3746860, 2.3746860 ]
#        3    1.397E-15  [ 3.5191434E+00  ]    -0.146912322   1.9324173E-02  lx=[  2.3743726, 2.3743726, 2.3743726 ]
#        4    9.220E-16  [ 3.5274119E+00  ]    -0.146925808   1.8088343E-02  lx=[  2.3740762, 2.3740762, 2.3740762 ]
#        5    1.169E-15  [ 3.5352501E+00  ]    -0.146924993   1.6968165E-02  lx=[  2.3737965, 2.3737965, 2.3737965 ]
