import os
import sys
import time
import numpy as np
from scipy.io import savemat
from scipy.ndimage.filters import gaussian_filter
import scft

# OpenMP environment variables
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"  # 0, 1
os.environ["OMP_NUM_THREADS"] = "2"  # 1 ~ 4

# Major Simulation params
f = 0.4        # A-fraction of major BCP chain, f

params = {
    # "platform":"cpu-mkl",           # choose platform among [cuda, cpu-mkl]
    
    "nx":[32,32,32],            # Simulation grid numbers
    "lx":[3.654,3.654,3.654],   # Simulation box size as a_Ref * N_Ref^(1/2) unit,
                                # where "a_Ref" is reference statistical segment length
                                # and "N_Ref" is the number of segments of reference linear homopolymer chain.

    "box_is_altering":False,    # Find box size that minimizes the free energy during saddle point iteration.
    "chain_model":"continuous",   # "discrete" or "continuous" chain model
    "ds":1/50,                  # Contour step interval, which is equal to 1/N_Ref.

    "segment_lengths":{         # Relative statistical segment length compared to "a_Ref.
        "A":1.0, 
        "B":1.0, },

    "chi_n": [["A", "B", 15]],   # Interaction parameter, Flory-Huggins params * N_Ref

    "distinct_polymers":[{      # Distinct Polymers
        "volume_fraction":1.0,  # volume fraction of polymer chain
        "blocks":[              # AB diBlock Copolymer
            {"type":"A", "length":f, }, # A-block
            {"type":"B", "length":1-f}, # B-block
        ],},],

    "optimizer":"adam", # Select an optimizer among 'Anderson Mixing' and 'ADAM' for finding saddle point

    # "am":{
    #     "max_hist":20,          # Maximum number of history
    #     "start_error":1e-3,     # When switch to AM from simple mixing
    #     "mix_min":0.01,          # Minimum mixing rate of simple mixing
    #     "mix_init":0.01,         # Initial mixing rate of simple mixing
    # },

    "max_iter":10000,     # The maximum relaxation iterations
    "tolerance":1e-8     # Terminate iteration if the self-consistency error is less than tolerance
}

# # Set random seed
# # If you want to obtain different results for each execution, set random_seed=None
# np.random.seed(random_seed)

# Set initial fields
print("w_A and w_B are initialized to random Gaussian.")
w_A = np.random.normal(0.0, 1.0, params["nx"])
w_B = np.random.normal(0.0, 1.0, params["nx"])

# w_A = np.random.uniform(-1.0, 1.0, params["nx"])
# w_B = np.random.uniform(-1.0, 1.0, params["nx"])

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
if len(sys.argv) >= 2:
    file_name = "fields_%05d.mat" % (int(sys.argv[1]))
else:
    file_name = "fields.mat"
calculation.save_results(file_name)

# Recording first a few iteration results for debugging and refactoring
    #    1   -2.554E-15  [ 1.0112006E+00  ]    -0.005556176   1.4462576E+00 
    #    2    6.439E-15  [ 1.0120959E+00  ]    -0.005282124   1.0825724E+00 
    #    3    3.553E-15  [ 1.0136723E+00  ]    -0.005608658   8.8710667E-01 
    #    4   -1.066E-14  [ 1.0156904E+00  ]    -0.006254574   7.8531908E-01 
    #    5   -6.772E-15  [ 1.0180879E+00  ]    -0.007113039   7.3257170E-01 
