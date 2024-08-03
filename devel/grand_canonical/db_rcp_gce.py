import os
import time
import numpy as np
from scipy.io import savemat, loadmat
import scft_gc as scft

# OpenMP environment variables
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"  # 0, 1
os.environ["OMP_NUM_THREADS"] = "2"  # 1 ~ 4

# Major Simulation params
f = 0.5             # A-fraction of major BCP chain, f
alpha = 0.2
mu_rcp = 1.0

params = {
    # "platform":"cuda",           # choose platform among [cuda, cpu-mkl]
    
    "nx":[100],          # Simulation grid numbers
    "lx":[1.67],    # Simulation box size as a_Ref * N_Ref^(1/2) unit,
                              # where "a_Ref" is reference statistical segment length
                              # and "N_Ref" is the number of segments of reference linear homopolymer chain.

    "box_is_altering":False,         # Find box size that minimizes the free energy during saddle point iteration.
    "scale_stress" : 0.1,            # Scaling factor for stress, w_diff[M:] = scale_stress*dlx
    
    "reduce_gpu_memory_usage":False, # Reduce gpu memory usage by storing propagators in main memory instead of gpu memory.
    "chain_model":"continuous",      # "discrete" or "continuous" chain model
    "ds":1/500,                      # Contour step interval, which is equal to 1/N_Ref.

    "segment_lengths":{         # Relative statistical segment length compared to "a_Ref.
        "A":1.0, 
        "B":1.0, },

    "chi_n": {"A,B": 15.0},       # Interaction parameter, Flory-Huggins params * N_Ref
    "ensemble": "gce",            # "ce": canonical ensemble or "gce": grand canonical ensemble

    "distinct_polymers":[{      # Distinct Polymers
        "chemical_potential": mu_rcp,
        "blocks":[              # AB Diblock
            {"type":"A", "length":f, },     # A-block
            {"type":"B", "length":1.0-f, }, # B-block
        ],},
        {
        "chemical_potential": 0.0,
        "blocks":[              # Random Copolymer. Only single block random copolymer is supported.
            {"type":"R", "length":alpha, "fraction":{"A":0.5, "B":0.5},},
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
for i in range(0,params["nx"][0]):
    w_A[i] =  10*np.cos(2*np.pi*i/params["nx"][0])+3.0
    w_B[i] = -10*np.cos(2*np.pi*i/params["nx"][0])+3.0

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
