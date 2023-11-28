import os
import time
import numpy as np
from scipy.io import savemat, loadmat
from scipy.ndimage import gaussian_filter
import scft

# OpenMP environment variables
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"  # 0, 1
os.environ["OMP_NUM_THREADS"] = "2"  # 1 ~ 4

# Major Simulation params
f = 0.43        # A-fraction of major BCP chain, f

params = {
    # "platform":"cuda",           # choose platform among [cuda, cpu-mkl]
    
    "nx":[16,32,56],            # Simulation grid numbers
    "lx":[1.59,3.17,5.58],      # Simulation box size as a_Ref * N_Ref^(1/2) unit,
                                # where "a_Ref" is reference statistical segment length
                                # and "N_Ref" is the number of segments of reference linear homopolymer chain.

    "box_is_altering":True,     # Find box size that minimizes the free energy during saddle point iteration.
    "chain_model":"continuous", # "discrete" or "continuous" chain model
    "ds":1/100,                  # Contour step interval, which is equal to 1/N_Ref.

    "segment_lengths":{         # Relative statistical segment length compared to "a_Ref.
        "A":1.0, 
        "B":1.0, },

    "chi_n": [["A", "B", 12.0]],  # Interaction parameter, Flory-Huggins params * N_Ref

    "distinct_polymers":[{      # Distinct Polymers
        "volume_fraction":1.0,  # volume fraction of polymer chain
        "blocks":[              # AB diBlock Copolymer
            {"type":"A", "length":f, }, # A-block
            {"type":"B", "length":1-f}, # B-block
        ],},],

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
w_A_k = np.zeros_like(np.fft.rfftn(np.zeros(list(params["nx"]), dtype=np.float64)))
w_A_k[ 0, 0,4] =  0.02+0.39j
w_A_k[ 0, 2,2] =  0.22+0.36j
w_A_k[ 0,30,2] =  0.37+0.20j
w_A_k[ 1, 1,1] = -0.81-0.32j
w_A_k[ 1,31,1] = -0.09+0.86j
w_A_k[15, 1,1] =  0.69+0.52j
w_A_k[15,31,1] = -0.37+0.80j

w_A = np.fft.irfftn(w_A_k, params["nx"])*np.prod(params["nx"])
w_B = -w_A

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
    