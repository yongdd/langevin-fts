import os
import sys
import time
import numpy as np
from scipy.io import savemat
from scipy.ndimage import gaussian_filter
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

    "chi_n": {"A,B": 15},       # Interaction parameter, Flory-Huggins params * N_Ref

    "distinct_polymers":[{      # Distinct Polymers
        "volume_fraction":1.0,  # volume fraction of polymer chain
        "blocks":[              # AB diBlock Copolymer
            {"type":"A", "length":f, }, # A-block
            {"type":"B", "length":1-f}, # B-block
        ],},],

    # Select an optimizer among "am"(Anderson Mixing) and 'adam'(ADAM) for finding saddle point

    "optimizer":{
        "name":"adam",     # ADAM optimizer
        "lr":1e-1,         # initial learning rate
        "gamma":0.9993,    # learning rate at Tth iteration is lr*γ^(T-1)
        
    #     "name":"am",            # Anderson Mixing
    #     "max_hist":20,          # Maximum number of history
    #     "start_error":1e-2,     # When switch to AM from simple mixing
    #     "mix_min":0.1,          # Minimum mixing rate of simple mixing
    #     "mix_init":0.1,         # Initial mixing rate of simple mixing
    },

    "max_iter":5000,     # The maximum relaxation iterations
    "tolerance":1e-8     # Terminate iteration if the self-consistency error is less than tolerance
}

# # Set random seed
# # If you want to obtain different results for each execution, set random_seed=None
# np.random.seed(random_seed)

# Set initial fields
# print("w_A and w_B are initialized to random Fourier components.")
# temp = np.zeros(params["nx"])
# temp_k = np.fft.rfftn(temp)

# cutoff=3
# w_A_k = np.zeros_like(temp_k)
# w_B_k = np.zeros_like(temp_k)
# phase_A = np.random.uniform(0.0, 2*np.pi, (cutoff,cutoff,cutoff))
# phase_B = np.random.uniform(0.0, 2*np.pi, (cutoff,cutoff,cutoff))
# w_A_k[0:cutoff,0:cutoff,0:cutoff] = np.random.normal(0.0, 1.0, (cutoff,cutoff,cutoff))*np.exp(1j*phase_A)
# w_B_k[0:cutoff,0:cutoff,0:cutoff] = np.random.normal(0.0, 1.0, (cutoff,cutoff,cutoff))*np.exp(1j*phase_B)
# w_A = np.fft.irfftn(w_A_k)
# w_B = np.fft.irfftn(w_B_k)

# print("w_A and w_B are initialized to random Gaussian.")
w_A = np.random.normal(0.0, 1.0, params["nx"])
w_B = np.random.normal(0.0, 1.0, params["nx"])

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