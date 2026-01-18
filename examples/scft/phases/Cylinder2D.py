import os
import time
import numpy as np
from scipy.io import savemat
from scipy.ndimage import gaussian_filter
from polymerfts import scft

# OpenMP environment variables
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"  # 0, 1
os.environ["OMP_NUM_THREADS"] = "2"  # 1 ~ 4

# Major Simulation params
f = 1.0/3.0     # A-fraction of major BCP chain, f

params = {
    # "platform":"cuda",           # choose platform among [cuda, cpu-mkl]
    
    "nx":[48,32],                  # Simulation grid numbers
    "lx":[2.7,1.6],                # Simulation box size as a_Ref * N_Ref^(1/2) unit,
                                   # where "a_Ref" is reference statistical segment length
                                   # and "N_Ref" is the number of segments of reference linear homopolymer chain.

    "reduce_memory":False,     # Reduce memory usage by storing only check points.
    "box_is_altering":True,          # Find box size that minimizes the free energy during saddle point iteration.
    "chain_model":"continuous",      # "discrete" or "continuous" chain model
    "ds":1/90,                       # Contour step interval, which is equal to 1/N_Ref.

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
print("w_A and w_B are initialized to cylindrical phase.")
cylinder_positions = [
[0.0,0.0],[1/2,1/2]]
for y,z in cylinder_positions:
    my, mz = np.round((np.array([y, z])*params["nx"])).astype(np.int32)
    w_A[my,mz] = -1/(np.prod(params["lx"])/np.prod(params["nx"]))
w_A = gaussian_filter(w_A, sigma=np.min(params["nx"])/15, mode='wrap')

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
calculation.save_results("C2D.json")

# Recording first a few iteration results for debugging and refactoring
#    1   -3.364E-15  [ 1.1072987E+00  ]    -0.057523751   1.5324187E+00  [  2.7000000, 1.6000000 ]
#    2    5.452E-15  [ 1.0699399E+00  ]    -0.024326094   1.0238845E+00  [  2.7025811, 1.6048410 ]
#    3   -4.180E-14  [ 1.0554717E+00  ]    -0.011536990   7.2534463E-01  [  2.7040263, 1.6076075 ]
#    4   -8.636E-15  [ 1.0490457E+00  ]    -0.006099893   5.5394703E-01  [  2.7050075, 1.6095148 ]
#    5    1.372E-14  [ 1.0459471E+00  ]    -0.003673710   4.5908209E-01  [  2.7057610, 1.6109921 ]