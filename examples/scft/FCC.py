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
f = 24/90       # A-fraction of major BCP chain, f

params = {
    "nx":[32,32,32],            # Simulation grid numbers
    "lx":[1.91,1.91,1.91],      # Simulation box size as a_Ref * N_Ref^(1/2) unit,
                                # where "a_Ref" is reference statistical segment length
                                # and "N_Ref" is the number of segments of reference linear homopolymer chain.

    "reduce_gpu_memory_usage":False, # Reduce gpu memory usage by storing propagators in main memory instead of gpu memory.
    "box_is_altering":True,     # Find box size that minimizes the free energy during saddle point iteration.
    "chain_model":"continuous", # "discrete" or "continuous" chain model
    "ds":1/90,                  # Contour step interval, which is equal to 1/N_Ref.

    "segment_lengths":{         # Relative statistical segment length compared to "a_Ref.
        "A":1.0, 
        "B":1.0, },

    "chi_n": {"A,B": 18.1},     # Interaction parameter, Flory-Huggins params * N_Ref

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
print("w_A and w_B are initialized to BCC phase.")
n_unitcell = 1 # number of unit cell for each direction. the number of total unit cells is n_unitcell^3
sphere_positions = []
for i in range(0,n_unitcell):
    for j in range(0,n_unitcell):
        for k in range(0,n_unitcell):
            sphere_positions.append([i/n_unitcell,j/n_unitcell,k/n_unitcell])
            sphere_positions.append([(i+0/2)/n_unitcell,(j+1/2)/n_unitcell,(k+1/2)/n_unitcell])
            sphere_positions.append([(i+1/2)/n_unitcell,(j+0/2)/n_unitcell,(k+1/2)/n_unitcell])
            sphere_positions.append([(i+1/2)/n_unitcell,(j+1/2)/n_unitcell,(k+0/2)/n_unitcell])
for x,y,z in sphere_positions:
    molecules, my, mz = np.round((np.array([x, y, z])*params["nx"])).astype(np.int32)
    w_A[molecules,my,mz] = -1/(np.prod(params["lx"])/np.prod(params["nx"]))
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
calculation.save_results("fields.mat")

# Recording first a few iteration results for debugging and refactoring
    #    1    3.490E-17  [ 1.2357278E+00  ]    -0.129993803   1.9797895E+00  [  1.9100000, 1.9100000, 1.9100000 ]
    #    2    3.659E-13  [ 1.1496105E+00  ]    -0.050647547   1.2179429E+00  [  1.9177727, 1.9177727, 1.9177727 ]
    #    3   -4.691E-13  [ 1.1240766E+00  ]    -0.023067731   8.5182029E-01  [  1.9219990, 1.9219990, 1.9219990 ]
    #    4    6.768E-13  [ 1.1144044E+00  ]    -0.010774352   6.5169543E-01  [  1.9250324, 1.9250324, 1.9250324 ]
    #    5    6.987E-13  [ 1.1107171E+00  ]    -0.004797123   5.3966576E-01  [  1.9274962, 1.9274962, 1.9274962 ]