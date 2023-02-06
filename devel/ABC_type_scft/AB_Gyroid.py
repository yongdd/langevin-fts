import os
import time
import numpy as np
from scipy.io import savemat, loadmat
from scipy.ndimage.filters import gaussian_filter
import scft

# OpenMP environment variables
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "2"  # 0, 1 or 2
os.environ["OMP_NUM_THREADS"] = "2"  # 1 ~ 4

# Major Simulation params
f = 0.36        # A-fraction of major BCP chain, f
eps = 1.0       # a_A/a_B, conformational asymmetry

params = {
    "platform":"cuda",           # choose platform among [cuda, cpu-mkl]
    
    "nx":[32,32,32],            # Simulation grid numbers
    "lx":[3.3,3.3,3.3],         # Simulation box size as a_Ref * N_Ref^(1/2) unit,
                                # where "a_Ref" is reference statistical segment length
                                # and "N_Ref" is the number of segments of reference linear homopolymer chain.

    "box_is_altering":True,     # Find box size that minimizes the free energy during saddle point iteration.
    "chain_model":"continuous", # "discrete" or "continuous" chain model
    "ds":1/100,                 # Contour step interval, which is equal to 1/N_Ref.

    "segment_lengths":{         # Relative statistical segment length compared to "a_Ref.
        "A":1.0, 
        "B":1.0,},

    "chi_n": [["A","B",20]],   # Interaction parameter, Flory-Huggins params * N

    "distinct_polymers":[
        {                               # Distinct Polymers
        "volume_fraction":1.0,          # volume fraction of polymer chain
        "blocks":[                      # AB diBlock Copolymer
            {"type":"A", "length":f, }, # A-block
            {"type":"B", "length":1-f}, # B-block
        ],},
        {
        "volume_fraction":0.0,  
        "blocks":[              # Random Copolymer. Currently, Only single block random copolymer is supported.
            {"type":"random", "length":0.4, "fraction":{"A":0.4, "B":0.6},},
        ],},
        {
        "volume_fraction":0.0,  
        "blocks":[              # Random Copolymer.
            {"type":"random", "length":0.4, "fraction":{"A":0.6, "B":0.4},},
        ],},
        ],
        
    "max_iter":2000,     # The maximum relaxation iterations
    "tolerance":1e-8     # Terminate iteration if the self-consistency error is less than tolerance
}
# Initialize calculation
calculation = scft.SCFT(params=params)

# Set initial fields
w_A = np.zeros(list(params["nx"]), dtype=np.float64)
w_B = np.zeros(list(params["nx"]), dtype=np.float64)
print("w_A and w_B are initialized to Gyroid phase.")
# [Ref: https://pubs.acs.org/doi/pdf/10.1021/ma951138i]
for i in range(0,params["nx"][0]):
    xx = (i+1)*2*np.pi/params["nx"][0]
    for j in range(0,params["nx"][1]):
        yy = (j+1)*2*np.pi/params["nx"][1]
        for k in range(0,params["nx"][2]):
            zz = (k+1)*2*np.pi/params["nx"][2]
            c1 = np.sqrt(8.0/3.0)*(np.cos(xx)*np.sin(yy)*np.sin(2.0*zz) +
                np.cos(yy)*np.sin(zz)*np.sin(2.0*xx)+np.cos(zz)*np.sin(xx)*np.sin(2.0*yy))
            c2 = np.sqrt(4.0/3.0)*(np.cos(2.0*xx)*np.cos(2.0*yy)+
                np.cos(2.0*yy)*np.cos(2.0*zz)+np.cos(2.0*zz)*np.cos(2.0*xx))
            idx = i*params["nx"][1]*params["nx"][2] + j*params["nx"][2] + k
            w_A[i,j,k] = -0.3164*c1 +0.1074*c2
            w_B[i,j,k] =  0.3164*c1 -0.1074*c2

# Set a timer
time_start = time.time()

# Run
calculation.run(initial_fields={"A": w_A, "B": w_B})

# Estimate execution time
time_duration = time.time() - time_start
print("total time: %f " % time_duration)

# Save final results
phi = calculation.get_concentrations()
w = calculation.get_fields()

mdic = {"params":params, "dim":len(params["nx"]), "nx":params["nx"], "lx":params["lx"], "ds":params["ds"],
        "f":f, "chi_n":params["chi_n"], "epsilon":eps, "chain_model":params["chain_model"],
        "w_a":w["A"], "w_b":w["B"], "phi_a":phi["A"], "phi_b":phi["B"]}
savemat("fields.mat", mdic)

# Recording first a few iteration results for debugging and refactoring
    #    1   -1.221E-14  [ 3.6339314E+01  ]    -0.005550595   1.4461834E+00  [  3.3000000, 3.3000000, 3.3000000 ]
    #    2   -4.996E-15  [ 3.6372091E+01  ]    -0.005274676   1.0820525E+00  [  3.3000217, 3.3000217, 3.3000217 ]
    #    3   -4.108E-15  [ 3.6428830E+01  ]    -0.005598893   8.8631868E-01  [  3.3000287, 3.3000287, 3.3000287 ]
    #    4   -9.659E-15  [ 3.6501158E+01  ]    -0.006242039   7.8458220E-01  [  3.3000282, 3.3000282, 3.3000282 ]
    #    5   -1.221E-15  [ 3.6586928E+01  ]    -0.007097169   7.3186726E-01  [  3.3000234, 3.3000234, 3.3000234 ]
    