import os
import time
import numpy as np
from scipy.io import savemat, loadmat
from scipy.ndimage.filters import gaussian_filter
import scft

# OpenMP environment variables
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "2"  # 0, 1 or 2
os.environ["OMP_NUM_THREADS"] = "2"  # 1 ~ 4

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
        "B":1.0,
        "C":1.0},

    "chi_n": [["A","B",19],   # Interaction parameter, Flory-Huggins params * N_Ref
              ["A","C",20],
              ["B","C",21],
             ],

    "distinct_polymers":[{      # Distinct Polymers
        "volume_fraction":1.0,  # volume fraction of polymer chain
        "blocks":[              # ABC triblock Copolymer
            {"type":"A", "length":0.2},     # A-block
            {"type":"B", "length":0.3},     # B-block
            {"type":"C", "length":0.5},     # C-block
        ],},],
        
    "max_iter":2000,     # The maximum relaxation iterations
    "tolerance":1e-8     # Terminate iteration if the self-consistency error is less than tolerance
}

# Initialize calculation
calculation = scft.SCFT(params=params)

# Set initial fields
w_A = np.zeros(list(params["nx"]), dtype=np.float64)
w_B = np.zeros(list(params["nx"]), dtype=np.float64)
w_C = np.zeros(list(params["nx"]), dtype=np.float64)
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
            w_C[i,j,k] =  0.3164*c1 -0.1074*c2

# w_A = np.zeros(list(params["nx"]), dtype=np.float64)
# w_B = np.zeros(list(params["nx"]), dtype=np.float64)
# w_C = np.zeros(list(params["nx"]), dtype=np.float64)

# Set a timer
time_start = time.time()

# Run
calculation.run(initial_fields={"A": w_A, "B": w_B, "C": w_C})

# Estimate execution time
time_duration = time.time() - time_start
print("total time: %f " % time_duration)

# Save final results
phi = calculation.get_concentrations()
w = calculation.get_fields()

mdic = {"params":params, "dim":len(params["nx"]), "nx":params["nx"], "lx":params["lx"], "ds":params["ds"],
        "chi_n":params["chi_n"], "chain_model":params["chain_model"],
        "w_a":w["A"], "w_b":w["B"], "w_c":w["C"], "phi_a":phi["A"], "phi_b":phi["B"], "phi_c":phi["C"]}
savemat("fields.mat", mdic)

# print(phi["A"])

# Recording first a few iteration results for debugging and refactoring
    #    1    5.551E-17  [ 3.6174136E+01  ]     6.573449234   4.1304847E+00  [  3.3000000, 3.3000000, 3.3000000 ]
    #    2   -5.107E-15  [ 3.6096529E+01  ]     6.563719988   3.2789176E-01  [  3.3000012, 3.3000012, 3.3000012 ]
    #    3    4.940E-15  [ 3.6089226E+01  ]     6.564058652   2.7369175E-01  [  3.2999732, 3.2999732, 3.2999732 ]
    #    4    1.615E-14  [ 3.6080571E+01  ]     6.564000530   2.7421913E-01  [  3.2999472, 3.2999472, 3.2999472 ]
    #    5    7.772E-15  [ 3.6072521E+01  ]     6.563917607   2.7325569E-01  [  3.2999237, 3.2999237, 3.2999237 ]