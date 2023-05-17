import os
import time
import numpy as np
from scipy.io import savemat
import scft

# OpenMP environment variables
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"  # 0, 1
os.environ["OMP_NUM_THREADS"] = "2"  # 1 ~ 4

# Major Simulation params
f = 0.36       # A-fraction of major BCP chain, f
eps = 1.0       # a_A/a_B, conformational asymmetry

params = {
    # "platform":"cpu-mkl",           # choose platform among [cuda, cpu-mkl]
    
    "nx":[64,64,64],        # Simulation grid numbers
    "lx":[7.15,7.15,7.15],  # Simulation box size as a_Ref * N_Ref^(1/2) unit,
                            # where "a_Ref" is reference statistical segment length
                            # and "N_Ref" is the number of segments of reference linear homopolymer chain.
                            
    "box_is_altering":True,       # Find box size that minimizes the free energy during saddle point iteration.
    "chain_model":"continuous",   # "discrete" or "continuous" chain model
    "ds":1/100,                   # Contour step interval, which is equal to 1/N_Ref.

    "segment_lengths":{         # Relative statistical segment length compared to "a_Ref.
        "A":np.sqrt(eps*eps/(eps*eps*f + (1-f))), 
        "B":np.sqrt(    1.0/(eps*eps*f + (1-f))), },

    "chi_n": [["A", "B", 10.0]],   # Interaction parameter, Flory-Huggins params * N_Ref

    "distinct_polymers":[{      # Distinct Polymers
        "volume_fraction":1.0,  # volume fraction of polymer chain
        "blocks":[              # 9-arm star-shaped AB Copolymer
            {"type":"A", "length":f,   "v":0, "u":1},    # A-block
            {"type":"A", "length":f,   "v":0, "u":2},    # A-block
            {"type":"A", "length":f,   "v":0, "u":3},    # A-block
            {"type":"A", "length":f,   "v":0, "u":4},    # A-block
            {"type":"A", "length":f,   "v":0, "u":5},    # A-block
            {"type":"A", "length":f,   "v":0, "u":6},    # A-block
            {"type":"A", "length":f,   "v":0, "u":7},    # A-block
            {"type":"A", "length":f,   "v":0, "u":8},    # A-block
            {"type":"A", "length":f,   "v":0, "u":9},    # A-block
            {"type":"B", "length":1-f, "v":1, "u":10},    # B-block
            {"type":"B", "length":1-f, "v":2, "u":11},    # B-block
            {"type":"B", "length":1-f, "v":3, "u":12},    # B-block
            {"type":"B", "length":1-f, "v":4, "u":13},    # B-block
            {"type":"B", "length":1-f, "v":5, "u":14},    # B-block
            {"type":"B", "length":1-f, "v":6, "u":15},    # B-block
            {"type":"B", "length":1-f, "v":7, "u":16},    # B-block
            {"type":"B", "length":1-f, "v":8, "u":17},    # B-block
            {"type":"B", "length":1-f, "v":9, "u":18},    # B-block
        ],},],

    "max_iter":2000,      # The maximum relaxation iterations
    "tolerance":1e-8,     # Terminate iteration if the self-consistency error is less than tolerance

    "am":{
        "max_hist":60,            # Maximum number of history
        "start_error":1e-2,       # When switch to AM from simple mixing
        "mix_min":0.02,           # Minimum mixing rate of simple mixing
        "mix_init":0.02,          # Initial mixing rate of simple mixing
    },
}

# Set initial fields
w_A = np.zeros(list(params["nx"]), dtype=np.float64)
w_B = np.zeros(list(params["nx"]), dtype=np.float64)
print("w_A and w_B are initialized to gyroid phase.")
# [Ref: https://pubs.acs.org/doi/pdf/10.1021/ma951138i]
for i in range(0,params["nx"][0]):
    xx = 2*(i+1)*2*np.pi/params["nx"][0]
    for j in range(0,params["nx"][1]):
        yy = 2*(j+1)*2*np.pi/params["nx"][1]
        zz = 2*np.arange(1,params["nx"][2]+1)*2*np.pi/params["nx"][2]
        
        c1 = np.sqrt(8.0/3.0)*(np.cos(xx)*np.sin(yy)*np.sin(2.0*zz) +
            np.cos(yy)*np.sin(zz)*np.sin(2.0*xx)+np.cos(zz)*np.sin(xx)*np.sin(2.0*yy))
        c2 = np.sqrt(4.0/3.0)*(np.cos(2.0*xx)*np.cos(2.0*yy)+
            np.cos(2.0*yy)*np.cos(2.0*zz)+np.cos(2.0*zz)*np.cos(2.0*xx))
        w_A[i,j,:] = -0.3164*c1 +0.1074*c2
        w_B[i,j,:] =  0.3164*c1 -0.1074*c2

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
phi = calculation.get_concentrations()
w = calculation.get_fields()

mdic = {"params":params, "dim":len(params["nx"]), "nx":params["nx"], "lx":params["lx"], "ds":params["ds"],
        "f":f, "chi_n":params["chi_n"], "epsilon":eps, "chain_model":params["chain_model"],
        "w_a":w["A"], "w_b":w["B"], "phi_a":phi["A"], "phi_b":phi["B"]}
savemat("fields.mat", mdic)

# Recording first a few iteration results for debugging and refactoring
    #    1   -7.638E-14  [ 1.1876171E+00  ]    -0.007941056   1.2529953E+00  [  7.1500000, 7.1500000, 7.1500000 ]
    #    2    4.379E-13  [ 1.1890070E+00  ]    -0.007739723   1.1394238E+00  [  7.1499870, 7.1499870, 7.1499870 ]
    #    3    1.581E-13  [ 1.1913107E+00  ]    -0.007620777   1.0431815E+00  [  7.1499733, 7.1499733, 7.1499733 ]
    #    4   -1.724E-13  [ 1.1943605E+00  ]    -0.007567283   9.6203651E-01  [  7.1499591, 7.1499591, 7.1499591 ]
    #    5   -7.205E-14  [ 1.1980285E+00  ]    -0.007566018   8.9394663E-01  [  7.1499446, 7.1499446, 7.1499446 ]
