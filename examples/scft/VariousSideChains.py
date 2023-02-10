import os
import time
import numpy as np
from scipy.io import savemat
import scft

# OpenMP environment variables
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "2"  # 0, 1 or 2
os.environ["OMP_NUM_THREADS"] = "2"  # 1 ~ 4

# Major Simulation params
f = 0.2         # A-fraction of major BCP chain, f
eps = 1.0       # a_A/a_B, conformational asymmetry

params = {
    # "platform":"cpu-mkl",           # choose platform among [cuda, cpu-mkl]
    
    "nx":[32,32,32],        # Simulation grid numbers
    "lx":[2.9,2.9,2.9],     # Simulation box size as a_Ref * N_Ref^(1/2) unit,
                            # where "a_Ref" is reference statistical segment length
                            # and "N_Ref" is the number of segments of reference linear homopolymer chain.

    "use_superposition":True,   # Superpose multiple partial partition functions when solving diffusion equations for speedup using superposition principle. 
                                # To obtain concentration of each block, disable this option.

    "box_is_altering":True,       # Find box size that minimizes the free energy during saddle point iteration.
    "chain_model":"discrete",   # "discrete" or "continuous" chain model
    "ds":1/10,                    # Contour step interval, which is equal to 1/N_Ref.
    "chi_n": 15.0,                # Interaction parameter, Flory-Huggins params * N

    "segment_lengths":{         # Relative statistical segment length compared to "a_Ref.
        "A":np.sqrt(eps*eps/(eps*eps*f + (1-f))), 
        "B":np.sqrt(    1.0/(eps*eps*f + (1-f))), },

    "distinct_polymers":[{      # Distinct Polymers
        "volume_fraction":1.0,  # volume fraction of polymer chain
        "blocks":[              # 9-arm star-shaped AB Copolymer
            {"type":"A", "length":f,   "v":0, "u":1},    # A-block
            {"type":"A", "length":f,   "v":1, "u":2},    # A-block
            {"type":"A", "length":f,   "v":2, "u":3},    # A-block
            {"type":"A", "length":f,   "v":3, "u":4},    # A-block
            {"type":"A", "length":f,   "v":4, "u":5},    # A-block

            {"type":"B", "length":(1-f)/1, "v":1, "u":6},    # B-block
            {"type":"B", "length":1-f, "v":2, "u":7},    # B-block
            {"type":"B", "length":(1-f)/2, "v":3, "u":8},    # B-block
            {"type":"B", "length":(1-f)/4, "v":4, "u":9},    # B-block

            {"type":"B", "length":1-f, "v":6, "u":10},    # B-block
            {"type":"B", "length":1-f, "v":7, "u":12},    # B-block

            {"type":"A", "length":f, "v":6, "u":11},    # B-block
            {"type":"A", "length":f, "v":7, "u":13},    # B-block
        ],},],

    "max_iter":1,      # The maximum relaxation iterations
    "tolerance":1e-8,     # Terminate iteration if the self-consistency error is less than tolerance

    "am":{
        "max_hist":60,           # Maximum number of history
        "start_error":1e-2,      # When switch to AM from simple mixing
        "mix_min":0.02,          # Minimum mixing rate of simple mixing
        "mix_init":0.02,         # Initial mixing rate of simple mixing
    },
}

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
phi_A, phi_B = calculation.get_concentrations()
w_A, w_B = calculation.get_fields()

mdic = {"params":params, "dim":len(params["nx"]), "nx":params["nx"], "lx":params["lx"], "ds":params["ds"],
        "f":f, "chi_n":params["chi_n"], "epsilon":eps, "chain_model":params["chain_model"],
        "w_a":w_A, "w_b":w_B, "phi_a":phi_A, "phi_b":phi_B}
savemat("fields.mat", mdic)

# Recording first a few iteration results for debugging and refactoring
    # 1    3.553E-15  [ 2.5612154E+01  ]    -0.002751834   1.1477009E+00  [  2.9000000, 2.9000000, 2.9000000 ]
    # 2   -1.199E-14  [ 2.5579319E+01  ]    -0.002374078   1.0665875E+00  [  2.9000198, 2.9000198, 2.9000198 ]
    # 3    7.327E-15  [ 2.5551103E+01  ]    -0.002039748   9.9023430E-01  [  2.9000380, 2.9000380, 2.9000380 ]
    # 4    2.132E-14  [ 2.5526964E+01  ]    -0.001744373   9.1850175E-01  [  2.9000550, 2.9000550, 2.9000550 ]
    # 5    3.109E-15  [ 2.5506412E+01  ]    -0.001483872   8.5122439E-01  [  2.9000707, 2.9000707, 2.9000707 ]
    # 6    6.217E-15  [ 2.5489005E+01  ]    -0.001254531   7.8821910E-01  [  2.9000854, 2.9000854, 2.9000854 ]