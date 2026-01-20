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
f = 0.25        # A-fraction of major BCP chain, f
eps = 2.0       # a_A/a_B, conformational asymmetry

params = {
    "nx":[32,64,64],            # Simulation grid numbers
    "lx":[4.0,7.0,7.0],         # Simulation box size as a_Ref * N_Ref^(1/2) unit,
                                # where "a_Ref" is reference statistical segment length
                                # and "N_Ref" is the number of segments of reference linear homopolymer chain.

    "reduce_memory":False,     # Reduce memory usage by storing only check points.
    "box_is_altering":True,
    "stress_interval":1,     # Compute stress every iteration (for reproducibility)          # Find box size that minimizes the free energy during saddle point iteration.
    "chain_model":"continuous",      # "discrete" or "continuous" chain model
    "ds":1/100,                      # Contour step interval, which is equal to 1/N_Ref.

    "segment_lengths":{         # Relative statistical segment length compared to "a_Ref.
        "A":np.sqrt(eps*eps/(eps*eps*f + (1-f))), 
        "B":np.sqrt(    1.0/(eps*eps*f + (1-f))), },

    "chi_n": {"A,B": 25},       # Interaction parameter, Flory-Huggins params * N_Ref

    "space_group" :{
        "symbol":"P4_2/mnm",   # IT symbol of the space group
        "number": 419,         # (optional) Hall number of the space group
    },

    "distinct_polymers":[{      # Distinct Polymers
        "volume_fraction":1.0,  # volume fraction of polymer chain
        "blocks":[              # AB diBlock Copolymer
            {"type":"B", "length":1-f}, # B-block
            {"type":"A", "length":f, }, # A-block
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
print("w_A and w_B are initialized to Sigma phase.")
# [Ref: https://doi.org/10.3390/app2030654]
sphere_positions = [
    [0.00, 0.00, 0.00], [0.50, 0.50, 0.50],  # A
    [0.00, 0.40, 0.40], [0.00, 0.60, 0.60], [0.50, 0.90, 0.10], [0.50, 0.10, 0.90],  # B
    [0.00, 0.46, 0.13], [0.00, 0.13, 0.46], [0.00, 0.87, 0.54], [0.00, 0.54, 0.87],  # C
    [0.50, 0.63, 0.04], [0.50, 0.04, 0.63], [0.50, 0.96, 0.37], [0.50, 0.37, 0.96],  # C
    [0.00, 0.74, 0.07], [0.00, 0.07, 0.74], [0.00, 0.93, 0.26], [0.00, 0.26, 0.93],  # D
    [0.50, 0.43, 0.24], [0.50, 0.24, 0.43], [0.50, 0.76, 0.57], [0.50, 0.56, 0.77],  # D
    [0.25, 0.18, 0.18], [0.25, 0.82, 0.82], [0.25, 0.68, 0.32], [0.25, 0.32, 0.68],  # E
    [0.75, 0.18, 0.18], [0.75, 0.82, 0.82], [0.75, 0.68, 0.32], [0.75, 0.32, 0.68]   # E
]

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
calculation.save_results("Sigma.json")

# Recording first a few iteration results for debugging and refactoring
#    1   -1.462E-15  [ 1.0040000E+00  ]    -0.001859583   1.9215121E+00  [  4.0000000, 7.0000000, 7.0000000 ]
#    2    4.736E-13  [ 1.0022061E+00  ]    -0.000570350   1.1293870E+00  [  4.0000299, 7.0000203, 7.0000203 ]
#    3    3.036E-12  [ 1.0016441E+00  ]    -0.000238806   6.6231791E-01  [  4.0000421, 7.0000294, 7.0000294 ]
#    4   -1.754E-12  [ 1.0014360E+00  ]    -0.000146152   4.2737822E-01  [  4.0000485, 7.0000346, 7.0000346 ]
#    5   -1.246E-12  [ 1.0013535E+00  ]    -0.000121015   3.3566219E-01  [  4.0000524, 7.0000379, 7.0000379 ]