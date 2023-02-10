import os
import time
import numpy as np
from scipy.io import savemat
from scipy.ndimage.filters import gaussian_filter
import scft

# OpenMP environment variables
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "2"  # 0, 1 or 2
os.environ["OMP_NUM_THREADS"] = "2"  # 1 ~ 4

# Major Simulation params
f = 0.25        # A-fraction of major BCP chain, f
eps = 2.0       # a_A/a_B, conformational asymmetry

params = {
    "nx":[64,64,32],            # Simulation grid numbers
    "lx":[7.0,7.0,4.0],         # Simulation box size as a_Ref * N_Ref^(1/2) unit,
                                # where "a_Ref" is reference statistical segment length
                                # and "N_Ref" is the number of segments of reference linear homopolymer chain.

    "box_is_altering":True,     # Find box size that minimizes the free energy during saddle point iteration.
    "chain_model":"continuous", # "discrete" or "continuous" chain model
    "ds":1/100,                 # Contour step interval, which is equal to 1/N_Ref.
    "chi_n": 25,                # Interaction parameter, Flory-Huggins params * N

    "segment_lengths":{         # Relative statistical segment length compared to "a_Ref.
        "A":np.sqrt(eps*eps/(eps*eps*f + (1-f))), 
        "B":np.sqrt(    1.0/(eps*eps*f + (1-f))), },

    "distinct_polymers":[{      # Distinct Polymers
        "volume_fraction":1.0,  # volume fraction of polymer chain
        "blocks":[              # AB diBlock Copolymer
            {"type":"B", "length":1-f}, # B-block
            {"type":"A", "length":f, }, # A-block
        ],},],
    
    "max_iter":2000,     # The maximum relaxation iterations
    "tolerance":1e-8     # Terminate iteration if the self-consistency error is less than tolerance
}

# Set initial fields
w_A = np.zeros(list(params["nx"]), dtype=np.float64)
w_B = np.zeros(list(params["nx"]), dtype=np.float64)
print("w_A and w_B are initialized to Sigma phase.")
# [Ref: https://doi.org/10.3390/app2030654]
sphere_positions = [[0.00,0.00,0.00],[0.50,0.50,0.50], #A
[0.40,0.40,0.00],[0.60,0.60,0.00],[0.10,0.90,0.50],[0.90,0.10,0.50], #B
[0.13,0.46,0.00],[0.46,0.13,0.00],[0.54,0.87,0.00],[0.87,0.54,0.00], #C
[0.04,0.63,0.50],[0.63,0.04,0.50],[0.37,0.96,0.50],[0.96,0.37,0.50], #C
[0.07,0.74,0.00],[0.74,0.07,0.00],[0.26,0.93,0.00],[0.93,0.26,0.00], #D
[0.24,0.43,0.50],[0.43,0.24,0.50],[0.57,0.76,0.50],[0.77,0.56,0.50], #D
[0.18,0.18,0.25],[0.82,0.82,0.25],[0.32,0.68,0.25],[0.68,0.32,0.25], #E
[0.18,0.18,0.75],[0.82,0.82,0.75],[0.32,0.68,0.75],[0.68,0.32,0.75]] #E

for x,y,z in sphere_positions:
    mx, my, mz = np.round((np.array([x, y, z])*params["nx"])).astype(np.int32)
    w_A[mx,my,mz] = -1/(np.prod(params["lx"])/np.prod(params["nx"]))
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

# Save final results
phi_A, phi_B = calculation.get_concentrations()
w_A, w_B = calculation.get_fields()

mdic = {"params":params, "dim":len(params["nx"]), "nx":params["nx"], "lx":params["lx"], "ds":params["ds"],
        "f":f, "chi_n":params["chi_n"], "epsilon":eps, "chain_model":params["chain_model"],
        "w_a":w_A, "w_b":w_B, "phi_a":phi_A, "phi_b":phi_B}
savemat("fields.mat", mdic)

# Recording first a few iteration results for debugging and refactoring
    #    1    1.776E-15  [ 1.0040000E+00  ]    -0.001859583   1.9215121E+00  [  7.0000000, 7.0000000, 4.0000000 ]
    #    2   -3.220E-15  [ 1.0022061E+00  ]    -0.000570350   1.1293870E+00  [  7.0000203, 7.0000203, 4.0000299 ]
    #    3    6.439E-15  [ 1.0016441E+00  ]    -0.000238806   6.6231790E-01  [  7.0000294, 7.0000294, 4.0000421 ]
    #    4    2.243E-14  [ 1.0014360E+00  ]    -0.000146151   4.2737821E-01  [  7.0000346, 7.0000346, 4.0000485 ]
    #    5   -1.943E-14  [ 1.0013535E+00  ]    -0.000121015   3.3566217E-01  [  7.0000379, 7.0000379, 4.0000524 ]
