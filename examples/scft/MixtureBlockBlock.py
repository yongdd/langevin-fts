import os
import time
import numpy as np
from scipy.io import savemat, loadmat
import scft

# # Major Simulation params
f1 = 0.5         # A-fraction of first BCP chain
f2 = 0.4         # A-fraction of second BCP chain
eps = 2.0        # a_A/a_B, conformational asymmetry

params = {
    # "platform":"cuda",           # choose platform among [cuda, cpu-mkl]
    
    "nx":[32,32,32],        # Simulation grid numbers
    "lx":[4.,4.,3.9],       # Simulation box size as a_Ref * N_Ref^(1/2) unit,
                            # where "a_Ref" is reference statistical segment length
                            # and "N_Ref" is the number of segments of reference linear homopolymer chain.

    "box_is_altering":True,      # Find box size that minimizes the free energy during saddle point iteration.
    "chain_model":"continuous",  # "discrete" or "continuous" chain model
    "ds":1/90,                   # Contour step interval, which is equal to 1/N_Ref.
    "chi_n": 14,                 # Interaction parameter, Flory-Huggins params * N

    "segment_lengths":{         # Relative statistical segment length compared to "a_Ref.
        "A":np.sqrt(eps*eps/(eps*eps*f1 + (1-f1))), 
        "B":np.sqrt(    1.0/(eps*eps*f1 + (1-f1))), },

    "distinct_polymers":[{      # Distinct Polymers
        "volume_fraction":0.7,  # volume fraction of polymer chain
        "blocks":[              # First AB homopolymer
            {"type":"A", "length":f1, },     # A-block
            {"type":"B", "length":1.0-f1, }, # B-block
        ],},
        {
        "volume_fraction":0.3,
        "blocks":[              # Second AB homopolymer
            {"type":"A", "length":f2/2, },       # A-block
            {"type":"B", "length":(1.0-f2)/2, }, # B-block
        ],},],

    "max_iter":2000,     # The maximum relaxation iterations
    "tolerance":1e-8     # Terminate iteration if the self-consistency error is less than tolerance
}

# Set initial fields
w_A = np.zeros(list(params["nx"]), dtype=np.float64)
w_B = np.zeros(list(params["nx"]), dtype=np.float64)
print("w_A and w_B are initialized to lamellar phase.")
for i in range(0,params["nx"][2]):
    w_A[:,:,i] =  np.cos(3*2*np.pi*i/params["nx"][2])
    w_B[:,:,i] = -np.cos(3*2*np.pi*i/params["nx"][2])

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
        "chi_n":params["chi_n"], "epsilon":eps, "chain_model":params["chain_model"],
        "w_a":w_A, "w_b":w_B, "phi_a":phi_A, "phi_b":phi_B}
savemat("fields.mat", mdic)

# Recording first a few iteration results for debugging and refactoring
    #    1    2.598E-14  [ 6.5597870E+01  6.3186606E+01  ]    -0.006786525   6.4245880E-01  [  4.0000000, 4.0000000, 3.9000000 ]
    #    2    4.885E-14  [ 6.5570420E+01  6.3122915E+01  ]    -0.004620644   4.7824635E-01  [  4.0000000, 4.0000000, 3.8997402 ]
    #    3   -6.550E-15  [ 6.5587711E+01  6.3086441E+01  ]    -0.003399957   3.5707236E-01  [  4.0000000, 4.0000000, 3.8995039 ]
    #    4   -1.267E-13  [ 6.5626011E+01  6.3066208E+01  ]    -0.002719688   2.6820888E-01  [  4.0000000, 4.0000000, 3.8992987 ]
    #    5   -6.328E-15  [ 6.5672788E+01  6.3055746E+01  ]    -0.002346503   2.0332903E-01  [  4.0000000, 4.0000000, 3.8991249 ]
