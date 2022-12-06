import os
import time
import numpy as np
from scipy.io import savemat
import scft

# OpenMP environment variables
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "2"  # 0, 1 or 2

# Major Simulation params
eps = 3.0              # a_A/a_B, conformational asymmetry
RCP_A_frac = 0.4       # fraction of A monomer in random copolymer

params = {
    # "platform":"cuda",           # choose platform among [cuda, cpu-mkl]
    
    "nx":[32,32,32],          # Simulation grid numbers
    "lx":[4.36,4.36,4.36],    # Simulation box size as a_Ref * N_Ref^(1/2) unit,
                              # where "a_Ref" is reference statistical segment length
                              # and "N_Ref" is the number of segments of reference linear homopolymer chain.

    "box_is_altering":False,    # Find box size that minimizes the free energy during saddle point iteration.
    "chain_model":"continuous", # "discrete" or "continuous" chain model
    "ds":1/50,                  # Contour step interval, which is equal to 1/N_Ref.
    "chi_n": 12,                # Interaction parameter, Flory-Huggins params * N

    "segment_lengths":{         # Relative statistical segment length compared to "a_Ref.
        "A":eps,
        "B":1.0},

    "distinct_polymers":[{      # Distinct Polymers
        "volume_fraction":0.8,  # volume fraction of polymer chain
        "blocks":[              # AB homopolymer
            {"type":"B", "length":1.0, },     # B-block
        ],},
        { 
        "volume_fraction":0.2,  
        "blocks":[              # Random Copolymer. Currently, Only single block random copolymer is supported.
            {"type":"random", "fraction":{"A":RCP_A_frac, "B":1-RCP_A_frac}, "length":0.3, },
        ],}],

    "max_iter":2000,     # The maximum relaxation iterations
    "tolerance":1e-8     # Terminate iteration if the self-consistency error is less than tolerance
}

# Set initial fields
w_A = np.zeros(list(params["nx"]), dtype=np.float64)
w_B = np.zeros(list(params["nx"]), dtype=np.float64)
print("w_A and w_B are initialized to lamellar phase.")
for i in range(0,params["nx"][2]):
    w_A[:,:,i] =  np.cos(2*np.pi*i/params["nx"][2])
    w_B[:,:,i] = -np.cos(2*np.pi*i/params["nx"][2])

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
    #    1    3.042E-14  [ 1.0247609E+02  8.2946705E+01  ]    -0.128625406   4.9746743E+00
    #    2   -8.760E-14  [ 9.8364588E+01  8.2917482E+01  ]    -0.067513288   3.1035349E+00
    #    3    4.707E-14  [ 9.4075112E+01  8.3220304E+01  ]    -0.007115336   1.9646286E+00
    #    4    4.330E-14  [ 9.0120763E+01  8.3738488E+01  ]     0.044558037   1.2333706E+00
    #    5   -3.553E-15  [ 8.6886192E+01  8.4342023E+01  ]     0.081298026   7.4298716E-01
