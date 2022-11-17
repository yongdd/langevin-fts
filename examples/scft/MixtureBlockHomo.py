import os
import time
import numpy as np
from scipy.io import savemat, loadmat
import scft

# # Major Simulation params
f = 0.4         # A-fraction of major BCP chain, f
eps = 2.0       # a_A/a_B, conformational asymmetry

params = {
    # "platform":"cuda",           # choose platform among [cuda, cpu-mkl]
    
    "nx":[32,32,32],          # Simulation grid numbers
    "lx":[4.,4.,6.5],         # Simulation box size as a_Ref * N_Ref^(1/2) unit,
                              # where "a_Ref" is reference statistical segment length
                              # and "N_Ref" is the number of segments of reference linear homopolymer chain.

    "box_is_altering":True,      # Find box size that minimizes the free energy during saddle point iteration.
    "chain_model":"continuous",  # "discrete" or "continuous" chain model
    "ds":1/90,                   # Contour step interval, which is equal to 1/N_Ref.
    "chi_n": 20,                 # Interaction parameter, Flory-Huggins params * N

    "segment_lengths":{         # Relative statistical segment length compared to "a_Ref.
        "A":np.sqrt(eps*eps/(eps*eps*f + (1-f))), 
        "B":np.sqrt(    1.0/(eps*eps*f + (1-f))), },

    "distinct_polymers":[{      # Distinct Polymers
        "volume_fraction":0.7,  # volume fraction of polymer chain
        "blocks":[              # AB homopolymer
            {"type":"A", "length":f, },     # A-block
            {"type":"B", "length":1.0-f, }, # B-block
        ],},
        {
        "volume_fraction":0.3,  
        "blocks":[              # A homopolymer
            {"type":"A", "length":0.5, }, # A-block
        ],}],

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
    #    1    7.616E-14  [ 1.0890682E+02  1.0848629E+02  ]    -0.032610986   1.3435247E+00  [  4.0000000, 4.0000000, 6.5000000 ]
    #    2    3.064E-14  [ 1.0986520E+02  1.1013029E+02  ]    -0.040863058   1.2806327E+00  [  4.0000000, 4.0000000, 6.4998393 ]
    #    3   -2.198E-14  [ 1.1132951E+02  1.1203817E+02  ]    -0.051682250   1.2555736E+00  [  4.0000000, 4.0000000, 6.4996526 ]
    #    4    3.775E-15  [ 1.1332183E+02  1.1438356E+02  ]    -0.065402931   1.2381097E+00  [  4.0000000, 4.0000000, 6.4994231 ]
    #    5    2.021E-14  [ 1.1594955E+02  1.1734267E+02  ]    -0.082641565   1.2207020E+00  [  4.0000000, 4.0000000, 6.4991324 ]
