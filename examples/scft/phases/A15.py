import os
import numpy as np
from scipy.io import savemat
from scipy.ndimage import gaussian_filter
from polymerfts import scft

# OpenMP environment variables
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"  # 0, 1
os.environ["OMP_NUM_THREADS"] = "2"  # 1 ~ 4

# Major Simulation params
f = 0.3         # A-fraction of major BCP chain, f
eps = 2.0       # a_A/a_B, conformational asymmetry

params = {
    "nx":[64,64,64],            # Simulation grid numbers
    "lx":[4.,4.,4.],            # Simulation box size as a_Ref * N_Ref^(1/2) unit,
                                # where "a_Ref" is reference statistical segment length
                                # and "N_Ref" is the number of segments of reference linear homopolymer chain.

    "reduce_memory":False,     # Reduce memory usage by storing only check points.
    "relax_cell":True,
    "stress_interval":1,        # Compute stress every iteration (for reproducibility)
    # Relax cell lengths and allowed angles during SCFT iteration.
    "chain_model":"continuous", # "discrete" or "continuous" chain model
    "ds":1/100,                 # Contour step interval, which is equal to 1/N_Ref.

    "segment_lengths":{         # Relative statistical segment length compared to "a_Ref.
        "A":np.sqrt(eps*eps/(eps*eps*f + (1-f))), 
        "B":np.sqrt(    1.0/(eps*eps*f + (1-f))), },

    "chi_n": {"A,B": 25},       # Interaction parameter, Flory-Huggins params * N_Ref

    "polymers":[{      # Polymer species
        "volume_fraction":1.0,  # volume fraction of polymer chain
        "blocks":[              # AB diBlock Copolymer
            {"type":"A", "length":f, }, # A-block
            {"type":"B", "length":1-f}, # B-block
        ],},],

    "space_group" :{
        "symbol":"Pm-3n",   # IT symbol of the space group
        "hall_number": 520,      # (optional) Hall number of the space group
    },

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
print("w_A and w_B are initialized to A15 phase.")
sphere_positions = [[0,0,0],[1/2,1/2,1/2],
[1/4,1/2,0],[3/4,1/2,0],[1/2,0,1/4],[1/2,0,3/4],[0,1/4,1/2],[0,3/4,1/2]]

for x,y,z in sphere_positions:
    molecules, my, mz = np.round((np.array([x, y, z])*params["nx"])).astype(np.int32)
    w_A[molecules,my,mz] = -1/(np.prod(params["lx"])/np.prod(params["nx"]))
w_A = gaussian_filter(w_A, sigma=np.min(params["nx"])/15, mode='wrap')

# Initialize calculation
calculation = scft.SCFT(params=params)

# Set a timer

# Run
calculation.run(initial_fields={"A": w_A, "B": w_B})

# Estimate execution time

# Save final results (.mat, .json or .yaml format)
calculation.save_results("A15.json")

# Recording first a few iteration results for debugging and refactoring
    #        1   -5.979E-16  [ 1.0035906E+00  ]     0.017618718   2.3633736E+00  lx=[  4.0000000, 4.0000000, 4.0000000 ]
    #        2   -3.295E-16  [ 1.0016463E+00  ]     0.003125567   1.0005568E+00  lx=[  4.0000289, 4.0000289, 4.0000289 ]
    #        3    6.876E-16  [ 1.0013348E+00  ]     0.000542568   4.2158641E-01  lx=[  4.0000367, 4.0000367, 4.0000367 ]
    #        4    6.916E-16  [ 1.0013226E+00  ]    -0.000200467   3.2209173E-01  lx=[  4.0000396, 4.0000396, 4.0000396 ]
    #        5    1.975E-15  [ 1.0014013E+00  ]    -0.000539175   3.7324500E-01  lx=[  4.0000405, 4.0000405, 4.0000405 ]
