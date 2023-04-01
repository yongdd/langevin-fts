import os
import time
import numpy as np
from scipy.io import savemat
from scipy.ndimage.filters import gaussian_filter
import scft

# OpenMP environment variables
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"  # 0, 1
os.environ["OMP_NUM_THREADS"] = "2"  # 1 ~ 4

# GPU environment variables
os.environ["LFTS_NUM_GPUS"] = "1" # 1 ~ 2

# Major Simulation params
f = 0.3         # A-fraction of major BCP chain, f
eps = 2.0       # a_A/a_B, conformational asymmetry

params = {
    # "platform":"cuda",           # choose platform among [cuda, cpu-mkl]
    
    "nx":[64,64,64],            # Simulation grid numbers
    "lx":[4.,4.,4.],            # Simulation box size as a_Ref * N_Ref^(1/2) unit,
                                # where "a_Ref" is reference statistical segment length
                                # and "N_Ref" is the number of segments of reference linear homopolymer chain.

    "box_is_altering":True,     # Find box size that minimizes the free energy during saddle point iteration.
    "chain_model":"continuous", # "discrete" or "continuous" chain model
    "ds":1/100,                 # Contour step interval, which is equal to 1/N_Ref.
    "chi_n": 25,                # Interaction parameter, Flory-Huggins params * N_Ref

    "segment_lengths":{         # Relative statistical segment length compared to "a_Ref.
        "A":np.sqrt(eps*eps/(eps*eps*f + (1-f))), 
        "B":np.sqrt(    1.0/(eps*eps*f + (1-f))), },

    "distinct_polymers":[{      # Distinct Polymers
        "volume_fraction":1.0,  # volume fraction of polymer chain
        "blocks":[              # AB diBlock Copolymer
            {"type":"A", "length":f, }, # A-block
            {"type":"B", "length":1-f}, # B-block
        ],},],

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
    #    1    2.220E-14  [ 1.0036657E+00  ]    -0.002335194   2.3607099E+00  [  4.0000000, 4.0000000, 4.0000000 ]
    #    2    1.021E-14  [ 1.0016847E+00  ]    -0.000653175   9.9782765E-01  [  4.0000297, 4.0000297, 4.0000297 ]
    #    3   -1.421E-14  [ 1.0013680E+00  ]    -0.000401154   4.2214365E-01  [  4.0000378, 4.0000378, 4.0000378 ]
    #    4   -4.996E-14  [ 1.0013556E+00  ]    -0.000371403   3.2386650E-01  [  4.0000408, 4.0000408, 4.0000408 ]
    #    5   -1.910E-14  [ 1.0014361E+00  ]    -0.000392140   3.7454571E-01  [  4.0000419, 4.0000419, 4.0000419 ]
