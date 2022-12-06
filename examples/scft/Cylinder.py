import os
import time
import numpy as np
from scipy.io import savemat
from scipy.ndimage.filters import gaussian_filter
import scft

# OpenMP environment variables
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "2"  # 0, 1 or 2

# Major Simulation params
f = 1.0/3.0     # A-fraction of major BCP chain, f
eps = 1.0       # a_A/a_B, conformational asymmetry

params = {
    # "platform":"cuda",           # choose platform among [cuda, cpu-mkl]
    
    "nx":[64,48,48],                        # Simulation grid numbers
    "lx":[6.4,5.52,np.sqrt(3.0/4.0)*5.52],  # Simulation box size as a_Ref * N_Ref^(1/2) unit,
                                            # where "a_Ref" is reference statistical segment length
                                            # and "N_Ref" is the number of segments of reference linear homopolymer chain.

    "box_is_altering":True,     # Find box size that minimizes the free energy during saddle point iteration.
    "chain_model":"discrete", # "discrete" or "continuous" chain model
    "ds":1/90,                  # Contour step interval, which is equal to 1/N_Ref.
    "chi_n": 15,                # Interaction parameter, Flory-Huggins params * N

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
print("w_A and w_B are initialized to cylindrical phase.")
cylinder_positions = [
[0.0,0.0],[0.0,1/3],[0.0,2/3],
[1/2,0.0],[1/2,1/3],[1/2,2/3],
[1/4,1/6],[1/4,3/6],[1/4,5/6],
[3/4,1/6],[3/4,3/6],[3/4,5/6]]
for y,z in cylinder_positions:
    _, my, mz = np.round((np.array([0, y, z])*params["nx"])).astype(np.int32)
    w_A[:,my,mz] = -1/(np.prod(params["lx"])/np.prod(params["nx"]))
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
    #    1    2.707E-13  [ 2.8734784E+02  ]    -0.347066724   1.9030564E+00  [  6.4000000, 5.5200000, 4.7804602 ]
    #    2    6.994E-14  [ 2.3659519E+02  ]    -0.167789774   1.3711348E+00  [  6.4000000, 5.5228865, 4.7847663 ]
    #    3   -1.609E-13  [ 2.1835837E+02  ]    -0.093092899   9.9029875E-01  [  6.4000000, 5.5238851, 4.7865062 ]
    #    4   -1.303E-13  [ 2.1083220E+02  ]    -0.059003091   7.1316836E-01  [  6.4000000, 5.5241884, 4.7872102 ]
    #    5   -1.297E-13  [ 2.0773597E+02  ]    -0.042734208   5.1525212E-01  [  6.4000000, 5.5242091, 4.7874487 ]
    