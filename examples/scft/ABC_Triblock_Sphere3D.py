import os
import numpy as np
from scipy.io import savemat, loadmat
from scipy.ndimage import gaussian_filter

# OpenMP environment variables (set before importing polymerfts)
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "2"  # 0, 1 or 2
os.environ["OMP_NUM_THREADS"] = "2"  # 1 ~ 4

from polymerfts import scft

# Major Simulation params
chin = 9.5        # Interaction parameter, Flory-Huggins params * N_Ref

params = {
    # "platform":"cuda",           # choose platform among [cuda, cpu-fftw, cpu-mkl]
    
    "nx":[32,32,32],        # Simulation grid numbers
    "lx":[2.9,2.9,2.9],     # Simulation box size as a_Ref * N_Ref^(1/2) unit,
                            # where "a_Ref" is reference statistical segment length
                            # and "N_Ref" is the number of segments of reference linear homopolymer chain.

    "reduce_memory":False,     # Reduce memory usage by storing only check points.
    "relax_cell":True,
    "stress_interval":1,        # Compute stress every iteration (for reproducibility)
    # Relax cell lengths and allowed angles during SCFT iteration.
    "chain_model":"continuous", # "discrete" or "continuous" chain model
    "ds":1/100,                 # Contour step interval, which is equal to 1/N_Ref. 

    "segment_lengths":{         # Relative statistical segment length compared to "a_Ref.
        "A":1.0,
        "B":1.0,
        "C":1.0},

    "chi_n": {"A,B": chin,      # Interaction parameter, Flory-Huggins params * N_Ref
              "A,C": chin*1.75,
              "B,C": chin,
             },

    "polymers":[{      # Polymer species
        "volume_fraction":1.0,  # volume fraction of polymer chain
        "blocks":[              # ABC triblock Copolymer
            {"type":"A", "length":0.47},       # A-block
            {"type":"B", "length":1.5},        # B-block
            {"type":"C", "length":0.35},       # C-block
        ],},],

    "optimizer":{
        "name":"am",            # Anderson Mixing
        "max_hist":20,          # Maximum number of history
        "start_error":1e-2,     # When switch to AM from simple mixing
        "mix_min":0.02,         # Minimum mixing rate of simple mixing
        "mix_init":0.02,        # Initial mixing rate of simple mixing
    },

    "max_iter":2000,     # The maximum relaxation iterations
    "tolerance":1e-8     # Terminate iteration if the self-consistency error is less than tolerance
}

# Initialize calculation
calculation = scft.SCFT(params=params)

# Set initial fields
w_A = np.zeros(list(params["nx"]), dtype=np.float64)
w_B = np.zeros(list(params["nx"]), dtype=np.float64)
w_C = np.zeros(list(params["nx"]), dtype=np.float64)
print("w_A and w_B are initialized to spherical phase.")
n_unitcell = 1 # number of unit cell for each direction. the number of total unit cells is n_unitcell^3
sphere_positions = []
for i in range(0,n_unitcell):
    for j in range(0,n_unitcell):
        for k in range(0,n_unitcell):
            sphere_positions.append([i/n_unitcell,j/n_unitcell,k/n_unitcell])
            sphere_positions.append([(i+1/2)/n_unitcell,(j+1/2)/n_unitcell,(k+1/2)/n_unitcell])
for x,y,z in sphere_positions:
    molecules, my, mz = np.round((np.array([x, y, z])*params["nx"])).astype(np.int32)
    w_A[molecules,my,mz] = -50/(np.prod(params["lx"])/np.prod(params["nx"]))
w_A = gaussian_filter(w_A, sigma=np.min(params["nx"])/5, mode='wrap')

# Set a timer

# Run
calculation.run(initial_fields={"A": w_A, "B": w_B, "C": w_C})

# Estimate execution time

# Save final results (.mat, .json or .yaml format)
calculation.save_results("fields.mat")

# Recording first a few iteration results for debugging and refactoring
    #        1   -5.216E-16  [ 3.6079092E+00  ]     2.701127013   4.9590932E+00  lx=[  2.9000000, 2.9000000, 2.9000000 ]
    #        2   -1.751E-15  [ 2.9250516E+00  ]     1.712520937   4.1292471E+00  lx=[  2.9006622, 2.9006622, 2.9006622 ]
    #        3   -2.176E-15  [ 2.5640063E+00  ]     1.184086102   3.5391508E+00  lx=[  2.9010649, 2.9010649, 2.9010649 ]
    #        4    1.467E-15  [ 2.3437839E+00  ]     0.860536919   3.0797200E+00  lx=[  2.9013215, 2.9013215, 2.9013215 ]
    #        5    2.519E-15  [ 2.1980075E+00  ]     0.645662224   2.7043781E+00  lx=[  2.9014864, 2.9014864, 2.9014864 ]
