import os
import time
import numpy as np
from scipy.io import savemat, loadmat
from scipy.ndimage import gaussian_filter
import lfts

# OpenMP environment variables
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "2"  # 0, 1 or 2
os.environ["OMP_NUM_THREADS"] = "2"  # 1 ~ 4

# Major Simulation params
chin = 9.5        # Interaction parameter, Flory-Huggins params * N_Ref

params = {
     "platform":"cuda",           # choose platform among [cuda, cpu-mkl]
    
    "nx":[32,32,32],        # Simulation grid numbers
    "lx":[2.9,2.9,2.9],     # Simulation box size as a_Ref * N_Ref^(1/2) unit,
                            # where "a_Ref" is reference statistical segment length
                            # and "N_Ref" is the number of segments of reference linear homopolymer chain.

    "chain_model":"continuous", # "discrete" or "continuous" chain model
    "ds":1/100,                 # Contour step interval, which is equal to 1/N_Ref. 

    "segment_lengths":{         # Relative statistical segment length compared to "a_Ref.
        "A":1.0,
        "B":1.0,
        "C":1.0},

    "chi_n": {"A,B":chin,     # Interaction parameter, Flory-Huggins params * N_Ref
              "A,C":chin*1.75,
              "B,C":chin},

    "distinct_polymers":[{      # Distinct Polymers
        "volume_fraction":1.0,  # volume fraction of polymer chain
        "blocks":[              # ABC triblock Copolymer
            {"type":"A", "length":0.47},       # A-block
            {"type":"B", "length":1.5},        # B-block
            {"type":"C", "length":0.35},       # C-block
        ],},],

    "langevin":{             # Langevin Dynamics
        "max_step":200000,    # Langevin steps for simulation
        "dt":8.0,            # Langevin step interval, delta tau*N_Ref
        "nbar":1.0e6,        # Invariant polymerization index, nbar of N_Ref
    },
    
    "recording":{                       # Recording Simulation Data
        "dir":"data_simulation",        # Directory name
        "recording_period":200,        # Period for recording concentrations and fields
        "sf_computing_period":10,       # Period for computing structure function
        "sf_recording_period":10000,    # Period for recording structure function
    },

    "saddle":{                # Iteration for the pressure field 
        "max_iter" :400,      # Maximum number of iterations
        "tolerance":1e-4,     # Tolerance of incompressibility 
    },

    "optimizer":{
        # "name":"am",                # Anderson Mixing
        # "name":"lr",                # Linear Response
        "name":"lram",              # Linear Response + Anderson Mixing
        "max_hist":20,              # Maximum number of history
        "start_error":5e-1,         # When switch to AM from simple mixing
        "mix_min":0.01,             # Minimum mixing rate of simple mixing
        "mix_init":0.01,            # Initial mixing rate of simple mixing
    },

    "verbose_level":1,      # 1 : Print at each Langevin step.
                            # 2 : Print at each saddle point iteration.
}

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

# Initialize calculation
simulation = lfts.LFTS(params=params, random_seed=12345)

# Set a timer
time_start = time.time()

# Run
simulation.run(initial_fields={"A": w_A, "B": w_B, "C": w_C})

# Estimate execution time
time_duration = time.time() - time_start
print("total time: %f, time per step: %f" %
    (time_duration, time_duration/params["langevin"]["max_step"]) )

# Recording first a few iteration results for debugging and refactoring
# (Anderson Mixing)
# ---------- Run  ----------
# iterations, mass error, total partitions, Hamiltonian, incompressibility error (or saddle point error)
#       53   -1.643E-16  [ 1.1878907E+01  ]     5.662175232   [9.4787668E-05 ]
# Langevin step:  1
#       47   -1.789E-15  [ 1.5278634E+03  ]     4.185883622   [9.5230592E-05 ]
# Langevin step:  2
#       48    9.779E-16  [ 4.9855374E+01  ]     4.051992910   [9.0729112E-05 ]
# Langevin step:  3
#       50   -1.905E-15  [ 2.3041191E+02  ]     4.013633667   [9.1691101E-05 ]
# Langevin step:  4
#       51    4.538E-16  [ 1.2099087E+02  ]     4.007848973   [9.6501273E-05 ]
# Langevin step:  5
#       51   -1.995E-15  [ 1.6642863E+02  ]     4.000168691   [9.1616075E-05 ]

# (LRAM)
# ---------- Run  ----------
# iterations, mass error, total partitions, Hamiltonian, incompressibility error (or saddle point error)
#        9   -1.973E-15  [ 1.1878232E+01  ]     5.662175237   [3.2262806E-05 ]
# Langevin step:  1
#        7   -1.217E-15  [ 1.5282007E+03  ]     4.185881487   [5.7770861E-05 ]
# Langevin step:  2
#        6   -7.928E-16  [ 4.9863796E+01  ]     4.051998339   [3.0608193E-05 ]
# Langevin step:  3
#        6    1.550E-15  [ 2.3044711E+02  ]     4.013610407   [3.9477143E-05 ]
# Langevin step:  4
#        6   -3.738E-16  [ 1.2098394E+02  ]     4.007838878   [3.6944707E-05 ]
# Langevin step:  5
#        5    1.688E-16  [ 1.6639899E+02  ]     4.000147580   [7.9780842E-05 ]