import os
import time
import numpy as np
from scipy.io import savemat, loadmat
from scipy.ndimage.filters import gaussian_filter
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
        "max_step":100,      # Langevin steps for simulation
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
        "max_iter" :100,      # Maximum number of iterations
        "tolerance":1e-4,     # Tolerance of incompressibility 
    },

    "am":{
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
    mx, my, mz = np.round((np.array([x, y, z])*params["nx"])).astype(np.int32)
    w_A[mx,my,mz] = -50/(np.prod(params["lx"])/np.prod(params["nx"]))
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

#       53   -1.599E-15  [ 1.1878907E+01  ]     5.662175232   9.4787433E-05 
# iteration, mass error, total partitions, total energy, incompressibility error
# ---------- Run  ----------
# Langevin step:  1
#       47   -2.249E-15  [ 4.9487578E+01  ]     4.788964055   [9.7535703E-05 ]
# Langevin step:  2
#       52   -2.557E-15  [ 1.6939952E+02  ]     4.512100450   [9.8681906E-05 ]
# Langevin step:  3
#       52   -1.446E-15  [ 4.5383235E+02  ]     4.344580196   [9.3440109E-05 ]
# Langevin step:  4
#       51    1.546E-15  [ 9.9284083E+02  ]     4.232714750   [9.6428622E-05 ]
# Langevin step:  5
#       50    1.958E-15  [ 1.9243007E+03  ]     4.149071041   [9.8916066E-05 ]