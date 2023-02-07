import os
import time
import numpy as np
from scipy.io import savemat
import scft

# OpenMP environment variables
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "2"  # 0, 1 or 2
os.environ["OMP_NUM_THREADS"] = "2"  # 1 ~ 4

# Major Simulation params
f = 0.5          # A-fraction of major BCP chain, f
eps = 1.0        # a_A/a_B, conformational asymmetry
n_sc = 50        # the number of side chains
sc_alpha = 0.3   # N_sc/ N_bb
chi_n = 36       # Interaction parameter, Flory-Huggins params * N_total

def create_bottle_brush(sc_alpha, n_sc, f):

    d = 1/n_sc
    N_BB_A = round(n_sc*f)

    assert(np.isclose(N_BB_A - n_sc*f, 0)), \
        "'n_sc*f' is not an integer."

    blocks = []
    # backbone (A region)
    blocks.append({"type":"A", "length":d/2, "v":0, "u":1})
    for i in range(1,N_BB_A):
        blocks.append({"type":"A", "length":d, "v":i, "u":i+1})

    # backbone (AB junction)
    blocks.append({"type":"A", "length":d/2,   "v":N_BB_A,   "u":N_BB_A+1}) 
    blocks.append({"type":"B", "length":d/2,   "v":N_BB_A+1, "u":N_BB_A+2})

    # backbone (B region)
    for i in range(N_BB_A+1,n_sc):
        blocks.append({"type":"B", "length":d,   "v":i+1, "u":i+2})
    blocks.append({"type":"B", "length":d/2,   "v":n_sc+1, "u":n_sc+2})

    # side chains (A)
    blocks.append({"type":"A", "length":sc_alpha, "v":1, "u":n_sc+3})
    for i in range(1,N_BB_A):
        blocks.append({"type":"A", "length":sc_alpha, "v":i+1, "u":i+n_sc+3})

    # side chains (B)
    for i in range(N_BB_A+1,n_sc):
        blocks.append({"type":"B", "length":sc_alpha, "v":i+1, "u":i+n_sc+2})
    blocks.append({"type":"B", "length":sc_alpha, "v":n_sc+1, "u":2*n_sc+2})
    
    return blocks

blocks = create_bottle_brush(sc_alpha, n_sc, f)
total_alpha = 1 + n_sc*sc_alpha

print("Blocks:", *blocks, sep = "\n")
print(total_alpha)

params = {
    # "platform":"cpu-mkl",           # choose platform among [cuda, cpu-mkl]
    
    "nx":[64,64,64],      # Simulation grid numbers
    "lx":[7.8,7.8,7.8],   # Simulation box size as a_Ref * N_Ref^(1/2) unit,
                          # where "a_Ref" is reference statistical segment length
                          # and "N_Ref" is the number of segments of reference linear homopolymer chain.

    "use_superposition":True,   # Superpose multiple partial partition functions when solving diffusion equations for speedup using superposition principle. 
                                # To obtain concentraions of each block, disable this option.

    "box_is_altering":True,       # Find box size that minimizes the free energy during saddle point iteration.
    "chain_model":"continuous",   # "discrete" or "continuous" chain model
    "ds":1/100,                   # Contour step interval, which is equal to 1/N_Ref.
    "chi_n": chi_n/total_alpha,   # Interaction parameter, Flory-Huggins params * N

    "segment_lengths":{         # Relative statistical segment length compared to "a_Ref.
        "A":np.sqrt(eps*eps/(eps*eps*f + (1-f))), 
        "B":np.sqrt(    1.0/(eps*eps*f + (1-f))), },

    "distinct_polymers":[{      # Distinct Polymers
        "volume_fraction":1.0,  # volume fraction of polymer chain
        "blocks": blocks,
        },],

    "max_iter":2000,      # The maximum relaxation iterations
    "tolerance":1e-8,     # Terminate iteration if the self-consistency error is less than tolerance

    "am":{
        "max_hist":20,           # Maximum number of history
        "start_error":1e-2,      # When switch to AM from simple mixing
        "mix_min":0.02,          # Minimum mixing rate of simple mixing
        "mix_init":0.02,         # Initial mixing rate of simple mixing
    },
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
        "f":f, "chi_n":params["chi_n"], "epsilon":eps, "chain_model":params["chain_model"],
        "w_a":w_A, "w_b":w_B, "phi_a":phi_A, "phi_b":phi_B}
savemat("fields.mat", mdic)

# Recording first a few iteration results for debugging and refactoring (continuous)
    #    1    1.761E-13  [ 1.1333811E+05  ]    -0.120012748   3.9037374E-01  [  7.8000000, 7.8000000, 7.8000000 ]
    #    2   -4.377E-02  [ 1.1899300E+05  ]    -0.120350302   3.3469859E-01  [  7.8000000, 7.8000000, 7.7997507 ]
    #    3   -7.634E-02  [ 1.2357567E+05  ]    -0.120535715   2.9546697E-01  [  7.8000000, 7.8000000, 7.7994852 ]
    #    4   -1.010E-01  [ 1.2728031E+05  ]    -0.120609303   2.6752115E-01  [  7.8000000, 7.8000000, 7.7992074 ]
    #    5   -1.198E-01  [ 1.3026647E+05  ]    -0.120599252   2.4742843E-01  [  7.8000000, 7.8000000, 7.7989198 ]
    #    6   -1.344E-01  [ 1.3266556E+05  ]    -0.120525912   2.3283207E-01  [  7.8000000, 7.8000000, 7.7986247 ]

# Recording first a few iteration results for debugging and refactoring (discrete)
    #    1   -1.107E-12  [ 1.1334786E+05  ]    -0.120018124   3.9039788E-01  [  7.8000000, 7.8000000, 7.8000000 ]
    #    2   -3.911E-13  [ 1.1900375E+05  ]    -0.120355746   3.8302551E-01  [  7.8000000, 7.8000000, 7.7997507 ]
    #    3    4.359E-13  [ 1.2488664E+05  ]    -0.120672341   3.7572307E-01  [  7.8000000, 7.8000000, 7.7995011 ]
    #    4   -8.771E-15  [ 1.3100023E+05  ]    -0.120968149   3.6849156E-01  [  7.8000000, 7.8000000, 7.7992511 ]
    #    5   -1.128E-13  [ 1.3734782E+05  ]    -0.121243447   3.6133186E-01  [  7.8000000, 7.8000000, 7.7990009 ]
    #    6    4.219E-14  [ 1.4393228E+05  ]    -0.121498540   3.5424473E-01  [  7.8000000, 7.8000000, 7.7987505 ]