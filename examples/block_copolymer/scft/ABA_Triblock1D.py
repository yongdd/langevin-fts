import os
import numpy as np
from scipy.io import savemat

# OpenMP environment variables (set before importing polymerfts)
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"  # 0, 1
os.environ["OMP_NUM_THREADS"] = "2"  # 1 ~ 4

from polymerfts import scft

# Major Simulation params
f = 0.5         # A-fraction of major BCP chain, f

params = {
    # "platform":"cpu-fftw",           # choose platform among [cuda, cpu-fftw, cpu-mkl]
    
    "nx":[256],         # Simulation grid numbers
    "lx":[1.38],        # Simulation box size as a_Ref * N_Ref^(1/2) unit,
                        # where "a_Ref" is reference statistical segment length
                        # and "N_Ref" is the number of segments of reference linear homopolymer chain.

    "reduce_memory":False,     # Reduce memory usage by storing only check points.
    "relax_cell":True,
    "stress_interval":1,     # Compute stress every iteration (for reproducibility)       # Relax cell lengths and allowed angles during SCFT iteration.
    "chain_model":"continuous",   # "discrete" or "continuous" chain model
    "ds":1/200,                   # Contour step interval, which is equal to 1/N_Ref.
    
    "segment_lengths":{         # Relative statistical segment length compared to "a_Ref.
        "A":1.0, 
        "B":1.0, },

    "chi_n": {"A,B": 9.5},      # Interaction parameter, Flory-Huggins params * N_Ref

    "polymers":[{      # Polymer species
        "volume_fraction":1.0,  # volume fraction of polymer chain
        "blocks":[              # ABA Triblock Copolymer
            {"type":"A", "length":f, },     # A-block
            {"type":"B", "length":2*(1-f)}, # B-block
            {"type":"A", "length":f, },     # A-block
        ],},],

    "optimizer":{
        "name":"am",
        "max_hist":20,           # Maximum number of history
        "start_error":1e-2,      # When switch to AM from simple mixing
        "mix_min":0.1,          # Minimum mixing rate of simple mixing
        "mix_init":0.1,         # Initial mixing rate of simple mixing
    },

    "max_iter":2000,     # The maximum relaxation iterations
    "tolerance":1e-8     # Terminate iteration if the self-consistency error is less than tolerance
}

# Set initial fields
w_A = np.zeros(list(params["nx"]), dtype=np.float64)
w_B = np.zeros(list(params["nx"]), dtype=np.float64)
print("w_A and w_B are initialized to lamellar phase.")
for i in range(0,params["nx"][0]):
    w_A[i] =  np.cos(2*np.pi*i/params["nx"][0])
    w_B[i] = -np.cos(2*np.pi*i/params["nx"][0])

# Initialize calculation
calculation = scft.SCFT(params=params)

# Set a timer

# Run
calculation.run(initial_fields={"A": w_A, "B": w_B})

# Estimate execution time

# Save final results (.mat, .json or .yaml format)
calculation.save_results("fields.mat")

# Recording first a few iteration results for debugging and refactoring
    #        1   -9.381E-17  [ 1.1201978E+00  ]    -0.001141218   2.0171039E-01  lx=[  1.3800000 ]
    #        2   -6.925E-16  [ 1.1207265E+00  ]    -0.002331904   1.5971632E-01  lx=[  1.3803115 ]
    #        3    2.309E-16  [ 1.1216322E+00  ]    -0.002961526   1.2739479E-01  lx=[  1.3805301 ]
    #        4    6.048E-16  [ 1.1227540E+00  ]    -0.003268099   1.0276243E-01  lx=[  1.3806868 ]
    #        5    2.025E-16  [ 1.1239965E+00  ]    -0.003395208   8.4214739E-02  lx=[  1.3808020 ]
