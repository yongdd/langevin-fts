import os
import time
import numpy as np
from scipy.io import savemat, loadmat
from scipy.ndimage import gaussian_filter
from polymerfts import scft

# OpenMP environment variables
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"  # 0, 1
os.environ["OMP_NUM_THREADS"] = "2"  # 1 ~ 4

# Major Simulation params
f = 0.43        # A-fraction of major BCP chain, f

params = {
    # "platform":"cuda",           # choose platform among [cuda, cpu-mkl]
    
    "nx":[24,48,84],            # Simulation grid numbers
    "lx":[1.59,3.17,5.58],      # Simulation box size as a_Ref * N_Ref^(1/2) unit,
                                # where "a_Ref" is reference statistical segment length
                                # and "N_Ref" is the number of segments of reference linear homopolymer chain.

    "reduce_gpu_memory_usage":False, # Reduce gpu memory usage by storing propagators in main memory instead of gpu memory.
    "box_is_altering":True,     # Find box size that minimizes the free energy during saddle point iteration.
    "chain_model":"continuous", # "discrete" or "continuous" chain model
    "ds":1/100,                  # Contour step interval, which is equal to 1/N_Ref.

    "segment_lengths":{         # Relative statistical segment length compared to "a_Ref.
        "A":1.0, 
        "B":1.0, },

    "chi_n": {"A,B": 12.0},     # Interaction parameter, Flory-Huggins params * N_Ref

    "distinct_polymers":[{      # Distinct Polymers
        "volume_fraction":1.0,  # volume fraction of polymer chain
        "blocks":[              # AB diBlock Copolymer
            {"type":"A", "length":f, }, # A-block
            {"type":"B", "length":1-f}, # B-block
        ],},],

    "optimizer":{       
        # "name":"adam",     # ADAM optimizer
        # "lr":1e-1,         # initial learning rate,
        # "gamma":0.9993,    # learning rate at Tth iteration is lr*γ^(T-1)
        
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
print("w_A and w_B are initialized to Fddd phase.")
x = np.arange(params["nx"][0])*2*np.pi/params["nx"][0]
y = np.arange(params["nx"][1])*2*np.pi/params["nx"][1]
z = np.arange(params["nx"][2])*2*np.pi/params["nx"][2]
xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

w_A = np.zeros(params["nx"], dtype=np.complex128)
w_A += ( 0.02+0.39j)*np.exp(             4j*zz)
w_A += ( 0.22+0.36j)*np.exp(       2j*yy+2j*zz)
w_A += ( 0.37+0.20j)*np.exp(      -2j*yy+2j*zz)
w_A += (-0.81-0.32j)*np.exp( 1j*xx+1j*yy+1j*zz)
w_A += (-0.09+0.86j)*np.exp( 1j*xx-1j*yy+1j*zz)
w_A += ( 0.69+0.52j)*np.exp(-1j*xx+1j*yy+1j*zz)
w_A += (-0.37+0.80j)*np.exp(-1j*xx-1j*yy+1j*zz)
w_A = (w_A + np.conjugate(w_A)).real

w_B = -w_A

# Initialize calculation
calculation = scft.SCFT(params=params)

# Set a timer
time_start = time.time()

# Run
calculation.run(initial_fields={"A": w_A, "B": w_B})

# Estimate execution time
time_duration = time.time() - time_start
print("total time: %f " % time_duration)

# Save final results (.mat, .json or .yaml format)
calculation.save_results("fields.mat")

# Recording first a few iteration results for debugging and refactoring
#     1   -1.310E-15  [ 1.9020053E+00  ]    -0.053408744   3.0154096E-01  [  1.5900000, 3.1700000, 5.5800000 ]
#     2   -4.743E-13  [ 1.8986511E+00  ]    -0.045996110   2.5355729E-01  [  1.5886964, 3.1698418, 5.5799942 ]
#     3    4.679E-13  [ 1.8971220E+00  ]    -0.040438763   2.1459982E-01  [  1.5873977, 3.1696502, 5.5799545 ]
#     4    4.350E-13  [ 1.8964810E+00  ]    -0.036232923   1.8280298E-01  [  1.5861457, 3.1694431, 5.5798945 ]
#     5    6.299E-13  [ 1.8961600E+00  ]    -0.033036051   1.5680863E-01  [  1.5849674, 3.1692325, 5.5798231 ]
