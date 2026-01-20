"""
HCP (Hexagonal Close-Packed) Phase SCFT Simulation - Orthorhombic Supercell

This uses an orthorhombic supercell representation of the HCP structure.
The orthorhombic unit cell contains 4 spheres and has dimensions:
    a = sqrt(3) * b,  c = c_hex

This is equivalent to the hexagonal primitive cell (2 spheres, gamma=120 deg)
and gives identical free energy when properly set up.

Relationship between hexagonal and orthorhombic representations:
    Hexagonal:    a_hex = b_hex,  gamma = 120 deg,  2 spheres/cell
    Orthorhombic: a = sqrt(3)*b,  all angles = 90 deg,  4 spheres/cell

Sphere positions in fractional coordinates [a, b, c]:
    (0, 0, 0), (1/2, 1/2, 0)       - Layer A at z=0
    (3/4, 1/2, 1/2), (1/4, 0, 1/2) - Layer B at z=1/2

Ideal c/a ratio: sqrt(8/3) / sqrt(3) = sqrt(8/9) ~ 0.943
Ideal c/b ratio: sqrt(8/3) ~ 1.633

Results:
- Free energy: F = -0.1345346 (identical to hexagonal representation)
- Equilibrium box: a ~ 2.977, b ~ 1.718, c ~ 2.798
- Ratio a/b ~ 1.733 (close to sqrt(3) ~ 1.732)
- Ratio c/b ~ 1.628 (close to ideal HCP sqrt(8/3) ~ 1.633)
"""

import os
import time
import numpy as np
from scipy.io import savemat
from scipy.ndimage import gaussian_filter
from polymerfts import scft

# OpenMP environment variables
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"  # 0, 1
os.environ["OMP_NUM_THREADS"] = "2"  # 1 ~ 4

# Major Simulation params
f = 0.25       # A-fraction of major BCP chain, f

params = {
    # Orthorhombic HCP: a = sqrt(3)*b, c ~ 1.63*b
    # Grid: nx[0]/nx[1] should be close to sqrt(3) for isotropic resolution
    "nx":[64, 32, 64],          # Simulation grid numbers [a, b, c]
    "lx":[3.0, 1.7, 2.8],       # Initial box size [a, b, c]
                                # Expected equilibrium: a/b ~ sqrt(3), c/b ~ 1.63

    "reduce_memory":False,      # Reduce memory usage by storing only check points.
    "box_is_altering":True,     # Find box size that minimizes the free energy during saddle point iteration.
    "chain_model":"continuous", # "discrete" or "continuous" chain model
    "ds":1/100,                 # Contour step interval, which is equal to 1/N_Ref.

    "segment_lengths":{         # Relative statistical segment length compared to "a_Ref.
        "A":1.0,
        "B":1.0, },

    "chi_n": {"A,B": 20},       # Interaction parameter, Flory-Huggins params * N_Ref

    "distinct_polymers":[{      # Distinct Polymers
        "volume_fraction":1.0,  # volume fraction of polymer chain
        "blocks":[              # AB diBlock Copolymer
            {"type":"A", "length":f, },   # A-block
            {"type":"B", "length":1-f},   # B-block
        ],}],

    "optimizer":{
        "name":"am",            # Anderson Mixing
        "max_hist":20,          # Maximum number of history
        "start_error":1e-2,     # When switch to AM from simple mixing
        "mix_min":0.1,          # Minimum mixing rate of simple mixing
        "mix_init":0.1,         # Initial mixing rate of simple mixing
    },

    "max_iter":2000,            # The maximum relaxation iterations
    "tolerance":1e-8            # Terminate iteration if the self-consistency error is less than tolerance
}

# Set initial fields for HCP structure (orthorhombic supercell)
# The orthorhombic HCP unit cell contains 4 spheres at:
#   Layer A (z=0):   (0, 0, 0) and (1/2, 1/2, 0)
#   Layer B (z=1/2): (3/4, 1/2, 1/2) and (1/4, 0, 1/2)
w_A = np.zeros(list(params["nx"]), dtype=np.float64)
w_B = np.zeros(list(params["nx"]), dtype=np.float64)
print("w_A and w_B are initialized to HCP phase (orthorhombic supercell).")

n_unitcell = 1  # number of unit cells in each direction
sphere_positions = []
for i in range(0, n_unitcell):
    for j in range(0, n_unitcell):
        for k in range(0, n_unitcell):
            # Layer A at z = 0
            sphere_positions.append([i/n_unitcell, j/n_unitcell, k/n_unitcell])
            sphere_positions.append([(i+1/2)/n_unitcell, (j+1/2)/n_unitcell, k/n_unitcell])
            # Layer B at z = 1/2
            sphere_positions.append([(i+3/4)/n_unitcell, (j+1/2)/n_unitcell, (k+1/2)/n_unitcell])
            sphere_positions.append([(i+1/4)/n_unitcell, j/n_unitcell, (k+1/2)/n_unitcell])

for x, y, z in sphere_positions:
    mx, my, mz = np.round((np.array([x, y, z]) * params["nx"])).astype(np.int32)
    mx = mx % params["nx"][0]
    my = my % params["nx"][1]
    mz = mz % params["nx"][2]
    w_A[mx, my, mz] = -1/(np.prod(params["lx"])/np.prod(params["nx"]))

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

# Save final results (.mat, .json or .yaml format)
calculation.save_results("HCP_Orthorhombic.json")

# Recording first a few iteration results for debugging and refactoring
# Equilibrium: F = -0.1345346, lx = [2.977, 1.718, 2.798]
#    1995   -1.952E-15  [ 3.7436442E+00  ]    -0.134534631   1.5855487E-05  lx=[  2.977184, 1.718356, 2.798248 ]
#    1996    1.984E-15  [ 3.7436426E+00  ]    -0.134534631   1.5741783E-05  lx=[  2.977182, 1.718356, 2.798245 ]
#    1997   -3.368E-16  [ 3.7436450E+00  ]    -0.134534631   1.6364910E-05  lx=[  2.977177, 1.718356, 2.798250 ]
#    1998   -9.343E-16  [ 3.7436508E+00  ]    -0.134534631   1.9036314E-05  lx=[  2.977174, 1.718343, 2.798263 ]
#    1999    1.975E-15  [ 3.7437253E+00  ]    -0.134534628   2.0546199E-05  lx=[  2.977335, 1.718379, 2.798334 ]
#    2000   -8.081E-16  [ 3.7437354E+00  ]    -0.134534627   1.9114511E-05  lx=[  2.977367, 1.718396, 2.798333 ]
