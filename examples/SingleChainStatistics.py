import os
import numpy as np
from langevinfts import *

# OpenMP environment variables
os.environ["MKL_NUM_THREADS"] = "1"  # always 1
os.environ["OMP_STACKSIZE"] = "1G"
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "2"  # 0, 1 or 2

# GPU environment variables
os.environ["LFTS_GPU_NUM_BLOCKS"]  = "256"
os.environ["LFTS_GPU_NUM_THREADS"] = "256"

# simulation parameters
nx = [64,64,64]                      # grid number
lx = [5.0, 5.0, 5.0]                 # box size
ds = 0.01                            # contour step Interval
stat_seg_length = {"A":1.0, "B":2.0} # statistical segment lengths
block_length = [0.5, 0.7, 0.3]       # contour length of each block (triblock)
block_type = ["A","B","A"]           # type of each block (triblock)

# select platform and chain model  ("cuda" or "cpu-mkl"), ("continuous" or "discrete")
factory = PlatformSelector.create_factory("cuda", "continuous")
factory.display_info()

# create instances
pc     = factory.create_polymer_chain(block_type, block_length, stat_seg_length, ds)
cb     = factory.create_computation_box(nx, lx)
pseudo = factory.create_pseudo(cb, pc)

print(type(pseudo))

# fields
w = {"A": np.random.normal(0.0, 1.0, np.prod(nx)),
     "B": np.random.normal(0.0, 1.0, np.prod(nx))}

# free end initial condition. q1 is q and q2 is qdagger.
# q1 starts from A end and q2 starts from B end.
q1_init = np.ones (np.prod(nx), dtype=np.float64)
q2_init = np.ones (np.prod(nx), dtype=np.float64)

# compute ensemble average concentration (phi) and total partition function (Q)
phi, Q = pseudo.compute_statistics(q1_init,q2_init,{"A":w["A"],"B":w["B"]})

# print
print(phi[0],phi[1],phi[2]) # concentration for each block
print(Q)