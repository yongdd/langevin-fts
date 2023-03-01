# This example script computes end-to-end distance of polymer chain

import os
import numpy as np
from langevinfts import *

# OpenMP environment variables
os.environ["MKL_NUM_THREADS"] = "1"  # always 1
os.environ["OMP_STACKSIZE"] = "1G"
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"  # 0, 1
os.environ["OMP_NUM_THREADS"] = "2"  # 1 ~ 4

# GPU environment variables
os.environ["LFTS_GPU_NUM_BLOCKS"]  = "256"
os.environ["LFTS_GPU_NUM_THREADS"] = "256"

# simulation parameters
nx = [32,32,32]                                   # grid number
lx = [4.0,4.0,4.0]                                # box size
ds = 0.01                                         # contour step interval
stat_seg_length = {"A":1.0, "B":1.0}              # statistical segment lengths
f = 0.5                                           # A-fraction of major BCP chain, f

use_superposition = False
reduce_gpu_memory_usage = False

# polymer
volume_faction = 1.0                               # volume faction
block_lengths  = [f, f, f, 1-f, 1-f, 1-f]          # contour length of each block (Star3Arm)
block_monomer_types = ["A","A","A","B","B","B"]    # type of each block (Star3Arm)
v = [0,0,0,1,2,3]                                  # vertices v (Star3Arm)
u = [1,2,3,4,5,6]                                  # vertices u (Star3Arm)

# grafting points
grafting_point = {4:"G",5:"G"}  # vertices 4 and 5 will be initialized with q_init["G"]

# select platform and chain model  ("cuda" or "cpu-mkl"), ("continuous" or "discrete")
factory = PlatformSelector.create_factory("cuda", "continuous")
factory.display_info()

# create instances
cb = factory.create_computation_box(nx, lx)
mixture = factory.create_mixture(ds, stat_seg_length, use_superposition)

mixture.add_polymer(volume_faction,block_monomer_types, block_lengths,v, u, grafting_point)
pseudo = factory.create_pseudo(cb, mixture, reduce_gpu_memory_usage)

mixture.display_unique_blocks()
mixture.display_unique_branches()

# fields
w = {"A": np.zeros(np.prod(nx)),
     "B": np.zeros(np.prod(nx))}

q_init = {"G":np.zeros(np.prod(nx))}
q_init["G"][0] = 1.0/cb.get_dv(0)

# compute ensemble average concentration (phi) and total partition function (Q)
pseudo.compute_statistics({"A":w["A"],"B":w["B"]}, q_init)

N = round(1.0/ds)
for n in range(10, round(f*N)+1, 10):
                                # output, p, v ,u, n
     q_out = pseudo.get_partial_partition(0, 4, 1, n)
     sum = 0.0
     x_square = 0.0
     for i in range(nx[0]):
          xx = cb.get_dx(0)*min([i, nx[0]-i])
          for j in range(nx[1]):
               yy = cb.get_dx(1)*min([j, nx[1]-j])
               for k in range(nx[2]):
                    zz = cb.get_dx(2)*min([k, nx[2]-k])
                    idx = i*nx[1]*nx[2] + j*nx[2] + k
                    x_square += q_out[idx]*cb.get_dv(idx)*(xx*xx + yy*yy + zz*zz)
                    sum += q_out[idx]*cb.get_dv(idx)
                    
     if mixture.get_model_name() == "continuous":
          x_square *= N/n
          print("n, <x^2>N/n:", n, x_square)
     elif mixture.get_model_name() == "discrete":
          x_square *= N/(n-1)
          print("n, <x^2>N/(n-1):", n, x_square)
