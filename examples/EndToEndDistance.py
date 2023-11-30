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

# Simulation parameters
nx = [32,32,32]               # grid number
lx = [6.0,6.0,6.0]            # box size
ds = 0.01                     # contour step interval
stat_seg_length = {"A":1.0}   # statistical segment lengths

aggregate_propagator_computation = False
reduce_gpu_memory_usage = False

# Grafting points
grafting_point = {0:"G"}  # vertex 0 will be initialized with q_init["G"]

# Select platform ("cuda" or "cpu-mkl")
factory = PlatformSelector.create_factory("cuda", reduce_gpu_memory_usage)
factory.display_info()

# Create an instance for computation box
cb = factory.create_computation_box(nx, lx) 
# Create an instance for molecule information with block segment information and chain model ("continuous" or "discrete")
molecules = factory.create_molecules_information("continuous", ds, stat_seg_length, aggregate_propagator_computation)

molecules.add_polymer(
     1.0,
     [
     ["A", 1.0, 0, 1],  # first block (type, length, starting node, ending node)
     ],
     grafting_point,
)
solver = factory.create_pseudospectral_solver(cb, molecules, propagators)

molecules.display_blocks()
molecules.display_propagators()

# Fields
w = {"A": np.zeros(np.prod(nx))}
q_init = {"G":np.zeros(np.prod(nx))}
q_init["G"][0] = 1.0/cb.get_dv(0)

# Compute ensemble average concentration (phi) and total partition function (Q)
solver.compute_statistics({"A":w["A"]}, q_init)

N = round(1.0/ds)
for n in range(10, round(N)+1, 10):
                                       # p, v ,u, n
     q_out = solver.get_chain_propagator(0, 0, 1, n)
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
                    
     if molecules.get_model_name() == "continuous":
          x_square *= N/n
          print("n, <x^2>N/n:", n, x_square)
     elif molecules.get_model_name() == "discrete":
          x_square *= N/(n-1)
          print("n, <x^2>N/(n-1):", n, x_square)
