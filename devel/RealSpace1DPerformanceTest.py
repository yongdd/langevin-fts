# This example script computes propagators and concentrations of polymer chains with a nano particle at center
# Propagators inside of the nano particle is zero, e.g., q(r,s) = 0.

import os
import time
import numpy as np
import matplotlib.pyplot as plt

from polymerfts import *

# OpenMP environment variables
os.environ["MKL_NUM_THREADS"] = "1"  # always 1
os.environ["OMP_STACKSIZE"] = "1G"
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"  # 0, 1
os.environ["OMP_NUM_THREADS"] = "2"  # 1 ~ 4

# GPU environment variables
os.environ["LFTS_GPU_NUM_BLOCKS"]  = "256"
os.environ["LFTS_GPU_NUM_THREADS"] = "256"
os.environ["LFTS_NUM_GPUS"] = "1" # 1 ~ 2

# Simulation parameters
nx = [128000]                 # grid number
lx = [6.0]                    # box size
ds = 0.01                     # contour step interval
stat_seg_length = {"A":1.0}   # statistical segment lengths

# boundary_conditions = ["periodic", "periodic"]
boundary_conditions = ["reflecting", "reflecting"]
# boundary_conditions = ["absorbing", "absorbing"]

aggregate_propagator_computation = False
reduce_memory_usage = False

# Select platform ("cuda" or "cpu-mkl")
factory = PlatformSelector.create_factory("cpu-mkl", reduce_memory_usage)
factory.display_info()

# Create an instance for computation box
cb = factory.create_computation_box(nx, lx, bc=boundary_conditions)

# Create an instance for molecule information with block segment information and chain model ("continuous" or "discrete")
molecules = factory.create_molecules_information("continuous", ds, stat_seg_length)

# Add polymer
molecules.add_polymer(
     1.0,
     [
     ["A", 1.0, 0, 1],  # first block (type, length, starting node, ending node)
     ],
     {0:"G"}
)

# Optimizer to avoid redundant computations
propagator_computation_optimizer = factory.create_propagator_computation_optimizer(molecules, aggregate_propagator_computation)
propagator_computation_optimizer.display_blocks()
propagator_computation_optimizer.display_propagators()

# Create Solver
solver = factory.create_propagator_computation(cb, molecules, propagator_computation_optimizer, "cn-adi4")

# Fields
w = {"A": np.zeros(nx)}
q_init = {"G":np.zeros(nx)}
q_init["G"][0] = 1.0/(lx[0]/nx[0])

# Compute ensemble average concentration (phi) and total partition function (Q)
time_start = time.time()
for i in range(10):
     print(i)
     solver.compute_statistics({"A":w["A"]}, q_init=q_init)
     solver.check_total_partition()
elapsed_time = time.time() - time_start
print("Elapsed time: ", elapsed_time)

phi = np.reshape(solver.get_total_concentration("A"), nx)
print(np.std(phi))