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
os.environ["LFTS_NUM_GPUS"] = "1" # 1 ~ 2

# Simulation parameters
nx = [32,32,32]               # grid number
lx = [6.0,6.0,6.0]            # box size
ds = 0.01                     # contour step interval
stat_seg_length = {"A":1.0}   # statistical segment lengths

aggregate_propagator_computation = False
reduce_gpu_memory_usage = False

# Grafting points
grafting_point = {0:"G"}  # node 0 will be initialized with q_init["G"]

# Select platform ("cuda" or "cpu-mkl")
factory = PlatformSelector.create_factory("cuda", reduce_gpu_memory_usage)
factory.display_info()

# Create an instance for computation box
cb = factory.create_computation_box(nx, lx) 
# Create an instance for molecule information with block segment information and chain model ("continuous" or "discrete")
molecules = factory.create_molecules_information("continuous", ds, stat_seg_length)

molecules.add_polymer(
     1.0,
     [
     ["A", 1.0, 0, 1],  # first block (type, length, starting node, ending node)
     ],
     grafting_point,
)

# Propagators analyzer for optimal propagator computation
propagator_analyzer = factory.create_propagator_analyzer(molecules, aggregate_propagator_computation)
propagator_analyzer.display_blocks()
propagator_analyzer.display_propagators()

# Create Solver
solver = factory.create_pseudospectral_solver(cb, molecules, propagator_analyzer)

# Fields
w = {"A": np.zeros(np.prod(nx))}
q_init = {"G":np.zeros(np.prod(nx))}
q_init["G"][0] = 1.0/cb.get_dv(0)

# Compute ensemble average concentration (phi) and total partition function (Q)
solver.compute_propagators({"A":w["A"]}, q_init=q_init)
solver.compute_concentrations()

# Distance array
dx = np.array(lx)/np.array(nx)
x = []
for i in range(3):
     x.append(np.min(np.abs(np.stack((np.linspace(0.0, lx[i], num=nx[i], endpoint=False), np.linspace(-lx[i], 0.0, num=nx[i], endpoint=False)))), axis=0))
xv, yv, zv = np.meshgrid(x[0], x[1], x[2], indexing='ij')
distance_square = np.reshape(xv**2+yv**2+zv**2, np.prod(nx))

# Compute <x^2>
N = round(1.0/ds)
for n in range(10, round(N)+1, 10):
                                       # p, v, u, n
     q_out = solver.get_chain_propagator(0, 0, 1, n)
     x_square = np.sum(q_out*distance_square)/np.sum(q_out)
     
     if molecules.get_model_name() == "continuous":
          x_square *= N/n
          print("n, <x^2>N/n:", n, x_square)
     elif molecules.get_model_name() == "discrete":
          x_square *= N/(n-1)
          print("n, <x^2>N/(n-1):", n, x_square)
          
# q_out = solver.get_chain_propagator(0, 0, 1, round(N/2))
# x_square_1 = np.sum(q_out*distance_square)/np.sum(q_out)
# q_out = solver.get_chain_propagator(0, 0, 1, N)
# x_square_2 = np.sum(q_out*distance_square)/np.sum(q_out)
# exponent = np.log(x_square_2/x_square_1)/np.log(2)/2
# print(exponent)