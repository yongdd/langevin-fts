# This example script computes propagators and concentrations of polymer chains of a brush absence of external field.

import os
import numpy as np
import matplotlib.pyplot as plt
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
nx = [256]                  # grid number
lx = [10.0]                 # box size
ds = 0.01                   # contour step interval
stat_seg_length = {"A":1.0} # statistical segment lengths

# Target density profile for film
T = 1.0
t = 0.4
T_mask = 1.0
L = lx[0] - 2*T_mask
dx = lx[0]/nx[0]
I_range = round(L/dx)-3
offset = round(T_mask/dx)+1
offset_grafting = np.max([round((T_mask+0.05)/dx), round((T_mask)/dx)+1])

# Set a mask to set q(r,s) = 0
mask = np.zeros(nx)
for i in range(I_range):
     mask[i+offset] = 1.0
     mask[nx[0]-offset-i-1] = 1.0

# Initial condition of q (grafting point)
q_init = {"G":np.zeros(list(nx), dtype=np.float64)}
q_init["G"][offset_grafting] = 1.0/dx
q_init["G"][nx[0]-offset_grafting-1] = 1.0/dx

print(mask[:])
print(mask[20:40])
print(q_init["G"][20:40])

aggregate_propagator_computation = False
reduce_gpu_memory_usage = False

# Select platform ("cuda" or "cpu-mkl")
factory = PlatformSelector.create_factory("cuda", reduce_gpu_memory_usage)
factory.display_info()

# Create an instance for computation box
cb = factory.create_computation_box(nx, lx, mask=mask)

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

# Propagators analyzer for optimal propagator computation
propagators_analyzer = factory.create_propagators_analyzer(molecules, aggregate_propagator_computation)
propagators_analyzer.display_blocks()
propagators_analyzer.display_propagators()

# Create Solver
solver = factory.create_pseudospectral_solver(cb, molecules, propagators_analyzer)

# Fields
w = {"A": np.zeros(nx)}

# Compute ensemble average concentration (phi) and total partition function (Q)
solver.compute_statistics({"A":w["A"]}, q_init=q_init)

x = np.linspace(-T_mask, lx[0]-T_mask, num=nx[0], endpoint=False)

phi = np.reshape(solver.get_total_concentration("A"), nx)
file_name = "phi"
plt.plot(x, phi)
plt.xlim([-0.01, 3])
plt.savefig(file_name)
print("phi(r) is written to file '%s'." % (file_name))
plt.close()

N = round(1.0/ds)
for n in range(0, round(N)+1, 20):
     file_name = "q_forward_%03d.png" % (n)
                                                  # p, v, u, n
     q_out = np.reshape(solver.get_chain_propagator(0, 0, 1, n), nx)
     plt.plot(x, q_out)
     plt.savefig(file_name)
     print("q(%3.1f,r) is written to file '%s'." % (n*ds, file_name))
     plt.close()
     
     file_name = "q_backward_%03d.png" % (n)
                                                  # p, v, u, n
     q_out = np.reshape(solver.get_chain_propagator(0, 1, 0, n), nx)

     plt.plot(x, q_out)
     plt.savefig(file_name)
     print("q^†(%3.1f,r) is written to file '%s'." % (n*ds, file_name))
     plt.close()