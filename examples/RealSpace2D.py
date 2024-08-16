# This example script computes propagators and concentrations of polymer chains with a nano particle at center
# Propagators inside of the nano particle is zero, e.g., q(r,s) = 0.

import os
import time
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
nx = [120,100]                # grid number
lx = [4.0,5.0]                # box size
ds = 0.01                     # contour step interval
stat_seg_length = {"A":1.0}   # statistical segment lengths

boundary_conditions = ["periodic", "periodic",
                       "absorbing", "reflecting"]

# Set a mask to set q(r,s) = 0
x = np.linspace(-lx[0]/2, lx[0]/2, num=nx[0], endpoint=False)
y = np.linspace(-lx[1]/2, lx[1]/2, num=nx[1], endpoint=False)

xv, yv = np.meshgrid(x, y, indexing='ij')
nano_particle_radius = 1.0
mask = np.ones(nx)
mask[np.sqrt(xv**2 + yv**2) < nano_particle_radius] = 0.0
print(1.0-np.mean(mask), (np.pi*nano_particle_radius**2)/np.prod(lx))

aggregate_propagator_computation = False
reduce_gpu_memory_usage = False

# Select platform ("cuda" or "cpu-mkl")
factory = PlatformSelector.create_factory("cuda", reduce_gpu_memory_usage)
factory.display_info()

# Create an instance for computation box
cb = factory.create_computation_box(nx, lx, bc=boundary_conditions, mask=mask)

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
propagator_analyzer = factory.create_propagator_analyzer(molecules, aggregate_propagator_computation)
propagator_analyzer.display_blocks()
propagator_analyzer.display_propagators()

# Create Solver
solver = factory.create_realspace_solver(cb, molecules, propagator_analyzer)

# Fields
w = {"A": np.zeros(nx)}
q_init = {"G":np.zeros(nx)}
q_init["G"][30,:-30] = 1.0/(lx[0]/nx[0])
q_init["G"][:-30,30] = 1.0/(lx[1]/nx[1])

# Compute ensemble average concentration (phi) and total partition function (Q)
time_start = time.time()
solver.compute_propagators({"A":w["A"]}, q_init=q_init)
solver.compute_concentrations()
elapsed_time = time.time() - time_start
print("Elapsed time: ", elapsed_time)

phi = np.reshape(solver.get_total_concentration("A"), nx)
file_name = "phi"
plt.imshow(phi, extent=[0, lx[0], 0, lx[1]], aspect='auto', interpolation='nearest')
plt.colorbar()
plt.savefig(file_name)
print("phi(r) is written to file '%s'." % (file_name))
plt.close()

N = round(1.0/ds)
for n in range(0, round(N)+1, round(N/5)):
     file_name = "q_forward_%03d.png" % (n)
                                                  # p, v, u, n
     q_out = np.reshape(solver.get_chain_propagator(0, 0, 1, n), nx)
     plt.imshow(q_out, extent=[0, lx[0], 0, lx[1]], aspect='auto', interpolation='nearest')
     plt.colorbar()
     plt.savefig(file_name)
     print("q(%3.1f,r) is written to file '%s'." % (n*ds, file_name))
     plt.close()