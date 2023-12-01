# This example script computes propagators and concentrations of polymer chains with a nano particle at center
# Propagators inside of the nano particle is zero, e.g., q(r,s) = 0.
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

# Set a mask to set q(r,s) = 0
x = np.linspace(-lx[0]/2, lx[0]/2, num=nx[0], endpoint=False)
y = np.linspace(-lx[1]/2, lx[1]/2, num=nx[1], endpoint=False)
z = np.linspace(-lx[2]/2, lx[2]/2, num=nx[2], endpoint=False)
xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')
nano_particle_radius = 1.0
q_mask = np.ones(nx)
q_mask[np.sqrt(xv**2 + yv**2 + zv**2) < nano_particle_radius] = 0.0
print(np.mean(q_mask), (4/3*np.pi*nano_particle_radius**3)/np.prod(lx))

aggregate_propagator_computation = False
reduce_gpu_memory_usage = False

# Select platform ("cuda" or "cpu-mkl")
factory = PlatformSelector.create_factory("cuda", reduce_gpu_memory_usage)
factory.display_info()

# Create an instance for computation box
cb = factory.create_computation_box(nx, lx) #, impenetrable_region)

# Create an instance for molecule information with block segment information and chain model ("continuous" or "discrete")
molecules = factory.create_molecules_information("continuous", ds, stat_seg_length)

# Add polymer
molecules.add_polymer(
     1.0,
     [
     ["A", 1.0, 0, 1],  # first block (type, length, starting node, ending node)
     ],
)

# Propagators analyzer for optimal propagator computation
propagators_analyzer = factory.create_propagators_analyzer(molecules, aggregate_propagator_computation)
propagators_analyzer.display_blocks()
propagators_analyzer.display_propagators()

# Create Solver
solver = factory.create_pseudospectral_solver(cb, molecules, propagators_analyzer)

# Fields
w = {"A": np.zeros(np.prod(nx))}

# Compute ensemble average concentration (phi) and total partition function (Q)
solver.compute_statistics({"A":w["A"]}, q_mask=q_mask)

# q_out = solver.get_chain_propagator(0, 0, 1, n)
