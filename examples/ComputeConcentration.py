# This example script computes partition functions and concentrations of polymer chains for given fields.
# This script does not perform SCFT iteration.

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
nx = [64,64,64]                                   # grid number
lx = [5.0,5.0,5.0]                                # box size
ds = 0.01                                         # contour step interval
stat_seg_length = {"A":1.0, "B":2.0, "C":1.5}     # statistical segment lengths

aggregate_propagator_computation = False
reduce_gpu_memory_usage = False

# Select platform ("cuda" or "cpu-mkl")
factory = PlatformSelector.create_factory("cuda", reduce_gpu_memory_usage)
factory.display_info()

# Create an instance for computation box
cb = factory.create_computation_box(nx, lx) 
# Create an instance for molecule information with block segment information and chain model ("continuous" or "discrete")
molecules = factory.create_molecules_information("continuous", ds, stat_seg_length)

# First Polymer (homopolymer)
molecules.add_polymer(
     0.2,      # volume faction
     [
     ["B", 1.0,  0,  1],  # first block (type, length, starting node, ending node)
     ]
)

# Second Polymer (diblock copolymer)
molecules.add_polymer(
     0.3,                # volume faction
     [
     ["A", 0.4, 0, 1],   # first block
     ["C", 0.5, 1, 2],   # second block
     ]
)

# Third Polymer (triblock copolymer)
molecules.add_polymer(
     0.5,                # volume faction
     [
     ["A", 0.5, 0, 1],   # first block
     ["B", 0.7, 1, 2],   # second block
     ["A", 0.3, 2, 3],   # third block
     ]
)

# Propagators analyzer for optimal propagator computation
propagator_analyzer = factory.create_propagator_analyzer(molecules, aggregate_propagator_computation)
propagator_analyzer.display_blocks()
propagator_analyzer.display_propagators()

# Create Solver
solver = factory.create_pseudospectral_solver(cb, molecules, propagator_analyzer)
print(type(solver))

# External fields
w = {"A": np.random.normal(0.0, 1.0, np.prod(nx)),
     "B": np.random.normal(0.0, 1.0, np.prod(nx)),
     "C": np.random.normal(0.0, 1.0, np.prod(nx))}

# Compute ensemble average concentration (phi) and total partition function (Q)
solver.compute_propagators({"A":w["A"],"B":w["B"],"C":w["C"]})
solver.compute_concentrations()

# Compute total concentration for each monomer type
phi_a = solver.get_total_concentration("A")
phi_b = solver.get_total_concentration("B")
phi_c = solver.get_total_concentration("C")

print(phi_a, phi_b, phi_c)
print("Total phi:", np.mean(phi_a) + np.mean(phi_b) + np.mean(phi_c))

# For each polymer chain type
for p in range(molecules.get_n_polymer_types()):

     print(f"\nPolymer: {p}")

     # Total partition function
     Q = solver.get_total_partition(p)
     print(f"Q({p}):", Q)

     # Compute total concentration for a given polymer type index and monomer type
     phi_p_a = solver.get_total_concentration(p, "A")
     phi_p_b = solver.get_total_concentration(p, "B")
     phi_p_c = solver.get_total_concentration(p, "C")

     print(f"Total phi({p}):", np.mean(phi_p_a), np.mean(phi_p_b), np.mean(phi_p_c))
     print(phi_p_a, phi_p_b, phi_p_c)

     # Compute concentration of each block for a given polymer type index
     phi = solver.get_block_concentration(p)
     print(f"Block phi({p}):", np.mean(np.sum(phi, axis=0)))
     for b in range(molecules.get_polymer(p).get_n_blocks()):
          print(phi[b])