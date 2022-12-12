# This example script compute statistics of polymer chains in external fields

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
nx = [64,64,64]                                   # grid number
lx = [5.0, 5.0, 5.0]                              # box size
ds = 0.01                                         # contour step Interval
stat_seg_length = {"A":1.0, "B":2.0, "C":1.5}     # statistical segment lengths

block_lengths = []
block_species = []
volume_faction = []
v = []
u = []

# First Polymer
volume_faction.append(0.5)             # volume faction
block_lengths.append([0.5, 0.7, 0.3])   # contour length of each block (triblock)
block_species.append(["A","B","A"])     # type of each block (triblock)
v.append([0,1,2])                       # vertices v (triblock)
u.append([1,2,3])                       # vertices u (triblock)

# Second Polymer
volume_faction.append(0.3)           # volume faction
block_lengths.append([0.4, 0.5])      # contour length of each block (diblock)
block_species.append(["A","C"])       # type of each block (diblock)
v.append([0,1])                       # vertices v (diblock)
u.append([1,2])                       # vertices u (diblock)

# Third Polymer
volume_faction.append(0.2)       # volume faction
block_lengths.append([1.0])       # contour length of each block (homo)
block_species.append(["B"])       # type of each block (homo)
v.append([0])                     # vertices v (homo)
u.append([1])                     # vertices u (homo)

# select platform and chain model  ("cuda" or "cpu-mkl"), ("continuous" or "discrete")
factory = PlatformSelector.create_factory("cuda", "continuous")
factory.display_info()

# create instances
cb     = factory.create_computation_box(nx, lx)
mx     = factory.create_mixture(ds, stat_seg_length)
for p in range(len(block_lengths)):
     mx.add_polymer(
     volume_faction[p], block_species[p],
     block_lengths[p],v[p], u[p], {})
pseudo = factory.create_pseudo(cb, mx)

print(type(pseudo))

# fields
w = {"A": np.random.normal(0.0, 1.0, np.prod(nx)),
     "B": np.random.normal(0.0, 1.0, np.prod(nx)),
     "C": np.random.normal(0.0, 1.0, np.prod(nx))}

# compute ensemble average concentration (phi) and total partition function (Q)
pseudo.compute_statistics({}, {"A":w["A"],"B":w["B"],"C":w["C"]})

phi_a = pseudo.get_species_concentration("A")
phi_b = pseudo.get_species_concentration("B")
phi_c = pseudo.get_species_concentration("C")

print(phi_a, phi_b, phi_c)
print(np.mean(phi_a) + np.mean(phi_b) + np.mean(phi_c))

# for each polymer chain
for p in range(mx.get_n_polymers()):
     phi = pseudo.get_polymer_concentration(p)
     Q = pseudo.get_total_partition(p)

     print(Q)           # total partition function
     print(np.mean(np.sum(phi, axis=0)))
     for b in range(mx.get_polymer(p).get_n_blocks()):
          print(phi[b]) # concentration for each block
