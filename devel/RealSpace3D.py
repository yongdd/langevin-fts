# This example script computes propagators and concentrations of polymer chains with a nano particle at center
# Propagators inside of the nano particle is zero, e.g., q(r,s) = 0.

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
nx = [120,100,140]            # grid number
lx = [4.0,5.0,6.0]            # box size
ds = 0.01                     # contour step interval
stat_seg_length = {"A":1.0}   # statistical segment lengths

boundary_conditions = ["absorbing", "absorbing",
                       "reflecting", "reflecting",
                       "reflecting", "reflecting"]

# Set a mask to set q(r,s) = 0
x = np.linspace(-lx[0]/2, lx[0]/2, num=nx[0], endpoint=False)
y = np.linspace(-lx[1]/2, lx[1]/2, num=nx[1], endpoint=False)
z = np.linspace(-lx[2]/2, lx[2]/2, num=nx[2], endpoint=False)

xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')
nano_particle_radius = 1.0
mask = np.ones(nx)
mask[np.sqrt(xv**2 + yv**2 + zv**2) < nano_particle_radius] = 0.0
print(1.0-np.mean(mask), (4/3*np.pi*nano_particle_radius**3)/np.prod(lx))

aggregate_propagator_computation = False
reduce_gpu_memory_usage = False

# Select platform ("cuda" or "cpu-mkl")
factory = PlatformSelector.create_factory("cpu-mkl", reduce_gpu_memory_usage)
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
q_init["G"][30,:,:] = 1.0/(lx[0]/nx[0])
q_init["G"][:,30,:] = 1.0/(lx[1]/nx[1])
q_init["G"][:,:,30] = 1.0/(lx[2]/nx[2])

# Compute ensemble average concentration (phi) and total partition function (Q)
solver.compute_statistics({"A":w["A"]}, q_init=q_init)

phi = np.reshape(solver.get_total_concentration("A"), nx)
file_name = "phi"
plt.imshow(phi[:,:,round(nx[2]/2)], extent=[0, lx[0], 0, lx[1]], interpolation='nearest')
plt.colorbar()
plt.savefig(file_name)
print("phi(r) is written to file '%s'." % (file_name))
plt.close()

N = round(1.0/ds)
for n in range(0, round(N)+1, round(N/5)):
                                                  # p, v, u, n
     q_out = np.reshape(solver.get_chain_propagator(0, 0, 1, n), nx)
     plt.imshow(q_out[:,:,round(nx[2]/2)], extent=[0, lx[0], 0, lx[1]], interpolation='nearest')
     plt.colorbar()
     file_name = "q_forward_%03d_xy.png" % (n)
     plt.savefig(file_name)
     print("q(%3.1f,x,y,Lz/2) is written to file '%s'." % (n*ds, file_name))
     plt.close()
     
     plt.imshow(q_out[:,round(nx[1]/2),:], extent=[0, lx[0], 0, lx[2]], interpolation='nearest')
     plt.colorbar()
     file_name = "q_forward_%03d_xz.png" % (n)
     plt.savefig(file_name)
     print("q(%3.1f,x,Ly/2,z) is written to file '%s'." % (n*ds, file_name))
     plt.close()

     plt.imshow(q_out[round(nx[0]/2),:,:], extent=[0, lx[1], 0, lx[2]], interpolation='nearest')
     plt.colorbar()
     file_name = "q_forward_%03d_yz.png" % (n)
     plt.savefig(file_name)
     print("q(%3.1f,Lx/2,y,z) is written to file '%s'." % (n*ds, file_name))
     plt.close()