import os
import time
import numpy as np
from scipy.io import savemat
# import scft_brush as scft
import scft

# OpenMP environment variables
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"  # 0, 1
os.environ["OMP_NUM_THREADS"] = "2"  # 1 ~ 4

linear_polymer =[{"type":"A", "length":1.0, "v":0, "u":1}]

n_backbone_node = 4
# Backbone chain
branched_polymer = [{"type":"A", "length":0.6, "v":0, "u":1}]
for i in range(1,n_backbone_node+1):
    branched_polymer.append({"type":"A", "length":0.4/n_backbone_node, "v":i, "u":i+1})

# Side chain
for i in range(1, n_backbone_node):
    branched_polymer.append({"type":"A", "length":0.3, "v":i+1, "u":n_backbone_node+i+1})
print(branched_polymer)

boundary_conditions = ["absorbing", "reflecting"]

params = {
    "platform":"cpu-mkl",           # choose platform among [cuda, cpu-mkl]
    
    "nx":[200],          # Simulation grid numbers
    "lx":[5],            # Simulation box size as a_Ref * N_Ref^(1/2) unit,
                         # where "a_Ref" is reference statistical segment length
                         # and "N_Ref" is the number of segments of reference linear homopolymer chain.
    "boundary_conditions": boundary_conditions,

    "box_is_altering":False,     # Find box size that minimizes the free energy during saddle point iteration.
    "chain_model":"continuous",  # "discrete" or "continuous" chain model
    "ds":1/500,                  # Contour step interval, which is equal to 1/N_Ref.

    "segment_lengths":{         # Relative statistical segment length compared to "a_Ref.
        "A":1.0},

    "distinct_polymers":[  # Distinct Polymers
        {   # A Grafted Brush
            "volume_fraction":1.0,
            "blocks":linear_polymer,
            "initial_conditions":{0:"G"}},
        ],

    "optimizer":{
        "name":"am",            # Anderson Mixing
        "max_hist":20,          # Maximum number of history
        "start_error":1e-2,     # When switch to AM from simple mixing
        "mix_min":0.1,         # Minimum mixing rate of simple mixing
        "mix_init":0.1,        # Initial mixing rate of simple mixing
    },

    "max_iter":100,     # The maximum relaxation iterations
    "tolerance":1e-8     # Terminate iteration if the self-consistency error is less than tolerance
}

# Target density profile for film
T = 1.0
t = 0.4
L = params["lx"][0]
dx = params["lx"][0]/params["nx"][0]

offset_grafting = np.max([round((0.05)/dx), 1])

mask = np.ones(params["nx"])

# Set initial fields
w_A = np.zeros(list(params["nx"]), dtype=np.float64)
print("w_A and w_B are initialized to lamellar phase.")
for i in range(params["nx"][0]):
    w_A[i] = -np.cos(2*np.pi*i/params["nx"][0])

# Initial condition of q (grafting point)
q_init = {"G":np.zeros(list(params["nx"]), dtype=np.float64)}
q_init["G"][offset_grafting] = 1.0/dx

# Mask for Nano Particle
mask = np.ones(params["nx"])
# nano_particle_radius = 0.7
# x = np.linspace(0.0-T-1.5, params["lx"][0]-T-1.5, num=params["nx"][0], endpoint=False)
# mask[np.sqrt(x**2) < nano_particle_radius] = 0.0
print(mask[:])
print(q_init["G"][:])

mask = mask*np.flip(mask, axis=0)

# Initialize calculation
calculation = scft.SCFT(params=params, mask=mask)

# Set a timer
time_start = time.time()

# Run
calculation.run(initial_fields={"A": w_A}, q_init=q_init)

# Estimate execution time
time_duration = time.time() - time_start
print("total time: %f " % time_duration)

# Save final results
calculation.save_results("fields.mat")

###############################################################
from langevinfts import *
import matplotlib.pyplot as plt
from scipy.io import savemat, loadmat

nx = params["nx"]
lx = params["lx"]
ds = params["ds"]
stat_seg_length = params["segment_lengths"]

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
input_data = loadmat("fields.mat", squeeze_me=True)
w_A = input_data["w_A"]
w = {"A": w_A}

# Compute ensemble average concentration (phi) and total partition function (Q)
solver.compute_statistics({"A":w["A"]}, q_init=q_init)

x = np.linspace(0.0, lx[0], num=nx[0], endpoint=False)

phi = np.reshape(solver.get_total_concentration("A"), nx)
phi = phi*molecules.get_polymer(0).get_alpha()/params["lx"][0]
file_name = "phi"
plt.plot(x, phi)
# plt.xlim([2, 4])
plt.savefig(file_name)
print("phi(r) is written to file '%s'." % (file_name))
plt.close()

N = round(1.0/ds)
for n in range(0, round(N)+1, round(N/5)):
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