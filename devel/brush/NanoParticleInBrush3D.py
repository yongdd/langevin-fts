import os
import sys
import shutil
import time
import pathlib
import numpy as np
from scipy.io import savemat
# import scft_brush as scft
import scft

# OpenMP environment variables
os.environ["MKL_NUM_THREADS"] = "1"  # always 1
os.environ["OMP_STACKSIZE"] = "1G"
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"  # 0, 1
os.environ["OMP_NUM_THREADS"] = "2"  # 1 ~ 4

# GPU environment variables
os.environ["LFTS_GPU_NUM_BLOCKS"]  = "256"
os.environ["LFTS_GPU_NUM_THREADS"] = "256"
os.environ["LFTS_NUM_GPUS"] = "2" # 1 ~ 2

linear_polymer =[{"type":"A", "length":1.0, "v":0, "u":1}]

n_backbone_node = 4
# Backbone chain
branched_polymer = [{"type":"A", "length":0.6, "v":0, "u":1}]
for i in range(1, n_backbone_node+1):
    branched_polymer.append({"type":"A", "length":0.4/n_backbone_node, "v":i, "u":i+1})

# Side chain
for i in range(1, n_backbone_node):
    branched_polymer.append({"type":"A", "length":0.3, "v":i+1, "u":n_backbone_node+i+1})
print(branched_polymer)


params = {
    # "platform":"cuda",           # choose platform among [cuda, cpu-mkl]
    
    "nx":[360,300,300],          # Simulation grid numbers
    "lx":[12,10,10],           # Simulation box size as a_Ref * N_Ref^(1/2) unit,
                              # where "a_Ref" is reference statistical segment length
                              # and "N_Ref" is the number of segments of reference linear homopolymer chain.
                              
    "reduce_gpu_memory_usage":True, # Reduce gpu memory usage by storing propagators in main memory instead of gpu memory.
    "box_is_altering":False,         # Find box size that minimizes the free energy during saddle point iteration.
    "chain_model":"continuous",      # "discrete" or "continuous" chain model
    "ds":1/60,                      # Contour step interval, which is equal to 1/N_Ref.

    "segment_lengths":{         # Relative statistical segment length compared to "a_Ref.
        "A":1.0},

    "distinct_polymers":[  # Distinct Polymers
        {   # A Grafted Brush
            "volume_fraction":1.0,
            "blocks":branched_polymer,
            "initial_conditions":{0:"G"}},
        ],

    "optimizer":{
        "name":"am",            # Anderson Mixing
        "max_hist":20,          # Maximum number of history
        "start_error":1e-2,     # When switch to AM from simple mixing
        "mix_min":0.1,         # Minimum mixing rate of simple mixing
        "mix_init":0.1,        # Initial mixing rate of simple mixing
    },

    "max_iter":80,       # The maximum relaxation iterations
    "tolerance":1e-8     # Terminate iteration if the self-consistency error is less than tolerance
}

# Target density profile for film
T = 1.0
t = 0.4
T_mask = 1.0
L = params["lx"][0] - 2*T_mask
dx = params["lx"][0]/params["nx"][0]
I_range = round(L/dx)-3
offset = round(T_mask/dx)+1
offset_grafting = np.max([round((T_mask+0.05)/dx), round((T_mask)/dx)+1])

phi_target = np.zeros(params["nx"])
for i in range(I_range):
    phi_target[i+offset,:,:] = 1.0
    phi_target[params["nx"][0]-offset-i-1,:,:] = 1.0
print(phi_target[:30,0,0])
print(phi_target[-30:,0,0])

# Set initial fields
w_A = np.zeros(list(params["nx"]), dtype=np.float64)
print("w_A and w_B are initialized to lamellar phase.")
for i in range(I_range):
    w_A[i+offset,:,:] = -np.cos(2*np.pi*i/I_range)
# w_A = np.random.normal(0.0, 0.1, params["nx"])

# Initial condition of q (grafting point)
q_init = {"G":np.zeros(list(params["nx"]), dtype=np.float64)}
q_init["G"][offset_grafting,:,:] = 1.0/dx
q_init["G"][params["nx"][0]-offset_grafting-1,:,:] = 1.0/dx

# Mask for Nano Particle
mask = np.ones(params["nx"])

dis_from_sub = 0.0
delta_z = 3.0

#---Sphere
nano_particle_radius = 1.0

x = np.linspace(0.0-T-nano_particle_radius-dis_from_sub, params["lx"][0]-T-nano_particle_radius-dis_from_sub, num=params["nx"][0], endpoint=False)
# # One Particle
# y = np.linspace(-params["lx"][1]/2, params["lx"][1]/2, num=params["nx"][1], endpoint=False)
# Two Particles
y = np.linspace(-params["lx"][1]/2+nano_particle_radius+delta_z/2, params["lx"][1]/2+nano_particle_radius+delta_z/2, num=params["nx"][1], endpoint=False)
z = np.linspace(-params["lx"][2]/2, params["lx"][2]/2, num=params["nx"][2], endpoint=False)
xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')

mask[np.sqrt(xv**2+yv**2+zv**2) < nano_particle_radius] = 0.0

# #---Cylinder
# nano_particle_radius = 1.0
# nano_particle_height = (4/3*np.pi*nano_particle_radius**3)/(np.pi*nano_particle_radius**2)

# sphere_below_volume = (nano_particle_radius + dis_from_sub)*np.pi*nano_particle_radius**2 - 2/3*np.pi*nano_particle_radius**3
# print(sphere_below_volume)
# # dis_from_sub_cyl = sphere_below_volume/np.pi*nano_particle_radius**2
# # dis_from_sub_cyl = 3.0
# # print(dis_from_sub_cyl)

# x = np.linspace(0.0-T, params["lx"][0]-T, num=params["nx"][0], endpoint=False)
# # One Particle
# y = np.linspace(-params["lx"][1]/2, params["lx"][1]/2, num=params["nx"][1], endpoint=False)
# # # Two Particles
# y = np.linspace(-params["lx"][1]/2+nano_particle_radius+delta_z/2, params["lx"][1]/2+nano_particle_radius+delta_z/2, num=params["nx"][1], endpoint=False)
# z = np.linspace(-params["lx"][2]/2, params["lx"][2]/2, num=params["nx"][2], endpoint=False)
# xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')

# aa = np.sqrt(yv**2+zv**2) < nano_particle_radius
# bb = xv > dis_from_sub
# cc = xv < nano_particle_height+dis_from_sub

# print(aa.shape)
# print(bb.shape)
# print(cc.shape)
# mask[aa & bb & cc] = 0.0

#---------------------------
mask[np.isclose(phi_target, 0.0)] = 0.0
print(mask[:,0])
print(q_init["G"][:])
print(mask[20:40,0])
print(q_init["G"][20:40,0])

mask = mask*np.flip(mask, axis=0)
mask = mask*np.flip(mask, axis=1)

print(np.sum(mask)*np.prod(params["lx"])/np.prod(params["nx"]))

# Create folder for saving results
if len(sys.argv) >= 2:
    folder_name = "temp_%05d" % (int(sys.argv[1]))
    pathlib.Path(folder_name).mkdir(parents=True, exist_ok=True)
    path = os.path.join(folder_name, "fields.mat")
    src = os.path.abspath(__file__)
    shutil.copy2(src, folder_name)
else:
    path = "fields.mat"

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
calculation.save_results(path)