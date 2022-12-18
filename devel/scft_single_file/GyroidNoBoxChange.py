# For the start, change "Major Simulation Parameters", currently in lines 20-27
# and "Initial Fields", currently in lines 70-84
import os
import numpy as np
import time
from scipy.io import savemat
from langevinfts import *
from find_saddle_point import *

# -------------- initialize ------------

# OpenMP environment variables
os.environ["MKL_NUM_THREADS"] = "1"  # always 1
os.environ["OMP_STACKSIZE"] = "1G"
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "2"  # 0, 1 or 2
os.environ["OMP_NUM_THREADS"] = "2"  # 1 ~ 4

max_scft_iter = 1000
tolerance = 1e-8

# Major Simulation Parameters
f = 0.36            # A-fraction, f
n_segment = 100     # segment number, N
chi_n = 20          # Flory-Huggins Parameters * N
epsilon = 1.0       # a_A/a_B, conformational asymmetry
nx = [32,32,32]     # grid numbers
lx = [3.3,3.3,3.3]  # as aN^(1/2) unit, a = sqrt(f*a_A^2 + (1-f)*a_B^2)
ds = 1/n_segment      # contour step interval
chain_model = "Discrete" # choose among [Continuous, Discrete]

# Anderson mixing
am_n_var = 2*np.prod(nx)  # w_a (w[0]) and w_b (w[1]) + lx
am_max_hist= 20           # maximum number of history
am_start_error = 1e-2     # when switch to AM from simple mixing
am_mix_min = 0.1          # minimum mixing rate of simple mixing
am_mix_init = 0.1         # initial mixing rate of simple mixing

# calculate chain parameters
# a : statistical segment length, N: n_segment
# a_sq_n = [a_A^2 * N, a_B^2 * N]
a_sq_n = [epsilon*epsilon/(f*epsilon*epsilon + (1.0-f)),
            1.0/(f*epsilon*epsilon + (1.0-f))]
N_pc = [int(f*n_segment),int((1-f)*n_segment)]

# choose platform among [cuda, cpu-mkl]
if "cuda" in PlatformSelector.avail_platforms():
    platform = "cuda"
else:
    platform = PlatformSelector.avail_platforms()[0]
print("platform :", platform)
factory = PlatformSelector.create_factory(platform, chain_model)

# create instances
pc     = factory.create_polymer_chain(N_pc, np.sqrt(a_sq_n), ds)
cb     = factory.create_computation_box(nx, lx)
pseudo = factory.create_pseudo(cb, pc)
am     = factory.create_anderson_mixing(am_n_var,
            am_max_hist, am_start_error, am_mix_min, am_mix_init)

# -------------- print simulation parameters ------------
print("---------- Simulation Parameters ----------")
print("Box Dimension: %d" % (cb.get_dim()))
print("chi_n: %f, f: %f, N: %d" % (chi_n, f, pc.get_n_segment_total()) )
print("%s chain model" % (pc.get_model_name()) )
print("Conformational asymmetry (epsilon): %f" % (epsilon) )
print("Nx: %d, %d, %d" % (cb.get_nx(0), cb.get_nx(1), cb.get_nx(2)) )
print("Lx: %f, %f, %f" % (cb.get_lx(0), cb.get_lx(1), cb.get_lx(2)) )
print("dx: %f, %f, %f" % (cb.get_dx(0), cb.get_dx(1), cb.get_dx(2)) )
print("Volume: %f" % (cb.get_volume()) )

#-------------- allocate array ------------
# free end initial condition. q1 is q and q2 is qdagger.
# q1 starts from A end and q2 starts from B end.
w       = np.zeros([2]+list(cb.get_nx()), dtype=np.float64)
q1_init = np.ones (    cb.get_n_grid(),   dtype=np.float64)
q2_init = np.ones (    cb.get_n_grid(),   dtype=np.float64)

# Initial Fields
print("w_A and w_B are initialized to gyroid phase.")
# [Ref: https://pubs.acs.org/doi/pdf/10.1021/ma951138i]
for i in range(0,cb.get_nx(0)):
    xx = (i+1)*2*np.pi/cb.get_nx(0)
    for j in range(0,cb.get_nx(1)):
        yy = (j+1)*2*np.pi/cb.get_nx(1)
        for k in range(0,cb.get_nx(2)):
            zz = (k+1)*2*np.pi/cb.get_nx(2)
            c1 = np.sqrt(8.0/3.0)*(np.cos(xx)*np.sin(yy)*np.sin(2.0*zz) +
                np.cos(yy)*np.sin(zz)*np.sin(2.0*xx)+np.cos(zz)*np.sin(xx)*np.sin(2.0*yy))
            c2 = np.sqrt(4.0/3.0)*(np.cos(2.0*xx)*np.cos(2.0*yy)+
                np.cos(2.0*yy)*np.cos(2.0*zz)+np.cos(2.0*zz)*np.cos(2.0*xx))
            idx = i*cb.get_nx(1)*cb.get_nx(2) + j*cb.get_nx(2) + k
            w[0,i,j,k] = -0.3164*c1 +0.1074*c2
            w[1,i,j,k] =  0.3164*c1 -0.1074*c2

w = np.reshape(w, [2, cb.get_n_grid()])

# keep the level of field value
cb.zero_mean(w[0])
cb.zero_mean(w[1])

#------------------ run ----------------------
print("---------- Run ----------")
time_start = time.time()

phi, Q, energy_total = find_saddle_point(pc, cb, pseudo, am, lx, chi_n,
    q1_init, q2_init, w, max_scft_iter, tolerance, is_box_altering=False)

# estimate execution time
time_duration = time.time() - time_start
print("total time: %f " % time_duration)

# save final results
mdic = {"dim":cb.get_dim(), "nx":cb.get_nx(), "lx":cb.get_lx(),
        "N":pc.get_n_segment_total(), "f":f, "chi_n":chi_n, "epsilon":epsilon,
        "chain_model":chain_model, "w_a":w[0], "w_b":w[1], "phi_a":phi[0], "phi_b":phi[1]}
savemat("fields.mat", mdic)