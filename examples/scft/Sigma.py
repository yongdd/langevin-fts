import os
import numpy as np
import time
from scipy.io import loadmat, savemat
from scipy.ndimage.filters import gaussian_filter
from langevinfts import *
from find_saddle_point import *

# -------------- initialize ------------

# OpenMP environment variables
os.environ["MKL_NUM_THREADS"] = "1"  # always 1
os.environ["OMP_STACKSIZE"] = "1G"
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "2"  # 0, 1 or 2

max_scft_iter = 2000
tolerance = 1e-6

# Major Simulation Parameters
f = 0.25          # A-fraction, f
n_contour = 100   # segment number, N
chi_n = 25        # Flory-Huggins Parameters * N
epsilon = 2.0     # a_A/a_B, conformational asymmetry, default = 1.0
nx = [64,64,32]   # grid numbers
lx = [7.0,7.0,4.0]   # as aN^(1/2) unit, a = sqrt(f*a_A^2 + (1-f)*a_B^2)
chain_model = "Gaussian"    # choose among [Gaussian, Discrete]

# Anderson mixing
am_n_var = 2*np.prod(nx).item()+len(lx)  # w_a (w[0]) and w_b (w[1]) + lx
am_max_hist= 20                          # maximum number of history
am_start_error = 1e-2                    # when switch to AM from simple mixing
am_mix_min = 0.1                         # minimum mixing rate of simple mixing
am_mix_init = 0.1                        # initial mixing rate of simple mixing

# choose platform among [cuda, cpu-mkl, cpu-fftw]
if "cuda" in PlatformSelector.avail_platforms():
    platform = "cuda"
else:
    platform = PlatformSelector.avail_platforms()[0]
print("platform :", platform)
factory = PlatformSelector.create_factory(platform)

# create instances
pc = factory.create_polymer_chain(f, n_contour, chi_n, chain_model, epsilon)
sb = factory.create_simulation_box(nx, lx)
pseudo = factory.create_pseudo(sb, pc)
am = factory.create_anderson_mixing(sb, am_n_var,
    am_max_hist, am_start_error, am_mix_min, am_mix_init)

# -------------- print simulation parameters ------------
print("---------- Simulation Parameters ----------")
print("Box Dimension: %d" % (sb.get_dim()))
print("Precision: 8")
print("chi_n: %f, f: %f, N: %d" % (pc.get_chi_n(), pc.get_f(), pc.get_n_contour()) )
print("%s chain model" % (pc.get_model_name()) )
print("Conformational asymmetry (epsilon): %f" % (pc.get_epsilon()) )
print("Nx: %d, %d, %d" % (sb.get_nx(0), sb.get_nx(1), sb.get_nx(2)) )
print("Lx: %f, %f, %f" % (sb.get_lx(0), sb.get_lx(1), sb.get_lx(2)) )
print("dx: %f, %f, %f" % (sb.get_dx(0), sb.get_dx(1), sb.get_dx(2)) )
print("Volume: %f" % (sb.get_volume()) )

#-------------- allocate array ------------
# free end initial condition. q1 is q and q2 is qdagger.
# q1 starts from A end and q2 starts from B end.
w       = np.zeros([2]+list(sb.get_nx()), dtype=np.float64)
w_out   = np.zeros([2, sb.get_n_grid()],  dtype=np.float64)
q1_init = np.ones (    sb.get_n_grid(),   dtype=np.float64)
q2_init = np.ones (    sb.get_n_grid(),   dtype=np.float64)

# Initial Fields
print("w_A and w_B are initialized to Sigma phase.")
# [Ref: https://doi.org/10.3390/app2030654]
sphere_positions = [[0.01,0.01,0.01],[0.51,0.51,0.52], #A
[0.40,0.40,0.01],[0.61,0.61,0.01],[0.10,0.91,0.52],[0.91,0.10,0.52], #B
[0.13,0.47,0.01],[0.26,0.95,0.01],[0.47,0.13,0.01],[0.95,0.26,0.01], #C
[0.04,0.64,0.52],[0.38,0.97,0.52],[0.64,0.04,0.52],[0.97,0.38,0.52], #C
[0.07,0.76,0.01],[0.54,0.89,0.01],[0.76,0.07,0.01],[0.89,0.54,0.01], #D
[0.25,0.44,0.52],[0.44,0.25,0.52],[0.57,0.77,0.52],[0.77,0.57,0.52], #D
[0.19,0.19,0.26],[0.32,0.70,0.26],[0.70,0.32,0.26],[0.83,0.83,0.26], #E
[0.19,0.19,0.77],[0.32,0.70,0.77],[0.70,0.32,0.77],[0.83,0.83,0.77]] #E

for x,y,z in sphere_positions:
    mx, my, mz = np.round((np.array([x, y, z])*sb.get_nx())).astype(np.int32)
    w[0,mx,my,mz] = -1/np.prod(sb.get_dx())
w[0] = gaussian_filter(w[0], sigma=np.min(sb.get_nx())/15, mode='wrap')
w = np.reshape(w, [2, sb.get_n_grid()])

# keep the level of field value
sb.zero_mean(w[0])
sb.zero_mean(w[1])

#------------------ run ----------------------
print("---------- Run ----------")
time_start = time.time()

phi_a, phi_b, Q, energy_total = find_saddle_point(pc, sb, pseudo, am, lx,
    q1_init, q2_init, w, max_scft_iter, tolerance)

# estimate execution time
time_duration = time.time() - time_start
print("total time: %f " % time_duration)

# save final results
mdic = {"dim":sb.get_dim(), "nx":sb.get_nx(), "lx":sb.get_lx(),
        "N":pc.get_n_contour(), "f":pc.get_f(), "chi_n":pc.get_chi_n(), "epsilon":pc.get_epsilon(),
        "chain_model":chain_model, "w_a":w[0], "w_b":w[1], "phi_a":phi_a, "phi_b":phi_b}
savemat("fields.mat", mdic)
