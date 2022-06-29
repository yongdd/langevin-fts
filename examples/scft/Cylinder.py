# For the start, change "Major Simulation Parameters", currently in lines 20-27
# and "Initial Fields", currently in lines 70-84
import os
import numpy as np
import time
from scipy.io import savemat
from scipy.ndimage.filters import gaussian_filter
from langevinfts import *
from find_saddle_point import *

# -------------- initialize ------------

# OpenMP environment variables
os.environ["MKL_NUM_THREADS"] = "1"  # always 1
os.environ["OMP_STACKSIZE"] = "1G"
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "2"  # 0, 1 or 2

max_scft_iter = 1000
tolerance = 1e-8

# Major Simulation Parameters
f = 1.0/3.0         # A-fraction, f
n_segment = 90      # segment number, N
chi_n = 15.0        # Flory-Huggins Parameters * N
epsilon = 1.0       # a_A/a_B, conformational asymmetry
nx = [64,48,48]     # grid numbers
lx = [6.4,5.52,np.sqrt(3.0/4.0)*5.52]  # as aN^(1/2) unit, a = sqrt(f*a_A^2 + (1-f)*a_B^2)
chain_model = "Discrete" # choose among [Continuous, Discrete]

# Anderson mixing
am_n_var = 2*np.prod(nx)+len(lx)  # w_a (w[0]) and w_b (w[1]) + lx
am_max_hist= 20                   # maximum number of history
am_start_error = 1e-2             # when switch to AM from simple mixing
am_mix_min = 0.1                  # minimum mixing rate of simple mixing
am_mix_init = 0.1                 # initial mixing rate of simple mixing

# choose platform among [cuda, cpu-mkl, cpu-pocketfft]
if "cuda" in PlatformSelector.avail_platforms():
    platform = "cuda"
else:
    platform = PlatformSelector.avail_platforms()[0]
print("platform :", platform)
factory = PlatformSelector.create_factory(platform)

# create instances
pc     = factory.create_polymer_chain(f, n_segment, chi_n, chain_model, epsilon)
sb     = factory.create_simulation_box(nx, lx)
pseudo = factory.create_pseudo(sb, pc)
am     = factory.create_anderson_mixing(am_n_var,
            am_max_hist, am_start_error, am_mix_min, am_mix_init)

# -------------- print simulation parameters ------------
print("---------- Simulation Parameters ----------")
print("Box Dimension: %d" % (sb.get_dim()))
print("chi_n: %f, f: %f, N: %d" % (pc.get_chi_n(), pc.get_f(), pc.get_n_segment()) )
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
q1_init = np.ones (    sb.get_n_grid(),   dtype=np.float64)
q2_init = np.ones (    sb.get_n_grid(),   dtype=np.float64)

# Initial Fields
print("w_A and w_B are initialized to cylindrical phase.")
cylinder_positions = [
[0.0,0.0],[0.0,1/3],[0.0,2/3],
[1/2,0.0],[1/2,1/3],[1/2,2/3],
[1/4,1/6],[1/4,3/6],[1/4,5/6],
[3/4,1/6],[3/4,3/6],[3/4,5/6]]
for y,z in cylinder_positions:
    _, my, mz = np.round((np.array([0, y, z])*sb.get_nx())).astype(np.int32)
    w[0,:,my,mz] = -1/np.prod(sb.get_dx())
w[0] = gaussian_filter(w[0], sigma=np.min(sb.get_nx())/15, mode='wrap')
w = np.reshape(w, [2, sb.get_n_grid()])

# keep the level of field value
sb.zero_mean(w[0])
sb.zero_mean(w[1])

#------------------ run ----------------------
print("---------- Run ----------")
time_start = time.time()

phi_a, phi_b, Q, energy_total = find_saddle_point(pc, sb, pseudo, am, lx,
    q1_init, q2_init, w, max_scft_iter, tolerance, is_box_altering=True)

# estimate execution time
time_duration = time.time() - time_start
print("total time: %f " % time_duration)

# save final results
mdic = {"dim":sb.get_dim(), "nx":sb.get_nx(), "lx":sb.get_lx(),
        "N":pc.get_n_segment(), "f":pc.get_f(), "chi_n":pc.get_chi_n(), "epsilon":pc.get_epsilon(),
        "chain_model":chain_model, "w_a":w[0], "w_b":w[1], "phi_a":phi_a, "phi_b":phi_b}
savemat("fields.mat", mdic)