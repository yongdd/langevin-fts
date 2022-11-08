import sys
import os
import time
import pathlib
import numpy as np
from langevinfts import *
from find_saddle_point import *

# -------------- simulation parameters ------------
# Cuda environment variables 
#os.environ["CUDA_VISIBLE_DEVICES"]= "1"
# OpenMP environment variables
os.environ["MKL_NUM_THREADS"] = "1"  # always 1 
os.environ["OMP_STACKSIZE"] = "1G"
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "2"  # 0, 1 or 2

verbose_level = 1  # 1 : print at each langevin step.
                   # 2 : print at each saddle point iteration.

# Simulation Grids and Lengths
nx = [48, 48, 48]
lx = [9, 9, 9]

# Polymer Chain
n_segment = 100
f = 0.34
chi_n = 10.0
epsilon = 1.0            # a_A/a_B, conformational asymmetry
chain_model = "Discrete" # choose among [Continuous, Discrete]

# Anderson Mixing 
saddle_tolerance = 1e-4
saddle_max_iter = 200
am_n_var = np.prod(nx) # W+
am_max_hist= 20
am_start_error = 1e-1
am_mix_min = 0.1
am_mix_init = 0.1

# Langevin Dynamics
langevin_dt = 1.0     # langevin step interval, delta tau*N
langevin_nbar = 1000  # invariant polymerization index
langevin_max_step = 2000

# -------------- initialize ------------
# calculate chain parameters
# a : statistical segment length, N: n_segment
# a_sq_n = a^2 * N
a_sq_n = [epsilon*epsilon/(f*epsilon*epsilon + (1.0-f)),
                                 1.0/(f*epsilon*epsilon + (1.0-f))]
N_pc = [int(f*n_segment),int((1-f)*n_segment)]

# choose platform among [cuda, cpu-mkl]
if "cuda" in PlatformSelector.avail_platforms():
    platform = "cuda"
else:
    platform = PlatformSelector.avail_platforms()[0]
print("platform :", platform)
computation = SingleChainStatistics.create_computation(platform, chain_model)

# create instances
pc     = computation.create_polymer_chain(N_pc, a_sq_n)
cb     = computation.create_computation_box(nx, lx)
pseudo = computation.create_pseudo(cb, pc)
am     = computation.create_anderson_mixing(am_n_var,
            am_max_hist, am_start_error, am_mix_min, am_mix_init)

# standard deviation of normal noise
langevin_sigma = np.sqrt(2*langevin_dt*cb.get_n_grid()/ 
    (cb.get_volume()*np.sqrt(langevin_nbar)))
    
# random seed for MT19937
np.random.seed(5489)

# arrays for exponential time differencing
space_ky, space_kx, space_kz = np.meshgrid(
    2*np.pi/cb.get_lx(1)*np.concatenate([np.arange((cb.get_nx(1)+1)//2), cb.get_nx(1)//2-np.arange(cb.get_nx(1)//2)]),
    2*np.pi/cb.get_lx(0)*np.concatenate([np.arange((cb.get_nx(0)+1)//2), cb.get_nx(0)//2-np.arange(cb.get_nx(0)//2)]),
    2*np.pi/cb.get_lx(2)*np.arange(cb.get_nx(2)//2+1))

mag2_k = (space_kx**2 + space_ky**2 + space_kz**2)/6.0
mag2_k[0,0,0] = 1.0e-5 # to prevent 'division by zero' error

g_k = 2*(mag2_k+np.exp(-mag2_k)-1.0)/mag2_k**2
g_k[0,0,0] = 1.0

kernel_minus = 2/chi_n
exp_kernel_minus  =         (1.0 - np.exp(-kernel_minus*langevin_dt))/kernel_minus
exp_kernel_noise  = np.sqrt((1.0 - np.exp(-2*kernel_minus*langevin_dt))/(2*kernel_minus*langevin_dt))
exp_kernel_second = (kernel_minus*langevin_dt + np.exp(-kernel_minus*langevin_dt) - 1.0)/(kernel_minus**2*langevin_dt)

# -------------- print simulation parameters ------------
print("---------- Simulation Parameters ----------")
print("Box Dimension: %d"  % (cb.get_dim()) )
print("chi_n: %f, f: %f, N: %d" % (chi_n, f, pc.get_n_segment_total()) )
print("%s chain model" % (pc.get_model_name()) )
print("Nx: %d, %d, %d" % (cb.get_nx(0), cb.get_nx(1), cb.get_nx(2)) )
print("Lx: %f, %f, %f" % (cb.get_lx(0), cb.get_lx(1), cb.get_lx(2)) )
print("dx: %f, %f, %f" % (cb.get_dx(0), cb.get_dx(1), cb.get_dx(2)) )
print("Volume: %f" % (cb.get_volume()) )

print("Invariant Polymerization Index: %d" % (langevin_nbar) )
print("Langevin Sigma: %f" % (langevin_sigma) )
print("Random Number Generator: ", np.random.RandomState().get_state()[0])

#-------------- allocate array ------------
# free end initial condition. q1 is q and q2 is qdagger.
# q1 starts from A end and q2 starts from B end.
q1_init = np.ones(cb.get_n_grid(), dtype=np.float64)
q2_init = np.ones(cb.get_n_grid(), dtype=np.float64)

print("w_minus and w_plus are initialized to random")
w_plus  = np.random.normal(0, langevin_sigma, cb.get_n_grid())
w_minus = np.random.normal(0, langevin_sigma, cb.get_n_grid())

# keep the level of field value
cb.zero_mean(w_plus)

# find saddle point of the pressure field
phi, _ = find_saddle_point(pc, cb, pseudo, am, chi_n,
    q1_init, q2_init, w_plus, w_minus,
    saddle_max_iter, saddle_tolerance, verbose_level)
#------------------ run ----------------------
print("---------- Run ----------")
time_start = time.time()

lnQ_list = []
print("iteration, mass error, total_partition, energy_total, error_level")
for langevin_step in range(1, langevin_max_step+1):
    
    print("langevin step: ", langevin_step)
    # update w_minus
    # Runge-Kuta step 1
    normal_noise = np.random.normal(0.0, langevin_sigma, cb.get_n_grid())
    g_minus = phi[0]-phi[1] + 2*w_minus/chi_n
    w_minus_copy = w_minus.copy()
    g_minus_copy = g_minus.copy()
    w_minus += -exp_kernel_minus*g_minus + exp_kernel_noise*normal_noise
    phi, _ = find_saddle_point(pc, cb, pseudo, am, chi_n,
        q1_init, q2_init, w_plus, w_minus, 
        saddle_max_iter, saddle_tolerance, verbose_level)

    # Runge-Kuta step 2    
    g_minus = phi[0]-phi[1] + 2*w_minus/chi_n
    w_minus += exp_kernel_second*(
         - g_minus      + kernel_minus*w_minus
         + g_minus_copy - kernel_minus*w_minus_copy)
    phi, Q = find_saddle_point(pc, cb, pseudo, am, chi_n,
        q1_init, q2_init, w_plus, w_minus, 
        saddle_max_iter, saddle_tolerance, verbose_level)
    lnQ_list.append(-np.log(Q/cb.get_volume()))

print(langevin_dt, np.mean(lnQ_list[1000:]))

# estimate execution time
time_duration = time.time() - time_start
print("total time: %f, time per step: %f" %
    (time_duration, time_duration/langevin_max_step) )
