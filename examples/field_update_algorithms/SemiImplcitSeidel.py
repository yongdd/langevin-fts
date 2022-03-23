import sys
import os
import time
import pathlib
import numpy as np
from langevinfts import *
from find_saddle_point import *

# -------------- simulation parameters ------------
# Cuda environment variables 
#os.environ["CUDA_VISIBLE_DEVICES"]= "0"
# OpenMP environment variables 
os.environ["MKL_NUM_THREADS"] = "1"  # always 1
os.environ["OMP_STACKSIZE"] = "1G"
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "0"  # 0, 1 or 2

verbose_level = 1  # 1 : print at each langevin step.
                   # 2 : print at each saddle point iteration.

# Simulation Box
nx = [48, 48, 48]
lx = [9, 9, 9]

# Polymer Chain
n_contour = 100
f = 0.34
chi_n = 10.0
chain_model = "Discrete" # choose among [Gaussian, Discrete]

# Anderson Mixing 
saddle_tolerance = 1e-4
saddle_max_iter = 200
am_n_comp = 1  # W+
am_max_hist= 20
am_start_error = 1e-1
am_mix_min = 0.1
am_mix_init = 0.1

# Langevin Dynamics
langevin_dt = 1.0     # langevin step interval, delta tau*N
langevin_nbar = 1000  # invariant polymerization index
langevin_max_iter = 2000

# -------------- initialize ------------
# choose platform among [cuda, cpu-mkl, cpu-fftw]
factory = PlatformSelector.create_factory("cuda")

# create instances and assign to the variables of base classs
# for the dynamic binding
pc = factory.create_polymer_chain(f, n_contour, chi_n, chain_model)
sb = factory.create_simulation_box(nx, lx)
pseudo = factory.create_pseudo(sb, pc)
am = factory.create_anderson_mixing(sb, am_n_comp,
    am_max_hist, am_start_error, am_mix_min, am_mix_init)

# standard deviation of normal noise for single segment
langevin_sigma = np.sqrt(2*langevin_dt*sb.get_n_grid()/ 
    (sb.get_volume()*np.sqrt(langevin_nbar)))
    
# random seed for MT19937
np.random.seed(5489)

# arrays for semi implicit Seidel
kernel_minus = langevin_dt/(1.0 + 2/pc.get_chi_n()*langevin_dt)
kernel_noise = 1/(1.0 + 2/pc.get_chi_n()*langevin_dt)

# -------------- print simulation parameters ------------
print("---------- Simulation Parameters ----------")
print("Box Dimension: %d"  % (sb.get_dim()) )
print("Precision: 8")
print("chi_n: %f, f: %f, N: %d" % (pc.get_chi_n(), pc.get_f(), pc.get_n_contour()) )
print("%s chain model" % (pc.get_model_name()) )
print("Nx: %d, %d, %d" % (sb.get_nx(0), sb.get_nx(1), sb.get_nx(2)) )
print("Lx: %f, %f, %f" % (sb.get_lx(0), sb.get_lx(1), sb.get_lx(2)) )
print("dx: %f, %f, %f" % (sb.get_dx(0), sb.get_dx(1), sb.get_dx(2)) )
print("Volume: %f" % (sb.get_volume()) )

print("Invariant Polymerization Index: %d" % (langevin_nbar) )
print("Langevin Sigma: %f" % (langevin_sigma) )
print("Random Number Generator: ", np.random.RandomState().get_state()[0])

#-------------- allocate array ------------
# free end initial condition. q1 is q and q2 is qdagger.
# q1 starts from A end and q2 starts from B end.
q1_init = np.ones (sb.get_n_grid(), dtype=np.float64)
q2_init = np.ones (sb.get_n_grid(), dtype=np.float64)
phi_a   = np.zeros(sb.get_n_grid(), dtype=np.float64)
phi_b   = np.zeros(sb.get_n_grid(), dtype=np.float64)

print("wminus and wplus are initialized to random")
w_plus  = np.random.normal(0.0, langevin_sigma, sb.get_n_grid())
w_minus = np.random.normal(0.0, langevin_sigma, sb.get_n_grid())

# keep the level of field value
sb.zero_mean(w_plus)
sb.zero_mean(w_minus)

find_saddle_point(pc, sb, pseudo, am,
    q1_init, q2_init, w_plus, w_minus, 
    phi_a, phi_b, 
    saddle_max_iter, saddle_tolerance, verbose_level)
#------------------ run ----------------------
print("---------- Run ----------")
time_start = time.time()

lnQ_list = []
print("iteration, mass error, total_partition, energy_total, error_level")
for langevin_step in range(0, langevin_max_iter):
    
    print("langevin step: ", langevin_step)
    # update w_minus
    normal_noise = np.random.normal(0.0, langevin_sigma, sb.get_n_grid())
    g_minus = phi_a-phi_b + 2*w_minus/pc.get_chi_n()
    w_minus += -kernel_minus*g_minus + kernel_noise*normal_noise
    _, QQ, = find_saddle_point(pc, sb, pseudo, am,
        q1_init, q2_init, w_plus, w_minus, 
        phi_a, phi_b, 
        saddle_max_iter, saddle_tolerance, verbose_level)
    lnQ_list.append(-np.log(QQ/sb.get_volume()))

print(langevin_dt, np.mean(lnQ_list[1000:]))

# estimate execution time
time_duration = time.time() - time_start
print( "total time: %f, time per step: %f" %
    (time_duration, time_duration/langevin_max_iter) )
