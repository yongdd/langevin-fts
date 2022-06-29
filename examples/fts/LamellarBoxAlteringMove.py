# (Caution!) In my experiment, box-altering move is accurate 
# only when simulation cell dV is close to cubic.
# If the cell changes too much from cubic during box-altering move,
# redo the simulation with prefered sized cubic box.
# In this example, initial box size is [4.46,4.46,4.46],
# but prefered size of cubic box is about [4.36,4.36,4.36].

import sys
import os
import time
import pathlib
import numpy as np
import scipy.special as sp
from scipy.io import loadmat, savemat
from langevinfts import *
from find_saddle_point import *

def renormal_psum(lx, nx, n_segment, nbar, summax=100):

    # cell volume * rho_0
    dx = np.array(lx)/np.array(nx)
    dv = np.prod(dx)
    vcellrho = n_segment*np.sqrt(nbar)*dv

    # z_infinity
    sum_p_i = 0.0
    prod_alpha = lambda i, dx, n_segment: \
        dx*np.sqrt(3*n_segment/(2*np.pi*i))*sp.erf(np.pi/dx*np.sqrt(i/6/n_segment))
    prod_alpha_array = np.zeros(summax)
    for i in range(0,summax):
        prod_alpha_array[i] = prod_alpha(i+1,lx[0]/nx[0],n_segment) \
                               *prod_alpha(i+1,lx[1]/nx[1],n_segment) \
                               *prod_alpha(i+1,lx[2]/nx[2],n_segment)
        sum_p_i += prod_alpha_array[i]
    sum_p_i += np.power(3*n_segment/(2*np.pi),1.5)*dv*2/np.sqrt(0.5+summax)
    z_inf = 1-(1+2*sum_p_i)/vcellrho
    
    # d(z_infinity)/dl
    sum_p_i = np.array([0.0, 0.0, 0.0])
    for i in range(0,summax):
        for n in range(0,3):
            sum_p_i[n] += np.exp(-(i+1)*np.pi**2/(6*dx[n]**2*n_segment)) \
                *prod_alpha_array[i]/prod_alpha(i+1,dx[n],n_segment)
    dz_inf_dl = (1+2*sum_p_i)/vcellrho/np.array(lx)
    
    return z_inf, dz_inf_dl

# -------------- simulation parameters ------------
# Cuda environment variables
# os.environ["CUDA_VISIBLE_DEVICES"]= "1"
# OpenMP environment variables

os.environ["MKL_NUM_THREADS"] = "1"  # always 1
os.environ["OMP_STACKSIZE"] = "1G"
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "2"  # 0, 1 or 2

verbose_level = 1  # 1 : print at each langevin step.
                   # 2 : print at each saddle point iteration.

input_data = loadmat("LamellarInput.mat", squeeze_me=True)

# Simulation Box
nx = [40, 40, 40]
lx = [4.46,4.46,4.46]

# Polymer Chain
n_segment = 90
f = 0.5
effective_chi_n = 12.75
epsilon = 1.0            # a_A/a_B, conformational asymmetry
chain_model = "Discrete" 

# Anderson Mixing
saddle_tolerance = 1e-4
saddle_max_iter = 100
am_n_var = np.prod(nx)  # W+
am_max_hist= 20
am_start_error = 8e-1
am_mix_min = 0.1
am_mix_init = 0.1

# Langevin Dynamics
langevin_dt = 0.8     # langevin step interval, delta tau*N
langevin_nbar = 10000  # invariant polymerization index
langevin_max_step = 2000

# -------------- initialize ------------
# choose platform among [cuda, cpu-mkl, cpu-pocketfft]
if "cuda" in PlatformSelector.avail_platforms():
    platform = "cuda"
else:
    platform = PlatformSelector.avail_platforms()[0]
print("platform :", platform)
factory = PlatformSelector.create_factory(platform)

# calculate bare chi_n
z_inf, dz_inf_dl = renormal_psum(lx, nx, n_segment, langevin_nbar)
chi_n = effective_chi_n/z_inf

# create instances
pc     = factory.create_polymer_chain(f, n_segment, chi_n, chain_model, epsilon)
sb     = factory.create_simulation_box(nx, lx)
pseudo = factory.create_pseudo(sb, pc)
am     = factory.create_anderson_mixing(am_n_var,
            am_max_hist, am_start_error, am_mix_min, am_mix_init)

if( np.abs(pc.get_epsilon() - 1.0) > 1e-7):
	raise Exception("Currently, only conformationally symmetric chains (epsilon==1) are supported.") 

# standard deviation of normal noise
langevin_sigma = np.sqrt(2*langevin_dt*sb.get_n_grid()/
    (sb.get_volume()*np.sqrt(langevin_nbar)))

## random seed for MT19937
#np.random.seed(5489)
# -------------- print simulation parameters ------------
print("---------- Simulation Parameters ----------")
print("Box Dimension: %d"  % (sb.get_dim()) )
print("chi_n: %f, f: %f, N: %d" % (pc.get_chi_n(), pc.get_f(), pc.get_n_segment()) )
print("%s chain model" % (pc.get_model_name()) )
print("Conformational asymmetry (epsilon): %f" % (pc.get_epsilon()) )
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
q1_init = np.ones(sb.get_n_grid(), dtype=np.float64)
q2_init = np.ones(sb.get_n_grid(), dtype=np.float64)

print("w_minus and w_plus are initialized to lamellar")
w_plus  = (input_data["w_a"] + input_data["w_b"])/2
w_minus = (input_data["w_a"] - input_data["w_b"])/2

# keep the level of field value
sb.zero_mean(w_plus)

phi_a, phi_b, _ = find_saddle_point(pc, sb, pseudo, am,
    q1_init, q2_init, w_plus, w_minus,
    saddle_max_iter, saddle_tolerance, verbose_level)

# for box move
init_lx = sb.get_lx()
box_lambda = 1.0
#------------------ run ----------------------
print("---------- Run ----------")
time_start = time.time()
print("iteration, mass error, total_partition, energy_total, error_level")
for langevin_step in range(1, langevin_max_step+1):

    # calculate bare chi_n
    z_inf, dz_inf_dl = renormal_psum(sb.get_lx(), sb.get_nx(), pc.get_n_segment(), langevin_nbar)
    chi_n = effective_chi_n/z_inf
    pc.set_chi_n(chi_n)

    print("langevin step: ", langevin_step)
    # update w_minus: predict step
    w_minus_copy = w_minus.copy()
    normal_noise = np.random.normal(0.0, langevin_sigma, sb.get_n_grid())
    lambda1 = phi_a-phi_b + 2*w_minus/pc.get_chi_n()
    w_minus += -lambda1*langevin_dt + normal_noise
    phi_a, phi_b, _ = find_saddle_point(pc, sb, pseudo, am,
        q1_init, q2_init, w_plus, w_minus,
        saddle_max_iter, saddle_tolerance, verbose_level)

    # update w_minus: correct step
    lambda2 = phi_a-phi_b + 2*w_minus/pc.get_chi_n()
    w_minus = w_minus_copy - 0.5*(lambda1+lambda2)*langevin_dt + normal_noise
    phi_a, phi_b, Q = find_saddle_point(pc, sb, pseudo, am,
        q1_init, q2_init, w_plus, w_minus,
        saddle_max_iter, saddle_tolerance, verbose_level)

    # write density and field data
    if langevin_step % 100 == 0:
        mdic = {"dim":sb.get_dim(), "nx":sb.get_nx(), "lx":sb.get_lx(),
            "N":pc.get_n_segment(), "f":pc.get_f(), "chi_n":pc.get_chi_n(), "epsilon":pc.get_epsilon(),
            "chain_model":pc.get_model_name(), "nbar":langevin_nbar,
            "random_generator":np.random.RandomState().get_state()[0],
            "random_seed":np.random.RandomState().get_state()[1],
            "w_plus":w_plus, "w_minus":w_minus, "phi_a":phi_a, "phi_b":phi_b}
        savemat( "fields_%06d.mat" % (langevin_step), mdic)
        
    # caculate stress
    dlogQ_dl = np.array(pseudo.dq_dl())/Q
    dfield_dchin = 1/4 - sb.inner_product(w_minus,w_minus)/pc.get_chi_n()**2/sb.get_volume()
    dfield_dl = -dfield_dchin*pc.get_chi_n()/z_inf*dz_inf_dl
    dH_dl = -dlogQ_dl + dfield_dl
    #print(-dlogQ_dl, dfield_dl)

    # box move
    box_lambda = box_lambda - 0.01*(dH_dl[0]*sb.get_lx(0)-dH_dl[1]*sb.get_lx(1)/2-dH_dl[2]*sb.get_lx(2)/2)/box_lambda
    new_lx = np.array([init_lx[0]*box_lambda, init_lx[1]/np.sqrt(box_lambda), init_lx[2]/np.sqrt(box_lambda)])
    print("new Lx:", new_lx)
    
    # change box size
    sb.set_lx(new_lx)
    # update bond parameters using new lx
    pseudo.update()

# estimate execution time
time_duration = time.time() - time_start
print("total time: %f, time per step: %f" %
    (time_duration, time_duration/langevin_max_step) )
