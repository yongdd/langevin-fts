# (Caution!) In my experiment, box-altering move is accurate 
# only when simulation cell dV is close to cubic.
# If the cell changes too much from cubic during box-altering move,
# redo the simulation with preferred sized cubic box.
# In this example, initial box size is [4.46,4.46,4.46],
# but preferred size of cubic box is about [4.36,4.36,4.36].

import os
import time
import numpy as np
import scipy.special as sp
from scipy.io import loadmat, savemat
from langevinfts import *

def find_saddle_point(cb, pseudo, am, chi_n,
    q1_init, q2_init, w_plus, w_minus, 
    saddle_max_iter, saddle_tolerance, verbose_level):
        
    # assign large initial value for the energy and error
    energy_total = 1e20
    error_level = 1e20

    # reset Anderson mixing module
    am.reset_count()

    # saddle point iteration begins here
    for saddle_iter in range(1,saddle_max_iter+1):
        
        # for the given fields find the polymer statistics
        phi, Q = pseudo.compute_statistics(
            q1_init, q2_init,
            np.stack((w_plus+w_minus,w_plus-w_minus), axis=0))
        phi_plus = phi[0] + phi[1]
        
        # calculate output fields
        g_plus = phi_plus-1.0
        w_plus_out = w_plus + g_plus 
        cb.zero_mean(w_plus_out)

        # error_level measures the "relative distance" between the input and output fields
        old_error_level = error_level
        error_level = np.sqrt(cb.inner_product(phi_plus-1.0,phi_plus-1.0)/cb.get_volume())

        # print iteration # and error levels
        if(verbose_level == 2 or
         verbose_level == 1 and
         (error_level < saddle_tolerance or saddle_iter == saddle_max_iter)):
             
            # calculate the total energy
            energy_total  = -np.log(Q/cb.get_volume())
            energy_total += cb.inner_product(w_minus,w_minus)/chi_n/cb.get_volume()
            energy_total += chi_n/4
            energy_total -= cb.integral(w_plus)/cb.get_volume()

            # check the mass conservation
            mass_error = cb.integral(phi_plus)/cb.get_volume() - 1.0
            print("%8d %12.3E %15.7E %15.9f %15.7E" %
                (saddle_iter, mass_error, Q, energy_total, error_level))
        # conditions to end the iteration
        if error_level < saddle_tolerance:
            break
            
        # calculate new fields using simple and Anderson mixing
        am.calculate_new_fields(w_plus, w_plus_out, g_plus, old_error_level, error_level)
    return phi, Q

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

# Simulation Grids and Lengths
nx = [40, 40, 40]
lx = [4.46,4.46,4.46]

# Polymer Chain
n_segment = 90
f = 0.5
effective_chi_n = 12.75
epsilon = 1.0            # a_A/a_B, conformational asymmetry
chain_model = "Discrete" 
ds = 1/n_segment         # contour step interval

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

# calculate bare chi_n
z_inf, dz_inf_dl = renormal_psum(lx, nx, n_segment, langevin_nbar)
chi_n = effective_chi_n/z_inf

# create instances
pc     = factory.create_polymer_chain(N_pc, np.sqrt(a_sq_n), ds)
cb     = factory.create_computation_box(nx, lx)
pseudo = factory.create_pseudo(cb, pc)
am     = factory.create_anderson_mixing(am_n_var,
            am_max_hist, am_start_error, am_mix_min, am_mix_init)

if( np.abs(epsilon - 1.0) > 1e-7):
    raise Exception("Currently, only conformationally symmetric chains (epsilon==1) are supported.") 

# standard deviation of normal noise
langevin_sigma = np.sqrt(2*langevin_dt*cb.get_n_grid()/
    (cb.get_volume()*np.sqrt(langevin_nbar)))

## random seed for MT19937
#np.random.seed(5489)
# -------------- print simulation parameters ------------
print("---------- Simulation Parameters ----------")
print("Box Dimension: %d"  % (cb.get_dim()) )
print("chi_n: %f, f: %f, N: %d" % (chi_n, f, pc.get_n_segment_total()) )
print("%s chain model" % (pc.get_model_name()) )
print("Conformational asymmetry (epsilon): %f" % (epsilon) )
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

print("w_minus and w_plus are initialized to lamellar")
w_plus  = (input_data["w_a"] + input_data["w_b"])/2
w_minus = (input_data["w_a"] - input_data["w_b"])/2

# keep the level of field value
cb.zero_mean(w_plus)

phi, _ = find_saddle_point(cb, pseudo, am, chi_n,
    q1_init, q2_init, w_plus, w_minus,
    saddle_max_iter, saddle_tolerance, verbose_level)

# for box move
init_lx = cb.get_lx()
box_lambda = 1.0
#------------------ run ----------------------
print("---------- Run ----------")
time_start = time.time()
print("iteration, mass error, total_partition, energy_total, error_level")
for langevin_step in range(1, langevin_max_step+1):

    # calculate bare chi_n
    z_inf, dz_inf_dl = renormal_psum(cb.get_lx(), cb.get_nx(), pc.get_n_segment_total(), langevin_nbar)
    chi_n = effective_chi_n/z_inf

    print("langevin step: ", langevin_step)
    # update w_minus: predict step
    w_minus_copy = w_minus.copy()
    normal_noise = np.random.normal(0.0, langevin_sigma, cb.get_n_grid())
    lambda1 = phi[0]-phi[1] + 2*w_minus/chi_n
    w_minus += -lambda1*langevin_dt + normal_noise
    phi, _ = find_saddle_point(cb, pseudo, am, chi_n,
        q1_init, q2_init, w_plus, w_minus,
        saddle_max_iter, saddle_tolerance, verbose_level)

    # update w_minus: correct step
    lambda2 = phi[0]-phi[1] + 2*w_minus/chi_n
    w_minus = w_minus_copy - 0.5*(lambda1+lambda2)*langevin_dt + normal_noise
    phi, Q = find_saddle_point(cb, pseudo, am, chi_n,
        q1_init, q2_init, w_plus, w_minus,
        saddle_max_iter, saddle_tolerance, verbose_level)

    # write density and field data
    if langevin_step % 100 == 0:
        mdic = {"dim":cb.get_dim(), "nx":cb.get_nx(), "lx":cb.get_lx(),
            "N":pc.get_n_segment_total(), "f":f, "chi_n":chi_n, "epsilon":epsilon,
            "chain_model":pc.get_model_name(), "nbar":langevin_nbar,
            "random_generator":np.random.RandomState().get_state()[0],
            "random_seed":np.random.RandomState().get_state()[1],
            "w_plus":w_plus, "w_minus":w_minus, "phi_a":phi[0], "phi_b":phi[1]}
        savemat( "fields_%06d.mat" % (langevin_step), mdic)
        
    # caculate stress
    dlogQ_dl = np.array(pseudo.dq_dl())/Q
    dfield_dchin = 1/4 - cb.inner_product(w_minus,w_minus)/chi_n**2/cb.get_volume()
    dfield_dl = -dfield_dchin*chi_n/z_inf*dz_inf_dl
    dH_dl = -dlogQ_dl + dfield_dl
    #print(-dlogQ_dl, dfield_dl)

    # box move
    box_lambda = box_lambda - 0.01*(dH_dl[0]*cb.get_lx(0)-dH_dl[1]*cb.get_lx(1)/2-dH_dl[2]*cb.get_lx(2)/2)/box_lambda
    new_lx = np.array([init_lx[0]*box_lambda, init_lx[1]/np.sqrt(box_lambda), init_lx[2]/np.sqrt(box_lambda)])
    print("new Lx:", new_lx)
    
    # change box size
    cb.set_lx(new_lx)
    # update bond parameters using new lx
    pseudo.update()

# estimate execution time
time_duration = time.time() - time_start
print("total time: %f, time per step: %f" %
    (time_duration, time_duration/langevin_max_step) )
