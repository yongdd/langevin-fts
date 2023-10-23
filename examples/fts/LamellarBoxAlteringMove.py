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

def find_saddle_point(cb, mixture, pseudo, am, chi_n,
    w_plus, w_minus, saddle_max_iter, saddle_tolerance, verbose_level):
        
    # assign large initial value for the energy and error
    energy_total = 1e20
    error_level = 1e20

    # reset Anderson mixing module
    am.reset_count()

    # saddle point iteration begins here
    for saddle_iter in range(1,saddle_max_iter+1):

        # for the given fields compute the polymer statistics
        pseudo.compute_statistics({"A":w_plus+w_minus,"B":w_plus-w_minus})
        phi_a = pseudo.get_total_concentration("A")
        phi_b = pseudo.get_total_concentration("B")
        
        # calculate incompressibility error
        old_error_level = error_level
        g_plus = phi_a + phi_b - 1.0
        error_level = np.sqrt(np.dot(g_plus, g_plus)/cb.get_n_grid())

        # print iteration # and error levels
        if(verbose_level == 2 or
         verbose_level == 1 and
         (error_level < saddle_tolerance or saddle_iter == saddle_max_iter)):
             
            # calculate the total energy
            energy_total = np.dot(w_minus,w_minus)/chi_n/cb.get_n_grid()
            energy_total += chi_n/4
            energy_total -= np.mean(w_plus)
            for p in range(mixture.get_n_polymers()):
                energy_total  -= np.log(pseudo.get_total_partition(p))

            # check the mass conservation
            mass_error = np.mean(g_plus)
            print("%8d %12.3E " % (saddle_iter, mass_error), end=" [ ")
            for p in range(mixture.get_n_polymers()):
                print("%13.7E " % (pseudo.get_total_partition(p)), end=" ")
            print("] %15.9f %15.7E " % (energy_total, error_level))

        # conditions to end the iteration
        if error_level < saddle_tolerance:
            break
            
        # calculate new fields using simple and Anderson mixing
        w_plus[:] = am.calculate_new_fields(w_plus, g_plus, old_error_level, error_level)

    w_plus -= np.mean(w_plus)
    Q = []
    for p in range(mixture.get_n_polymers()):
        Q.append(pseudo.get_total_partition(p))

    return phi_a, phi_b, Q, energy_total

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
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"  # 0, 1
os.environ["OMP_NUM_THREADS"] = "2"  # 1 ~ 4

verbose_level = 1  # 1 : print at each langevin step.
                   # 2 : print at each saddle point iteration.

input_data = loadmat("LamellarInput.mat", squeeze_me=True)

# Simulation Grids and Lengths
nx = [40, 40, 40]
lx = [4.46,4.46,4.46]

# Polymer Chain
n_segment = 90          # the number of segments
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
langevin_dt = 0.8      # Langevin step interval, delta tau*N
langevin_nbar = 10000  # Invariant polymerization index
langevin_max_step = 2000

use_superposition = False
reduce_gpu_memory_usage = False

# -------------- initialize ------------
# calculate chain parameters, dict_a_n = [a_A, a_B]
dict_a_n = {"A":np.sqrt(epsilon*epsilon/(f*epsilon*epsilon + (1.0-f))),
            "B":np.sqrt(1.0/(f*epsilon*epsilon + (1.0-f)))}

# choose platform among [cuda, cpu-mkl]
if "cuda" in PlatformSelector.avail_platforms():
    platform = "cuda"
else:
    platform = PlatformSelector.avail_platforms()[0]
print("platform :", platform)
factory = PlatformSelector.create_factory(platform, chain_model, reduce_gpu_memory_usage)

# create instances
cb = factory.create_computation_box(nx, lx)
mixture = factory.create_mixture(ds, dict_a_n, use_superposition)
mixture.add_polymer(1.0, ["A","B"], [f, 1-f], [0, 1], [1, 2])
pseudo = factory.create_pseudo(cb, mixture)
am = factory.create_anderson_mixing(am_n_var,
            am_max_hist, am_start_error, am_mix_min, am_mix_init)

# calculate bare chi_n
z_inf, dz_inf_dl = renormal_psum(lx, nx, n_segment, langevin_nbar)
chi_n = effective_chi_n/z_inf

if( np.abs(epsilon - 1.0) > 1e-7):
    raise Exception("Currently, only conformationally symmetric chains (epsilon==1) are supported.") 

# standard deviation of normal noise
langevin_sigma = np.sqrt(2*langevin_dt*cb.get_n_grid()/
    (cb.get_volume()*np.sqrt(langevin_nbar)))

## random seed for MT19937
np.random.seed(5489)
# -------------- print simulation parameters ------------
print("---------- Simulation Parameters ----------")
print("Box Dimension: %d" % (cb.get_dim()))
print("chi_n: %f, f: %f, N: %d" % (chi_n, f, mixture.get_polymer(0).get_n_segment_total()) )
print("%s chain model" % (mixture.get_model_name()) )
print("Conformational asymmetry (epsilon): %f" % (epsilon) )
print("Nx: %d, %d, %d" % (cb.get_nx(0), cb.get_nx(1), cb.get_nx(2)) )
print("Lx: %f, %f, %f" % (cb.get_lx(0), cb.get_lx(1), cb.get_lx(2)) )
print("dx: %f, %f, %f" % (cb.get_dx(0), cb.get_dx(1), cb.get_dx(2)) )
print("Volume: %f" % (cb.get_volume()) )

print("Invariant Polymerization Index: %d" % (langevin_nbar) )
print("Langevin Sigma: %f" % (langevin_sigma) )
print("Random Number Generator: ", np.random.RandomState().get_state()[0])

print("w_A and w_B are initialized to lamellar phase.")
w_plus  = (input_data["w_A"] + input_data["w_B"])/2
w_minus = (input_data["w_A"] - input_data["w_B"])/2

# keep the level of field value
w_plus -= np.mean(w_plus)

phi_a, phi_b, _, _ = find_saddle_point(cb, mixture, pseudo, am, chi_n,
    w_plus, w_minus, saddle_max_iter, saddle_tolerance, verbose_level)

# for box move
init_lx = cb.get_lx()
box_lambda = 1.0
#------------------ run ----------------------
print("---------- Run ----------")
time_start = time.time()
print("iteration, mass error, total_partition, energy_total, error_level")
for langevin_step in range(1, langevin_max_step+1):

    # calculate bare chi_n
    z_inf, dz_inf_dl = renormal_psum(cb.get_lx(), cb.get_nx(), mixture.get_polymer(0).get_n_segment_total(), langevin_nbar)
    chi_n = effective_chi_n/z_inf

    print("langevin step: ", langevin_step)
    w_minus_copy = w_minus.copy()
    normal_noise = np.random.normal(0.0, langevin_sigma, cb.get_n_grid())
    lambda1 = phi_a-phi_b + 2*w_minus/chi_n
    w_minus += -lambda1*langevin_dt + normal_noise
    phi_a, phi_b, _, _ = find_saddle_point(cb, mixture, pseudo, am, chi_n,
        w_plus, w_minus, saddle_max_iter, saddle_tolerance, verbose_level)

    # update w_minus: correct step
    lambda2 = phi_a-phi_b + 2*w_minus/chi_n
    w_minus = w_minus_copy - 0.5*(lambda1+lambda2)*langevin_dt + normal_noise
    phi_a, phi_b, _, _ = find_saddle_point(cb, mixture, pseudo, am, chi_n,
        w_plus, w_minus, saddle_max_iter, saddle_tolerance, verbose_level)

    # write density and field data
    if langevin_step % 100 == 0:
        mdic = {"dim":cb.get_dim(), "nx":cb.get_nx(), "lx":cb.get_lx(),
            "N":mixture.get_polymer(0).get_n_segment_total(), "f":f, "chi_n":chi_n, "epsilon":epsilon,
            "chain_model":mixture.get_model_name(), "nbar":langevin_nbar,
            "random_generator":np.random.RandomState().get_state()[0],
            "random_seed":np.random.RandomState().get_state()[1],
            "w_A":w_plus+w_minus, "w_B":w_plus-w_minus, "phi_A":phi_a, "phi_B":phi_b}
        savemat("fields_%06d.mat" % (langevin_step), mdic, do_compression=True)
        
    # caculate stress
    dlogQ_dl = -np.array(pseudo.compute_stress())
    dfield_dchin = 1/4 - np.dot(w_minus,w_minus)/chi_n**2/cb.get_n_grid()
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
    pseudo.update_bond_function()

# estimate execution time
time_duration = time.time() - time_start
print("total time: %f, time per step: %f" %
    (time_duration, time_duration/langevin_max_step) )

# Recording first a few iteration results for debugging and refactoring

# w_A and w_B are initialized to lamellar
#        3   -5.818E-14  [ 3.7439547E+00  ]     3.753877750   1.3019734E-05 
# ---------- Run ----------
# iteration, mass error, total_partition, energy_total, error_level
# langevin step:  1
#       32    8.882E-16  [ 4.3473228E+00  ]     4.351649260   9.9454838E-05 
#       17    2.665E-15  [ 4.2996180E+00  ]     4.301192083   8.1994433E-05 
# new Lx: [4.46123986 4.4593802  4.4593802 ]
# langevin step:  2
#       33    5.773E-15  [ 4.9316123E+00  ]     4.806826643   9.2777865E-05 
#       17    1.554E-15  [ 4.8829079E+00  ]     4.760009298   8.6524667E-05 
# new Lx: [4.46245645 4.45877228 4.45877228]
# langevin step:  3
#       33   -7.550E-15  [ 5.4527607E+00  ]     5.203397403   8.9344207E-05 
#       17    1.776E-15  [ 5.4037111E+00  ]     5.159082640   8.7022224E-05 
# new Lx: [4.46367781 4.45816223 4.45816223]
# langevin step:  4
#       33   -2.220E-15  [ 5.9254153E+00  ]     5.539058563   8.7304445E-05 
#       17   -1.554E-15  [ 5.8776521E+00  ]     5.497353437   8.6202925E-05 
# new Lx: [4.46487651 4.45756374 4.45756374]
# langevin step:  5
#       33   -2.998E-15  [ 6.4033675E+00  ]     5.801582131   9.7216658E-05 
#       17    3.109E-15  [ 6.3567886E+00  ]     5.762916498   8.7964949E-05 
# new Lx: [4.46606802 4.45696908 4.45696908]