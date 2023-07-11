# This program will produce a lamellar phase from random initial condition
# For test purpose, this program stops after 2000 Langevin steps
# change "langevin_max_step" to a larger number for the actual simulation

import os
import time
import numpy as np
from scipy.io import savemat
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

# -------------- simulation parameters ------------

# OpenMP environment variables
os.environ["OMP_STACKSIZE"] = "1G"
os.environ["MKL_NUM_THREADS"] = "1"  # always 1
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"  # 0, 1
os.environ["OMP_NUM_THREADS"] = "2"

verbose_level = 1  # 1 : print at each langevin step.
                   # 2 : print at each saddle point iteration.

# Simulation Grids and Lengths
nx = [32, 32, 32]
lx = [8.0, 8.0, 8.0]

# Polymer Chain
f = 0.5
chi_n = 20                   # Bare interaction parameter
epsilon = 1.0                # a_A/a_B, conformational asymmetry
chain_model = "Continuous"   # choose among [Continuous, Discrete]
ds = 1/16                    # contour step interval

# Anderson Mixing
saddle_tolerance = 1e-4
saddle_max_iter = 100
am_n_var = np.prod(nx)  # W+
am_max_hist= 20
am_start_error = 8e-1
am_mix_min = 0.1
am_mix_init = 0.1

# Langevin Dynamics
langevin_dt = 0.8        # Langevin step interval, delta tau*N_Ref
langevin_nbar = 1024     # invariant polymerization index
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

print("w_A and w_B are initialized to random Gaussian.")
w_plus  = np.random.normal(0.0, 1.0, cb.get_n_grid())
w_minus = np.random.normal(0.0, 1.0, cb.get_n_grid())

# keep the level of field value
w_plus -= np.mean(w_plus)

phi_a, phi_b, _, _ = find_saddle_point(cb, mixture, pseudo, am, chi_n,
    w_plus, w_minus, saddle_max_iter, saddle_tolerance, verbose_level)
    
# init structure function
sf_average = np.zeros_like(np.fft.rfftn(np.reshape(w_minus, cb.get_nx())),np.float64)

#------------------ run ----------------------
print("---------- Run ----------")
time_start = time.time()

print("iteration, mass error, total_partition, energy_total, error_level")
for langevin_step in range(1, langevin_max_step+1):

    print("langevin step: ", langevin_step)
    # update w_minus: predict step
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
        
    # calcaluate structure function
    if langevin_step % 10 == 0:
        sf_average += np.absolute(np.fft.rfftn(np.reshape(w_minus, cb.get_nx()))/cb.get_n_grid())**2

    # save structure function
    if langevin_step % 1000 == 0:
        sf_average *= 10/1000*cb.get_volume()*np.sqrt(langevin_nbar)/chi_n**2
        sf_average -= 1.0/(2*chi_n)
        mdic = {"dim":cb.get_dim(), "nx":cb.get_nx(), "lx":cb.get_lx(),
        "N":mixture.get_polymer(0).get_n_segment_total(), "f":f, "chi_n":chi_n, "epsilon":epsilon,
        "chain_model":mixture.get_model_name(),
        "dt":langevin_dt, "nbar":langevin_nbar,
        "structure_function":sf_average}
        savemat( "structure_function_%06d.mat" % (langevin_step), mdic)
        sf_average[:,:,:] = 0.0

    # write density and field data
    if langevin_step % 1000 == 0:
        mdic = {"dim":cb.get_dim(), "nx":cb.get_nx(), "lx":cb.get_lx(),
            "N":mixture.get_polymer(0).get_n_segment_total(), "f":f, "chi_n":chi_n, "epsilon":epsilon,
            "chain_model":mixture.get_model_name(), "nbar":langevin_nbar,
            "random_generator":np.random.RandomState().get_state()[0],
            "random_seed":np.random.RandomState().get_state()[1],
            "w_plus":w_plus, "w_minus":w_minus, "phi_a":phi_a, "phi_b":phi_b}
        savemat( "fields_%06d.mat" % (langevin_step), mdic)

# estimate execution time
time_duration = time.time() - time_start
print("total time: %f, time per step: %f" %
    (time_duration, time_duration/langevin_max_step) )

# Recording first a few iteration results for debugging and refactoring

# w_A and w_B are initialized to random
#       21    8.882E-16  [ 1.0419908E+00  ]     5.009168908   9.3083443E-05 
# ---------- Run ----------
# iteration, mass error, total_partition, energy_total, error_level
# langevin step:  1
#       18   -2.887E-15  [ 1.1837270E+00  ]     5.040091197   8.1538804E-05 
#        7    1.110E-15  [ 1.1830971E+00  ]     5.037712444   8.9411454E-05 
# langevin step:  2
#       19   -2.442E-15  [ 1.3403324E+00  ]     5.070141937   7.9617551E-05 
#        8    4.441E-16  [ 1.3389027E+00  ]     5.067732088   8.0144714E-05 
# langevin step:  3
#       19   -2.887E-15  [ 1.5022276E+00  ]     5.099658828   9.5416355E-05 
#        8    2.220E-16  [ 1.5001386E+00  ]     5.097283111   9.5933782E-05 
# langevin step:  4
#       20   -3.331E-15  [ 1.6789946E+00  ]     5.127049567   7.4926202E-05 
#        9   -3.442E-15  [ 1.6764255E+00  ]     5.124597150   7.2687167E-05 
# langevin step:  5
#       20   -3.109E-15  [ 1.8673914E+00  ]     5.154916422   8.3472264E-05 
#        9   -1.998E-15  [ 1.8639734E+00  ]     5.152498970   7.9975233E-05 
