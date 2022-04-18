import sys
import os
import numpy as np
import time
from scipy.io import loadmat, savemat
from scipy.ndimage.filters import gaussian_filter
import scipy.optimize
from langevinfts import *

def find_saddle_point(lx):

    # set box size
    sb.set_lx(lx)
    pseudo.update()

    # assign large initial value for the energy and error
    energy_total = 1.0e20
    error_level = 1.0e20

    # reset Anderson mixing module
    am.reset_count()

    # iteration begins here
    for scft_iter in range(1,max_scft_iter+1):
        # for the given fields find the polymer statistics
        phi_a, phi_b, Q = pseudo.find_phi(q1_init,q2_init,w[0],w[1])

        # calculate the total energy
        energy_old = energy_total
        w_minus = (w[0]-w[1])/2
        w_plus  = (w[0]+w[1])/2

        energy_total  = -np.log(Q/sb.get_volume())
        energy_total += sb.inner_product(w_minus,w_minus)/pc.get_chi_n()/sb.get_volume()
        energy_total -= sb.integral(w_plus)/sb.get_volume()

        # calculate pressure field for the new field calculation, the method is modified from Fredrickson's
        xi = 0.5*(w[0]+w[1]-chi_n)

        # calculate output fields
        w_out[0] = chi_n*phi_b + xi
        w_out[1] = chi_n*phi_a + xi
        sb.zero_mean(w_out[0])
        sb.zero_mean(w_out[1])

        # error_level measures the "relative distance" between the input and output fields
        old_error_level = error_level
        w_diff = w_out - w
        multi_dot = sb.inner_product(w_diff[0],w_diff[0]) + sb.inner_product(w_diff[1],w_diff[1])
        multi_dot /= sb.inner_product(w[0],w[0]) + sb.inner_product(w[1],w[1]) + 1.0
        error_level = np.sqrt(multi_dot)

        # print iteration # and error levels and check the mass conservation
        mass_error = (sb.integral(phi_a) + sb.integral(phi_b))/sb.get_volume() - 1.0
        print( "%8d %12.3E %15.7E %13.9f %13.9f" %
            (scft_iter, mass_error, Q, energy_total, error_level) )

        # conditions to end the iteration
        if error_level < tolerance:
            break
        # calculte new fields using simple and Anderson mixing
        am.caculate_new_fields(
            np.reshape(w,      2*sb.get_n_grid()),
            np.reshape(w_out,  2*sb.get_n_grid()),
            np.reshape(w_diff, 2*sb.get_n_grid()),
            old_error_level, error_level)

    return energy_total

# -------------- initialize ------------

# OpenMP environment variables
os.environ["MKL_NUM_THREADS"] = "1"  # always 1
os.environ["OMP_STACKSIZE"] = "1G"
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "2"  # 0, 1 or 2

max_scft_iter = 2000
tolerance = 1e-7

# Major Simulation Parameters
f = 0.25          # A-fraction, f
n_contour = 100   # segment number, N
chi_n = 25        # Flory-Huggins Parameters * N
epsilon = 2.0     # a_A/a_B, conformational asymmetry, default = 1.0
nx = [64,64,32]   # grid numbers
lx = [7.0,7.0,4.0]   # as aN^(1/2) unit, a = sqrt(f*a_A^2 + (1-f)*a_B^2)
chain_model = "Gaussian"    # choose among [Gaussian, Discrete]

# Anderson mixing
am_n_comp = 2         # w_a (w[0]) and w_b (w[1])
am_max_hist= 20       # maximum number of history
am_start_error = 1e-2 # when switch to AM from simple mixing
am_mix_min = 0.1      # minimum mixing rate of simple mixing
am_mix_init = 0.1     # initial mixing rate of simple mixing

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
am = factory.create_anderson_mixing(sb, am_n_comp,
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
print("iteration, mass error, total_partition, energy_total, error_level")
time_start = time.time()

# find the natural period of gyroid
res = scipy.optimize.minimize(find_saddle_point, lx, tol=1e-4, options={'disp':True})
print('Unit cell that minimizes the free energy: ', np.round(res.x, 4), '(aN^1/2)')
print('Free energy per chain: ', np.round(res.fun,6), 'kT')

#find_saddle_point(lx)

# estimate execution time
time_duration = time.time() - time_start
print("total time: %f " % time_duration)

# save final results
phi_a, phi_b, Q = pseudo.find_phi(q1_init,q2_init,w[0],w[1])
mdic = {"dim":sb.get_dim(), "nx":sb.get_nx(), "lx":sb.get_lx(),
        "N":pc.get_n_contour(), "f":pc.get_f(), "chi_n":pc.get_chi_n(), "epsilon":pc.get_epsilon(),
        "chain_model":chain_model, "w_a":w[0], "w_b":w[1], "phi_a":phi_a, "phi_b":phi_b}
savemat("fields.mat", mdic)
