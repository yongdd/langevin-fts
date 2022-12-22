import os
import numpy as np
import time
from scipy.io import loadmat, savemat
from scipy.ndimage.filters import gaussian_filter
from langevinfts import *


def find_saddle_point(cb, mixture, pseudo, am, lx, chi_n, w, max_iter, tolerance, is_box_altering=True):

    # assign large initial value for the energy and error
    energy_total = 1.0e20
    error_level = 1.0e20

    # reset Anderson mixing module
    am.reset_count()

    # array for output fields
    w_out = np.zeros([2, cb.get_n_grid()], dtype=np.float64)

    # iteration begins here
    if (is_box_altering):
        print("iteration, mass error, total_partition, energy_total, error_level, box size")
    else:
        print("iteration, mass error, total_partition, energy_total, error_level")
    
    for scft_iter in range(1,max_iter+1):
        # for the given fields find the polymer statistics
        pseudo.compute_statistics({"A":w[0],"B":w[1]})

        phi_a = pseudo.get_monomer_concentration("A")
        phi_b = pseudo.get_monomer_concentration("B")

        # calculate the total energy, Hamiltonian
        w_minus = (w[0]-w[1])/2
        w_plus  = (w[0]+w[1])/2

        energy_total = cb.inner_product(w_minus,w_minus)/chi_n/cb.get_volume()
        energy_total -= cb.integral(w_plus)/cb.get_volume()
        for p in range(mixture.get_n_polymers()):
            energy_total  -= np.log(pseudo.get_total_partition(p)/cb.get_volume())

        # calculate pressure field for the new field calculation, the method is modified from Fredrickson's
        xi = 0.5*(w[0]+w[1]-chi_n)

        # calculate output fields
        w_out[0] = chi_n*phi_b + xi
        w_out[1] = chi_n*phi_a + xi

        # error_level measures the "relative distance" between the input and output fields
        old_error_level = error_level
        w_diff = w_out - w

        # keep the level of field value
        cb.zero_mean(w_diff[0])
        cb.zero_mean(w_diff[1])

        multi_dot = cb.inner_product(w_diff[0],w_diff[0]) + cb.inner_product(w_diff[1],w_diff[1])
        multi_dot /= cb.inner_product(w[0],w[0]) + cb.inner_product(w[1],w[1]) + 1.0

        # print iteration # and error levels and check the mass conservation
        mass_error = (cb.integral(phi_a) + cb.integral(phi_b))/cb.get_volume() - 1.0
        error_level = np.sqrt(multi_dot)
        
        if (is_box_altering):
            # Calculate stress
            stress_array = np.array(pseudo.compute_stress())
            error_level += np.sqrt(np.sum(stress_array**2))
            print("%8d %12.3E " % (scft_iter, mass_error), end=" [ ")
            for p in range(mixture.get_n_polymers()):
                print("%13.7E " % (pseudo.get_total_partition(p)), end=" ")
            print("] %15.9f %15.7E " % (energy_total, error_level), end=" ")
            print("[", ",".join(["%10.7f" % (x) for x in cb.get_lx()[-cb.get_dim():]]), "]")

        else:
            print("%8d %12.3E " %
            (scft_iter, mass_error), end=" [ ")
            for p in range(mixture.get_n_polymers()):
                print("%13.7E " % (pseudo.get_total_partition(p)), end=" ")
            print("] %15.9f %15.7E " % (energy_total, error_level), end=" ")

        # conditions to end the iteration
        if error_level < tolerance:
            break

        # calculate new fields using simple and Anderson mixing
        if (is_box_altering):
            am_current = np.concatenate((np.reshape(w,      2*cb.get_n_grid()), lx))
            am_diff    = np.concatenate((np.reshape(w_diff, 2*cb.get_n_grid()), -stress_array))
            am_new = am.calculate_new_fields(am_current, am_diff, old_error_level, error_level)

            # copy fields
            w = np.reshape(am_new[0:2*cb.get_n_grid()], (2, cb.get_n_grid()))

            # set box size
            # restricting |dLx| to be less than 10 % of Lx
            old_lx = lx
            new_lx = np.array(am_new[-cb.get_dim():])
            new_dlx = np.clip((new_lx-old_lx)/old_lx, -0.1, 0.1)
            lx = (1 + new_dlx)*old_lx
            cb.set_lx(lx)

            # update bond parameters using new lx
            pseudo.update()
        else:
            w = am.calculate_new_fields(
            np.reshape(w,      2*cb.get_n_grid()),
            np.reshape(w_diff, 2*cb.get_n_grid()), old_error_level, error_level)
            w = np.reshape(w, (2, cb.get_n_grid()))

    # get total partition functions
    Q = []
    for p in range(mixture.get_n_polymers()):
        Q.append(pseudo.get_total_partition(p))

    return phi_a, phi_b, Q, energy_total

# -------------- initialize ------------

# OpenMP environment variables
os.environ["MKL_NUM_THREADS"] = "1"  # always 1
os.environ["OMP_STACKSIZE"] = "1G"
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "2"  # 0, 1 or 2
os.environ["OMP_NUM_THREADS"] = "2"  # 1 ~ 4

max_scft_iter = 2000
tolerance = 1e-8

# major Simulation Parameters
f = 0.25            # A-fraction, f
chi_n = 25           # Flory-Huggins Parameters * N
epsilon = 2.0        # a_A/a_B, conformational asymmetry
nx = [64,64,32]      # grid numbers
lx = [7.0,7.0,4.0]   # as aN^(1/2) unit
ds = 1/100           # contour step interval
chain_model = "Continuous"    # choose among [Continuous, Discrete]

# Anderson mixing
am_n_var = 2*np.prod(nx)+len(lx)  # w_a (w[0]) and w_b (w[1]) + lx
am_max_hist= 20                   # maximum number of history
am_start_error = 1e-2             # when switch to AM from simple mixing
am_mix_min = 0.1                  # minimum mixing rate of simple mixing
am_mix_init = 0.1                 # initial mixing rate of simple mixing

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
factory = PlatformSelector.create_factory(platform, chain_model)

# create instances
cb = factory.create_computation_box(nx, lx)
mixture = factory.create_mixture(ds, dict_a_n)
mixture.add_polymer(1.0, ["A","B"], [f, 1-f], [0, 1], [1, 2])
pseudo = factory.create_pseudo(cb, mixture)
am = factory.create_anderson_mixing(am_n_var,
            am_max_hist, am_start_error, am_mix_min, am_mix_init)

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

#-------------- allocate array ------------
# free end initial condition. q1 is q and q2 is qdagger.
# q1 starts from A end and q2 starts from B end.
w       = np.zeros([2]+list(cb.get_nx()), dtype=np.float64)
q1_init = np.ones (    cb.get_n_grid(),   dtype=np.float64)
q2_init = np.ones (    cb.get_n_grid(),   dtype=np.float64)

# initial fields
print("w_A and w_B are initialized to Sigma phase.")
# [Ref: https://doi.org/10.3390/app2030654]
sphere_positions = [[0.00,0.00,0.00],[0.50,0.50,0.50], #A
[0.40,0.40,0.00],[0.60,0.60,0.00],[0.10,0.90,0.50],[0.90,0.10,0.50], #B
[0.13,0.46,0.00],[0.46,0.13,0.00],[0.54,0.87,0.00],[0.87,0.54,0.00], #C
[0.04,0.63,0.50],[0.63,0.04,0.50],[0.37,0.96,0.50],[0.96,0.37,0.50], #C
[0.07,0.74,0.00],[0.74,0.07,0.00],[0.26,0.93,0.00],[0.93,0.26,0.00], #D
[0.24,0.43,0.50],[0.43,0.24,0.50],[0.57,0.76,0.50],[0.77,0.56,0.50], #D
[0.18,0.18,0.25],[0.82,0.82,0.25],[0.32,0.68,0.25],[0.68,0.32,0.25], #E
[0.18,0.18,0.75],[0.82,0.82,0.75],[0.32,0.68,0.75],[0.68,0.32,0.75]] #E

for x,y,z in sphere_positions:
    _mx, _my, _mz = np.round((np.array([x, y, z])*cb.get_nx())).astype(np.int32)
    w[0,_mx,_my,_mz] = -1/np.prod(cb.get_dx())
w[0] = gaussian_filter(w[0], sigma=np.min(cb.get_nx())/15, mode='wrap')
w = np.reshape(w, [2, cb.get_n_grid()])

# keep the level of field value
cb.zero_mean(w[0])
cb.zero_mean(w[1])

#------------------ run ----------------------
print("---------- Run ----------")
time_start = time.time()

phi_a, phi_b, Q, energy_total = find_saddle_point(cb, mixture, pseudo, am, lx, chi_n,
    w, max_scft_iter, tolerance, is_box_altering=True)

# estimate execution time
time_duration = time.time() - time_start
print("total time: %f " % time_duration)

# save final results
mdic = {"dim":cb.get_dim(), "nx":cb.get_nx(), "lx":cb.get_lx(),
        "N":mixture.get_polymer(0).get_n_segment_total(), "f":f, "chi_n":chi_n, "epsilon":epsilon,
        "chain_model":chain_model, "w_a":w[0], "w_b":w[1], "phi_a":phi_a, "phi_b":phi_b}
savemat("fields.mat", mdic)

# Recording first a few iteration results for debugging and refactoring
    #    1   -4.663E-15  [ 1.9678400E+02  ]    -0.001859583   1.9215121E+00  [  7.0000000, 7.0000000, 4.0000000 ]
    #    2    8.216E-15  [ 1.9643500E+02  ]    -0.000570350   1.1293870E+00  [  7.0000203, 7.0000203, 4.0000299 ]
    #    3    6.661E-16  [ 1.9632596E+02  ]    -0.000238806   6.6231791E-01  [  7.0000294, 7.0000294, 4.0000421 ]
    #    4   -5.107E-15  [ 1.9628576E+02  ]    -0.000146152   4.2737822E-01  [  7.0000346, 7.0000346, 4.0000485 ]
    #    5    8.882E-16  [ 1.9626999E+02  ]    -0.000121015   3.3566219E-01  [  7.0000379, 7.0000379, 4.0000524 ]
