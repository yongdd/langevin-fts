import os
import numpy as np
import time
from scipy.io import savemat
import scipy.optimize
from langevinfts import *

def find_saddle_point(lx):

    # set box size
    cb.set_lx(lx)
    # update bond parameters using new lx
    solver.update_laplacian_operator()

    # assign large initial value for the energy and error
    energy_total = 1.0e20
    error_level = 1.0e20

    # reset Anderson mixing module
    am.reset_count()

    # iteration begins here
    print("iteration, mass error, total_partition, energy_total, error_level")
    for scft_iter in range(1,max_scft_iter+1):
        # for the given fields find the polymer statistics
        phi, Q = solver.compute_statistics(q1_init,q2_init,{"A":w[0],"B":w[1]})

        # calculate the total energy
        energy_old = energy_total
        w_minus = (w[0]-w[1])/2
        w_plus  = (w[0]+w[1])/2

        energy_total  = -np.log(Q/cb.get_volume())
        energy_total += cb.inner_product(w_minus,w_minus)/chi_n/cb.get_volume()
        energy_total -= cb.integral(w_plus)/cb.get_volume()

        # calculate pressure field for the new field calculation, the method is modified from Fredrickson's
        xi = 0.5*(w[0]+w[1]-chi_n)

        # calculate output fields
        w_out[0] = chi_n*phi[1] + xi
        w_out[1] = chi_n*phi[0] + xi
        cb.zero_mean(w_out[0])
        cb.zero_mean(w_out[1])

        # error_level measures the "relative distance" between the input and output fields
        old_error_level = error_level
        w_diff = w_out - w
        multi_dot = cb.inner_product(w_diff[0],w_diff[0]) + cb.inner_product(w_diff[1],w_diff[1])
        multi_dot /= cb.inner_product(w[0],w[0]) + cb.inner_product(w[1],w[1]) + 1.0
        error_level = np.sqrt(multi_dot)

        # print iteration # and error levels and check the mass conservation
        mass_error = (cb.integral(phi[0]) + cb.integral(phi[1]))/cb.get_volume() - 1.0
        print( "%8d %12.3E %15.7E %13.9f %15.11f" %
            (scft_iter, mass_error, Q, energy_total, error_level) )

        # conditions to end the iteration
        if error_level < tolerance:
            break
        # calculte new fields using simple and Anderson mixing
        am.calculate_new_fields(
            np.reshape(w,      2*cb.get_total_grid()),
            np.reshape(w_out,  2*cb.get_total_grid()),
            np.reshape(w_diff, 2*cb.get_total_grid()),
            old_error_level, error_level)
    
    if use_stress:
        stress_array = np.array(solver.compute_stress()[-cb.get_dim():])/Q
        return stress_array
    else:
        return energy_total

# -------------- initialize ------------

# OpenMP environment variables
os.environ["MKL_NUM_THREADS"] = "1"  # always 1
os.environ["OMP_STACKSIZE"] = "1G"
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "2"  # 0, 1 or 2
os.environ["OMP_NUM_THREADS"] = "2"  # 1 ~ 4

max_scft_iter = 2000
tolerance = 1e-11

# Major Simulation Parameters
f = 0.30              # A-fraction, f
chi_n = 25            # Flory-Huggins Parameters * N
epsilon = 2.0         # a_A/a_B, conformational asymmetry
nx = [32]#,32,32]     # grids number
lx = [3.4]#,3.5,3.7]  # as aN^(1/2) unit, a = sqrt(f*a_A^2 + (1-f)*a_B^2)
ds = 1/100            # contour step interval
chain_model = "Continuous" # choose among [Continuous, Discrete]

am_n_var = 2*np.prod(nx) # w_a (w[0]) and w_b (w[1])
am_max_hist= 20          # maximum number of history
am_start_error = 1e-1    # when switch to AM from simple mixing
am_mix_min = 0.1         # minimum mixing rate of simple mixing
am_mix_init = 0.1        # initial mixing rate of simple mixing

# use stress for finding unit cell
use_stress = True

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
pc     = factory.create_polymer_chain(["A","B"], [f, 1-f], dict_a_n, ds)
cb     = factory.create_computation_box(nx, lx)
solver = factory.create_pseudospectral_solver(cb, pc)
am     = factory.create_anderson_mixing(am_n_var,
            am_max_hist, am_start_error, am_mix_min, am_mix_init)

# -------------- print simulation parameters ------------
print("---------- Simulation Parameters ----------")
print("Box Dimension: %d" % (cb.get_dim()))
print("chi_n: %f, f: %f, N: %d" % (chi_n, f, pc.get_n_segment_total()) )
print("%s chain model" % (pc.get_model_name()) )
print("Conformational asymmetry (epsilon): %f" % (epsilon) )
print("Nx: %d, %d, %d" % (cb.get_nx(0), cb.get_nx(1), cb.get_nx(2)) )
print("Lx: %f, %f, %f" % (cb.get_lx(0), cb.get_lx(1), cb.get_lx(2)) )
print("dx: %f, %f, %f" % (cb.get_dx(0), cb.get_dx(1), cb.get_dx(2)) )
print("Volume: %f" % (cb.get_volume()) )

#-------------- allocate array ------------
# free end initial condition. q1 is q and q2 is qdagger.
# q1 starts from A end and q2 starts from B end.
w       = np.zeros([2]+list(cb.get_nx()), dtype=np.float64)
w_out   = np.zeros([2, cb.get_total_grid()],  dtype=np.float64)
q1_init = np.ones (    cb.get_total_grid(),   dtype=np.float64)
q2_init = np.ones (    cb.get_total_grid(),   dtype=np.float64)

print("w_A and w_B are initialized to gyroid phase.")
for i in range(0,cb.get_nx(0)):
    xx = (i+1)*2*np.pi/cb.get_nx(0)
    for j in range(0,cb.get_nx(1)):
        yy = (j+1)*2*np.pi/cb.get_nx(1)
        for k in range(0,cb.get_nx(2)):
            zz = (k+1)*2*np.pi/cb.get_nx(2)
            c1 = np.sqrt(8.0/3.0)*(np.cos(xx)*np.sin(yy)*np.sin(2.0*zz) +
                np.cos(yy)*np.sin(zz)*np.sin(2.0*xx)+np.cos(zz)*np.sin(xx)*np.sin(2.0*yy))
            c2 = np.sqrt(4.0/3.0)*(np.cos(2.0*xx)*np.cos(2.0*yy)+
                np.cos(2.0*yy)*np.cos(2.0*zz)+np.cos(2.0*zz)*np.cos(2.0*xx))
            idx = i*cb.get_nx(1)*cb.get_nx(2) + j*cb.get_nx(2) + k
            w[0,i,j,k] = -0.364*c1+0.133*c2
            w[1,i,j,k] = 0.302*c1-0.106*c2
w = np.reshape(w, [2, cb.get_total_grid()])

# keep the level of field value
cb.zero_mean(w[0])
cb.zero_mean(w[1])

#------------------ run ----------------------
print("---------- Run ----------")
time_start = time.time()

# find the natural period of gyroid
if (use_stress):
    res = scipy.optimize.root(find_saddle_point, lx, tol=1e-6)
    print('Unit cell that make the stress zero: ', res.x, '(aN^1/2)')
    print('Stress in each direction: ', res.fun, )
else:
    res = scipy.optimize.minimize(find_saddle_point, lx, tol=1e-6, options={'disp':True})
    print('Unit cell that minimizes the free energy: ', res.x, '(aN^1/2)')
    print('Free energy per chain: ', res.fun, 'kT')

# estimate execution time
time_duration = time.time() - time_start
print("total time: %f " % time_duration)