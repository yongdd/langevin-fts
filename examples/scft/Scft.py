import sys
import os
import numpy as np
import time
from scipy.io import savemat
from langevinfts import *

# -------------- initialize ------------

# OpenMP environment variables 
os.environ["MKL_NUM_THREADS"] = "1"  # always 1
os.environ["OMP_STACKSIZE"] = "1G"
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "0"  # 0, 1 or 2

max_scft_iter = 20
tolerance = 1e-9

f = 0.3
n_contour = 50
chi_n = 20
nx = [31,49,63]
lx = [4.0,3.0,2.0] # as aN^(1/2) unit

### for 2D
# nx = [31,49]
# lx = [4.0,3.0] 

### for 1D
#nx = [31]
#lx = [4.0] 

chain_model = "Discrete" # choose among [Gaussian, Discrete]

am_n_comp = 2  # w[0] and w[1]
am_max_hist= 20
am_start_error = 8e-1
am_mix_min = 0.1
am_mix_init = 0.1

# choose platform among [cuda, cpu-mkl, cpu-fftw]
print("Available Platforms: ", PlatformSelector.avail_platforms())
factory = PlatformSelector.create_factory("cuda")

# create instances
pc = factory.create_polymer_chain(f, n_contour, chi_n, chain_model)
sb = factory.create_simulation_box(nx, lx)
pseudo = factory.create_pseudo(sb, pc)
am = factory.create_anderson_mixing(sb, am_n_comp,
    am_max_hist, am_start_error, am_mix_min, am_mix_init)

# assign large initial value for the energy and error
energy_total = 1.0e20;
error_level = 1.0e20;

# -------------- print simulation parameters ------------
print("---------- Simulation Parameters ----------");
print("Box Dimension: %d" % (sb.get_dim()))
print("Precision: 8")
print("chi_n: %f, f: %f, N: %d" % (pc.get_chi_n(), pc.get_f(), pc.get_n_contour()) )
print("%s chain model" % (pc.get_model_name()) )
print("Nx: %d, %d, %d" % (sb.get_nx(0), sb.get_nx(1), sb.get_nx(2)) )
print("Lx: %f, %f, %f" % (sb.get_lx(0), sb.get_lx(1), sb.get_lx(2)) )
print("dx: %f, %f, %f" % (sb.get_dx(0), sb.get_dx(1), sb.get_dx(2)) )
print("Volume: %f" % (sb.get_volume()) )

#-------------- allocate array ------------
# free end initial condition. q1 is q and q2 is qdagger.
# q1 starts from A end and q2 starts from B end.
w       = np.zeros([2, sb.get_n_grid()], dtype=np.float64)
w_out   = np.zeros([2, sb.get_n_grid()], dtype=np.float64)
phi_a   = np.zeros(    sb.get_n_grid(),  dtype=np.float64)
phi_b   = np.zeros(    sb.get_n_grid(),  dtype=np.float64)
q1_init = np.ones (    sb.get_n_grid(),  dtype=np.float64)
q2_init = np.ones (    sb.get_n_grid(),  dtype=np.float64)

print("w_minus and w_plus are initialized to a given test fields.")
for i in range(0,sb.get_nx(0)):
    for j in range(0,sb.get_nx(1)):
        for k in range(0,sb.get_nx(2)):
            idx = i*sb.get_nx(1)*sb.get_nx(2) + j*sb.get_nx(2) + k;
            phi_a[idx]= np.cos(2.0*np.pi*i/4.68)*np.cos(2.0*np.pi*j/3.48)*np.cos(2.0*np.pi*k/2.74)*0.1;

phi_b = 1.0 - phi_a;
w[0] = chi_n*phi_b;
w[1] = chi_n*phi_a;

# keep the level of field value
sb.zero_mean(w[0]);
sb.zero_mean(w[1]);

#------------------ run ----------------------
print("---------- Run ----------")
print("iteration, mass error, total_partition, energy_total, error_level")
time_start = time.time()
# iteration begins here
for scft_iter in range(0,max_scft_iter):
    # for the given fields find the polymer statistics
    QQ = pseudo.find_phi(phi_a, phi_b, q1_init,q2_init,w[0],w[1])
    
    # calculate the total energy
    energy_old = energy_total
    w_minus = (w[0]-w[1])/2;
    w_plus  = (w[0]+w[1])/2;
    
    energy_total  = -np.log(QQ/sb.get_volume())
    energy_total += sb.inner_product(w_minus,w_minus)/pc.get_chi_n()/sb.get_volume()
    energy_total -= sb.integral(w_plus)/sb.get_volume()
    
    # calculate pressure field for the new field calculation, the method is modified from Fredrickson's
    xi = 0.5*(w[0]+w[1]-chi_n)

    # calculate output fields
    w_out[0] = chi_n*phi_b + xi;
    w_out[1] = chi_n*phi_a + xi;
    sb.zero_mean(w_out[0]);
    sb.zero_mean(w_out[1]);
    
    # error_level measures the "relative distance" between the input and output fields
    old_error_level = error_level
    w_diff = w_out - w
    multi_dot = sb.inner_product(w_diff[0],w_diff[0]) + sb.inner_product(w_diff[1],w_diff[1])
    multi_dot /= sb.inner_product(w[0],w[0]) + sb.inner_product(w[1],w[1]) + +1.0
    error_level = np.sqrt(multi_dot)
    
    # print iteration # and error levels and check the mass conservation
    mass_error = (sb.integral(phi_a) + sb.integral(phi_b))/sb.get_volume() - 1.0
    print( "%8d %12.3E %15.7E %13.9f %13.9f" %
        (scft_iter, mass_error, QQ, energy_total, error_level) )

    # conditions to end the iteration
    if(error_level < tolerance):
        break;
    # calculte new fields using simple and Anderson mixing
    # (Note! we are now passing entire w, w_out and w_diff not just w[0], w_out[0] and w_diff[0])
    am.caculate_new_fields(w[0], w_out[0], w_diff[0], old_error_level, error_level);

# estimate execution time
time_duration = time.time() - time_start; 
print( "total time: %f, time per step: %f" %
    (time_duration, time_duration/max_scft_iter) )

# save final results
mdic = {"dim":sb.get_dim(), "nx":sb.get_nx(), "lx":sb.get_lx(),
        "N":pc.get_n_contour(), "f":pc.get_f(), "chi_n":pc.get_chi_n(),
        "chain_model":chain_model, "w_a":w[0], "w_b":[1], "phi_a":phi_a, "phi_b":phi_b}
savemat("fields.mat", mdic)
