import sys
import numpy as np
import time
from langevinfts import *

# -------------- initialize ------------
#pp = ParamParser.get_instance()
#pp.read_param_file(sys.argv[1], False);

max_scft_iter = 20
tolerance = 1e-9
n_comp = 2  # A and B

f = 0.3
NN = 50
chi_n = 20
nx = [31,49,63]
lx = [4.0,3.0,2.0]

am_max_hist= 20
am_start_error = 8e-1
am_mix_min = 0.1
am_mix_init = 0.1

# choose platform
factory = KernelFactory("CUDA")

# create instances and assign to the variables of base classs
# for the dynamic binding
pc = factory.create_polymer_chain(f, NN, chi_n)
sb = factory.create_simulation_box(nx, lx)
pseudo = factory.create_pseudo(sb, pc)
am = factory.create_anderson_mixing(sb, n_comp,
    am_max_hist, am_start_error, am_mix_min, am_mix_init)

# assign large initial value for the energy and error
energy_total = 1.0e20;
error_level = 1.0e20;

# -------------- print simulation parameters ------------
print("---------- Simulation Parameters ----------");
print("Box Dimension: 3")
print("Precision: 8")
print("chi_n: %f, f: %f, NN: %d" % (pc.chi_n, pc.f, pc.NN) )
print("Nx: %d, %d, %d" % (sb.nx[0], sb.nx[1], sb.nx[2]) )
print("Lx: %f, %f, %f" % (sb.lx[0], sb.lx[1], sb.lx[2]) )
print("dx: %f, %f, %f" % (sb.dx[0], sb.dx[1], sb.dx[2]) )

total = 0.0;
for i in range(0, sb.MM):
    total += sb.dv_at(i);
print("volume: %f, sum(dv): %f" % (sb.volume, total) )

#-------------- allocate array ------------
w       = np.zeros([2, sb.MM], dtype=np.float64)
w_out   = np.zeros([2, sb.MM], dtype=np.float64)
w_diff  = np.zeros([2, sb.MM], dtype=np.float64)
xi      = np.zeros(    sb.MM,  dtype=np.float64)
phia    = np.zeros(    sb.MM,  dtype=np.float64)
phib    = np.zeros(    sb.MM,  dtype=np.float64)
phitot  = np.zeros(    sb.MM,  dtype=np.float64)
w_plus  = np.zeros(    sb.MM,  dtype=np.float64)
w_minus = np.zeros(    sb.MM,  dtype=np.float64)
q1_init = np.zeros(    sb.MM,  dtype=np.float64)
q2_init = np.zeros(    sb.MM,  dtype=np.float64)

print("wminus and wplus are initialized to a given test fields.")
for i in range(0,sb.nx[0]):
    for j in range(0,sb.nx[1]):
        for k in range(0,sb.nx[2]):
            idx = i*sb.nx[1]*sb.nx[2] + j*sb.nx[2] + k;
            phia[idx]= np.cos(2.0*np.pi*i/4.68)*np.cos(2.0*np.pi*j/3.48)*np.cos(2.0*np.pi*k/2.74)*0.1;

phib[:] = 1.0 - phia[:];
w[0,:] = chi_n*phib[:];
w[1,:] = chi_n*phia[:];

# keep the level of field value
sb.zero_mean(w[0]);
sb.zero_mean(w[1]);

# free end initial condition. q1 is q and q2 is qdagger.
# q1 starts from A end and q2 starts from B end.
q1_init[:] = 1.0;
q2_init[:] = 1.0;

#------------------ run ----------------------
print("---------- Run ----------")
print("iteration, mass error, total_partition, energy_total, error_level")
time_start = time.time()
# iteration begins here
for scft_iter in range(0,max_scft_iter):
    # for the given fields find the polymer statistics
    QQ = pseudo.find_phi(phia, phib, q1_init,q2_init,w[0],w[1])
    
    # calculate the total energy
    energy_old = energy_total
    w_minus[:] = (w[0,:]-w[1,:])/2;
    w_plus[:]  = (w[0,:]+w[1,:])/2;
    
    energy_total  = -np.log(QQ/sb.volume)
    energy_total += sb.inner_product(w_minus,w_minus)/chi_n/sb.volume
    energy_total += sb.integral(w_plus)/sb.volume
    
    # calculate pressure field for the new field calculation, the method is modified from Fredrickson's
    xi[:] = 0.5*(w[0,:]+w[1,:]-chi_n)

    # calculate output fields
    w_out[0,:] = chi_n*phib[:] + xi[:];
    w_out[1,:] = chi_n*phia[:] + xi[:];
    sb.zero_mean(w_out[0]);
    sb.zero_mean(w_out[1]);
    
    # error_level measures the "relative distance" between the input and output fields
    old_error_level = error_level
    w_diff[:,:] = w_out[:,:]- w[:,:]
    multi_dot = sb.inner_product(w_diff[0],w_diff[0]) + sb.inner_product(w_diff[1],w_diff[1])
    multi_dot /= sb.inner_product(w[0],w[0]) + sb.inner_product(w[1],w[1]) + +1.0
    error_level = np.sqrt(multi_dot)
    
    # print iteration # and error levels and check the mass conservation
    mass_error = (sb.integral(phia) + sb.integral(phib))/sb.volume - 1.0
    print( "%8d %12.3E %15.7E %13.9f %13.9f" %
        (scft_iter, mass_error, QQ, energy_total, error_level) )

    # conditions to end the iteration
    if(error_level < tolerance):
        break;
    # calculte new fields using simple and Anderson mixing
    # (Caution! We are passing w, w_out and w_diff[0] not just w[0], w_out[0], w_diff[0])
    am.caculate_new_fields(w[0], w_out[0], w_diff[0], old_error_level, error_level);

# estimate execution time
time_duration = time.time() - time_start; 
print( "total time: %f, time per step: %f" %
    (time_duration, time_duration/max_scft_iter) )
