import sys
import os
import numpy as np
import time
from langevinfts import *

# -------------- initialize ------------

# OpenMP environment variables 
os.environ["KMP_STACKSIZE"] = "1G"
os.environ["MKL_NUM_THREADS"] = "1"  # always 1
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"  # 0, 1 or 2

# Iteration Parameter
max_scft_iter = 10000
tolerance = 1e-8

# Simulation Box
nx = [64,  64,  64]
lx = [9.0, 9.0, 9.0]/np.sqrt(6)

# Polymer Chain
NN = 100
f = 0.37
chi_n = 20
polymer_model = "Discrete" # choose among [Gaussian, Discrete]

# choose platform among [CUDA, CPU_MKL, CPU_FFTW]
factory = PlatformSelector.create_factory("CUDA")

# create instances and assign to the variables of base classs
# for the dynamic binding
pc = factory.create_polymer_chain(f, NN, chi_n)
sb = factory.create_simulation_box(nx, lx)
pseudo = factory.create_pseudo(sb, pc, polymer_model)

# arrays for exponential time differencing
space_kx, space_ky, space_kz = np.meshgrid(
    2*np.pi/sb.get_lx(1)*np.concatenate([np.arange((sb.get_nx(1)+1)//2), sb.get_nx(1)//2-np.arange(sb.get_nx(1)//2)]),
    2*np.pi/sb.get_lx(0)*np.concatenate([np.arange((sb.get_nx(0)+1)//2), sb.get_nx(0)//2-np.arange(sb.get_nx(0)//2)]),
    2*np.pi/sb.get_lx(2)*np.arange(sb.get_nx(2)//2+1))

mag2_k = (space_kx**2 + space_ky**2 + space_kz**2)/6.0
mag2_k[0,0,0] = 1.0e-5 # to prevent 'division by zero' error
g_k = 2*(mag2_k+np.exp(-mag2_k)-1.0)/mag2_k**2
g_k[0,0,0] = 1.0

lambda_plus = 100.0
lambda_minus = 20.0
exp_kernel_plus_k = (1.0 - np.exp(-g_k*lambda_plus))/g_k
exp_kernel_minus_k = (1.0 - np.exp(-(2/pc.get_chi_n())*lambda_minus))/(2/pc.get_chi_n())
print(exp_kernel_plus_k[0,0,0:5])
print(exp_kernel_minus_k)

# assign large initial value for the energy and error
energy_total = 1.0e20;
error_level = 1.0e20;

# -------------- print simulation parameters ------------
print("---------- Simulation Parameters ----------");
print("Box Dimension: %d" % (sb.get_dimension()))
print("Precision: 8")
print("chi_n: %f, f: %f, NN: %d" % (pc.get_chi_n(), pc.get_f(), pc.get_NN()) )
print("Nx: %d, %d, %d" % (sb.get_nx(0), sb.get_nx(1), sb.get_nx(2)) )
print("Lx: %f, %f, %f" % (sb.get_lx(0), sb.get_lx(1), sb.get_lx(2)) )
print("dx: %f, %f, %f" % (sb.get_dx(0), sb.get_dx(1), sb.get_dx(2)) )
print("volume: %f" % (sb.get_volume()) )

#-------------- allocate array ------------
phi_a   = np.zeros(sb.get_MM(), dtype=np.float64)
phi_b   = np.zeros(sb.get_MM(), dtype=np.float64)
q1_init = np.zeros(sb.get_MM(), dtype=np.float64)
q2_init = np.zeros(sb.get_MM(), dtype=np.float64)

#print("wminus and wplus are initialized to random")
w_minus = np.random.normal(0, 1, sb.get_MM())
w_plus  = np.random.normal(0, 1, sb.get_MM())

#print("wminus and wplus are initialized with an input file")
#input_data = np.load("fields_scft_EM1.npz")
#w = input_data['w']

# keep the level of field value
sb.zero_mean(w_minus);
sb.zero_mean(w_plus);

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
    QQ = pseudo.find_phi(phi_a, phi_b, q1_init,q2_init,w_plus+w_minus, w_plus-w_minus)
    
    # calculate the total energy
    energy_old = energy_total    
    energy_total  = -np.log(QQ/sb.get_volume())
    energy_total += sb.inner_product(w_minus,w_minus)/pc.get_chi_n()/sb.get_volume()
    energy_total -= sb.integral(w_plus)/sb.get_volume()
    
    # calculate output fields
    # stagger step 1 for w_plus
    g_plus = phi_a + phi_b - 1.0
    g_plus_k = np.fft.rfftn(np.reshape(g_plus, sb.get_nx()))
    w_plus_k = np.fft.rfftn(np.reshape(w_plus, sb.get_nx()))
    w_plus_k += exp_kernel_plus_k*g_plus_k
    w_plus = np.reshape(np.fft.irfftn(w_plus_k, sb.get_nx()), sb.get_MM())
    sb.zero_mean(w_plus)
    
    # stagger step 2 for w_minus
    QQ = pseudo.find_phi(phi_a, phi_b, q1_init,q2_init,w_plus+w_minus, w_plus-w_minus)
    g_minus = phi_a-phi_b + 2*w_minus/pc.get_chi_n()
    g_minus_k = np.fft.rfftn(np.reshape(g_minus, sb.get_nx()))
    w_minus_k = np.fft.rfftn(np.reshape(w_minus, sb.get_nx()))
    w_minus_k -= exp_kernel_minus_k*g_minus_k
    w_minus = np.reshape(np.fft.irfftn(w_minus_k, sb.get_nx()), sb.get_MM())
    sb.zero_mean(w_minus)
    
    # error_level measures the "relative distance" between the input and output fields
    old_error_level = error_level
    multi_dot = sb.inner_product(g_minus,g_minus) + sb.inner_product(g_plus,g_plus)
    multi_dot  /= sb.inner_product(w_minus,w_minus) + sb.inner_product(w_plus,w_plus) + 1.0
    error_level = np.sqrt(multi_dot)
    
    # print iteration # and error levels and check the mass conservation
    mass_error = (sb.integral(phi_a) + sb.integral(phi_b))/sb.get_volume() - 1.0
    print( "%8d %12.3E %15.7E %13.9f %13.9f" %
        (scft_iter, mass_error, QQ, energy_total, error_level) )

    # conditions to end the iteration
    if(error_level < tolerance):
        break;
        
# estimate execution time
time_duration = time.time() - time_start; 
print( "total time: %f, time per step: %f" %
    (time_duration, time_duration/max_scft_iter) )

# save final results
w = [w_plus + w_minus, w_plus - w_minus]
np.savez("fields_scft_ETD1.npz",
        nx=sb.get_nx(), lx=sb.get_lx(), N=pc.get_NN(), f=pc.get_f(), chi_n=pc.get_chi_n(),
        polymer_model=polymer_model, w=w, 
        phi_a=phi_a, phi_b=phi_b)
