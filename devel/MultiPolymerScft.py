# For the start, change "Major Simulation Parameters", currently in lines 20-27
# and "Initial Fields", currently in lines 70-84
import os
import numpy as np
import time
from scipy.io import savemat
from langevinfts import *

# -------------- initialize ------------

# OpenMP environment variables
os.environ["MKL_NUM_THREADS"] = "1"  # always 1
os.environ["OMP_STACKSIZE"] = "1G"
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "2"  # 0, 1 or 2

max_scft_iter = 1000
tolerance = 1e-8

# Example number
# Change this parameter to run different examples
example_number = 2

# Major Simulation Parameters
f = 0.5             # A-fraction of major BCP chain, f
n_segment = 90     # segment number of major BCP chain, N
chi_n = 13.27        # Flory-Huggins Parameters * N
epsilon = 1.0       # a_A/a_B, conformational asymmetry
nx = [32,32,32]     # grid numbers
lx = [4.36,4.36,4.36]  # as aN^(1/2) unit, a = sqrt(f*a_A^2 + (1-f)*a_B^2)
chain_model = "Discrete" # choose among [Continuous, Discrete]

frac_bcp = 0.7      # volume fraction of major BCP chain
                    # second polymer chain parameters are defined below

# Anderson mixing
am_n_var = 2*np.prod(nx)          # w_a (w[0]) and w_b (w[1])
am_max_hist= 20                   # maximum number of history
am_start_error = 1e-2             # when switch to AM from simple mixing
am_mix_min = 0.1                  # minimum mixing rate of simple mixing
am_mix_init = 0.1                 # initial mixing rate of simple mixing

# choose platform among [cuda, cpu-mkl]
if "cuda" in PlatformSelector.avail_platforms():
    platform = "cuda"
else:
    platform = PlatformSelector.avail_platforms()[0]
print("platform :", platform)
computation = SingleChainStatistics.create_computation(platform, chain_model)

# create instances
cb     = computation.create_computation_box(nx, lx)
## create diblock copolymer with a_A, a_B such that  a = sqrt(f*a_A^2 + (1-f)*a_B^2), a_A/a_B = epsilon
## a_A_sq_n = a_A^2*N, a_Bq_n = a_B^2*N 
a_A_sq_n = epsilon*epsilon/(f*epsilon*epsilon + (1.0-f))
a_B_sq_n = 1.0/(f*epsilon*epsilon + (1.0-f))
N_pc = [int(f*n_segment),int((1-f)*n_segment)]
pc = computation.create_polymer_chain(N_pc, [a_A_sq_n, a_A_sq_n])

###### Example 01 ######
## this code adds homopolymer A with length N_A = 45
## homopolymer A takes 30 percent of the system (70 percent of bcp)
## it uses previously calculated a_A_sq_n
if example_number == 1:
    N_homo = 45
    pc_homo_a = computation.create_polymer_chain([N_homo], [a_A_sq_n])

###### Example 02 ######
## this code adds A,B random copolymer with A fraction of 0.7, length = N_rcp = 70
## you need to calculate effective bond length manually if two monomers have different bond lengths
if example_number == 2:
    N_rcp = 70
    random_A_frac = 0.7
    a_R_sq_n = random_A_frac*a_A_sq_n + (1-random_A_frac)*a_B_sq_n
    pc_rand = computation.create_polymer_chain([N_rcp], [a_R_sq_n])

###### Example 03 ######
## this code adds another A,B block copolymer with fraction f2 = 0.4, length = N2 = 45
## additional chain shares previously calculated bond_len_a/a_B_sq_n
if example_number == 3:
    f2 = 0.4
    N2 = 45
    N_pc2 = [int(f2*N2),int((1-f2)*N2)]
    pc2 = computation.create_polymer_chain(N_pc2, [a_A_sq_n,a_B_sq_n])

## pseudo should be created for each chain used in simulation
pseudo = computation.create_pseudo(cb, pc)
###### Example 01 ######
if example_number == 1:
    pseudo_homo_a = computation.create_pseudo(cb, pc_homo_a)
###### Example 02 ######
if example_number == 2:
    pseudo_rand   = computation.create_pseudo(cb, pc_rand)
###### Example 03 ######
if example_number == 3:
    pseudo_pc2    = computation.create_pseudo(cb, pc2)
am     = computation.create_anderson_mixing(am_n_var,
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
q1_init = np.ones (    cb.get_n_grid(),   dtype=np.float64)
q2_init = np.ones (    cb.get_n_grid(),   dtype=np.float64)
phi     = np.zeros((2, cb.get_n_grid()),  dtype=np.float64)
# Initial Fields
print("w_A and w_B are initialized to lamellar phase.")
for i in range(0,cb.get_nx(2)):
    w[0,:,:,i] =  np.cos(3*2*np.pi*i/cb.get_nx(2))
    w[1,:,:,i] = -np.cos(3*2*np.pi*i/cb.get_nx(2))
w = np.reshape(w, [2, cb.get_n_grid()])

# keep the level of field value
cb.zero_mean(w[0])
cb.zero_mean(w[1])

#------------------ run ----------------------
print("---------- Run ----------")
time_start = time.time()

# assign large initial value for the energy and error
energy_total = 1.0e20
error_level = 1.0e20

# reset Anderson mixing module
am.reset_count()

# array for output fields
w_out = np.zeros([2, cb.get_n_grid()], dtype=np.float64)

# iteration begins here
print("iteration, mass error, total_partition, energy_total, error_level")
for scft_iter in range(1,max_scft_iter+1):
    # for the given fields find the polymer statistics
    phi_bcp, Q_bcp = pseudo.compute_statistics(q1_init,q2_init,w)
    phi_bcp = phi_bcp.reshape(pc.get_n_block(),cb.get_n_grid())
    ###### Example 01 ######
    if example_number == 1:
        phi_homo_a, Q_homo_a = pseudo_homo_a.compute_statistics(q1_init,q2_init,w[0])
        phi[0] = phi_bcp[0]*frac_bcp + phi_homo_a*(1.0-frac_bcp)
        phi[1] = phi_bcp[1]*frac_bcp
    ###### Example 02 ######
    if example_number == 2:
        phi_rand, Q_rand = pseudo_rand.compute_statistics(q1_init,q2_init,random_A_frac*w[0]+(1.0-random_A_frac)*w[1])
        phi[0] = phi_bcp[0]*frac_bcp + phi_rand*random_A_frac*(1.0-frac_bcp)
        phi[1] = phi_bcp[1]*frac_bcp + phi_rand*(1.0-random_A_frac)*(1.0-frac_bcp)
    ###### Example 03 ######
    if example_number == 3:
        phi_bcp_2, Q_bcp_2 = pseudo_pc2.compute_statistics(q1_init,q2_init,w)
        phi[0] = phi_bcp[0]*frac_bcp + phi_bcp_2[0]*(1.0-frac_bcp)
        phi[1] = phi_bcp[1]*frac_bcp + phi_bcp_2[1]*(1.0-frac_bcp)
    # calculate the total energy
    w_minus = (w[0]-w[1])/2
    w_plus  = (w[0]+w[1])/2

    energy_total = -frac_bcp*(np.log(Q_bcp/cb.get_volume()))
    energy_total += cb.inner_product(w_minus,w_minus)/chi_n/cb.get_volume()
    energy_total -= cb.integral(w_plus)/cb.get_volume()
    
    ###### Example 01 ######
    if example_number == 1:
        n_segment_ratio  = N_homo/n_segment
        energy_total -= ((1.0-frac_bcp)/n_segment_ratio)*(np.log(Q_homo_a/cb.get_volume()))
    ###### Example 02 ######
    if example_number == 2:
        n_segment_ratio = N_rcp/n_segment
        energy_total -= ((1.0-frac_bcp)/n_segment_ratio)*(np.log(Q_rand/cb.get_volume()))
    ###### Example 03 ######
    if example_number == 3:
        n_segment_ratio = N2/n_segment
        energy_total -= ((1.0-frac_bcp)/n_segment_ratio)*(np.log(Q_bcp_2/cb.get_volume()))

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

    # print iteration # and error levels and check the mass conservation
    mass_error = (cb.integral(phi[0]) + cb.integral(phi[1]))/cb.get_volume() - 1.0
    error_level = np.sqrt(multi_dot)
    if example_number == 1:
        print("%8d %12.3E %15.7E %15.7E %15.9f %15.7E" %
        (scft_iter, mass_error, Q_bcp, Q_homo_a, energy_total, error_level))
    ###### Example 02 ######
    if example_number == 2:
        print("%8d %12.3E %15.7E %15.7E %15.9f %15.7E" %
        (scft_iter, mass_error, Q_bcp, Q_rand, energy_total, error_level))
    ###### Example 03 ######
    if example_number == 3:
        print("%8d %12.3E %15.7E %15.7E %15.9f %15.7E" %
        (scft_iter, mass_error, Q_bcp, Q_bcp_2, energy_total, error_level))

    # conditions to end the iteration
    if error_level < tolerance:
        break

    # calculte new fields using simple and Anderson mixing
    am.caculate_new_fields(
    np.reshape(w,      2*cb.get_n_grid()),
    np.reshape(w_out,  2*cb.get_n_grid()),
    np.reshape(w_diff, 2*cb.get_n_grid()),
    old_error_level, error_level)

# estimate execution time
time_duration = time.time() - time_start
print("total time: %f " % time_duration)

# save final results
mdic = {"dim":cb.get_dim(), "nx":cb.get_nx(), "lx":cb.get_lx(),
        "N":pc.get_n_segment_total(), "f":f, "chi_n":chi_n, "epsilon":epsilon,
        "chain_model":chain_model, "w_a":w[0], "w_b":w[1], "phi_a":phi[0], "phi_b":phi[1]}
savemat("fields.mat", mdic)