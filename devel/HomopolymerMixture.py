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

# Major Simulation Parameters
frac_A = 0.5           # A polymer volume fraction
n_segment_A = 50       # segment number of A polymer, N_A
n_segment_B = 50       # segment number of B polymer, N_B
chi_n = 5          # Flory-Huggins Parameters * N_B
epsilon = 3          # a_A/a_B, conformational asymmetry
nx = [32,32,32]        # grid numbers
lx = [4.36,4.36,4.36]  # as a_B*N_B^(1/2) unit
chain_model = "Discrete" # choose among [Continuous, Discrete]


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
simulation = FieldTheoreticSimulation.create_simulation(platform, chain_model)

# create instances
sb     = simulation.create_simulation_box(nx, lx)
## create diblock copolymer with a_A, a_B such that  a = sqrt(f*a_A^2 + (1-f)*a_B^2), a_A/a_B = epsilon

bond_length_b = np.power(n_segment_B,-0.5) # a_B*N_B^(1/2) = 1
bond_length_a = epsilon * bond_length_b    # epsilon = a_A/a_B

pc_A = simulation.create_polymer_chain([n_segment_A], [bond_length_a])
pc_B = simulation.create_polymer_chain([n_segment_B], [bond_length_b])

###### Additional example ######
## this code adds AB random copolymer instead of polymer A
## for this example, please comment every lines using polymer chain A
#
# frac_rcp = frac_A    # RCP fraction
# rcp_A_frac = 0.5     # fraction of A monomer in RCP
# bond_length_rcp = np.sqrt(rcp_A_frac*bond_length_a**2 + (1-rcp_A_frac)*bond_length_b**2)
# n_segment_rcp = n_segment_A
# pc_rcp = simulation.create_polymer_chain([n_segment_rcp], [bond_length_rcp])
##

## pseudo should be created for each chain used in simulation
pseudo_A = simulation.create_pseudo(sb, pc_A)
pseudo_B = simulation.create_pseudo(sb, pc_B)

###### Additional example ######
# pseudo_rcp   = simulation.create_pseudo(sb, pc_rcp)

am     = simulation.create_anderson_mixing(am_n_var,
            am_max_hist, am_start_error, am_mix_min, am_mix_init)

# -------------- print simulation parameters ------------
print("---------- Simulation Parameters ----------")
print("Box Dimension: %d" % (sb.get_dim()))
print("chi_n: %f, f: %f, N_A: %d, N_B: %d" % (chi_n, frac_A, pc_A.get_n_segment_total(), pc_B.get_n_segment_total()) ) 
print("%s chain model" % (simulation.get_model_name()) )
print("Conformational asymmetry (epsilon): %f" % (epsilon) )
print("Nx: %d, %d, %d" % (sb.get_nx(0), sb.get_nx(1), sb.get_nx(2)) )
print("Lx: %f, %f, %f" % (sb.get_lx(0), sb.get_lx(1), sb.get_lx(2)) )
print("dx: %f, %f, %f" % (sb.get_dx(0), sb.get_dx(1), sb.get_dx(2)) )
print("Volume: %f" % (sb.get_volume()) )

#-------------- allocate array ------------
# free end initial condition. q1 is q and q2 is qdagger.
# q1 starts from A end and q2 starts from B end.
w       = np.zeros([2]+list(sb.get_nx()), dtype=np.float64)
q1_init = np.ones (    sb.get_n_grid(),   dtype=np.float64)
q2_init = np.ones (    sb.get_n_grid(),   dtype=np.float64)
phi     = np.zeros((2, sb.get_n_grid()),  dtype=np.float64)
# Initial Fields
print("w_A and w_B are initialized to lamellar phase.")
for i in range(0,sb.get_nx(2)):
    w[0,:,:,i] =  np.cos(2*np.pi*i/sb.get_nx(2))
    w[1,:,:,i] = -np.cos(2*np.pi*i/sb.get_nx(2))
w = np.reshape(w, [2, sb.get_n_grid()])

# keep the level of field value
sb.zero_mean(w[0])
sb.zero_mean(w[1])

#------------------ run ----------------------
print("---------- Run ----------")
time_start = time.time()

# assign large initial value for the energy and error
energy_total = 1.0e20
error_level = 1.0e20

# reset Anderson mixing module
am.reset_count()

# array for output fields
w_out = np.zeros([2, sb.get_n_grid()], dtype=np.float64)

# iteration begins here
print("iteration, mass error, total_partition, energy_total, error_level")
for scft_iter in range(1,max_scft_iter+1):
    # for the given fields find the polymer statistics
    phi_A, Q_A = pseudo_A.find_phi(q1_init,q2_init,w[0])
    phi_B, Q_B = pseudo_B.find_phi(q1_init,q2_init,w[1])
    
    phi[0] = phi_A*frac_A
    phi[1] = phi_B*(1.0-frac_A)
    ###### Additional example ######
    # phi_rcp, Q_rcp = pseudo_rcp.find_phi(q1_init,q2_init,rcp_A_frac*w[0]+(1.0-rcp_A_frac)*w[1])
    # phi[0] = phi_rcp*rcp_A_frac*frac_rcp
    # phi[1] = phi_rcp*(1.0-rcp_A_frac)*frac_rcp + phi_B*(1.0-frac_rcp)

    # calculate the total energy
    w_minus = (w[0]-w[1])/2
    w_plus  = (w[0]+w[1])/2

    alpha = n_segment_B/n_segment_A
    energy_total  = -frac_A*alpha*np.log(Q_A/sb.get_volume())
    energy_total -= (1.0-frac_A)*np.log(Q_B/sb.get_volume())
    energy_total += sb.inner_product(w_minus,w_minus)/chi_n/sb.get_volume()
    energy_total -= sb.integral(w_plus)/sb.get_volume()
    ###### Additional example ######
    #alpha = n_segment_B/n_segment_rcp
    #energy_total  = -frac_rcp*alpha*np.log(Q_rcp/sb.get_volume())
    #energy_total -= (1.0-frac_rcp)*np.log(Q_B/sb.get_volume())
    #energy_total += sb.inner_product(w_minus,w_minus)/chi_n/sb.get_volume()
    #energy_total -= sb.integral(w_plus)/sb.get_volume()


    # calculate pressure field for the new field calculation, the method is modified from Fredrickson's
    xi = 0.5*(w[0]+w[1]-chi_n)

    # calculate output fields
    w_out[0] = chi_n*phi[1] + xi
    w_out[1] = chi_n*phi[0] + xi
    sb.zero_mean(w_out[0])
    sb.zero_mean(w_out[1])

    # error_level measures the "relative distance" between the input and output fields
    old_error_level = error_level
    w_diff = w_out - w
    multi_dot = sb.inner_product(w_diff[0],w_diff[0]) + sb.inner_product(w_diff[1],w_diff[1])
    multi_dot /= sb.inner_product(w[0],w[0]) + sb.inner_product(w[1],w[1]) + 1.0

    # print iteration # and error levels and check the mass conservation
    mass_error = (sb.integral(phi[0]) + sb.integral(phi[1]))/sb.get_volume() - 1.0
    error_level = np.sqrt(multi_dot)
    print("%8d %12.3E %15.7E %15.7E %15.9f %15.7E" %
    (scft_iter, mass_error, Q_A, Q_B, energy_total, error_level))
    ###### Additional example ######
    # print("%8d %12.3E %15.7E %15.7E %15.9f %15.7E" %
    # (scft_iter, mass_error, Q_rcp, Q_B, energy_total, error_level))

    # conditions to end the iteration
    if error_level < tolerance:
        break

    # calculte new fields using simple and Anderson mixing
    am.caculate_new_fields(
    np.reshape(w,      2*sb.get_n_grid()),
    np.reshape(w_out,  2*sb.get_n_grid()),
    np.reshape(w_diff, 2*sb.get_n_grid()),
    old_error_level, error_level)

# estimate execution time
time_duration = time.time() - time_start
print("total time: %f " % time_duration)

# save final results
mdic = {"dim":sb.get_dim(), "nx":sb.get_nx(), "lx":sb.get_lx(),
        "N_A":pc_A.get_n_segment_total(),"N_B":pc_B.get_n_segment_total(), "frac_A":frac_A, "chi_n":chi_n, "epsilon":epsilon,
        "chain_model":chain_model, "w_a":w[0], "w_b":w[1], "phi_a":phi[0], "phi_b":phi[1]}
savemat("fields.mat", mdic)
###### Additional example ######
# mdic = {"dim":sb.get_dim(), "nx":sb.get_nx(), "lx":sb.get_lx(),
#         "N_rcp":pc_rcp.get_n_segment_total(),"N_B":pc_B.get_n_segment_total(), "frac_rcp":frac_rcp, "rcp_A_frac":rcp_A_frac, "chi_n":chi_n, "epsilon":epsilon,
#         "chain_model":chain_model, "w_a":w[0], "w_b":w[1], "phi_a":phi[0], "phi_b":phi[1]}
# savemat("fields.mat", mdic)