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
example_number = 3

# Major Simulation Parameters
f = 0.5             # A-fraction, f
n_segment = 90     # segment number, N
chi_n = 13.27        # Flory-Huggins Parameters * N
epsilon = 1.0       # a_A/a_B, conformational asymmetry
nx = [32,32,32]     # grid numbers
lx = [4.36,4.36,4.36]  # as aN^(1/2) unit, a = sqrt(f*a_A^2 + (1-f)*a_B^2)
chain_model = "Discrete" # choose among [Continuous, Discrete]

frac_bcp = 0.7      # fraction of major BCP chain
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
simulation = FieldTheoreticSimulation.create_simulation(platform, chain_model)

# create instances
sb     = simulation.create_simulation_box(nx, lx)
## create diblock copolymer with a_A, a_B such that  a = sqrt(f*a_A^2 + (1-f)*a_B^2), a_A/a_B = epsilon

bond_length_a = np.sqrt(epsilon*epsilon/(f*epsilon*epsilon + (1.0-f)))*np.power(n_segment,-0.5)
bond_length_b = np.sqrt(1.0/(f*epsilon*epsilon + (1.0-f)))*np.power(n_segment,-0.5)
N_pc = [int(f*n_segment),int((1-f)*n_segment)]
pc = simulation.create_polymer_chain(N_pc, [bond_length_a,bond_length_b])

###### Example 01 ######
## this code adds homopolymer A with length N_A = 45
## homopolymer A takes 30 percent of the system (70 percent of bcp)
## it uses previously calculated bond_length_a
if example_number == 1:
    N_homo = 45
    pc_homo_a = simulation.create_polymer_chain([N_homo], [bond_length_a])

###### Example 02 ######
## this code adds A,B random copolymer with A fraction of 0.7, length = N_rcp = 70
## you need to calculate effective bond length manually if two monomers have different bond lengths
if example_number == 2:
    N_rcp = 70
    random_A_frac = 0.7
    bond_length_rand = np.sqrt(random_A_frac*bond_length_a**2 + (1-random_A_frac)*bond_length_b**2) ###맞는지 좀 생각해 볼 것
    pc_rand = simulation.create_polymer_chain([N_rcp], [bond_length_rand])

###### Example 03 ######
## this code adds another A,B block copolymer with fraction f2 = 0.4, length = N2 = 45
## additional chain shares previously calculated bond_length_a/b
if example_number == 3:
    f2 = 0.4
    N2 = 45
    N_pc2 = [int(f2*N2),int((1-f2)*N2)]
    pc2 = simulation.create_polymer_chain(N_pc2, [bond_length_a,bond_length_b])

## pseudo should be created for each chain used in simulation
pseudo = simulation.create_pseudo(sb, pc)
###### Example 01 ######
if example_number == 1:
    pseudo_homo_a = simulation.create_pseudo(sb, pc_homo_a)
###### Example 02 ######
if example_number == 2:
    pseudo_rand   = simulation.create_pseudo(sb, pc_rand)
###### Example 03 ######
if example_number == 3:
    pseudo_pc2    = simulation.create_pseudo(sb, pc2)
am     = simulation.create_anderson_mixing(am_n_var,
            am_max_hist, am_start_error, am_mix_min, am_mix_init)

# -------------- print simulation parameters ------------
print("---------- Simulation Parameters ----------")
print("Box Dimension: %d" % (sb.get_dim()))
print("chi_n: %f, f: %f, N: %d" % (chi_n, f, pc.get_n_segment_total()) ) 
print("%s chain model" % (pc.get_model_name()) )
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
    w[0,:,:,i] =  np.cos(3*2*np.pi*i/sb.get_nx(2))
    w[1,:,:,i] = -np.cos(3*2*np.pi*i/sb.get_nx(2))
w = np.reshape(w, [2, sb.get_n_grid()])

# keep the level of field value
sb.zero_mean(w[0])
sb.zero_mean(w[1])

#------------------ run ----------------------
print("---------- Run ----------")
time_start = time.time()

#phi, Q, energy_total = find_saddle_point(pc, sb, pseudo, am, lx, chi_n,
#    q1_init, q2_init, w, max_scft_iter, tolerance, is_box_altering=True)
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
    phi_bcp, Q_bcp = pseudo.find_phi(q1_init,q2_init,w)
    phi_bcp = phi_bcp.reshape(pc.get_n_block(),sb.get_n_grid())
    ###### Example 01 ######
    if example_number == 1:
        phi_homo_a, Q_homo_a = pseudo_homo_a.find_phi(q1_init,q2_init,w[0])
        phi[0] = phi_bcp[0]*frac_bcp + phi_homo_a*(1.0-frac_bcp)
        phi[1] = phi_bcp[1]*frac_bcp
    ###### Example 02 ######
    if example_number == 2:
        phi_rand, Q_rand = pseudo_rand.find_phi(q1_init,q2_init,random_A_frac*w[0]+(1.0-random_A_frac)*w[1])
        phi[0] = phi_bcp[0]*frac_bcp + phi_rand*random_A_frac*(1.0-frac_bcp)
        phi[1] = phi_bcp[1]*frac_bcp + phi_rand*(1.0-random_A_frac)*(1.0-frac_bcp)
    ###### Example 03 ######
    if example_number == 3:
        phi_bcp_2, Q_bcp_2 = pseudo_pc2.find_phi(q1_init,q2_init,w)
        phi[0] = phi_bcp[0]*frac_bcp + phi_bcp_2[0]*(1.0-frac_bcp)
        phi[1] = phi_bcp[1]*frac_bcp + phi_bcp_2[1]*(1.0-frac_bcp)
    # calculate the total energy
    w_minus = (w[0]-w[1])/2
    w_plus  = (w[0]+w[1])/2

    volume_bcp = frac_bcp*sb.get_volume()
    energy_total  = -np.log(Q_bcp/volume_bcp)
    energy_total += sb.inner_product(w_minus,w_minus)/chi_n/volume_bcp
    energy_total -= sb.integral(w_plus)/volume_bcp
    ###### Example 01 ######
    if example_number == 1:
        volume_homo_a = (1.0-frac_bcp)*sb.get_volume()
        energy_total += -volume_homo_a/(N_homo/n_segment*volume_bcp)*np.log(Q_homo_a/volume_homo_a)
    ###### Example 02 ######
    if example_number == 2:
        volume_rand = (1.0-frac_bcp)*sb.get_volume()
        energy_total += -volume_rand/(N_rcp/n_segment*volume_bcp)*np.log(Q_rand/volume_rand) ####나중에 다시 볼 것
    ###### Example 03 ######
    if example_number == 3:
        volume_bcp2 = (1.0-frac_bcp)*sb.get_volume()
        energy_total += -volume_bcp2/(N2/n_segment*volume_bcp)*np.log(Q_bcp_2/volume_bcp2)

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
    np.reshape(w,      2*sb.get_n_grid()),
    np.reshape(w_out,  2*sb.get_n_grid()),
    np.reshape(w_diff, 2*sb.get_n_grid()),
    old_error_level, error_level)

# estimate execution time
time_duration = time.time() - time_start
print("total time: %f " % time_duration)

# save final results
mdic = {"dim":sb.get_dim(), "nx":sb.get_nx(), "lx":sb.get_lx(),
        "N":pc.get_n_segment_total(), "f":f, "chi_n":chi_n, "epsilon":epsilon,
        "chain_model":chain_model, "w_a":w[0], "w_b":w[1], "phi_a":phi[0], "phi_b":phi[1]}
savemat("fields.mat", mdic)