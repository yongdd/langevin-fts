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
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "2"  # 0, 1 or 2

# GPU environment variables
os.environ["LFTS_GPU_NUM_BLOCKS"]  = "256"
os.environ["LFTS_GPU_NUM_THREADS"] = "256"

max_scft_iter = 2
tolerance = 1e-9

f = 0.3            # A-fraction, f
chi_n = 20         # Flory-Huggins Parameters * N
epsilon = 2.5      # a_A/a_B, conformational asymmetry
ds = 1/50          # contour step interval

am_n_comp = 2         # w[0] and w[1]
am_max_hist= 20       # maximum number of history
am_start_error = 8e-1 # when switch to AM from simple mixing
am_mix_min = 0.1      # minimum mixing rate of simple mixing
am_mix_init = 0.1     # initial mixing rate of simple mixing

# calculate chain parameters, dict_a_n = [a_A, a_B]
dict_a_n = {"A":np.sqrt(epsilon*epsilon/(f*epsilon*epsilon + (1.0-f))),
            "B":np.sqrt(1.0/(f*epsilon*epsilon + (1.0-f)))}

# choose platform among [cuda, cpu-mkl]
print("Available Platforms: ", PlatformSelector.avail_platforms())
print("*" * 30, "Run", "*" * 30)
for dim in [1,2,3]:
    ### for 1D
    if (dim == 1):
        nx = [31]
        lx = [4.0]
        block_lengths = [f, 1-f]
        ds = 1/100
    ### for 2D
    elif (dim == 2): 
        nx = [31,49]
        lx = [4.0,3.0]
        block_lengths = [2*f, 2*(1-f)]
        ds = 1/90
    ### for 3D
    elif (dim == 3): 
        nx = [31,49,63]    # grids number
        lx = [4.0,3.0,2.0] # as aN^(1/2) unit
        block_lengths = [f/2, (1-f)/2]
        ds = 1/100

    for chain_model in ["Discrete", "Continuous"]:
        print("dimension: %d, chain_model: %s" % (dim, chain_model))
        print("platform, time per iter, mass error, output1 (error), output2 (dqdl)")
        test_output = []
        test_dqdl = []
        for platform in PlatformSelector.avail_platforms():

            factory = PlatformSelector.create_factory(platform, chain_model)
            #computation.display_info()

            # create instances
            pc     = factory.create_polymer_chain(["A","B"], block_lengths, dict_a_n, ds)
            cb     = factory.create_computation_box(nx, lx)
            pseudo = factory.create_pseudo(cb, pc)
            am     = factory.create_anderson_mixing(am_n_comp*np.prod(nx),
                        am_max_hist, am_start_error, am_mix_min, am_mix_init)

            # assign large initial value for the energy and error
            energy_total = 1.0e20
            error_level = 1.0e20

            #-------------- allocate array ------------
            # free end initial condition. q1 is q and q2 is qdagger.
            # q1 starts from A end and q2 starts from B end.
            w       = np.zeros([2, cb.get_n_grid()], dtype=np.float64)
            w_out   = np.zeros([2, cb.get_n_grid()], dtype=np.float64)
            q1_init = np.ones (    cb.get_n_grid(),  dtype=np.float64)
            q2_init = np.ones (    cb.get_n_grid(),  dtype=np.float64)

            #print("w_minus and w_plus are initialized to a given test fields.")
            for i in range(0,cb.get_nx(0)):
                for j in range(0,cb.get_nx(1)):
                    for k in range(0,cb.get_nx(2)):
                        idx = i*cb.get_nx(1)*cb.get_nx(2) + j*cb.get_nx(2) + k
                        w[0,idx]= np.cos(2.0*np.pi*i/4.68)*np.cos(2.0*np.pi*j/3.48)*np.cos(2.0*np.pi*k/2.74)*0.1
            w[1] = -w[0]

            # keep the level of field value
            cb.zero_mean(w[0])
            cb.zero_mean(w[1])

            #------------------ run ----------------------
            #print("iteration, mass error, total_partition, energy_total, error_level")
            time_start = time.time()
            # iteration begins here
            for scft_iter in range(1,max_scft_iter+1):
                # for the given fields find the polymer statistics
                phi, Q = pseudo.compute_statistics(q1_init,q2_init,{"A":w[0],"B":w[1]})
                
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

                # conditions to end the iteration
                if error_level < tolerance:
                    break
                # calculte new fields using simple and Anderson mixing
                am.calculate_new_fields(
                    np.reshape(w,      2*cb.get_n_grid()),
                    np.reshape(w_out,  2*cb.get_n_grid()),
                    np.reshape(w_diff, 2*cb.get_n_grid()),
                    old_error_level, error_level)
            
            dqdl = np.linalg.norm(np.array(pseudo.dq_dl()[-cb.get_dim():]))/Q
            test_output.append(error_level)
            test_dqdl.append(dqdl)
            
            # estimate execution time
            time_duration = time.time() - time_start
            print("%8s: %13.5f, %10.3E, %14.5E, %14.5E" %
                (platform, time_duration/max_scft_iter, mass_error, error_level, dqdl) )
        
        # Test error
        print("Standard deviations: %9.5E, %9.5E" % (np.std(test_output), np.std(test_dqdl)))
        if np.std(test_output) > 1e-8 or np.std(test_dqdl) > 1e-8 \
        or np.isnan(np.std(test_output)) or np.isnan(np.std(test_dqdl)):
            print("Each platform should give the same result")
            sys.exit(-1);
        print("-" * 50)
