from posixpath import lexists
import numpy as np
from langevinfts import *

def find_saddle_point(pc, cb, pseudo, am, lx, chi_n,
    q1_init, q2_init, w, max_iter, tolerance, is_box_altering=True):

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
        phi, Q = pseudo.compute_statistics(q1_init,q2_init,w)

        # calculate the total energy
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

        # print iteration # and error levels and check the mass conservation
        mass_error = (cb.integral(phi[0]) + cb.integral(phi[1]))/cb.get_volume() - 1.0
        error_level = np.sqrt(multi_dot)
        
        if (is_box_altering):
            # Calculate stress
            stress_array = np.array(pseudo.compute_stress()[-cb.get_dim():])/Q
            error_level += np.sqrt(np.sum(stress_array)**2)
            print("%8d %12.3E %15.7E %15.9f %15.7E" %
            (scft_iter, mass_error, Q, energy_total, error_level), end=" ")
            print("\t[", ",".join(["%10.7f" % (x) for x in lx]), "]")
        else:
            print("%8d %12.3E %15.7E %15.9f %15.7E" %
            (scft_iter, mass_error, Q, energy_total, error_level))

        # conditions to end the iteration
        if error_level < tolerance:
            break

        # calculte new fields using simple and Anderson mixing
        if (is_box_altering):
            am_new  = np.concatenate((np.reshape(w,      2*cb.get_n_grid()), lx))
            am_out  = np.concatenate((np.reshape(w_out,  2*cb.get_n_grid()), lx + stress_array))
            am_diff = np.concatenate((np.reshape(w_diff, 2*cb.get_n_grid()), stress_array))
            am.calculate_new_fields(am_new, am_out, am_diff, old_error_level, error_level)

            # set box size
            w[0] = am_new[0:cb.get_n_grid()]
            w[1] = am_new[cb.get_n_grid():2*cb.get_n_grid()]
            lx = am_new[-cb.get_dim():]
            cb.set_lx(lx)
            # update bond parameters using new lx
            pseudo.update_bond_function()
        else:
            am.calculate_new_fields(
            np.reshape(w,      2*cb.get_n_grid()),
            np.reshape(w_out,  2*cb.get_n_grid()),
            np.reshape(w_diff, 2*cb.get_n_grid()),
            old_error_level, error_level)

    return phi, Q, energy_total