from posixpath import lexists
import numpy as np
from langevinfts import *

def find_saddle_point(pc, sb, pseudo, am, lx,
    q1_init, q2_init, w, max_iter, tolerance):

    # assign large initial value for the energy and error
    energy_total = 1.0e20
    error_level = 1.0e20

    # reset Anderson mixing module
    am.reset_count()

    # array for output fields
    w_out = np.zeros([2, sb.get_n_grid()], dtype=np.float64)

    # iteration begins here
    print("iteration, mass error, total_partition, energy_total, error_level, box size")
    for scft_iter in range(1,max_iter+1):
        # for the given fields find the polymer statistics
        phi_a, phi_b, Q = pseudo.find_phi(q1_init,q2_init,w[0],w[1])

        # calculate the total energy
        w_minus = (w[0]-w[1])/2
        w_plus  = (w[0]+w[1])/2

        energy_total  = -np.log(Q/sb.get_volume())
        energy_total += sb.inner_product(w_minus,w_minus)/pc.get_chi_n()/sb.get_volume()
        energy_total -= sb.integral(w_plus)/sb.get_volume()

        # calculate pressure field for the new field calculation, the method is modified from Fredrickson's
        xi = 0.5*(w[0]+w[1]-pc.get_chi_n())

        # calculate output fields
        w_out[0] = pc.get_chi_n()*phi_b + xi
        w_out[1] = pc.get_chi_n()*phi_a + xi
        sb.zero_mean(w_out[0])
        sb.zero_mean(w_out[1])

        # Calculate stress
        stress_array = np.array(pseudo.dq_dl()[-sb.get_dim():])/Q

        # error_level measures the "relative distance" between the input and output fields
        old_error_level = error_level
        w_diff = w_out - w
        multi_dot = sb.inner_product(w_diff[0],w_diff[0]) + sb.inner_product(w_diff[1],w_diff[1])
        multi_dot /= sb.inner_product(w[0],w[0]) + sb.inner_product(w[1],w[1]) + 1.0
        error_level = np.sqrt(multi_dot) + np.sqrt(np.sum(stress_array)**2)

        # print iteration # and error levels and check the mass conservation
        mass_error = (sb.integral(phi_a) + sb.integral(phi_b))/sb.get_volume() - 1.0
        print("%8d %12.3E %15.7E %15.9f %15.7E" %
            (scft_iter, mass_error, Q, energy_total, error_level), end=' ')

        # conditions to end the iteration
        if error_level < tolerance:
            break

        # calculte new fields using simple and Anderson mixing
        am_new  = np.concatenate((np.reshape(w,      2*sb.get_n_grid()), lx))
        am_out  = np.concatenate((np.reshape(w_out,  2*sb.get_n_grid()), lx + stress_array))
        am_diff = np.concatenate((np.reshape(w_diff, 2*sb.get_n_grid()), stress_array))
        am.caculate_new_fields(am_new, am_out, am_diff, old_error_level, error_level)

        # set box size
        w[0] = am_new[0:sb.get_n_grid()]
        w[1] = am_new[sb.get_n_grid():2*sb.get_n_grid()]
        lx = am_new[-sb.get_dim():]
        print(np.round(lx,7))
        sb.set_lx(lx)
        # update bond parameters using new lx
        pseudo.update()

    return phi_a, phi_b, Q, energy_total