import sys
import os
import numpy as np
import time
from scipy.io import savemat
import scipy.optimize
from langevinfts import *

def find_saddle_point(lx):

    # set box size
    sb.set_lx(lx)
    pseudo.update()

    # assign large initial value for the energy and error
    energy_total = 1.0e20
    error_level = 1.0e20

    # reset Anderson mixing module
    am.reset_count()

    # iteration begins here
    for scft_iter in range(1,max_scft_iter+1):
        # for the given fields find the polymer statistics
        phi_a, phi_b, Q = pseudo.find_phi(q1_init,q2_init,w[0],w[1])

        # calculate the total energy
        energy_old = energy_total
        w_minus = (w[0]-w[1])/2
        w_plus  = (w[0]+w[1])/2

        energy_total  = -np.log(Q/sb.get_volume())
        energy_total += sb.inner_product(w_minus,w_minus)/pc.get_chi_n()/sb.get_volume()
        energy_total -= sb.integral(w_plus)/sb.get_volume()

        # calculate pressure field for the new field calculation, the method is modified from Fredrickson's
        xi = 0.5*(w[0]+w[1]-chi_n)

        # calculate output fields
        w_out[0] = chi_n*phi_b + xi
        w_out[1] = chi_n*phi_a + xi
        sb.zero_mean(w_out[0])
        sb.zero_mean(w_out[1])

        # error_level measures the "relative distance" between the input and output fields
        old_error_level = error_level
        w_diff = w_out - w
        multi_dot = sb.inner_product(w_diff[0],w_diff[0]) + sb.inner_product(w_diff[1],w_diff[1])
        multi_dot /= sb.inner_product(w[0],w[0]) + sb.inner_product(w[1],w[1]) + 1.0
        error_level = np.sqrt(multi_dot)

        # print iteration # and error levels and check the mass conservation
        mass_error = (sb.integral(phi_a) + sb.integral(phi_b))/sb.get_volume() - 1.0
        print( "%8d %12.3E %15.7E %13.9f %13.9f" %
            (scft_iter, mass_error, Q, energy_total, error_level) )

        # conditions to end the iteration
        if error_level < tolerance:
            break
        # calculte new fields using simple and Anderson mixing
        am.caculate_new_fields(
            np.reshape(w,      2*sb.get_n_grid()),
            np.reshape(w_out,  2*sb.get_n_grid()),
            np.reshape(w_diff, 2*sb.get_n_grid()),
            old_error_level, error_level)
    
    if use_stress:
        
        #[3.45369971] D, 31, FE
        #[3.45369961] D, 31, CPU
        #[3.45369961] D, 31, GPU
        #[3.45168121] D, 32, FE
        #[3.45168187] D, 32, CPU
        #[3.45168187] D, 32, GPU

        #[3.13305012 3.61800653] D, 31, FE
        #[3.13305186 3.61799861] D, 31, CPU
        #[3.13305186 3.61799861] D, 31, GPU

        #[3.13303768 3.6185239 ]

        #[3.1330375  3.61852269] D, 32, FE
        #[3.13303996 3.61851564] D, 32, CPU
        #[3.13303996 3.61851565] D, 32, GPU

        #[3.89024855 3.89041043 3.88891139] D, 31, FE
        #[3.89023387 3.89040209 3.88891398] D, 31, CPU
        #[3.89023386 3.89040205 3.88891401] D, 31, GPU

        #[3.88985882 3.88985806 3.88985764] D, 32, FE
        #[3.88986151 3.88986161 3.88986162] D, 32, CPU
        #[3.88986151 3.8898615  3.88986174] D, 32, GPU

        #[3.45875285] G, 31, FE
        #[3.45387881] G, 31
        #[3.45387881] G, 31, CPU
        #[3.45387881] G, 31, GPU

        #[3.4568217]  G, 32, FE
        #[3.45267581] G, 32
        #[3.45267581] G, 32, CPU
        #[3.45267581] G, 32, GPU

        #[3.13518938 3.62048445] G, 31, FE
        #[3.13220859 3.61696481] G, 31
        #[3.13220859 3.61696482] G, 31, CPU
        #[3.13220859 3.61696482] G, 31, GPU

        #[3.13518961 3.62091294] G, 32, FE
        #[3.13223572 3.61744164] G, 32
        #[3.13223573 3.61744165] G, 32, CPU
        #[3.13223573 3.61744164] G, 32, GPU

        #[3.89278648 3.89296259 3.89201472] G, 31, FE
        #[3.88905361 3.88914876 3.88848473] G, 31
        #[3.88905355 3.88914882 3.88848473] G, 31, GPU
        #[3.88905363 3.88914873 3.88848473] G, 31, CPU

        #[3.89260891 3.8926072  3.89260755] G, 32, FE
        #[3.88892214 3.88892232 3.88892231] G, 32
        #[3.88892215 3.88892231 3.88892231] G, 32, GPU
        #[3.88892212 3.88892234 3.8889223]  G, 32, CPU
        
        # #initialize arrays for stress calculation
        # space_ky, space_kx, space_kz = np.meshgrid(
            # 2*np.pi/sb.get_lx(1)*np.concatenate([np.arange((sb.get_nx(1)+1)//2), sb.get_nx(1)//2-np.arange(sb.get_nx(1)//2)]),
            # 2*np.pi/sb.get_lx(0)*np.concatenate([np.arange((sb.get_nx(0)+1)//2), sb.get_nx(0)//2-np.arange(sb.get_nx(0)//2)]),
            # 2*np.pi/sb.get_lx(2)*np.concatenate([np.arange((sb.get_nx(2)+1)//2), sb.get_nx(2)//2-np.arange(sb.get_nx(2)//2)]))

        # mag2_k = space_kx**2 + space_ky**2 + space_kz**2
        # g_k = np.exp(-mag2_k/6.0/pc.get_n_contour())

        # space_kx2 = space_kx**2
        # space_ky2 = space_ky**2
        # space_kz2 = space_kz**2

        # g_k_x = g_k * space_kx2
        # g_k_y = g_k * space_ky2
        # g_k_z = g_k * space_kz2
        
        # eps = pc.get_epsilon()
        # f = pc.get_f()
        # N_A =  pc.get_n_contour_a()
        # N =  pc.get_n_contour()
        # bond_length_a = eps*eps/(f*eps*eps + (1.0-f))
        # bond_length_b = 1.0/(f*eps*eps + (1.0-f))
        # bond_length_ab = 0.5*bond_length_a + 0.5*bond_length_b
        
        # # caculating stress
        # stress_x = 0.0
        # stress_y = 0.0
        # stress_z = 0.0
        
        # if pc.get_model_name().lower() == "discrete" :
            # for n in range(1, pc.get_n_contour()):
                # q1_out, q2_out = pseudo.get_partition(n, n+1)
                # q1_out_k = np.fft.fftn(np.reshape(q1_out, sb.get_nx()))
                # q2_out_k = np.fft.fftn(np.reshape(q2_out, sb.get_nx()))
                # #print(q1_out_k)
                # #sys.exit(-1)
                
                # stress_x += np.sum(q1_out_k*np.conj(q2_out_k)*g_k_x)
                # stress_y += np.sum(q1_out_k*np.conj(q2_out_k)*g_k_y)
                # stress_z += np.sum(q1_out_k*np.conj(q2_out_k)*g_k_z)
        # else:
            # for n in range(0, pc.get_n_contour()+1):
                # q1_out, q2_out = pseudo.get_partition(n, n)
                # q1_out_k = np.fft.fftn(np.reshape(q1_out, sb.get_nx()))
                # q2_out_k = np.fft.fftn(np.reshape(q2_out, sb.get_nx()))
                
                # if n == 0:
                    # bond_length = bond_length_a/2;
                # elif n < N_A:
                    # bond_length = bond_length_a;
                # elif n == N_A:
                    # bond_length = bond_length_ab;
                # elif n < N:
                    # bond_length = bond_length_b;
                # elif n == N:
                    # bond_length = bond_length_b/2;

                # stress_x += bond_length*np.sum(q1_out_k*np.conj(q2_out_k)*space_kx2)
                # stress_y += bond_length*np.sum(q1_out_k*np.conj(q2_out_k)*space_ky2)
                # stress_z += bond_length*np.sum(q1_out_k*np.conj(q2_out_k)*space_kz2)
                
        # stress_x *= 1.0/(3.0*sb.get_lx(0))
        # stress_y *= 1.0/(3.0*sb.get_lx(1))
        # stress_z *= 1.0/(3.0*sb.get_lx(2))
        
        # stress_array = np.multiply(np.real([stress_x, stress_y, stress_z]),
            # 1.0/(sb.get_n_grid())**2/pc.get_n_contour()/(Q/sb.get_volume()))
        
        stress_array = np.array(pseudo.dq_dl()[-sb.get_dim():])/Q
        return stress_array
    else:
        return energy_total

# -------------- initialize ------------

# OpenMP environment variables
os.environ["MKL_NUM_THREADS"] = "1"  # always 1
os.environ["OMP_STACKSIZE"] = "1G"
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "2"  # 0, 1 or 2

max_scft_iter = 1000
tolerance = 1e-8

# Major Simulation Parameters
f = 0.30            # A-fraction, f
n_contour = 100     # segment number, N
chi_n = 20          # Flory-Huggins Parameters * N
epsilon = 2.0       # a_A/a_B, conformational asymmetry
nx = [32,32,32]     # grids number
lx = [3.4,3.5,3.7]  # as aN^(1/2) unit, a = sqrt(f*a_A^2 + (1-f)*a_B^2)

chain_model = "Discrete" # choose among [Gaussian, Discrete]

am_n_comp = 2         # w_a (w[0]) and w_b (w[1])
am_max_hist= 20       # maximum number of history
am_start_error = 1e-1 # when switch to AM from simple mixing
am_mix_min = 0.1      # minimum mixing rate of simple mixing
am_mix_init = 0.1     # initial mixing rate of simple mixing

# use stress for finding unit cell
use_stress = False

# choose platform among [cuda, cpu-mkl, cpu-fftw]
if "cuda" in PlatformSelector.avail_platforms():
    platform = "cuda"
else:
    platform = PlatformSelector.avail_platforms()[0]
print("platform :", platform)
factory = PlatformSelector.create_factory(platform)

# create instances
pc = factory.create_polymer_chain(f, n_contour, chi_n, chain_model, epsilon)
sb = factory.create_simulation_box(nx, lx)
pseudo = factory.create_pseudo(sb, pc)
am = factory.create_anderson_mixing(sb, am_n_comp,
    am_max_hist, am_start_error, am_mix_min, am_mix_init)

# -------------- print simulation parameters ------------
print("---------- Simulation Parameters ----------")
print("Box Dimension: %d" % (sb.get_dim()))
print("Precision: 8")
print("chi_n: %f, f: %f, N: %d" % (pc.get_chi_n(), pc.get_f(), pc.get_n_contour()) )
print("%s chain model" % (pc.get_model_name()) )
print("Conformational asymmetry (epsilon): %f" % (pc.get_epsilon()) )
print("Nx: %d, %d, %d" % (sb.get_nx(0), sb.get_nx(1), sb.get_nx(2)) )
print("Lx: %f, %f, %f" % (sb.get_lx(0), sb.get_lx(1), sb.get_lx(2)) )
print("dx: %f, %f, %f" % (sb.get_dx(0), sb.get_dx(1), sb.get_dx(2)) )
print("Volume: %f" % (sb.get_volume()) )

#-------------- allocate array ------------
# free end initial condition. q1 is q and q2 is qdagger.
# q1 starts from A end and q2 starts from B end.
w       = np.zeros([2]+list(sb.get_nx()),  dtype=np.float64)
w_out   = np.zeros([2, sb.get_n_grid()],  dtype=np.float64)
q1_init = np.ones (    sb.get_n_grid(),   dtype=np.float64)
q2_init = np.ones (    sb.get_n_grid(),   dtype=np.float64)

# Initial Fields
# for i in range(0,sb.get_nx(0)):
   # for j in range(0,sb.get_nx(1)):
       # for k in range(0,sb.get_nx(2)):
           # w[0,i,j,k] =  np.cos(2*np.pi*i/sb.get_nx(0))
           # w[1,i,j,k] = -np.cos(2*np.pi*i/sb.get_nx(0))   

print("w_A and w_B are initialized to gyroid phase.")
for i in range(0,sb.get_nx(0)):
    xx = (i+1)*2*np.pi/sb.get_nx(0)
    for j in range(0,sb.get_nx(1)):
        yy = (j+1)*2*np.pi/sb.get_nx(1)
        for k in range(0,sb.get_nx(2)):
            zz = (k+1)*2*np.pi/sb.get_nx(2)
            c1 = np.sqrt(8.0/3.0)*(np.cos(xx)*np.sin(yy)*np.sin(2.0*zz) +
                np.cos(yy)*np.sin(zz)*np.sin(2.0*xx)+np.cos(zz)*np.sin(xx)*np.sin(2.0*yy))
            c2 = np.sqrt(4.0/3.0)*(np.cos(2.0*xx)*np.cos(2.0*yy)+
                np.cos(2.0*yy)*np.cos(2.0*zz)+np.cos(2.0*zz)*np.cos(2.0*xx))
            idx = i*sb.get_nx(1)*sb.get_nx(2) + j*sb.get_nx(2) + k
            w[0,i,j,k] = -0.364*c1+0.133*c2
            w[1,i,j,k] = 0.302*c1-0.106*c2
            
w = np.reshape(w, [2, sb.get_n_grid()])

#print("w_A and w_B are initialized to random field.")
#np.random.seed(5489) # random seed for MT19937
#w[0] = np.random.normal(0.0, 1.0, sb.get_n_grid())
#w[1] = np.random.normal(0.0, 1.0, sb.get_n_grid())

# keep the level of field value
sb.zero_mean(w[0])
sb.zero_mean(w[1])

#------------------ run ----------------------
print("---------- Run ----------")
print("iteration, mass error, total_partition, energy_total, error_level")
time_start = time.time()

# find the natural period of gyroid
if (use_stress):
    res = scipy.optimize.root(find_saddle_point, lx, tol=1e-5)
    print('Unit cell that make the stress zero: ', res.x, '(aN^1/2)')
    print('Stress in each direction: ', res.fun, )
else:
    res = scipy.optimize.minimize(find_saddle_point, lx, tol=1e-5, options={'disp':True})
    print('Unit cell that minimizes the free energy: ', res.x, '(aN^1/2)')
    print('Free energy per chain: ', res.fun, 'kT')

# estimate execution time
time_duration = time.time() - time_start
print("total time: %f " % time_duration)

# save final results
phi_a, phi_b, Q = pseudo.find_phi(q1_init,q2_init,w[0],w[1])
mdic = {"dim":sb.get_dim(), "nx":sb.get_nx(), "lx":sb.get_lx(),
        "N":pc.get_n_contour(), "f":pc.get_f(), "chi_n":pc.get_chi_n(),
        "chain_model":chain_model, "w_a":w[0], "w_b":w[1], "phi_a":phi_a, "phi_b":phi_b}
savemat("fields.mat", mdic)
