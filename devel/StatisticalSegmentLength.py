import os
import numpy as np
from scipy.io import savemat
from langevinfts import *

# -------------- initialize ------------

# OpenMP environment variables 
os.environ["MKL_NUM_THREADS"] = "1"  # always 1
os.environ["OMP_STACKSIZE"] = "1G"
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "2"  # 0, 1 or 2

# Major Simulation Parameters
f = 0.3                  # A-fraction, f
n_segment = 100           # segment number, N
chi_n = 20               # Flory-Huggins Parameters * N
epsilon = 2.0            # a_A/a_B, conformational asymmetry
nx = [64,64,64]          # grids number
lx = [18.,6.,12.]          # as aN^(1/2) unit, a = sqrt(f*a_A^2 + (1-f)*a_B^2)
chain_model = "Continuous" # choose among [Continuous, Discrete]

# choose platform among [cuda, cpu-mkl, cpu-pocketfft]
if "cuda" in PlatformSelector.avail_platforms():
    platform = "cuda"
else:
    platform = PlatformSelector.avail_platforms()[0]
print("platform :", platform)
factory = PlatformSelector.create_factory(platform)

# create instances
pc = factory.create_polymer_chain(f, n_segment, chi_n, chain_model, epsilon)
sb = factory.create_simulation_box(nx, lx)
pseudo = factory.create_pseudo(sb, pc)

# -------------- print simulation parameters ------------
print("---------- Simulation Parameters ----------")
print("Box Dimension: %d" % (sb.get_dim()))
print("chi_n: %f, f: %f, N: %d" % (pc.get_chi_n(), pc.get_f(), pc.get_n_segment()) )
print("%s chain model" % (pc.get_model_name()) )
print("Conformational asymmetry (epsilon): %f" % (pc.get_epsilon()) )
print("Nx: %d, %d, %d" % (sb.get_nx(0), sb.get_nx(1), sb.get_nx(2)) )
print("Lx: %f, %f, %f" % (sb.get_lx(0), sb.get_lx(1), sb.get_lx(2)) )
print("dx: %f, %f, %f" % (sb.get_dx(0), sb.get_dx(1), sb.get_dx(2)) )
print("Volume: %f" % (sb.get_volume()) )

#-------------- allocate array ------------
# free end initial condition. q1 is q and q2 is qdagger.
# q1 starts from A end and q2 starts from B end.
w       = np.zeros([2, sb.get_n_grid()], dtype=np.float64)
q1_init = np.zeros (   sb.get_n_grid(),  dtype=np.float64)
q2_init = np.zeros (   sb.get_n_grid(),  dtype=np.float64)
q1_init[0] = np.prod(sb.get_nx())/np.prod(sb.get_lx())

space_y, space_x, space_z = np.meshgrid(
    sb.get_lx(1)/sb.get_nx(1)*np.concatenate([np.arange((sb.get_nx(1)+1)//2), sb.get_nx(1)//2-np.arange(sb.get_nx(1)//2)]),
    sb.get_lx(0)/sb.get_nx(0)*np.concatenate([np.arange((sb.get_nx(0)+1)//2), sb.get_nx(0)//2-np.arange(sb.get_nx(0)//2)]),
    sb.get_lx(2)/sb.get_nx(2)*np.concatenate([np.arange((sb.get_nx(2)+1)//2), sb.get_nx(2)//2-np.arange(sb.get_nx(2)//2)]))
squared_x = space_x**2 + space_y**2 + space_z**2

eps = pc.get_epsilon()
f = pc.get_f()
norm_segment = (f*eps**2 + (1-f))

print("---------- Statistical Segment Length <x^2> ----------")
print("n'th segment, theory, caculation")
phi_a, phi_b, Q = pseudo.find_phi(q1_init,q2_init,w[0],w[1])
pred_mean_squared_x = 0
if(pc.get_model_name().lower() == "continuous"):
    for n in range(0, pc.get_n_segment()+1):
        q1_out, _ = pseudo.get_partition(n, 0)
        q1_out = np.reshape(q1_out, sb.get_nx())
        mean_squared_x = np.sum(q1_out*squared_x)/np.sum(q1_out)
                    
        print("%8d: %10.4f, %10.4f"
            % (n,
              sb.get_dim()/3*pred_mean_squared_x,
              pc.get_n_segment()*norm_segment*mean_squared_x))
        
        if (n < pc.get_n_segment_a()):
            pred_mean_squared_x += eps**2
        else:
            pred_mean_squared_x += 1
            
elif(pc.get_model_name().lower() == "discrete"):
    for n in range(1, pc.get_n_segment()+1):
        q1_out, _ = pseudo.get_partition(n, 0)
        q1_out = np.reshape(q1_out, sb.get_nx())

        mean_squared_x = np.sum(q1_out*squared_x)/np.sum(q1_out)
        print("%8d: %10.4f, %10.4f"
            % (n,
               sb.get_dim()/3*pred_mean_squared_x,
               pc.get_n_segment()*norm_segment*mean_squared_x))

        if (n < pc.get_n_segment_a()):
            pred_mean_squared_x += eps**2
        elif (n == pc.get_n_segment_a()):
            pred_mean_squared_x += (eps**2 + 1)/2
        else:
            pred_mean_squared_x += 1
