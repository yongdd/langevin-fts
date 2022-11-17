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
chi_n = 20               # Flory-Huggins Parameters * N
epsilon = 2.0            # a_A/a_B, conformational asymmetry
nx = [64,64,64]          # grids number
lx = [18.,6.,12.]        # as aN^(1/2) unit, a = sqrt(f*a_A^2 + (1-f)*a_B^2)
ds = 1/100               # contour step interval
chain_model = "Continuous" # choose among [Continuous, Discrete]

# calculate chain parameters, dict_a_n = [a_A, a_B]
dict_a_n = {"A":np.sqrt(epsilon*epsilon/(f*epsilon*epsilon + (1.0-f))),
            "B":np.sqrt(1.0/(f*epsilon*epsilon + (1.0-f)))}

# choose platform among [cuda, cpu-mkl]
if "cuda" in PlatformSelector.avail_platforms():
    platform = "cuda"
else:
    platform = PlatformSelector.avail_platforms()[0]
print("platform :", platform)
factory = PlatformSelector.create_factory(platform, chain_model)

# create instances
pc     = factory.create_polymer_chain(["A","B"], [f, 1-f], dict_a_n, ds)
cb     = factory.create_computation_box(nx, lx)
pseudo = factory.create_pseudo(cb, pc)

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
w       = np.zeros([2, cb.get_n_grid()], dtype=np.float64)
q1_init = np.zeros (   cb.get_n_grid(),  dtype=np.float64)
q2_init = np.zeros (   cb.get_n_grid(),  dtype=np.float64)
q1_init[0] = np.prod(cb.get_nx())/np.prod(cb.get_lx())

space_y, space_x, space_z = np.meshgrid(
    cb.get_lx(1)/cb.get_nx(1)*np.concatenate([np.arange((cb.get_nx(1)+1)//2), cb.get_nx(1)//2-np.arange(cb.get_nx(1)//2)]),
    cb.get_lx(0)/cb.get_nx(0)*np.concatenate([np.arange((cb.get_nx(0)+1)//2), cb.get_nx(0)//2-np.arange(cb.get_nx(0)//2)]),
    cb.get_lx(2)/cb.get_nx(2)*np.concatenate([np.arange((cb.get_nx(2)+1)//2), cb.get_nx(2)//2-np.arange(cb.get_nx(2)//2)]))
squared_x = space_x**2 + space_y**2 + space_z**2

eps = epsilon
f = f
norm_segment = (f*eps**2 + (1-f))

print("---------- Statistical Segment Length <x^2> ----------")
print("n'th segment, theory, caculation")
phi, Q = pseudo.compute_statistics(q1_init,q2_init,{"A":w[0],"B":w[1]})
pred_mean_squared_x = 0
if(pc.get_model_name().lower() == "continuous"):
    for n in range(0, pc.get_n_segment_total()+1):
        q1_out, _ = pseudo.get_partition(n, 0)
        q1_out = np.reshape(q1_out, cb.get_nx())
        mean_squared_x = np.sum(q1_out*squared_x)/np.sum(q1_out)
                    
        print("%8d: %10.4f, %10.4f"
            % (n,
              cb.get_dim()/3*pred_mean_squared_x,
              pc.get_n_segment_total()*norm_segment*mean_squared_x))
        
        if (n < pc.get_n_segment(0)):
            pred_mean_squared_x += eps**2
        else:
            pred_mean_squared_x += 1
            
elif(pc.get_model_name().lower() == "discrete"):
    for n in range(1, pc.get_n_segment_total()+1):
        q1_out, _ = pseudo.get_partition(n, 1)
        q1_out = np.reshape(q1_out, cb.get_nx())

        mean_squared_x = np.sum(q1_out*squared_x)/np.sum(q1_out)
        print("%8d: %10.4f, %10.4f"
            % (n,
               cb.get_dim()/3*pred_mean_squared_x,
               pc.get_n_segment_total()*norm_segment*mean_squared_x))

        if (n < pc.get_n_segment(0)):
            pred_mean_squared_x += eps**2
        elif (n == pc.get_n_segment(0)):
            pred_mean_squared_x += (eps**2 + 1)/2
        else:
            pred_mean_squared_x += 1
