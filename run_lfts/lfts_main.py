import sys
import time
import numpy as np
from langevinfts import *
from lfts_torch import *

# -------------- simulation parameters ------------
#pp = ParamParser.get_instance()
#pp.read_param_file(sys.argv[1], False);
#pp.get("platform")

verbose_level = 1  # 1 : print at each langevin step.
                   # 2 : print at each saddle point iteration.

# Simulation Box
nx = [32,32,32]
lx = [8.0,8.0,8.0]

# Polymer Chain
f = 0.5
NN = 16
chi_n = 20

# Anderson Mixing 
saddle_tolerance = 1e-4
saddle_max_iter = 100
am_n_comp = 1  # A and B
am_max_hist= 20
am_start_error = 8e-1
am_mix_min = 0.1
am_mix_init = 0.1

# Langevin Dynamics
langevin_dt = 0.1         # langevin step interval, delta tau
langevin_nbar = 1024;     # invariant polymerization index
langevin_max_iter = 10;

# -------------- initialize ------------
# choose platform among [CUDA, CPU_MKL, CPU_FFTW]
factory = KernelFactory("CUDA")

# create instances and assign to the variables of base classs
# for the dynamic binding
pc = factory.create_polymer_chain(f, NN, chi_n)
sb = factory.create_simulation_box(nx, lx)
pseudo = factory.create_pseudo(sb, pc)
am = factory.create_anderson_mixing(sb, am_n_comp,
    am_max_hist, am_start_error, am_mix_min, am_mix_init)

# standard deviation of normal noise
langevin_sigma = np.sqrt(2*langevin_dt/ 
    (sb.dx[0]*sb.dx[1]*sb.dx[2]*np.sqrt(langevin_nbar/NN)*NN**1.5))

# -------------- print simulation parameters ------------
print("---------- Simulation Parameters ----------");
print("Box Dimension: 3")
print("Precision: 8")
print("chi_n: %f, f: %f, NN: %d" % (pc.chi_n, pc.f, pc.NN) )
print("Nx: %d, %d, %d" % (sb.nx[0], sb.nx[1], sb.nx[2]) )
print("Lx: %f, %f, %f" % (sb.lx[0], sb.lx[1], sb.lx[2]) )
print("dx: %f, %f, %f" % (sb.dx[0], sb.dx[1], sb.dx[2]) )
print("Volume: %f" % (sb.volume) )

print("Invariant Polymerization Index" % () )
#-------------- allocate array ------------
w         = np.zeros([2, sb.MM], dtype=np.float64)
w_out     = np.zeros([2, sb.MM], dtype=np.float64)
w_diff    = np.zeros([2, sb.MM], dtype=np.float64)
w_plus    = np.zeros(    sb.MM,  dtype=np.float64)
w_minus   = np.zeros(    sb.MM,  dtype=np.float64)
xi        = np.zeros(    sb.MM,  dtype=np.float64)
q1_init   = np.zeros(    sb.MM,  dtype=np.float64)
q2_init   = np.zeros(    sb.MM,  dtype=np.float64)
phi_a     = np.zeros(    sb.MM,  dtype=np.float64)
phi_b     = np.zeros(    sb.MM,  dtype=np.float64)
phi_plus  = np.zeros(    sb.MM,  dtype=np.float64)
phi_minus = np.zeros(    sb.MM,  dtype=np.float64)

print("wminus and wplus are initialized to random")
w_plus[:] = np.random.normal(0, langevin_sigma, sb.MM)
w_minus[:] = np.random.normal(0, langevin_sigma, sb.MM)

# keep the level of field value
sb.zero_mean(w_plus);
sb.zero_mean(w_minus);

# free end initial condition. q1 is q and q2 is qdagger.
# q1 starts from A end and q2 starts from B end.
q1_init[:] = 1.0;
q2_init[:] = 1.0;

# Initialize deep learning module
# if (use_pretrained_model) :
    # deeplfts = DeepLFTS(lfts.wminus.shape, train_model, model_file=pretrained_model)
# else :
    # deeplfts = DeepLFTS(lfts.wminus.shape, train_model)

#------------------ run ----------------------
print("---------- Run ----------")
time_start = time.time()
#print("iteration, mass error, total_partition, energy_total, error_level")
for langevin_step in range(0, langevin_max_iter):
    
    print("langevin step: ", langevin_step)
    
    # update w_minus
    w[0] = w_plus[:] + w_minus[:]
    w[1] = w_plus[:] - w_minus[:]
    QQ = pseudo.find_phi(phi_a, phi_b, q1_init,q2_init,w[0],w[1])
    normal_noise = np.random.normal(0, langevin_sigma, sb.MM)
    g_minus = phi_a[:]-phi_b[:] + 2/chi_n*w_minus[:]
    w_minus[:] -= g_minus*langevin_dt*NN + normal_noise[:]*NN
    sb.zero_mean(w_minus)
    
    # guess w_plus using FCN
    #if ( deeplfts.model_is_trained ):
    #    w_plus[:] = deeplfts.predict(w_minus)
    
    # assign large initial value for the energy and error
    energy_total = 1e20
    error_level = 1e20

    # reset Anderson mixing module
    am.reset_count()
    
    # saddle point iteration begins here
    for saddle_iter in range(0,saddle_max_iter):
        
        w[0] = w_plus[:] + w_minus[:]
        w[1] = w_plus[:] - w_minus[:]
        # for the given fields find the polymer statistics
        QQ = pseudo.find_phi(phi_a, phi_b, q1_init,q2_init,w[0],w[1])
        phi_plus[:] = phi_a[:] + phi_b[:]
        phi_minus[:] = phi_a[:] - phi_b[:]
        
        # calculate the total energy
        energy_old = energy_total
        energy_total  = -np.log(QQ/sb.volume)
        energy_total += sb.inner_product(w_minus,w_minus)/chi_n/sb.volume
        energy_total -= sb.integral(w_plus)/sb.volume
        
        # calculate output fields
        g_plus = 1.0*(phi_plus[:]-1.0)
        w_plus_out = w_plus[:] + g_plus[:] 
        sb.zero_mean(w_plus_out);

        # error_level measures the "relative distance" between the input and output fields
        old_error_level = error_level
        error_level = np.sqrt(sb.inner_product(phi_plus-1.0,phi_plus-1.0)/sb.volume)
        
        # print iteration # and error levels and check the mass conservation
        mass_error = (sb.integral(phi_a) + sb.integral(phi_b))/sb.volume - 1.0

        # print iteration # and error levels
        if(verbose_level == 2 or
         verbose_level == 1 and
         (error_level < saddle_tolerance or saddle_iter == saddle_max_iter )):
            # check the mass conservation
            mass_error = sb.integral(phi_plus)/sb.volume - 1.0
            print("%8d %12.3E %15.7E %13.9f %13.9f" %
                (saddle_iter, mass_error, QQ, energy_total, error_level))
        # conditions to end the iteration
        if(error_level < saddle_tolerance):
            break;
            
        # calculte new fields using simple and Anderson mixing
        # (Caution! we are now passing entire w, w_out and w_diff not just w[0], w_out[0] and w_diff[0])
        am.caculate_new_fields(w_plus, w_plus_out, g_plus, old_error_level, error_level);

# estimate execution time
time_duration = time.time() - time_start; 
print( "total time: %f, time per step: %f" %
    (time_duration, time_duration/langevin_max_iter) )
