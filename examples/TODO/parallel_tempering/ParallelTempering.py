import sys
import os
import time
import pathlib
from mpi4py import MPI
import numpy as np
from langevinfts import *

class ParallelTempering:
    def __init__(self, langevin_nbar):
        super().__init__()

        comm = MPI.COMM_WORLD
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        self.langevin_nbar = langevin_nbar
        self.even_turn = False
        
        print("MPI rank/size: %d/%d" % (self.rank, self.size))
        
    def get_my_chi_n(self, chi_n_from, chi_n_to):
        if self.size > 1 :
            return chi_n_from + self.rank*(chi_n_to - chi_n_from)/(self.size-1)
        else:
            return chi_n_from
            
    def attempt(self, sb, chi_n, w_minus, w_plus):
        
        wswap = np.empty(len(w_minus), dtype=np.float64)
        
        chi_n_swap   = 0.0
        p_swap       = 0.0
        is_receiver  = False
        is_sender    = False
        is_swap      = False
        
        comm = MPI.COMM_WORLD
        rank = self.rank
        size = self.size

        # even case
        if self.even_turn :
          if rank%2 == 0 and 0 < rank:
            is_receiver = True
          elif rank%2 == 1 and rank < size-1:
            is_sender = True
          self.even_turn = False
        # odd case
        else:
          if rank%2 == 0 and rank < size-1:
            is_sender = True
          elif rank%2 == 1 and 0 < rank:
            is_receiver = True
          self.even_turn = True

        print( rank, ": is_sender, is_receiver", is_sender, is_receiver)
        if is_sender :
          receiver_rank = rank+1
          print(rank, ": receiver id ", receiver_rank)
          print(rank, ": sending chi_n...", chi_n)
          comm.send(chi_n, dest=receiver_rank, tag=55)

          print(rank, ": sending w_minus to other...")
          comm.Send([w_minus, MPI.REAL8], dest=receiver_rank, tag=55)

          print(rank, ": receiving swap decision...")
          is_swap = comm.recv(source=receiver_rank, tag=55)

          print(rank, ": is_swap", is_swap)
          if is_swap:
            # swap w_minus
            print(rank, ": receiving w_minus from other...")
            comm.Recv([wswap, MPI.REAL8], source=receiver_rank, tag=55)
            print(rank, ": RECV w_minus")

            w_minus[:] = wswap[:]

            # swap w_plus
            print(rank, ": receiving w_plus from other...")
            comm.Recv([wswap, MPI.REAL8], source=receiver_rank, tag=55)
            print(rank, ": RECV w_plus")

            print(rank, ": sending w_plus to other...")
            comm.Send([w_plus, MPI.REAL8], source=receiver_rank, tag=55)
            w_plus[:] = wswap[:]

        if is_receiver :
          sender_rank = rank-1
          print(rank, ": sender id ", sender_rank)
          print(rank, ": receiving b_chi_n...")
          chi_n_swap = comm.recv(source=sender_rank, tag=55)
          print(rank, ": RECV b_chi_n")

          print(rank, ": receiving w_minus from other ...")
          comm.Recv([wswap,  MPI.REAL8], source=sender_rank, tag=55)
          print(rank, ": RECV w_minus")

          p_swap = np.exp(np.sqrt(self.langevin_nbar) *(1.0/chi_n - 1.0/chi_n_swap) * sb.integral(w_minus**2 - wswap**2))
          p_random = np.random.uniform()
          print(rank, ": p_swap, p_random,", p_swap, p_random)

          if p_random < p_swap :
            b_is_swap = True
          else :
            b_is_swap = False

          print(rank, ": sending swap decision...", b_is_swap)
          comm.send(is_swap, dest=sender_rank, tag=55)
          
          if is_swap :
            # swap w_minus
            print(rank, ": sending w_minus...")
            comm.Send([w_minus, MPI.REAL8], dest=sender_rank, tag=55)
            w_minus[:] = wswap[:]

            # swap w_plus
            print(rank, ": sending w_plus...")
            comm.Send([w_plus, MPI.REAL8], dest=sender_rank, tag=55)
            
            print(rank, ": receiving w_plus...")
            comm.Recv([wswap, MPI.REAL8], source=sender_rank, tag=55)
            print(rank, ": RECV w_plus") 

            w_plus[:] = wswap[:]

        print(rank, ": waiting for synchronization...")
        comm.barrier()
        print(rank, ": END.")

def find_saddle_point():
    # assign large initial value for the energy and error
    energy_total = 1e20
    error_level = 1e20

    # reset Anderson mixing module
    am.reset_count()

    # saddle point iteration begins here
    for saddle_iter in range(0,saddle_max_iter):
        
        # for the given fields find the polymer statistics
        QQ = pseudo.find_phi(phi_a, phi_b, 
                q1_init,q2_init,
                w_plus + w_minus,
                w_plus - w_minus)
        phi_plus = phi_a + phi_b
        
        # calculate output fields
        g_plus = 1.0*(phi_plus-1.0)
        w_plus_out = w_plus + g_plus 
        sb.zero_mean(w_plus_out);

        # error_level measures the "relative distance" between the input and output fields
        old_error_level = error_level
        error_level = np.sqrt(sb.inner_product(phi_plus-1.0,phi_plus-1.0)/sb.get_volume())

        # print iteration # and error levels
        if(verbose_level == 2 or
         verbose_level == 1 and
         (error_level < saddle_tolerance or saddle_iter == saddle_max_iter-1 )):
             
            # calculate the total energy
            energy_old = energy_total
            energy_total  = -np.log(QQ/sb.get_volume())
            energy_total += sb.inner_product(w_minus,w_minus)/pc.get_chi_n()/sb.get_volume()
            energy_total -= sb.integral(w_plus)/sb.get_volume()

            # check the mass conservation
            mass_error = sb.integral(phi_plus)/sb.get_volume() - 1.0
            print("%8d %12.3E %15.7E %13.9f %13.9f" %
                (saddle_iter, mass_error, QQ, energy_total, error_level))
        # conditions to end the iteration
        if(error_level < saddle_tolerance):
            break;
            
        # calculte new fields using simple and Anderson mixing
        # (Caution! we are now passing entire w, w_out and w_diff not just w[0], w_out[0] and w_diff[0])
        am.caculate_new_fields(w_plus, w_plus_out, g_plus, old_error_level, error_level);
        
# -------------- simulation parameters ------------

# OpenMP environment variables
os.environ["KMP_STACKSIZE"] = "1G"
os.environ["MKL_NUM_THREADS"] = "1"  # always 1
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "0"  # 0, 1 or 2

verbose_level = 1  # 1 : print at each langevin step.
                   # 2 : print at each saddle point iteration.

# Simulation Box
nx = [64,64,64]
lx = [8.0,8.0,8.0]

# Polymer Chain
f = 0.5
n_contour = 16
chi_n_from = 16.0
chi_n_to   = 20.0
polymer_model = "Gaussian"  # choose among [Gaussian, Discrete]

# Anderson Mixing 
saddle_tolerance = 1e-4
saddle_max_iter = 100
am_n_comp = 1  # A and B
am_max_hist= 20
am_start_error = 8e-1
am_mix_min = 0.1
am_mix_init = 0.1

# Langevin Dynamics
langevin_dt = 0.8         # langevin step interval, delta tau*N
langevin_nbar = 1024;     # invariant polymerization index
langevin_max_iter = 50;

# Parall Tempering
pt_period = 10

# -------------- initialize ------------
# parallel tempering
pt = ParallelTempering(langevin_nbar) 
chi_n = pt.get_my_chi_n(chi_n_from, chi_n_to)

# choose platform among [cuda, cpu-mkl, cpu-fftw]
factory = PlatformSelector.create_factory("cuda")

# create instances 
pc = factory.create_polymer_chain(f, n_contour, chi_n)
sb = factory.create_simulation_box(nx, lx)
pseudo = factory.create_pseudo(sb, pc, polymer_model)
am = factory.create_anderson_mixing(sb, am_n_comp,
    am_max_hist, am_start_error, am_mix_min, am_mix_init)

# standard deviation of normal noise for single segment
langevin_sigma = np.sqrt(2*langevin_dt*sb.get_n_grid()/ 
    (sb.get_volume()*np.sqrt(langevin_nbar)))
    
# random seed for MT19937
np.random.seed(5489);  
print("Random Number Generator: ", np.random.RandomState().get_state()[0])

# -------------- print simulation parameters ------------
print("---------- Simulation Parameters ----------");
print("Box Dimension: %d"  % (sb.get_dim()) )
print("Precision: 8")
print("chi_n: %f, f: %f, NN: %d" % (pc.get_chi_n(), pc.get_f(), pc.get_n_contour()) )
print("Nx: %d, %d, %d" % (sb.get_nx(0), sb.get_nx(1), sb.get_nx(2)) )
print("Lx: %f, %f, %f" % (sb.get_lx(0), sb.get_lx(1), sb.get_lx(2)) )
print("dx: %f, %f, %f" % (sb.get_dx(0), sb.get_dx(1), sb.get_dx(2)) )
print("Volume: %f" % (sb.get_volume()) )

print("Invariant Polymerization Index: %d" % (langevin_nbar) )
print("Langevin Sigma: %f" % (langevin_sigma) )
print("Random Number Generator: ", np.random.RandomState().get_state()[0])

#-------------- allocate array ------------
# free end initial condition. q1 is q and q2 is qdagger.
# q1 starts from A end and q2 starts from B end.
q1_init = np.ones (sb.get_n_grid(), dtype=np.float64)
q2_init = np.ones (sb.get_n_grid(), dtype=np.float64)
phi_a   = np.zeros(sb.get_n_grid(), dtype=np.float64)
phi_b   = np.zeros(sb.get_n_grid(), dtype=np.float64)

print("w_minus and w_plus are initialized to random")
w_plus  = np.random.normal(0.0, langevin_sigma, sb.get_n_grid())
w_minus = np.random.normal(0.0, langevin_sigma, sb.get_n_grid())

# keep the level of field value
sb.zero_mean(w_plus);
sb.zero_mean(w_minus);

find_saddle_point()
#------------------ run ----------------------
print("---------- Run ----------")
time_start = time.time()

print("iteration, mass error, total_partition, energy_total, error_level")
for langevin_step in range(0, langevin_max_iter):
    
    print("langevin step: ", langevin_step)
    # attempt parallel tempering
    if( langevin_step % pt_period == 0 ):
        pt.attempt(sb, chi_n, w_minus, w_plus)
            
    # update w_minus: predict step
    w_minus_copy = w_minus.copy()
    normal_noise = np.random.normal(0.0, langevin_sigma, sb.get_n_grid())
    lambda1 = phi_a-phi_b + 2*w_minus/pc.get_chi_n()
    w_minus += -lambda1*langevin_dt + normal_noise
    sb.zero_mean(w_minus)
    find_saddle_point()
    
    # update w_minus: correct step 
    lambda2 = phi_a-phi_b + 2*w_minus/pc.get_chi_n()
    w_minus = w_minus_copy - 0.5*(lambda1+lambda2)*langevin_dt + normal_noise
    sb.zero_mean(w_minus)
    find_saddle_point()

    # if( langevin_step % 1000 == 0 ):
        # np.savez("data/fields_%06d.npz" % (langevin_step),
        # nx=nx, lx=lx, N=NN, f=pc.get_f(), chi_n=pc.get_chi_n(),
        # polymer_model=polymer_model, n_bar=langevin_nbar,
        # random_seed=np.random.RandomState().get_state()[0],
        # w_minus=w_minus, w_plus=w_plus)

# estimate execution time
time_duration = time.time() - time_start; 
print( "total time: %f, time per step: %f" %
    (time_duration, time_duration/langevin_max_iter) )
