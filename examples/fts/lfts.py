import os
import time
import pathlib
import numpy as np
from scipy.io import savemat, loadmat
from langevinfts import *

# OpenMP environment variables
os.environ["MKL_NUM_THREADS"] = "1"  # always 1
os.environ["OMP_STACKSIZE"] = "1G"
os.environ["OMP_MAX_ACTIVE_LEVELS"] = "2"  # 0, 1 or 2

def calculate_sigma(langevin_nbar, langevin_dt, n_grids, volume):
        return np.sqrt(2*langevin_dt*n_grids/(volume*np.sqrt(langevin_nbar)))

class LFTS:
    def __init__(self, params):

        # choose platform among [cuda, cpu-mkl]
        avail_platforms = PlatformSelector.avail_platforms()
        if "platform" in params:
            platform = params["platform"]
        elif "cpu-mkl" in avail_platforms and len(params["nx"]) == 1: # for 1D simulation, use CPU
            platform = "cpu-mkl"
        elif "cuda" in avail_platforms: # If cuda is available, use GPU
            platform = "cuda"
        else:
            platform = avail_platforms[0]

        distinct_polymers = []
        assert(len(params['segment_lengths']) == 2), \
            "Currently, only AB copolymers are supported."
        assert(len(set(["A","B"]).intersection(set(params['segment_lengths'].keys())))==2), \
            "Use letters 'A' and 'B' for two species."
        assert(len(params["distinct_polymers"]) >= 1), \
            "There is no polymer chain."

        # (c++ class) Create a factory for given platform and chain_model
        factory = PlatformSelector.create_factory(platform, params["chain_model"])

        # (C++ class) Computation box
        cb = factory.create_computation_box(params["nx"], params["lx"])

        # Polymer chains
        total_volume_fraction = 0.0
        for polymer in params["distinct_polymers"]:
            block_length_list = []
            type_list = []
            A_fraction = 0.0
            alpha = 0.0  #total_relative_contour_length
            for block in polymer["blocks"]:
                block_length_list.append(block["length"])
                type_list.append(block["type"])
                alpha += block["length"]
                if block["type"] == "A":
                    A_fraction += block["length"]
                elif block["type"] == "random":
                    A_fraction += block["length"]*block["fraction"]["A"]
            total_volume_fraction += polymer["volume_fraction"]

            total_A_fraction = A_fraction/alpha
            statistical_segment_length = \
                np.sqrt(params["segment_lengths"]["A"]**2*total_A_fraction + \
                        params["segment_lengths"]["B"]**2*(1-total_A_fraction))

            if "random" in set(bt.lower() for bt in type_list):
                assert(len(type_list) == 1), \
                    "Currently, Only single block random copolymer is supported."
                assert(np.isclose(polymer["blocks"][0]["fraction"]["A"]+polymer["blocks"][0]["fraction"]["B"],1.0)), \
                    "The sum of volume fraction of random copolymer must be equal to 1."
                segment_length_list = {"random":statistical_segment_length}
            else:
                segment_length_list = params["segment_lengths"]
            
            # (C++ class) Polymer chain
            pc = factory.create_polymer_chain(type_list, block_length_list, segment_length_list, params["ds"])

            # (C++ class) Solvers using Pseudo-spectral method
            pseudo = factory.create_pseudo(cb, pc)

            distinct_polymers.append(
                {"volume_fraction":polymer["volume_fraction"],
                 "block_types":type_list,
                 "total_A_fraction":total_A_fraction,
                 "statistical_segment_length":statistical_segment_length,
                 "alpha":alpha, "pc":pc, "pseudo":pseudo, })

        assert(np.isclose(total_volume_fraction,1.0)), "The sum of volume fraction must be equal to 1."

        # (C++ class) Fields Relaxation using Anderson Mixing
        am = factory.create_anderson_mixing(
            np.prod(params["nx"]),      # the number of variables
            params["am"]["max_hist"],     # maximum number of history
            params["am"]["start_error"],  # when switch to AM from simple mixing
            params["am"]["mix_min"],      # minimum mixing rate of simple mixing
            params["am"]["mix_init"])     # initial mixing rate of simple mixing

        # Langevin Dynamics
        # standard deviation of normal noise
        langevin_sigma = calculate_sigma(params["langevin"]["nbar"], params["langevin"]["dt"], np.prod(params["nx"]), np.prod(params["lx"]))

        # -------------- print simulation parameters ------------
        print("---------- Simulation Parameters ----------")
        print("Platform :", platform)
        print("Box Dimension: %d" % (cb.get_dim()))
        print("Nx: %d, %d, %d" % (cb.get_nx(0), cb.get_nx(1), cb.get_nx(2)) )
        print("Lx: %f, %f, %f" % (cb.get_lx(0), cb.get_lx(1), cb.get_lx(2)) )
        print("dx: %f, %f, %f" % (cb.get_dx(0), cb.get_dx(1), cb.get_dx(2)) )
        print("Volume: %f" % (cb.get_volume()) )
        
        print("%s chain model" % (params["chain_model"]))
        print("chi_n: %f," % (params["chi_n"]))
        print("Conformational asymmetry (epsilon): %f" %
            (params["segment_lengths"]["A"]/params["segment_lengths"]["B"]))
        idx = 0
        for polymer in distinct_polymers:
            print("distinct_polymers[%d]:" % (idx) )
            print("    volume fraction: %f, alpha: %f, N: %d" %
                (polymer["volume_fraction"], polymer["alpha"], polymer["pc"].get_n_segment_total()), end=",")
            print(" sequence of block types:", polymer["block_types"])
            print("    total A fraction: %f, average statistical segment length: %f" % 
                (polymer["total_A_fraction"], polymer["statistical_segment_length"]))
            idx += 1
        print("Invariant Polymerization Index: %d" % (params["langevin"]["nbar"]))
        print("Langevin Sigma: %f" % (langevin_sigma))
        print("Random Number Generator: ", np.random.RandomState().get_state()[0])

        #  Save Internal Variables
        self.params = params
        self.distinct_polymers = distinct_polymers
        self.chain_model = params["chain_model"]
        self.chi_n = params["chi_n"]
        self.ds = params["ds"]
        self.epsilon = params["segment_lengths"]["A"]/params["segment_lengths"]["B"]
        self.langevin = params["langevin"]
        self.langevin.update({"sigma":langevin_sigma})

        self.verbose_level = params["verbose_level"]
        self.saddle_max_iter = params["saddle"]["max_iter"]
        self.saddle_tolerance = params["saddle"]["tolerance"]
        self.recording = params["recording"]

        self.cb = cb
        self.am = am

    def save_simulation_data(self, path, w_plus, w_minus, phi):
        mdic = {"dim":self.cb.get_dim(), "nx":self.cb.get_nx(), "lx":self.cb.get_lx(),
            "chi_n":self.chi_n, "chain_model":self.chain_model, "ds":self.ds, "epsilon":self.epsilon,
            "dt":self.langevin["dt"], "nbar":self.langevin["nbar"], "params": self.params,
            "random_generator":np.random.RandomState().get_state()[0],
            "random_seed":np.random.RandomState().get_state()[1],
            "w_plus":w_plus, "w_minus":w_minus, "phi_a":phi["A"], "phi_a":phi["B"]}
        savemat(path, mdic)

    def run(self, w_plus, w_minus):

        # simulation data directory
        pathlib.Path(self.recording["dir"]).mkdir(parents=True, exist_ok=True)

        # flattening arrays
        w_plus  = np.reshape(w_plus,  self.cb.get_n_grid())
        w_minus = np.reshape(w_minus, self.cb.get_n_grid())

        # array for the initial condition
        # free end initial condition. q[0,:] is q and q[1,:] is qdagger.
        # q starts from one end and qdagger starts from the other end.
        self.q1_init = np.ones(self.cb.get_n_grid(), dtype=np.float64)
        self.q2_init = np.ones(self.cb.get_n_grid(), dtype=np.float64)

        # find saddle point 
        phi, _, _, = self.find_saddle_point(w_plus=w_plus, w_minus=w_minus)

        # structure function
        sf_average = np.zeros_like(np.fft.rfftn(np.reshape(w_minus, self.cb.get_nx())),np.float64)

        # init timers
        total_saddle_iter = 0
        time_start = time.time()

        #------------------ run ----------------------
        print("iteration, mass error, total_partitions, energy_total, error_level")
        print("---------- Run  ----------")
        for langevin_step in range(1, self.langevin["max_step"]+1):
            print("Langevin step: ", langevin_step)
            
            # update w_minus
            total_error_level = 0.0
            for w_step in ["predictor", "corrector"]:
                if w_step == "predictor":
                    w_minus_copy = w_minus.copy()
                    normal_noise = np.random.normal(0.0, self.langevin["sigma"], self.cb.get_n_grid())
                    lambda1 = phi["A"]-phi["B"] + 2*w_minus/self.chi_n
                    w_minus += -lambda1*self.langevin["dt"] + normal_noise
                elif w_step == "corrector": 
                    lambda2 = phi["A"]-phi["B"] + 2*w_minus/self.chi_n
                    w_minus = w_minus_copy - 0.5*(lambda1+lambda2)*self.langevin["dt"] + normal_noise
                    
                phi, saddle_iter, error_level = self.find_saddle_point(w_plus=w_plus, w_minus=w_minus)
                total_saddle_iter += saddle_iter
                total_error_level += error_level

            # calculate structure function
            if langevin_step % self.recording["sf_computing_period"] == 0:
                sf_average += np.absolute(np.fft.rfftn(np.reshape(w_minus, self.cb.get_nx()))/self.cb.get_n_grid())**2

            # save structure function
            if langevin_step % self.recording["sf_recording_period"] == 0:
                sf_average *= self.recording["sf_computing_period"]/self.recording["sf_recording_period"]* \
                        self.cb.get_volume()*np.sqrt(self.langevin["nbar"])/self.chi_n**2
                sf_average -= 1.0/(2*self.chi_n)
                mdic = {"dim":self.cb.get_dim(), "nx":self.cb.get_nx(), "lx":self.cb.get_lx(), "params": self.params,
                    "chi_n":self.chi_n, "chain_model":self.chain_model, "ds":self.ds, "epsilon":self.epsilon,
                    "dt": self.langevin["dt"], "nbar":self.langevin["nbar"], "structure_function":sf_average}
                savemat(os.path.join(self.recording["dir"], "structure_function_%06d.mat" % (langevin_step)), mdic)
                sf_average[:,:,:] = 0.0

            # save simulation data
            if (langevin_step) % self.recording["sf_recording_period"] == 0:
                self.save_simulation_data(
                    path=os.path.join(self.recording["dir"], "fields_%06d.mat" % (langevin_step)),
                    w_plus=w_plus, w_minus=w_minus, phi_A=phi["A"], phi_B=phi["B"])

        # estimate execution time
        time_duration = time.time() - time_start
        return total_saddle_iter, total_saddle_iter/self.langevin["max_step"], time_duration/self.langevin["max_step"], total_error_level

    def find_saddle_point(self,w_plus, w_minus):
            
        # assign large initial value for the energy and error
        energy_total = 1e20
        error_level = 1e20

        # reset Anderson mixing module
        self.am.reset_count()

        # array for concentrations
        phi = {"A":np.zeros([self.cb.get_n_grid()], dtype=np.float64),
               "B":np.zeros([self.cb.get_n_grid()], dtype=np.float64)}

        # saddle point iteration begins here
        for saddle_iter in range(1,self.saddle_max_iter+1):
            # for the given fields find the polymer statistics
            phi["A"][:] = 0.0
            phi["B"][:] = 0.0
            for polymer in self.distinct_polymers:
                frac_ = polymer["volume_fraction"]/polymer["alpha"]
                if not "random" in set(polymer["block_types"]):
                    phi_, Q_ = polymer["pseudo"].compute_statistics(self.q1_init,self.q2_init,
                        {"A":w_plus+w_minus,"B":w_plus-w_minus})
                    for i in range(len(polymer["block_types"])):
                        phi[polymer["block_types"][i]] += frac_*phi_[i]
                elif set(polymer["block_types"]) == set(["random"]):
                    phi_, Q_ = polymer["pseudo"].compute_statistics(self.q1_init,self.q2_init,
                        {"random":w_minus*(2*polymer["total_A_fraction"]-1)+w_plus})
                    phi["A"] += frac_*phi_[0]*polymer["total_A_fraction"]
                    phi["B"] += frac_*phi_[0]*(1.0-polymer["total_A_fraction"])
                else:
                    raise ValueError("Unknown species,", set(polymer["block_types"]))
                polymer.update({"phi":phi_})
                polymer.update({"Q": Q_})

            phi_plus = phi["A"] + phi["B"]

            # calculate output fields
            g_plus = phi_plus-1.0
            w_plus_out = w_plus + g_plus 
            self.cb.zero_mean(w_plus_out)

            # error_level measures the "relative distance" between the input and output fields
            old_error_level = error_level
            error_level = np.sqrt(self.cb.inner_product(g_plus,g_plus)/self.cb.get_volume())

            # print iteration # and error levels
            if(self.verbose_level == 2 or self.verbose_level == 1 and
            (error_level < self.saddle_tolerance or saddle_iter == self.saddle_max_iter)):
                # calculate the total energy
                energy_total = self.cb.inner_product(w_minus,w_minus)/self.chi_n/self.cb.get_volume()
                energy_total += self.chi_n/4
                energy_total -= self.cb.integral(w_plus)/self.cb.get_volume()

                for polymer in self.distinct_polymers:
                    energy_total -= polymer["volume_fraction"]/polymer["alpha"]*np.log(polymer["Q"]/self.cb.get_volume())

                # check the mass conservation
                mass_error = self.cb.integral(phi_plus)/self.cb.get_volume() - 1.0
                print("%8d %12.3E " % (saddle_iter, mass_error), end=" [ ")
                for polymer in self.distinct_polymers:
                    print("%13.7E " % (polymer["Q"]), end=" ")
                print("] %15.9f %15.7E " % (energy_total, error_level))

            # conditions to end the iteration
            if error_level < self.saddle_tolerance:
                break
                
            # calculate new fields using simple and Anderson mixing
            self.am.calculate_new_fields(w_plus, w_plus_out, g_plus, old_error_level, error_level)

        self.cb.zero_mean(w_plus)
        return phi, saddle_iter, error_level
