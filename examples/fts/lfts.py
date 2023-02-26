import os
import time
import pathlib
import numpy as np
from scipy.io import savemat, loadmat
from langevinfts import *

# OpenMP environment variables
os.environ["MKL_NUM_THREADS"] = "1"  # always 1
os.environ["OMP_STACKSIZE"] = "1G"

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

        assert(len(params['segment_lengths']) == 2), \
            "Currently, only AB-type polymers are supported."
        assert(len(set(["A","B"]).intersection(set(params['segment_lengths'].keys())))==2), \
            "Use letters 'A' and 'B' for monomer types."
        assert(len(params["distinct_polymers"]) >= 1), \
            "There is no polymer chain."

        # (c++ class) Create a factory for given platform and chain_model
        factory = PlatformSelector.create_factory(platform, params["chain_model"])
        factory.display_info()
        
        # (C++ class) Computation box
        cb = factory.create_computation_box(params["nx"], params["lx"])

        # Polymer chains
        total_volume_fraction = 0.0
        random_count = 0
        for polymer in params["distinct_polymers"]:
            block_length_list = []
            block_monomer_type_list = []
            v_list = []
            u_list = []
            A_fraction = 0.0
            alpha = 0.0  #total_relative_contour_length
            block_count = 0
            is_linear = not "v" in polymer["blocks"][0]
            for block in polymer["blocks"]:
                block_length_list.append(block["length"])
                block_monomer_type_list.append(block["type"])

                if is_linear:
                    assert(not "v" in block), \
                        "Index v should exist in all blocks, or it should not exist in all blocks for each polymer." 
                    assert(not "u" in block), \
                        "Index u should exist in all blocks, or it should not exist in all blocks for each polymer." 

                    v_list.append(block_count)
                    u_list.append(block_count+1)
                else:
                    assert("v" in block), \
                        "Index v should exist in all blocks, or it should not exist in all blocks for each polymer." 
                    assert("u" in block), \
                        "Index u should exist in all blocks, or it should not exist in all blocks for each polymer." 

                    v_list.append(block["v"])
                    u_list.append(block["u"])

                alpha += block["length"]
                if block["type"] == "A":
                    A_fraction += block["length"]
                elif block["type"] == "random":
                    A_fraction += block["length"]*block["fraction"]["A"]
                
                block_count += 1
            total_volume_fraction += polymer["volume_fraction"]
            total_A_fraction = A_fraction/alpha
            statistical_segment_length = \
                np.sqrt(params["segment_lengths"]["A"]**2*total_A_fraction + \
                        params["segment_lengths"]["B"]**2*(1-total_A_fraction))

            if "random" in set(bt.lower() for bt in block_monomer_type_list):
                random_count +=1
                assert(random_count == 1), \
                    "Only one random copolymer is allowed." 
                assert(len(block_monomer_type_list) == 1), \
                    "Only single block random copolymer is allowed."
                assert(np.isclose(polymer["blocks"][0]["fraction"]["A"]+polymer["blocks"][0]["fraction"]["B"],1.0)), \
                    "The sum of volume fraction of random copolymer must be equal to 1."
                params["segment_lengths"].update({"R":statistical_segment_length})
                block_monomer_type_list = ["R"]
                self.random_copolymer_exist = True
                self.random_A_fraction = total_A_fraction

            else:
                self.random_copolymer_exist = False
            
            polymer.update({"block_monomer_types":block_monomer_type_list})
            polymer.update({"block_lengths":block_length_list})
            polymer.update({"v":v_list})
            polymer.update({"u":u_list})

        assert(np.isclose(total_volume_fraction,1.0)), "The sum of volume fraction must be equal to 1."

        # (C++ class) Mixture box
        print(params["segment_lengths"])
        if "use_superposition" in params:
            mixture = factory.create_mixture(params["ds"], params["segment_lengths"], params["use_superposition"])
        else:
            mixture = factory.create_mixture(params["ds"], params["segment_lengths"], True)

        # Add polymer chains
        for polymer in params["distinct_polymers"]:
            # print(polymer["volume_fraction"], polymer["block_monomer_types"], polymer["block_lengths"], polymer["v"], polymer["u"])
            mixture.add_polymer(polymer["volume_fraction"], polymer["block_monomer_types"], polymer["block_lengths"], polymer["v"] ,polymer["u"])

        # (C++ class) Solver using Pseudo-spectral method
        if "reduce_gpu_memory_usage" in params and platform == "cuda":
            pseudo = factory.create_pseudo(cb, mixture, params["reduce_gpu_memory_usage"])
        else:
            pseudo = factory.create_pseudo(cb, mixture, False)

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
        print("Nx:", cb.get_nx())
        print("Lx:", cb.get_lx())
        print("dx:", cb.get_dx())
        print("Volume: %f" % (cb.get_volume()))
        
        print("%s chain model" % (params["chain_model"]))
        print("chi_n: %f," % (params["chi_n"]))
        print("Conformational asymmetry (epsilon): %f" %
            (params["segment_lengths"]["A"]/params["segment_lengths"]["B"]))

        for p in range(mixture.get_n_polymers()):
            print("distinct_polymers[%d]:" % (p) )
            print("\tvolume fraction: %f, alpha: %f, N_total: %d" %
                (mixture.get_polymer(p).get_volume_fraction(),
                 mixture.get_polymer(p).get_alpha(),
                 mixture.get_polymer(p).get_n_segment_total()))
            # add display monomer types and lengths

        print("Invariant Polymerization Index: %d" % (params["langevin"]["nbar"]))
        print("Langevin Sigma: %f" % (langevin_sigma))
        print("Random Number Generator: ", np.random.RandomState().get_state()[0])

        mixture.display_unique_blocks()
        mixture.display_unique_branches()

        #  Save Internal Variables
        self.params = params
        self.chain_model = params["chain_model"]
        self.chi_n = params["chi_n"]
        self.ds = params["ds"]
        self.epsilon = params["segment_lengths"]["A"]/params["segment_lengths"]["B"]
        self.langevin = params["langevin"]
        self.langevin.update({"sigma":langevin_sigma})

        self.verbose_level = params["verbose_level"]
        self.saddle = params["saddle"]
        self.recording = params["recording"]

        self.cb = cb
        self.mixture = mixture
        self.pseudo = pseudo
        self.am = am

    def save_simulation_data(self, path, w_plus, w_minus, phi):
        mdic = {"dim":self.cb.get_dim(), "nx":self.cb.get_nx(), "lx":self.cb.get_lx(),
            "chi_n":self.chi_n, "chain_model":self.chain_model, "ds":self.ds, "epsilon":self.epsilon,
            "dt":self.langevin["dt"], "nbar":self.langevin["nbar"], "params": self.params,
            "random_generator":np.random.RandomState().get_state()[0],
            "random_seed":np.random.RandomState().get_state()[1],
            "w_plus":w_plus, "w_minus":w_minus, "phi_a":phi["A"], "phi_b":phi["B"]}
        savemat(path, mdic)

    def run(self, w_plus, w_minus):

        # simulation data directory
        pathlib.Path(self.recording["dir"]).mkdir(parents=True, exist_ok=True)

        # flattening arrays
        w_plus  = np.reshape(w_plus,  self.cb.get_n_grid())
        w_minus = np.reshape(w_minus, self.cb.get_n_grid())

        # find saddle point 
        phi, _, _, = self.find_saddle_point(w_plus=w_plus, w_minus=w_minus)

        # structure function
        sf_average = np.zeros_like(np.fft.rfftn(np.reshape(w_minus, self.cb.get_nx())),np.float64)

        # init timers
        total_saddle_iter = 0
        time_start = time.time()

        #------------------ run ----------------------
        print("iteration, mass error, total partitions, total energy, incompressibility error")
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
            if (langevin_step) % self.recording["recording_period"] == 0:
                self.save_simulation_data(
                    path=os.path.join(self.recording["dir"], "fields_%06d.mat" % (langevin_step)),
                    w_plus=w_plus, w_minus=w_minus, phi=phi)

        # estimate execution time
        time_duration = time.time() - time_start
        return total_saddle_iter, total_saddle_iter/self.langevin["max_step"], time_duration/self.langevin["max_step"], total_error_level

    def find_saddle_point(self,w_plus, w_minus):
            
        # assign large initial value for the energy and error
        energy_total = 1e20
        error_level = 1e20

        # reset Anderson mixing module
        self.am.reset_count()

        # concentration of each monomer
        phi = {}

        # saddle point iteration begins here
        for saddle_iter in range(1,self.saddle["max_iter"]+1):
            # for the given fields find the polymer statistics
            if self.random_copolymer_exist:
                self.pseudo.compute_statistics({"A":w_plus+w_minus,"B":w_plus-w_minus,"R":w_minus*(2*self.random_A_fraction-1)+w_plus})
            else:
                self.pseudo.compute_statistics({"A":w_plus+w_minus,"B":w_plus-w_minus})

            phi["A"] = self.pseudo.get_monomer_concentration("A")
            phi["B"] = self.pseudo.get_monomer_concentration("B")

            if self.random_copolymer_exist:
                phi["R"] = self.pseudo.get_monomer_concentration("R")
                phi["A"] += phi["R"]*self.random_A_fraction
                phi["B"] += phi["R"]*(1.0-self.random_A_fraction)

            phi_plus = phi["A"] + phi["B"]

            # calculate output fields
            g_plus = phi_plus-1.0

            # error_level measures the "relative distance" between the input and output fields
            old_error_level = error_level
            error_level = np.sqrt(self.cb.inner_product(g_plus,g_plus)/self.cb.get_volume())

            # print iteration # and error levels
            if(self.verbose_level == 2 or self.verbose_level == 1 and
            (error_level < self.saddle["tolerance"] or saddle_iter == self.saddle["max_iter"])):
            
                # calculate the total energy
                energy_total = self.cb.inner_product(w_minus,w_minus)/self.chi_n/self.cb.get_volume()
                energy_total += self.chi_n/4
                energy_total -= self.cb.integral(w_plus)/self.cb.get_volume()
                for p in range(self.mixture.get_n_polymers()):
                    energy_total -= self.mixture.get_polymer(p).get_volume_fraction()/ \
                                    self.mixture.get_polymer(p).get_alpha() * \
                                    np.log(self.pseudo.get_total_partition(p))

                # check the mass conservation
                mass_error = self.cb.integral(phi_plus)/self.cb.get_volume() - 1.0
                print("%8d %12.3E " % (saddle_iter, mass_error), end=" [ ")
                for p in range(self.mixture.get_n_polymers()):
                    print("%13.7E " % (self.pseudo.get_total_partition(p)), end=" ")
                print("] %15.9f %15.7E " % (energy_total, error_level))

            # conditions to end the iteration
            if error_level < self.saddle["tolerance"]:
                break
                
            # calculate new fields using simple and Anderson mixing
            w_plus[:] = self.am.calculate_new_fields(w_plus, g_plus, old_error_level, error_level)
        self.cb.zero_mean(w_plus)
        return phi, saddle_iter, error_level
