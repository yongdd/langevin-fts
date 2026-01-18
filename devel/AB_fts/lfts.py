import os
import re
import time
import pathlib
import numpy as np
import itertools
from scipy.io import savemat, loadmat
from polymerfts import *

# OpenMP environment variables
os.environ["MKL_NUM_THREADS"] = "1"  # always 1
os.environ["OMP_STACKSIZE"] = "1G"

def calculate_sigma(langevin_nbar, langevin_dt, n_grids, volume):
        return np.sqrt(2*langevin_dt*n_grids/(volume*np.sqrt(langevin_nbar)))

class LFTS:
    def __init__(self, params, random_seed=None):

        # Segment length
        self.monomer_types = sorted(list(params["segment_lengths"].keys()))
        
        assert(len(self.monomer_types) == len(set(self.monomer_types))), \
            "There are duplicated monomer_types"

        # Choose platform among [cuda, cpu-mkl]
        avail_platforms = PlatformSelector.avail_platforms()
        if "platform" in params:
            platform = params["platform"]
        elif "cpu-mkl" in avail_platforms and len(params["nx"]) == 1: # for 1D simulation, use CPU
            platform = "cpu-mkl"
        elif "cuda" in avail_platforms: # If cuda is available, use GPU
            platform = "cuda"
        else:
            platform = avail_platforms[0]

        # (C++ class) Create a factory for given platform and chain_model
        if "use_checkpointing" in params and platform == "cuda":
            factory = PlatformSelector.create_factory(platform, params["use_checkpointing"])
        else:
            factory = PlatformSelector.create_factory(platform, False)
        factory.display_info()

        # (C++ class) Computation box
        cb = factory.create_computation_box(params["nx"], params["lx"])

        # Flory-Huggins parameters, χN
        self.chi_n = {}
        for monomer_pair_str, chin_value in params["chi_n"].items():
            monomer_pair = re.split(',| |_|/', monomer_pair_str)
            assert(monomer_pair[0] in params["segment_lengths"]), \
                f"Monomer type '{monomer_pair[0]}' is not in 'segment_lengths'."
            assert(monomer_pair[1] in params["segment_lengths"]), \
                f"Monomer type '{monomer_pair[1]}' is not in 'segment_lengths'."
            assert(monomer_pair[0] != monomer_pair[1]), \
                "Do not add self interaction parameter, " + monomer_pair_str + "."
            monomer_pair.sort()
            sorted_monomer_pair = monomer_pair[0] + "," + monomer_pair[1]
            assert(not sorted_monomer_pair in self.chi_n), \
                f"There are duplicated χN ({sorted_monomer_pair}) parameters."
            self.chi_n[sorted_monomer_pair] = chin_value

        for monomer_pair in itertools.combinations(self.monomer_types, 2):
            monomer_pair = list(monomer_pair)
            monomer_pair.sort()
            sorted_monomer_pair = monomer_pair[0] + "," + monomer_pair[1] 
            if not sorted_monomer_pair in self.chi_n:
                self.chi_n[sorted_monomer_pair] = 0.0
        
        # Total volume fraction
        assert(len(params["distinct_polymers"]) >= 1), \
            "There is no polymer chain."

        total_volume_fraction = 0.0
        for polymer in params["distinct_polymers"]:
            total_volume_fraction += polymer["volume_fraction"]
        assert(np.isclose(total_volume_fraction,1.0)), "The sum of volume fractions must be equal to 1."

        # Polymer Chains
        for polymer_counter, polymer in enumerate(params["distinct_polymers"]):
            blocks_input = []
            alpha = 0.0             # total_relative_contour_length
            has_node_number = not "v" in polymer["blocks"][0]
            for block in polymer["blocks"]:
                alpha += block["length"]
                if has_node_number:
                    assert(not "v" in block), \
                        "Index v should exist in all blocks, or it should not exist in all blocks for each polymer." 
                    assert(not "u" in block), \
                        "Index u should exist in all blocks, or it should not exist in all blocks for each polymer." 
                    blocks_input.append([block["type"], block["length"], len(blocks_input), len(blocks_input)+1])
                else:
                    assert("v" in block), \
                        "Index v should exist in all blocks, or it should not exist in all blocks for each polymer." 
                    assert("u" in block), \
                        "Index u should exist in all blocks, or it should not exist in all blocks for each polymer." 
                    blocks_input.append([block["type"], block["length"], block["v"], block["u"]])
            polymer.update({"blocks_input":blocks_input})

        # Random Copolymer Chains
        self.random_fraction = {}
        for polymer in params["distinct_polymers"]:

            is_random = False
            for block in polymer["blocks"]:
                if "fraction" in block:
                    is_random = True
            if not is_random:
                continue

            assert(len(polymer["blocks"]) == 1), \
                "Only single block random copolymer is allowed."

            statistical_segment_length = 0
            total_random_fraction = 0
            for monomer_type in polymer["blocks"][0]["fraction"]:
                statistical_segment_length += params["segment_lengths"][monomer_type]**2 * polymer["blocks"][0]["fraction"][monomer_type]
                total_random_fraction += polymer["blocks"][0]["fraction"][monomer_type]
            statistical_segment_length = np.sqrt(statistical_segment_length)

            assert(np.isclose(total_random_fraction, 1.0)), \
                "The sum of volume fractions of random copolymer must be equal to 1."

            random_type_string = polymer["blocks"][0]["type"]
            assert(not random_type_string in params["segment_lengths"]), \
                f"The name of random copolymer '{random_type_string}' is already used as a type in 'segment_lengths' or other random copolymer"

            # Add random copolymers
            polymer["block_monomer_types"] = [random_type_string]
            params["segment_lengths"].update({random_type_string:statistical_segment_length})
            self.random_fraction[random_type_string] = polymer["blocks"][0]["fraction"]

        # (C++ class) Molecules list
        molecules = factory.create_molecules_information(params["chain_model"], params["ds"], params["segment_lengths"])

        # Add polymer chains
        for polymer in params["distinct_polymers"]:
            molecules.add_polymer(polymer["volume_fraction"], polymer["blocks_input"])

        # (C++ class) Propagator Computation Optimizer
        if "aggregate_propagator_computation" in params:
            propagator_computation_optimizer = factory.create_propagator_computation_optimizer(molecules, params["aggregate_propagator_computation"])
        else:
            propagator_computation_optimizer = factory.create_propagator_computation_optimizer(molecules, True)

        # (C++ class) Solver using Pseudo-spectral method
        solver = factory.create_propagator_computation(cb, molecules, propagator_computation_optimizer, "rqm4")

        # (C++ class) Fields Relaxation using Anderson Mixing
        am = factory.create_anderson_mixing(
            np.prod(params["nx"]),        # the number of variables
            params["am"]["max_hist"],     # maximum number of history
            params["am"]["start_error"],  # when switch to AM from simple mixing
            params["am"]["mix_min"],      # minimum mixing rate of simple mixing
            params["am"]["mix_init"])     # initial mixing rate of simple mixing

        # Langevin Dynamics
        # standard deviation of normal noise
        langevin_sigma = calculate_sigma(params["langevin"]["nbar"], params["langevin"]["dt"], np.prod(params["nx"]), np.prod(params["lx"]))

        # Set random generator
        if random_seed is None:         
            self.random_bg = np.random.PCG64()  # Set random bit generator
        else:
            self.random_bg = np.random.PCG64(random_seed)
        self.random = np.random.Generator(self.random_bg)
        
        # -------------- print simulation parameters ------------
        print("---------- Simulation Parameters ----------")
        print("Platform :", platform)
        print("Box Dimension: %d" % (cb.get_dim()))
        print("Nx:", cb.get_nx())
        print("Lx:", cb.get_lx())
        print("dx:", cb.get_dx())
        print("Volume: %f" % (cb.get_volume()))

        print("Chain model: %s" % (params["chain_model"]))
        print("Segment lengths:\n\t", list(params["segment_lengths"].items()))
        print("Conformational asymmetry (epsilon): ")
        for monomer_pair in itertools.combinations(self.monomer_types,2):
            print("\t%s/%s: %f" % (monomer_pair[0], monomer_pair[1], params["segment_lengths"][monomer_pair[0]]/params["segment_lengths"][monomer_pair[1]]))

        print("χN: ")
        for key in self.chi_n:
            print("\t%s: %f" % (key, self.chi_n[key]))

        for p in range(molecules.get_n_polymer_types()):
            print("distinct_polymers[%d]:" % (p) )
            print("\tvolume fraction: %f, alpha: %f, N: %d" %
                (molecules.get_polymer(p).get_volume_fraction(),
                 molecules.get_polymer(p).get_alpha(),
                 molecules.get_polymer(p).get_n_segment_total()))

        print("Invariant Polymerization Index (N_Ref): %d" % (params["langevin"]["nbar"]))
        print("Langevin Sigma: %f" % (langevin_sigma))
        print("Random Number Generator: ", self.random_bg.state)

        propagator_computation_optimizer.display_blocks()
        propagator_computation_optimizer.display_propagators()

        #  Save Internal Variables
        self.params = params
        self.chain_model = params["chain_model"]
        self.ds = params["ds"]
        self.langevin = params["langevin"].copy()
        self.langevin.update({"sigma":langevin_sigma})

        self.verbose_level = params["verbose_level"]
        self.saddle = params["saddle"].copy()
        self.recording = params["recording"].copy()

        self.cb = cb
        self.molecules = molecules
        self.propagator_computation_optimizer = propagator_computation_optimizer
        self.solver = solver 
        self.am = am

    def save_simulation_data(self, path, w_plus, w_minus, phi):
        mdic = {"dim":self.cb.get_dim(), "nx":self.cb.get_nx(), "lx":self.cb.get_lx(),
            "chi_n":self.chi_n["A,B"], "chain_model":self.chain_model, "ds":self.ds, "epsilon":self.epsilon,
            "dt":self.langevin["dt"], "nbar":self.langevin["nbar"], "params": self.params,
            "random_generator": self.random_bg.state["bit_generator"],
            "random_state_state": str(self.random_bg.state["state"]["state"]),
            "random_state_inc": str(self.random_bg.state["state"]["inc"]),
            "w_plus":w_plus, "w_minus":w_minus, "phi_a":phi["A"], "phi_b":phi["B"]}
        savemat(path, mdic, do_compression=True)

    def run(self, w_plus, w_minus):

        # simulation data directory
        pathlib.Path(self.recording["dir"]).mkdir(parents=True, exist_ok=True)

        # flattening arrays
        w_plus  = np.reshape(w_plus,  self.cb.get_total_grid())
        w_minus = np.reshape(w_minus, self.cb.get_total_grid())

        # find saddle point 
        phi, _, _, = self.find_saddle_point(w_plus=w_plus, w_minus=w_minus)

        # structure function
        sf_average = np.zeros_like(np.fft.rfftn(np.reshape(w_minus, self.cb.get_nx())),np.float64)

        # create an empty array for field update algorithm
        normal_noise_prev = np.zeros(self.cb.get_total_grid(), dtype=np.float64)

        # init timers
        total_saddle_iter = 0
        total_error_level = 0
        time_start = time.time()

        #------------------ run ----------------------
        print("iteration, mass error, total partitions, total energy, incompressibility error")
        print("---------- Run  ----------")
        for langevin_step in range(1, self.langevin["max_step"]+1):
            print("Langevin step: ", langevin_step)
            
            # Update w_exchange using Leimkuhler-Matthews method
            normal_noise_current = self.random.normal(0.0, self.langevin["sigma"], self.cb.get_total_grid())
            lambda_minus = phi["A"]-phi["B"] + 2*w_minus/self.chi_n["A,B"]

            # print("w_minus:\n", w_minus)
            # print("w_lambda:\n", lambda_minus)
            # print("noise:\n",(normal_noise_prev + normal_noise_current)/2)
            # print("dw_lambda:\n", -lambda_minus*self.langevin["dt"] + (normal_noise_prev + normal_noise_current)/2)

            w_minus += -lambda_minus*self.langevin["dt"] + (normal_noise_prev + normal_noise_current)/2


            # swap two noise arrays
            normal_noise_prev, normal_noise_current = normal_noise_current, normal_noise_prev

            # find saddle point of the pressure field
            phi, saddle_iter, error_level = self.find_saddle_point(w_plus=w_plus, w_minus=w_minus)
            total_saddle_iter += saddle_iter
            total_error_level += error_level

            # calculate structure function
            if langevin_step % self.recording["sf_computing_period"] == 0:
                sf_average += np.absolute(np.fft.rfftn(np.reshape(w_minus, self.cb.get_nx()))/self.cb.get_total_grid())**2

            # save structure function
            if langevin_step % self.recording["sf_recording_period"] == 0:
                sf_average *= self.recording["sf_computing_period"]/self.recording["sf_recording_period"]* \
                        self.cb.get_volume()*np.sqrt(self.langevin["nbar"])/self.chi_n["A,B"]**2
                sf_average -= 1.0/(2*self.chi_n["A,B"])
                mdic = {"dim":self.cb.get_dim(), "nx":self.cb.get_nx(), "lx":self.cb.get_lx(), "params": self.params,
                    "chi_n":self.chi_n["A,B"], "chain_model":self.chain_model, "ds":self.ds, "epsilon":self.epsilon,
                    "dt": self.langevin["dt"], "nbar":self.langevin["nbar"], "structure_function":sf_average}
                savemat(os.path.join(self.recording["dir"], "structure_function_%06d.mat" % (langevin_step)), mdic, do_compression=True)
                sf_average[:,:,:] = 0.0

            # save simulation data
            if (langevin_step) % self.recording["recording_period"] == 0:
                self.save_simulation_data(
                    path=os.path.join(self.recording["dir"], "fields_%06d.mat" % (langevin_step)),
                    w_plus=w_plus, w_minus=w_minus, phi=phi)

        # estimate execution time
        time_duration = time.time() - time_start
        return total_saddle_iter, total_saddle_iter/self.langevin["max_step"], time_duration/self.langevin["max_step"], total_error_level/self.langevin["max_step"]

    def find_saddle_point(self,w_plus, w_minus):
            
        # assign large initial value for the energy and error
        energy_total = 1e20
        error_level = 1e20

        # reset Anderson mixing module
        self.am.reset_count()

        # concentration of each monomer
        phi = {}

        # compute hamiltonian part that is independent of w_plus
        energy_total_minus = np.dot(w_minus,w_minus)/self.chi_n["A,B"]/self.cb.get_total_grid()
        energy_total_minus += self.chi_n["A,B"]/4

        # saddle point iteration begins here
        for saddle_iter in range(1,self.saddle["max_iter"]+1):

            # for the given fields compute the polymer statistics
            if len(self.random_fraction.items()) > 0:
                self.solver.compute_propagators({"A":w_plus+w_minus,"B":w_plus-w_minus,"R":w_minus*(2*self.random_A_fraction-1)+w_plus})
            else:
                self.solver.compute_propagators({"A":w_plus+w_minus,"B":w_plus-w_minus})
            self.solver.compute_concentrations()

            phi["A"] = self.solver.get_total_concentration("A")
            phi["B"] = self.solver.get_total_concentration("B")

            if len(self.random_fraction.items()) > 0:
                phi["R"] = self.solver.get_total_concentration("R")
                phi["A"] += phi["R"]*self.random_A_fraction
                phi["B"] += phi["R"]*(1.0-self.random_A_fraction)

            # calculate incompressibility error
            old_error_level = error_level
            g_plus = phi["A"] + phi["B"] - 1.0
            error_level = np.sqrt(np.dot(g_plus, g_plus)/self.cb.get_total_grid())

            # print iteration # and error levels
            if(self.verbose_level == 2 or self.verbose_level == 1 and
            (error_level < self.saddle["tolerance"] or saddle_iter == self.saddle["max_iter"])):
            
                # calculate the total energy
                energy_total = energy_total_minus - np.mean(w_plus)
                for p in range(self.molecules.get_n_polymer_types()):
                    energy_total -= self.molecules.get_polymer(p).get_volume_fraction()/ \
                                    self.molecules.get_polymer(p).get_alpha() * \
                                    np.log(self.solver.get_total_partition(p))

                # check the mass conservation
                mass_error = np.mean(g_plus)
                print("%8d %12.3E " % (saddle_iter, mass_error), end=" [ ")
                for p in range(self.molecules.get_n_polymer_types()):
                    print("%13.7E " % (self.solver.get_total_partition(p)), end=" ")
                print("] %15.9f %15.7E " % (energy_total, error_level))

            # conditions to end the iteration
            if error_level < self.saddle["tolerance"]:
                break
                
            # calculate new fields using simple and Anderson mixing
            w_plus[:] = self.am.calculate_new_fields(w_plus, g_plus, old_error_level, error_level)
        w_plus -= np.mean(w_plus)
        return phi, saddle_iter, error_level
