import os
import numpy as np
from langevinfts import *

# OpenMP environment variables
os.environ["MKL_NUM_THREADS"] = "1"  # always 1
os.environ["OMP_STACKSIZE"] = "1G"

class SCFT:
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

        assert(np.isclose(total_volume_fraction,1.0)), "The sum of volume fraction must be equal to 1."

        # (C++ class) Fields Relaxation using Anderson Mixing
        if params["box_is_altering"] : 
            am_n_var = 2*np.prod(params["nx"]) + len(params["lx"])
        else :
            am_n_var = 2*np.prod(params["nx"])
        if "am" in params :
            am = factory.create_anderson_mixing(am_n_var,
                params["am"]["max_hist"],     # maximum number of history
                params["am"]["start_error"],  # when switch to AM from simple mixing
                params["am"]["mix_min"],      # minimum mixing rate of simple mixing
                params["am"]["mix_init"])     # initial mixing rate of simple mixing
        else : 
            am = factory.create_anderson_mixing(am_n_var, 60, 1e-2, 0.1,  0.1)

       # The maximum iteration steps
        if "max_iter" in params :
            max_iter = params["max_iter"]
        else :
            max_iter = 2000      # the number of maximum iterations

        # Tolerance
        if "tolerance" in params :
            tolerance = params["tolerance"]
        else :
            tolerance = 1e-8     # Terminate iteration if the self-consistency error is less than tolerance

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

        mixture.display_unique_blocks()
        mixture.display_unique_branches()

        #  Save Internal Variables
        self.chi_n = params["chi_n"]
        self.box_is_altering = params["box_is_altering"]

        self.max_iter = max_iter
        self.tolerance = tolerance

        self.cb = cb
        self.mixture = mixture
        self.pseudo = pseudo
        self.am = am

    def run(self, initial_fields):

        # assign large initial value for the energy and error
        energy_total = 1.0e20
        error_level = 1.0e20

        # reset Anderson mixing module
        self.am.reset_count()

        # concentration of each monomer
        phi = {}

        # array for output fields
        w_out = np.zeros([2, self.cb.get_n_grid()], dtype=np.float64)

        #------------------ run ----------------------
        print("---------- Run ----------")

        # iteration begins here
        if (self.box_is_altering):
            print("iteration, mass error, total partitions, total energy, error level, box size")
        else:
            print("iteration, mass error, total partitions, total energy, error level")

        # reshape initial fields
        w = np.reshape([initial_fields["A"], initial_fields["B"]], [2, self.cb.get_n_grid()])

        # keep the level of field value
        self.cb.zero_mean(w[0])
        self.cb.zero_mean(w[1])

        for scft_iter in range(1, self.max_iter+1):
            # for the given fields find the polymer statistics
            if self.random_copolymer_exist:
                self.pseudo.compute_statistics({"A":w[0],"B":w[1],"R":w[0]*self.random_A_fraction + w[1]*(1.0-self.random_A_fraction)})
            else:
                self.pseudo.compute_statistics({"A":w[0],"B":w[1]})

            phi["A"] = self.pseudo.get_monomer_concentration("A")
            phi["B"] = self.pseudo.get_monomer_concentration("B")

            if self.random_copolymer_exist:
                phi["R"] = self.pseudo.get_monomer_concentration("R")
                phi["A"] += phi["R"]*self.random_A_fraction
                phi["B"] += phi["R"]*(1.0-self.random_A_fraction)

            # calculate the total energy
            w_minus = (w[0]-w[1])/2
            w_plus  = (w[0]+w[1])/2
            energy_total = self.cb.inner_product(w_minus,w_minus)/self.chi_n/self.cb.get_volume()
            energy_total -= self.cb.integral(w_plus)/self.cb.get_volume()
            for p in range(self.mixture.get_n_polymers()):
                energy_total -= self.mixture.get_polymer(p).get_volume_fraction()/ \
                                self.mixture.get_polymer(p).get_alpha() * \
                                np.log(self.pseudo.get_total_partition(p))
                
            # calculate pressure field for the new field calculation
            xi = 0.5*(w[0]+w[1]-self.chi_n)

            # calculate output fields
            w_out[0] = self.chi_n*phi["B"] + xi
            w_out[1] = self.chi_n*phi["A"] + xi

            # error_level measures the "relative distance" between the input and output fields
            old_error_level = error_level
            w_diff = w_out - w

            # keep the level of field value
            self.cb.zero_mean(w_diff[0])
            self.cb.zero_mean(w_diff[1])

            multi_dot = self.cb.inner_product(w_diff[0],w_diff[0]) + self.cb.inner_product(w_diff[1],w_diff[1])
            multi_dot /= self.cb.inner_product(w[0],w[0]) + self.cb.inner_product(w[1],w[1]) + 1.0
            error_level = np.sqrt(multi_dot)

            # print iteration # and error levels and check the mass conservation
            mass_error = self.cb.integral(phi["A"] + phi["B"])/self.cb.get_volume() - 1.0
            
            if (self.box_is_altering):
                # calculate stress
                stress_array = np.array(self.pseudo.compute_stress())
                error_level += np.sqrt(np.sum(stress_array**2))

                print("%8d %12.3E " %
                (scft_iter, mass_error), end=" [ ")
                for p in range(self.mixture.get_n_polymers()):
                    print("%13.7E " % (self.pseudo.get_total_partition(p)), end=" ")
                print("] %15.9f %15.7E " % (energy_total, error_level), end=" ")
                print("[", ",".join(["%10.7f" % (x) for x in self.cb.get_lx()]), "]")
            else:
                print("%8d %12.3E " % (scft_iter, mass_error), end=" [ ")
                for p in range(self.mixture.get_n_polymers()):
                    print("%13.7E " % (self.pseudo.get_total_partition(p)), end=" ")
                print("] %15.9f %15.7E " % (energy_total, error_level))

            # conditions to end the iteration
            if error_level < self.tolerance:
                break

            # calculate new fields using simple and Anderson mixing
            if (self.box_is_altering):
                dlx = -stress_array
                am_current  = np.concatenate((np.reshape(w,      2*self.cb.get_n_grid()), self.cb.get_lx()))
                am_diff     = np.concatenate((np.reshape(w_diff, 2*self.cb.get_n_grid()), dlx))
                am_new = self.am.calculate_new_fields(am_current, am_diff, old_error_level, error_level)

                # copy fields
                w = np.reshape(am_new[0:2*self.cb.get_n_grid()], (2, self.cb.get_n_grid()))

                # set box size
                # restricting |dLx| to be less than 10 % of Lx
                old_lx = np.array(self.cb.get_lx())
                new_lx = np.array(am_new[-self.cb.get_dim():])
                new_dlx = np.clip((new_lx-old_lx)/old_lx, -0.1, 0.1)
                new_lx = (1 + new_dlx)*old_lx
                self.cb.set_lx(new_lx)

                # update bond parameters using new lx
                self.pseudo.update_bond_function()
            else:
                w = self.am.calculate_new_fields(
                np.reshape(w,      2*self.cb.get_n_grid()),
                np.reshape(w_diff, 2*self.cb.get_n_grid()), old_error_level, error_level)
                w = np.reshape(w, (2, self.cb.get_n_grid()))

        self.phi = phi
        self.w = w

    def get_concentrations(self,):
        return self.phi["A"], self.phi["B"]
    
    def get_fields(self,):
        return self.w[0], self.w[1]