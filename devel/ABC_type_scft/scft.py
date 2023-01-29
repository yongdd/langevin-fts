import os
import string
import numpy as np
import itertools
from langevinfts import *

# OpenMP environment variables
os.environ["MKL_NUM_THREADS"] = "1"  # always 1
os.environ["OMP_STACKSIZE"] = "1G"

class SCFT:
    def __init__(self, params):

        # Segment length
        self.monomer_types = sorted(list(params["segment_lengths"].keys()))
        assert(len(self.monomer_types) >=2 and len(self.monomer_types) <= 3), \
            "Currently, only AB- or ABC-type polymers are supported."
        if len(self.monomer_types) == 2:
            assert( len(set(["A","B"]).intersection(self.monomer_types)) == len(self.monomer_types)), \
                "Use letters 'A' and 'B' for monomer names."
        elif len(self.monomer_types) == 3:
            assert( len(set(["A","B","C"]).intersection(self.monomer_types)) == len(self.monomer_types)), \
                "Use letters 'A', 'B' and 'C' for monomer names."

        # Flory-Huggins parameters, chi*N
        self.chi_n = {}
        for pair_chi_n in params["chi_n"]:
            assert(pair_chi_n[0] in params["segment_lengths"]), \
                f"Monomer type '{pair_chi_n[0]}' is not in 'segment_lengths'."
            assert(pair_chi_n[1] in params["segment_lengths"]), \
                f"Monomer type '{pair_chi_n[1]}' is not in 'segment_lengths'."
            assert(len(set(pair_chi_n[0:2])) == 2), \
                "Do not add self interaction parameter, " + str(pair_chi_n[0:3]) + "."
            assert(pair_chi_n[2] >= 0), \
                f"chi N ({pair_chi_n[2]}) must be non-negative."
            assert(not frozenset(pair_chi_n[0:2]) in self.chi_n), \
                f"There are duplicated chi N ({pair_chi_n[0:2]}) parameters."
            self.chi_n[frozenset(pair_chi_n[0:2])] = pair_chi_n[2]

        for monomer_pair in itertools.combinations(self.monomer_types, 2):
            if not frozenset(list(monomer_pair)) in self.chi_n:
                self.chi_n[frozenset(list(monomer_pair))] = 0.0

        # exchange mapping matrix.
        S = len(self.monomer_types)
        self.matrix_o = np.zeros((S-1,S-1))
        self.matrix_a = np.zeros((S,S))
        self.matrix_a_inv = np.zeros((S,S))
        self.vector_s = np.zeros(S-1)

        for i in range(S-1):
            key = frozenset([self.monomer_types[i], self.monomer_types[S-1]])
            self.vector_s[i] = self.chi_n[key]

        # if S == 2: # if there are two monomers, use the traditional two-species exchange mapping
        #     key = frozenset([self.monomer_types[0], self.monomer_types[1]])
        #     self.exchange_eigenvalues = np.array([-self.chi_n[key]])
        #     self.matrix_o = np.array([0])
        #     self.matrix_a     = np.array([[1,1],[-1,1]])
        #     self.matrix_a_inv = np.array([[0.5,-0.5],[0.5,0.5]])

        # elif S >= 3:
        matrix_chi = np.zeros((S,S))
        matrix_chin = np.zeros((S-1,S-1))

        for i in range(S):
            for j in range(i+1,S):
                key = frozenset([self.monomer_types[i], self.monomer_types[j]])
                if key in self.chi_n:
                    matrix_chi[i,j] = self.chi_n[key]
                    matrix_chi[j,i] = self.chi_n[key]
        
        for i in range(S-1):
            for j in range(S-1):
                matrix_chin[i,j] = matrix_chi[i,j] - matrix_chi[i,S-1] - matrix_chi[j,S-1]

            
        # print(matrix_chi)
        # print(matrix_chin)

        self.exchange_eigenvalues, self.matrix_o = np.linalg.eig(matrix_chin)

        assert(self.exchange_eigenvalues < 0).all(), \
            f"There are non-negative eigenvalues {self.exchange_eigenvalues}. Cannot run with these chi_n parameters."

        self.matrix_a[0:S-1,0:S-1] = self.matrix_o[0:S-1,0:S-1]
        self.matrix_a[:,S-1] = 1
    
        self.matrix_a_inv[0:S-1,0:S-1] = np.transpose(self.matrix_o[0:S-1,0:S-1])
        for i in range(S-1):
            self.matrix_a_inv[i,S-1] =  -np.sum(self.matrix_o[:,i])
            self.matrix_a_inv[S-1,S-1] = 1

        # Total volume fraction
        assert(len(params["distinct_polymers"]) >= 1), \
            "There is no polymer chain."

        total_volume_fraction = 0.0
        for polymer in params["distinct_polymers"]:
            total_volume_fraction += polymer["volume_fraction"]
        assert(np.isclose(total_volume_fraction,1.0)), "The sum of volume fraction must be equal to 1."

        # Polymer Chains
        self.random_fraction = {}
        for polymer_counter, polymer in enumerate(params["distinct_polymers"]):
            block_length_list = []
            block_monomer_type_list = []
            v_list = []
            u_list = []

            alpha = 0.0             # total_relative_contour_length
            block_count = 0
            is_linear_chain = not "v" in polymer["blocks"][0]
            for block in polymer["blocks"]:
                block_length_list.append(block["length"])
                block_monomer_type_list.append(block["type"])
                alpha += block["length"]

                if is_linear_chain:
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
                block_count += 1

            polymer.update({"block_monomer_types":block_monomer_type_list})
            polymer.update({"block_lengths":block_length_list})
            polymer.update({"v":v_list})
            polymer.update({"u":u_list})

        # Random Copolymer Chains
        for polymer_counter, polymer in enumerate(params["distinct_polymers"]):

            is_random = False
            for block in polymer["blocks"]:
                if "random" in block["type"]:
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
                "The sum of volume fraction of random copolymer must be equal to 1."

            # assign polymer_counter to an arbitrary string for each distinct random copolymers
            alpha_dict = dict(zip(range(1, 10), string.ascii_uppercase[-20:]))
            random_string = "R_"
            for num in str(polymer_counter):
                random_string += alpha_dict[int(num)]

            # add random copolymers
            polymer["block_monomer_types"] = [random_string]
            params["segment_lengths"].update({random_string:statistical_segment_length})
            self.random_fraction[random_string] = polymer["blocks"][0]["fraction"]

        # (c++ class) Create a factory for given platform and chain_model
        factory = PlatformSelector.create_factory(params["platform"], params["chain_model"])

        # (C++ class) Computation box
        cb = factory.create_computation_box(params["nx"], params["lx"])

        # (C++ class) Mixture box
        mixture = factory.create_mixture(params["ds"], params["segment_lengths"])

        # Add polymer chains
        for polymer in params["distinct_polymers"]:
            # print(polymer["volume_fraction"], polymer["block_monomer_types"], polymer["block_lengths"], polymer["v"], polymer["u"])
            mixture.add_polymer(polymer["volume_fraction"], polymer["block_monomer_types"], polymer["block_lengths"], polymer["v"] ,polymer["u"])

        # (C++ class) Solvers using Pseudo-spectral method
        pseudo = factory.create_pseudo(cb, mixture)

        # (C++ class) Fields Relaxation using Anderson Mixing
        if params["box_is_altering"] : 
            am_n_var = len(self.monomer_types)*np.prod(params["nx"]) + len(params["lx"])
        else :
            am_n_var = len(self.monomer_types)*np.prod(params["nx"])
        if "am" in params :
            am = factory.create_anderson_mixing(am_n_var,
                params["am"]["max_hist"],     # maximum number of history
                params["am"]["start_error"],  # when switch to AM from simple mixing
                params["am"]["mix_min"],      # minimum mixing rate of simple mixing
                params["am"]["mix_init"])     # initial mixing rate of simple mixing
        else : 
            am = factory.create_anderson_mixing(am_n_var, 20, 1e-2, 0.1,  0.1)

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
        print("Platform:", params["platform"])
        print("Box Dimension: %d" % (cb.get_dim()))
        print("Nx: %d, %d, %d" % (cb.get_nx(0), cb.get_nx(1), cb.get_nx(2)) )
        print("Lx: %f, %f, %f" % (cb.get_lx(0), cb.get_lx(1), cb.get_lx(2)) )
        print("dx: %f, %f, %f" % (cb.get_dx(0), cb.get_dx(1), cb.get_dx(2)) )
        print("Volume: %f" % (cb.get_volume()) )

        print("Chain model: %s" % (params["chain_model"]))

        print("Segment lengths:\n\t", list(params["segment_lengths"].items()))
        print("Conformational asymmetry (epsilon): ")
        for monomer_pair in itertools.combinations(self.monomer_types,2):
            # print(monomer_pair)
            if "R_" in monomer_pair[0] or "R_" in monomer_pair[1]:
                continue
            print("\t%s/%s: %f" % (monomer_pair[0], monomer_pair[1], params["segment_lengths"][monomer_pair[0]]/params["segment_lengths"][monomer_pair[1]]))

        print("chiN: ")
        for pair in self.chi_n:
            # print("\t%s, %s: %f" % (list(chi_n[0])[0], list(chi_n[0])[1], chi_n[1]))
            print("\t%s, %s: %f" % (list(pair)[0], list(pair)[1], self.chi_n[pair]))

        print("Eigenvalues:\n\t", self.exchange_eigenvalues)
        print("Column eigenvectors O:\n\t", str(self.matrix_o).replace("\n", "\n\t"))
        print("Vector chi_iS:\n\t", str(self.vector_s).replace("\n", "\n\t"))
        print("Mapping matrix A:\n\t", str(self.matrix_a).replace("\n", "\n\t"))
        print("Inverse of A:\n\t", str(self.matrix_a_inv).replace("\n", "\n\t"))
        print("A*Inverse[A]:\n\t", str(np.matmul(self.matrix_a, self.matrix_a_inv)).replace("\n", "\n\t"))

        for p in range(mixture.get_n_polymers()):
            print("distinct_polymers[%d]:" % (p) )
            print("\tvolume fraction: %f, alpha: %f, N: %d" %
                (mixture.get_polymer(p).get_volume_fraction(),
                 mixture.get_polymer(p).get_alpha(),
                 mixture.get_polymer(p).get_n_segment_total()))
            # add display monomer types and lengths

        mixture.display_unique_branches()
        mixture.display_unique_blocks()

        #  Save Internal Variables
        self.box_is_altering = params["box_is_altering"]

        self.max_iter = max_iter
        self.tolerance = tolerance

        self.cb = cb
        self.mixture = mixture
        self.pseudo = pseudo
        self.am = am

    def run(self, initial_fields):

        # the number of components
        S = len(self.monomer_types)

        # assign large initial value for the energy and error
        energy_total = 1.0e20
        error_level = 1.0e20

        # reset Anderson mixing module
        self.am.reset_count()

        #------------------ run ----------------------
        print("---------- Run ----------")

        # iteration begins here
        print("iteration, mass error, total_partitions, energy_total, error_level", end="")
        if (self.box_is_altering):
            print(", box size")
        else:
            print("")

        # reshape initial fields
        w = np.zeros([S, self.cb.get_n_grid()], dtype=np.float64)
        for i in range(S):
            w[i,:] = np.reshape(initial_fields[self.monomer_types[i]],  self.cb.get_n_grid())

        # exchange-mapped chemical potential fields
        w_exchange = np.matmul(self.matrix_a_inv, w)

        # keep the level of field value
        for i in range(S):
            self.cb.zero_mean(w_exchange[i,:])

        for scft_iter in range(1, self.max_iter+1):

            # convert to species chemical potential fields
            w = np.matmul(self.matrix_a, w_exchange)

            # make a dictionary for input fields 
            w_input = {}
            for i in range(S):
                w_input[self.monomer_types[i]] = w[i,:]
            for random_polymer_name, random_fraction in self.random_fraction.items():
                w_input[random_polymer_name] = np.zeros(self.cb.get_n_grid(), dtype=np.float64)
                for monomer_type, fraction in random_fraction.items():
                    w_input[random_polymer_name] += w_input[monomer_type]*fraction

            # for the given fields find the polymer statistics
            self.pseudo.compute_statistics(w_input)

            # compute concentration for each monomer type
            phi = {}
            for monomer_type in self.monomer_types:
                phi[monomer_type] = self.pseudo.get_monomer_concentration(monomer_type)

            # add random copolymer concentration to each monomer type
            for random_polymer_name, random_fraction in self.random_fraction.items():
                phi[random_polymer_name] = self.pseudo.get_monomer_concentration(random_polymer_name)
                for monomer_type, fraction in random_fraction.items():
                    phi[monomer_type] += phi[random_polymer_name]*fraction

            # calculate the total energy
            energy_total = 0.0

            for i in range(S-1):
                energy_total -= 0.5/self.exchange_eigenvalues[i]*self.cb.inner_product(w_exchange[i],w_exchange[i])/self.cb.get_volume()
            for i in range(S-1):
                for j in range(S-1):
                    energy_total += 1.0/self.exchange_eigenvalues[i]*self.matrix_o[j,i]*self.vector_s[j]*self.cb.integral(w_exchange[i])/self.cb.get_volume()
            energy_total -= self.cb.integral(w_exchange[S-1])

            for p in range(self.mixture.get_n_polymers()):
                energy_total -= self.mixture.get_polymer(p).get_volume_fraction()/ \
                                self.mixture.get_polymer(p).get_alpha() * \
                                np.log(self.pseudo.get_total_partition(p)/self.cb.get_volume())
                
            # print(np.std(w_exchange[i] - (w[0] - w[1])))

            # calculate self-consistent error 
            w_diff = np.zeros([S, self.cb.get_n_grid()], dtype=np.float64) # array for output fields
            for i in range(S-1):
                w_diff[i] = 1.0/self.exchange_eigenvalues[i]*w_exchange[i]
                for j in range(S-1):
                    w_diff[i] -= 1.0/self.exchange_eigenvalues[i]*self.matrix_o[j,i]*self.vector_s[j]
                    w_diff[i] -= self.matrix_o[j,i]*phi[self.monomer_types[j]]

            for i in range(S):
                w_diff[S-1] += phi[self.monomer_types[i]]
            w_diff[S-1] -= 1.0

            # keep the level of field value
            for i in range(S):
                self.cb.zero_mean(w_diff[i,:])

            # error_level measures the "relative distance" between the input and output fields
            old_error_level = error_level
            error_level = 0.0
            error_normal = 1.0 # add 1.0 to prevent divergence
            for i in range(S):
                error_level += self.cb.inner_product(w_diff[i],w_diff[i])
                error_normal += self.cb.inner_product(w_exchange[i],w_exchange[i])
            error_level = np.sqrt(error_level/error_normal)

            # print iteration # and error levels and check the mass conservation
            mass_error = -1.0
            for monomer_type in self.monomer_types: # do not add random copolymer
                mass_error += self.cb.integral(phi[monomer_type])/self.cb.get_volume()
            
            if (self.box_is_altering):
                # calculate stress
                stress_array = np.array(self.pseudo.compute_stress())[-self.cb.get_dim():]
                error_level += np.sqrt(np.sum(stress_array**2))

                print("%8d %12.3E " %
                (scft_iter, mass_error), end=" [ ")
                for p in range(self.mixture.get_n_polymers()):
                    print("%13.7E " % (self.pseudo.get_total_partition(p)), end=" ")
                print("] %15.9f %15.7E " % (energy_total, error_level), end=" ")
                print("[", ",".join(["%10.7f" % (x) for x in self.cb.get_lx()[-self.cb.get_dim():]]), "]")
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
                am_current  = np.concatenate((np.reshape(w_exchange,      S*self.cb.get_n_grid()), self.cb.get_lx()[-self.cb.get_dim():]))
                am_diff     = np.concatenate((np.reshape(w_diff, S*self.cb.get_n_grid()), dlx))
                am_new = self.am.calculate_new_fields(am_current, am_diff, old_error_level, error_level)

                # copy fields
                w_exchange = np.reshape(am_new[0:S*self.cb.get_n_grid()], (S, self.cb.get_n_grid()))

                # set box size
                # restricting |dLx| to be less than 10 % of Lx
                old_lx = np.array(self.cb.get_lx()[-self.cb.get_dim():])
                new_lx = np.array(am_new[-self.cb.get_dim():])
                new_dlx = np.clip((new_lx-old_lx)/old_lx, -0.1, 0.1)
                new_lx = (1 + new_dlx)*old_lx
                self.cb.set_lx(new_lx)

                # update bond parameters using new lx
                self.pseudo.update()
            else:
                w_exchange = self.am.calculate_new_fields(
                np.reshape(w_exchange,      S*self.cb.get_n_grid()),
                np.reshape(w_diff, S*self.cb.get_n_grid()), old_error_level, error_level)
                w_exchange = np.reshape(w_exchange, (S, self.cb.get_n_grid()))

        self.phi = phi
        self.w_exchange = w_exchange

    def get_concentrations(self,):
        return self.phi
    
    def get_fields(self,):
        w_dict = {}
        w = np.matmul(self.matrix_a, self.w_exchange)
        for idx, monomer_type in enumerate(self.monomer_types):
            w_dict[monomer_type] = w[idx,:]
        return w_dict