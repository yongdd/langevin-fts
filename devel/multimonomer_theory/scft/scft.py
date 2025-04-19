import os
import time
import re
import numpy as np
import sys
import itertools
import networkx as nx
import matplotlib.pyplot as plt
from scipy.io import savemat, loadmat
from polymerfts import *

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.mse import *
from models.mse_no_const_term import *
from models.symmetric_original import *
from models.symmetric_traditional import *
from models.symmetric_orthonormal import *

# OpenMP environment variables
os.environ["MKL_NUM_THREADS"] = "1"  # always 1
os.environ["OMP_STACKSIZE"] = "1G"

# For ADAM optimizer, see https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
class Adam:
    def __init__(self, M,
                    lr = 1e-2,       # initial learning rate, γ
                    b1 = 0.9,        # β1
                    b2 = 0.999,      # β2
                    eps = 1e-8,      # epsilon, small number to prevent dividing by zero
                    gamma = 1.0,     # learning rate at Tth iteration is lr*γ^(T-1)
                    ):
        self.M = M
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.gamma = gamma
        self.count = 1
        
        self.m = np.zeros(M, dtype=np.float64) # first moment
        self.v = np.zeros(M, dtype=np.float64) # second moment
        
    def reset_count(self,):
        self.count = 1
        self.m[:] = 0.0
        self.v[:] = 0.0        
        
    def calculate_new_fields(self, w_current, w_diff, old_error_level, error_level):

        lr = self.lr*self.gamma**(self.count-1)
        
        self.m = self.b1*self.m + (1.0-self.b1)*w_diff
        self.v = self.b2*self.v + (1.0-self.b2)*w_diff**2
        m_hat = self.m/(1.0-self.b1**self.count)
        v_hat = self.v/(1.0-self.b2**self.count)
                
        w_new = w_current + lr*m_hat/(np.sqrt(v_hat) + self.eps)
        
        self.count += 1
        return w_new
    
class SCFT:
    def __init__(self, params): #, phi_target=None, mask=None):

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
        if "reduce_gpu_memory_usage" in params and platform == "cuda":
            factory = PlatformSelector.create_factory(platform, params["reduce_gpu_memory_usage"])
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
        
        # Multimonomer polymer field theory
        print("---------------------------- MSE ----------------------------")
        self.mpt1 = MSE(self.monomer_types, self.chi_n)
        print("---------------------------- MPT_MSE_No_Const ----------------------------")
        self.mpt2 = MPT_MSE_No_Const(self.monomer_types, self.chi_n)
        print("---------------------------- Symmetric_Original ----------------------------")
        self.mpt3 = Symmetric_Original(self.monomer_types, self.chi_n)
        print("---------------------------- Symmetric_Traditional ----------------------------")
        self.mpt4 = Symmetric_Traditional(self.monomer_types, self.chi_n)
        print("---------------------------- Symmetric_Orthonormal ----------------------------")
        self.mpt5 = Symmetric_Orthonormal(self.monomer_types, self.chi_n)
        
        # Matrix for field residuals.
        # See *J. Chem. Phys.* **2017**, 146, 244902
        S = len(self.monomer_types)
        matrix_chi = np.zeros((S,S))
        for i in range(S):
            for j in range(i+1,S):
                key = self.monomer_types[i] + "," + self.monomer_types[j]
                if key in self.chi_n:
                    matrix_chi[i,j] = self.chi_n[key]
                    matrix_chi[j,i] = self.chi_n[key]
        
        self.matrix_chi = matrix_chi
        matrix_chi_pinv = np.linalg.pinv(matrix_chi)
        matrix_pchip = np.matmul(np.matmul(matrix_chi_pinv, matrix_chi), matrix_chi_pinv)
        
        self.matrix_p = np.identity(S) - np.matmul(np.ones((S,S)), matrix_pchip)/np.sum(matrix_chi_pinv)
        # print(matrix_chi)
        # print(matrix_chin)

        self.non_zero_eigenvalues = self.mpt1.aux_fields_real_idx + self.mpt1.aux_fields_imag_idx

        # if phi_target is None:
        #     phi_target = np.ones(params["nx"])
        # self.phi_target = np.reshape(phi_target, (-1))

        # self.phi_target_pressure = self.phi_target/np.sum(matrix_chi_inv)
        # print(self.matrix_p)

        # # Scaling rate of total polymer concentration
        # if mask is None:
        #     mask = np.ones(params["nx"])
        # self.mask = np.reshape(mask, (-1))
        # self.phi_rescaling = np.mean((self.phi_target*self.mask)[np.isclose(self.mask, 1.0)])

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

        # Make a monomer color dictionary
        dict_color= {}
        colors = ["red", "blue", "green", "cyan", "magenta", "yellow"]
        for count, type in enumerate(params["segment_lengths"].keys()):
            if count < len(colors):
                dict_color[type] = colors[count]
            else:
                dict_color[type] = np.random.rand(3,)
        print("Monomer color: ", dict_color)
            
        # Draw polymer chain architectures
        for idx, polymer in enumerate(params["distinct_polymers"]):
        
            # Make a graph
            G = nx.Graph()
            for block in polymer["blocks_input"]:
                type = block[0]
                length = round(block[1]/params["ds"])
                v = block[2]
                u = block[3]
                G.add_edge(v, u, weight=length, monomer_type=type)

            # Set node colors
            color_map = []
            for node in G:
                if len(G.edges(node)) == 1:
                    color_map.append('yellow')
                else: 
                    color_map.append('gray')

            labels = nx.get_edge_attributes(G, 'weight')
            pos = nx.drawing.nx_agraph.graphviz_layout(G, prog='twopi')
            colors = [dict_color[G[u][v]['monomer_type']] for u,v in G.edges()]

            plt.figure(figsize=(20,20))
            title = "Polymer ID: %2d," % (idx)
            title += "\nColors of monomers: " + str(dict_color) + ","
            title += "\nColor of chain ends: 'yellow',"
            title += "\nColor of junctions: 'gray',"
            title += "\nPlease note that the length of each edge is not proportional to the number of monomers in this image."
            plt.title(title)
            nx.draw(G, pos, node_color=color_map, edge_color=colors, width=4, with_labels=True) #, node_size=100, font_size=15)
            nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, rotate=False, bbox=dict(boxstyle='round', ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0), alpha=0.5)) #, font_size=12)
            plt.savefig("polymer_%01d.png" % (idx))

        # (C++ class) Molecules list
        molecules = factory.create_molecules_information(params["chain_model"], params["ds"], params["segment_lengths"])

        # Add polymer chains
        for polymer in params["distinct_polymers"]:
            if "initial_conditions" in polymer:
                molecules.add_polymer(polymer["volume_fraction"], polymer["blocks_input"], polymer["initial_conditions"])
            else:
                molecules.add_polymer(polymer["volume_fraction"], polymer["blocks_input"])

        # (C++ class) Propagator Computation Optimizer
        if "aggregate_propagator_computation" in params:
            propagator_computation_optimizer = factory.create_propagator_computation_optimizer(molecules, params["aggregate_propagator_computation"])
        else:
            propagator_computation_optimizer = factory.create_propagator_computation_optimizer(molecules, True)

        # (C++ class) Solver using Pseudo-spectral method
        solver = factory.create_pseudospectral_solver(cb, molecules, propagator_computation_optimizer)

        # Scaling factor for stress when the fields and box size are simultaneously computed
        if "scale_stress" in params:
            self.scale_stress = params["scale_stress"]
        else:
            self.scale_stress = 1

        # Total number of variables to be adjusted to minimize the Hamiltonian
        if params["box_is_altering"]:
            n_var = len(self.non_zero_eigenvalues)*np.prod(params["nx"]) + len(params["lx"])
        else :
            n_var = len(self.non_zero_eigenvalues)*np.prod(params["nx"])
            
        # Select an optimizer among 'Anderson Mixing' and 'ADAM' for finding saddle point        
        # (C++ class) Anderson Mixing method for finding saddle point
        if params["optimizer"]["name"] == "am":
            self.field_optimizer = factory.create_anderson_mixing(n_var,
                params["optimizer"]["max_hist"],     # maximum number of history
                params["optimizer"]["start_error"],  # when switch to AM from simple mixing
                params["optimizer"]["mix_min"],      # minimum mixing rate of simple mixing
                params["optimizer"]["mix_init"])     # initial mixing rate of simple mixing
        # (Python class) ADAM optimizer for finding saddle point
        elif params["optimizer"]["name"] == "adam":
            self.field_optimizer = Adam(M = n_var,
                lr = params["optimizer"]["lr"],
                gamma = params["optimizer"]["gamma"])
        else:
            print("Invalid optimizer name: ", params["optimizer"], '. Choose among "am" and "adam".')

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

        # dH/dw_aux[i] is scaled by dt_scaling[i]
        S = len(self.monomer_types)
        self.dt_scaling = np.ones(S)
        for i in range(S-1):
            self.dt_scaling[i] = np.abs(self.mpt1.eigenvalues[i])/np.max(np.abs(self.mpt1.eigenvalues))

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

        print("P matrix for field residuals:\n\t", str(self.matrix_p).replace("\n", "\n\t"))

        for p in range(molecules.get_n_polymer_types()):
            print("distinct_polymers[%d]:" % (p) )
            print("\tvolume fraction: %f, alpha: %f, N: %d" %
                (molecules.get_polymer(p).get_volume_fraction(),
                 molecules.get_polymer(p).get_alpha(),
                 molecules.get_polymer(p).get_n_segment_total()))

        propagator_computation_optimizer.display_blocks()
        propagator_computation_optimizer.display_propagators()

        #  Save Internal Variables
        self.params = params
        self.chain_model = params["chain_model"]
        self.ds = params["ds"]
        self.box_is_altering = params["box_is_altering"]

        self.max_iter = max_iter
        self.tolerance = tolerance

        self.cb = cb
        self.molecules = molecules
        self.propagator_computation_optimizer = propagator_computation_optimizer
        self.solver = solver

    def compute_concentrations(self, w):
        S = len(self.monomer_types)
        elapsed_time = {}

        # Make a dictionary for input fields 
        w_input = {}
        for i in range(S):
            w_input[self.monomer_types[i]] = w[i]
        for random_polymer_name, random_fraction in self.random_fraction.items():
            w_input[random_polymer_name] = np.zeros(self.cb.get_total_grid(), dtype=np.float64)
            for monomer_type, fraction in random_fraction.items():
                w_input[random_polymer_name] += w_input[monomer_type]*fraction

        # For the given fields, compute the polymer statistics
        time_p_start = time.time()
        self.solver.compute_propagators(w_input)
        self.solver.compute_concentrations()
        elapsed_time["pseudo"] = time.time() - time_p_start

        # Compute total concentration for each monomer type
        phi = {}
        time_phi_start = time.time()
        for monomer_type in self.monomer_types:
            phi[monomer_type] = self.solver.get_total_concentration(monomer_type)
        elapsed_time["phi"] = time.time() - time_phi_start

        # Add random copolymer concentration to each monomer type
        for random_polymer_name, random_fraction in self.random_fraction.items():
            phi[random_polymer_name] = self.solver.get_total_concentration(random_polymer_name)
            for monomer_type, fraction in random_fraction.items():
                phi[monomer_type] += phi[random_polymer_name]*fraction
        
        return phi, elapsed_time

    def save_results(self, path):
        # Make a dictionary for chi_n
        chi_n_mat = {}
        for key in self.chi_n:
            chi_n_mat[key] = self.chi_n[key]
            
        # Make a dictionary for data
        mdic = {"dim":self.cb.get_dim(), "nx":self.cb.get_nx(), "lx":self.cb.get_lx(),
            "chi_n":chi_n_mat, "chain_model":self.chain_model, "ds":self.ds, "initial_params": self.params,
            "eigenvalues": self.mpt1.eigenvalues, "matrix_a": self.mpt1.matrix_a, "matrix_a_inverse": self.mpt1.matrix_a_inv,
            "monomer_types":self.monomer_types}

        # Add w fields to the dictionary
        for i, name in enumerate(self.monomer_types):
            mdic["w_" + name] = self.w[i]
        
        # Add concentrations to the dictionary
        for name in self.monomer_types:
            mdic["phi_" + name] = self.phi[name]

        # if self.mask is not None:
        #     mdic["mask"] = self.mask
            
        # if self.phi_target is not None:
        #     mdic["phi_target"] = self.phi_target

        # phi_total = np.zeros(self.cb.get_total_grid())
        # for name in self.monomer_types:
        #     phi_total += self.phi[name]
        # print(np.reshape(phi_total, self.cb.get_nx())[0,0,:])

        # Save data with matlab format
        savemat(path, mdic, long_field_names=True, do_compression=True)

    def run(self, initial_fields, q_init=None):

        # The number of components
        S = len(self.monomer_types)

        # Assign large initial value for the energy and error
        energy_total = 1.0e20
        error_level = 1.0e20

        # Reset Optimizer
        self.field_optimizer.reset_count()
        
        #------------------ run ----------------------
        print("---------- Run ----------")

        # Iteration begins here
        print("iteration, mass error, total_partitions, energy_total, error_level", end="")
        if (self.box_is_altering):
            print(", box size")
        else:
            print("")

        # Reshape initial fields
        w = np.zeros([S, self.cb.get_total_grid()], dtype=np.float64)
        
        for i in range(S):
            w[i,:] = np.reshape(initial_fields[self.monomer_types[i]],  self.cb.get_total_grid())

        # # Keep the level of field value
        # for i in range(S):
        #     w[i] -= self.cb.integral(w[i])/self.cb.get_volume()

        # w_aux_1 = self.mpt1.to_aux_fields(w)
        # for i in range(S-1):
        #     if np.isclose(self.mpt1.eigenvalues[i], 0.0):
        #         w_aux_1[i,:] = 0.0

        # w_mpt2 = self.mpt2.to_monomer_fields(w_aux_1)
        # w_mpt3 = self.mpt3.to_monomer_fields(w_aux_1)
        # w_mpt4 = self.mpt4.to_monomer_fields(w_aux_1)
        # w_mpt5 = self.mpt5.to_monomer_fields(w_aux_1)

        # print("w_mpt2", np.mean(w_mpt2-w, axis=1), np.std(w_mpt2-w, axis=1))
        # print("w_mpt3", np.mean(w_mpt3-w, axis=1), np.std(w_mpt3-w, axis=1))
        # print("w_mpt4", np.mean(w_mpt4-w, axis=1), np.std(w_mpt4-w, axis=1))
        # print("w_mpt5", np.mean(w_mpt5-w, axis=1), np.std(w_mpt5-w, axis=1))

        # w_aux_1 = self.mpt1.to_aux_fields(w)
        # w_aux_1[1,:] = 0.0
        # w = self.mpt1.to_monomer_fields(w_aux_1)

        # w_aux_2 = self.mpt2.to_aux_fields(w)
        # for i in range(S-1):
        #     if np.isclose(self.mpt2.eigenvalues[i], 0.0):
        #         w_aux_2[i,:] = -np.dot(self.mpt2.matrix_a_inv[i], self.mpt2.o_large_s)
        # w = self.mpt2.to_monomer_fields(w_aux_2)

        # w_aux_3 = self.mpt3.to_aux_fields(w)
        # for i in range(S-1):
        #     if np.isclose(self.mpt3.eigenvalues[i], 0.0):
        #         w_aux_3[i,:] = -np.dot(self.mpt3.matrix_a_inv[i], self.mpt3.o_large_s)
        # w = self.mpt3.to_monomer_fields(w_aux_3)

        # w_aux_4 = self.mpt4.to_aux_fields(w)
        # for i in range(S-1):
        #     if np.isclose(self.mpt4.eigenvalues[i], 0.0):
        #         w_aux_4[i,:] = 0.0
        # w = self.mpt4.to_monomer_fields(w_aux_4)

        # w_aux_5 = self.mpt5.to_aux_fields(w)
        # for i in range(S-1):
        #     if np.isclose(self.mpt5.eigenvalues[i], 0.0):
        #         w_aux_5[i,:] = 0.0
        # w = self.mpt5.to_monomer_fields(w_aux_5)

        # w_aux_1 = self.mpt1.to_aux_fields(w)
        # w_aux_4 = self.mpt4.to_aux_fields(w)
        # w_aux_5 = self.mpt5.to_aux_fields(w)

        # print("w_aux_1:\n", np.mean(w_aux_1[1]), np.std(w_aux_1[1]))
        # print("w_aux_4:\n", np.mean(w_aux_4[1]), np.std(w_aux_4[1]))
        # print("w_aux_5:\n", np.mean(w_aux_5[1]), np.std(w_aux_5[1]))

        # for i in range(S-1):
        #     if np.isclose(self.mpt1.eigenvalues[i], 0.0):
        #         w_aux_1[i,:] = 0.0
        #     if np.isclose(self.mpt2.eigenvalues[i], 0.0):
        #         w_aux_2[i,:] = -np.dot(self.mpt2.matrix_a_inv[i], self.mpt2.o_large_s)
        #     if np.isclose(self.mpt3.eigenvalues[i], 0.0):
        #         w_aux_3[i,:] = -np.dot(self.mpt3.matrix_a_inv[i], self.mpt3.o_large_s)
        #     if np.isclose(self.mpt4.eigenvalues[i], 0.0):
        #         w_aux_4[i,:] = 0.0
        #     if np.isclose(self.mpt5.eigenvalues[i], 0.0):
        #         w_aux_5[i,:] = 0.0
        
        # w_mpt1 = self.mpt1.to_monomer_fields(w_aux_1)
        # w_mpt2 = self.mpt2.to_monomer_fields(w_aux_2)
        # w_mpt3 = self.mpt3.to_monomer_fields(w_aux_3)
        # w_mpt4 = self.mpt4.to_monomer_fields(w_aux_4)
        # w_mpt5 = self.mpt5.to_monomer_fields(w_aux_5)

        # print("w_mpt1", np.mean(w_mpt1-w, axis=1), np.std(w_mpt1-w, axis=1))
        # print("w_mpt2", np.mean(w_mpt2-w, axis=1), np.std(w_mpt2-w, axis=1))
        # print("w_mpt3", np.mean(w_mpt3-w, axis=1), np.std(w_mpt3-w, axis=1))
        # print("w_mpt4", np.mean(w_mpt4-w, axis=1), np.std(w_mpt4-w, axis=1))
        # print("w_mpt5", np.mean(w_mpt5-w, axis=1), np.std(w_mpt5-w, axis=1))

        for scft_iter in range(1, self.max_iter+1):

            # Compute total concentration for each monomer type
            phi, _ = self.compute_concentrations(w)

            # # Scaling phi
            # for monomer_type in self.monomer_types:
            #     phi[monomer_type] *= self.phi_rescaling

            # Convert monomer fields to auxiliary fields
            w_aux_1 = self.mpt1.to_aux_fields(w)
            w_aux_2 = self.mpt2.to_aux_fields(w)
            w_aux_3 = self.mpt3.to_aux_fields(w)
            w_aux_4 = self.mpt4.to_aux_fields(w)
            w_aux_5 = self.mpt5.to_aux_fields(w)

            # Calculate the total energy
            # energy_total = - self.cb.integral(self.phi_target*w_aux[S-1])/self.cb.get_volume()
            total_partitions = [self.solver.get_total_partition(p) for p in range(self.molecules.get_n_polymer_types())]
            
            # Compute Hamiltonian part that total partition functions
            energy_total_0 = 0.0
            for p in range(self.molecules.get_n_polymer_types()):
                energy_total_0 -= self.molecules.get_polymer(p).get_volume_fraction()/ \
                                self.molecules.get_polymer(p).get_alpha() * \
                                np.log(total_partitions[p])
            for key in self.chi_n:
                monomer_pair = sorted(key.split(","))
                energy_total_0 += np.mean(phi[monomer_pair[0]]*phi[monomer_pair[1]]*self.chi_n[key])
            for count, type in enumerate(self.monomer_types):
                energy_total_0 -= np.mean(w[count]*phi[type])

            energy_total_1 = self.mpt1.compute_hamiltonian(self.molecules, w_aux_1, total_partitions)
            energy_total_2 = self.mpt2.compute_hamiltonian(self.molecules, w_aux_2, total_partitions)
            energy_total_3 = self.mpt3.compute_hamiltonian(self.molecules, w_aux_3, total_partitions)
            energy_total_4 = self.mpt4.compute_hamiltonian(self.molecules, w_aux_4, total_partitions)
            energy_total_5 = self.mpt5.compute_hamiltonian(self.molecules, w_aux_5, total_partitions)

            # # Compute functional derivatives of Hamiltonian w.r.t. imaginary auxiliary fields 
            # h_deriv, _ = self.mpt1.compute_func_deriv(w_aux_1, phi, self.non_zero_eigenvalues)

            # # Compute total error
            # old_error_level = error_level
            # error_level_array = np.std(h_deriv, axis=1)
            # error_level = np.max(error_level_array)

            # # Print iteration # and error levels and check the mass conservation
            # mass_error = np.mean(h_deriv[-1])
            
            # print("%8d %12.3E " % (scft_iter, mass_error), end=" [ ")
            # for p in range(self.molecules.get_n_polymer_types()):
            #     print("%13.7E " % (self.solver.get_total_partition(p)), end=" ")
            # print("] [%12.9f, %12.9f, %12.9f, %12.9f, %12.9f, %12.9f] %15.7E " 
            #         % (energy_total_0, energy_total_1, energy_total_2, energy_total_3, energy_total_4, energy_total_5, error_level))

            # # Conditions to end the iteration
            # if error_level < self.tolerance:
            #     break

            # # Scaling h_deriv
            # for count, i in enumerate(self.non_zero_eigenvalues):
            #     h_deriv[count] *= self.dt_scaling[i]

            # # Calculate new fields using simple and Anderson mixing
            # w_aux_1[self.non_zero_eigenvalues] = np.reshape(self.field_optimizer.calculate_new_fields(w_aux_1[self.non_zero_eigenvalues], -h_deriv, old_error_level, error_level), [len(self.non_zero_eigenvalues), self.cb.get_total_grid()])

            # # Convert auxiliary fields to monomer fields
            # w = self.mpt1.to_monomer_fields(w_aux_1)

            # Calculate difference between current total density and target density
            phi_total = np.zeros(self.cb.get_total_grid())
            for i in range(S):
                phi_total += phi[self.monomer_types[i]]
            # phi_diff = phi_total-self.phi_target
            phi_diff = phi_total-1.0

            # Compute functional derivatives of Hamiltonian w.r.t. imaginary auxiliary fields 
            w_diff, _ = self.mpt1.compute_func_deriv(w_aux_1, phi, self.non_zero_eigenvalues)

            # Calculate self-consistency error
            w_diff = np.zeros([S, self.cb.get_total_grid()], dtype=np.float64) # array for output fields
            for i in range(S):
                for j in range(S):
                    w_diff[i,:] += self.matrix_chi[i,j]*phi[self.monomer_types[j]] - self.matrix_p[i,j]*w[j,:]
                # w_diff[i,:] -= self.phi_target_pressure

            # Keep the level of functional derivatives
            for i in range(S):
                # w_diff[i] *= self.mask
                w_diff[i] -= self.cb.integral(w_diff[i])/self.cb.get_volume()

            # error_level measures the "relative distance" between the input and output fields
            old_error_level = error_level
            error_level = 0.0
            error_normal = 1.0  # add 1.0 to prevent divergence
            for i in range(S):
                error_level += self.cb.inner_product(w_diff[i],w_diff[i])
                error_normal += self.cb.inner_product(w_aux_1[i],w_aux_1[i])
            error_level = np.sqrt(error_level/error_normal)

            # Print iteration # and error levels and check the mass conservation
            mass_error = self.cb.integral(phi_diff)/self.cb.get_volume()
            
            if (self.box_is_altering):
                # Calculate stress
                self.solver.compute_stress()
                stress_array = np.array(self.solver.get_stress())
                error_level += np.sqrt(np.sum(stress_array**2))

                print("%8d %12.3E " %
                (scft_iter, mass_error), end=" [ ")
                for p in range(self.molecules.get_n_polymer_types()):
                    print("%13.7E " % (self.solver.get_total_partition(p)), end=" ")
                print("] [%12.9f, %12.9f, %12.9f, %12.9f, %12.9f, %12.9f] %15.7E " 
                      % (energy_total_0, energy_total_1, energy_total_2, energy_total_3, energy_total_4, energy_total_5, error_level), end=" ")
                print("[", ",".join(["%10.7f" % (x) for x in self.cb.get_lx()]), "]")
            else:
                print("%8d %12.3E " % (scft_iter, mass_error), end=" [ ")
                for p in range(self.molecules.get_n_polymer_types()):
                    print("%13.7E " % (self.solver.get_total_partition(p)), end=" ")
                print("] [%12.9f, %12.9f, %12.9f, %12.9f, %12.9f, %12.9f] %15.7E " 
                      % (energy_total_0, energy_total_1, energy_total_2, energy_total_3, energy_total_4, energy_total_5, error_level))

            # Conditions to end the iteration
            if error_level < self.tolerance:
                break

            # Calculate new fields using simple and Anderson mixing
            if (self.box_is_altering):
                dlx = -stress_array
                am_current  = np.concatenate((np.reshape(w,      S*self.cb.get_total_grid()), self.cb.get_lx()))
                am_diff     = np.concatenate((np.reshape(w_diff, S*self.cb.get_total_grid()), self.scale_stress*dlx))
                am_new = self.field_optimizer.calculate_new_fields(am_current, am_diff, old_error_level, error_level)

                # Copy fields
                w = np.reshape(am_new[0:S*self.cb.get_total_grid()], (S, self.cb.get_total_grid()))

                # Set box size
                # Restricting |dLx| to be less than 10 % of Lx
                old_lx = np.array(self.cb.get_lx())
                new_lx = np.array(am_new[-self.cb.get_dim():])
                new_dlx = np.clip((new_lx-old_lx)/old_lx, -0.1, 0.1)
                new_lx = (1 + new_dlx)*old_lx
                self.cb.set_lx(new_lx)

                # Update bond parameters using new lx
                self.solver.update_laplacian_operator()
            else:
                w = self.field_optimizer.calculate_new_fields(
                np.reshape(w,      S*self.cb.get_total_grid()),
                np.reshape(w_diff, S*self.cb.get_total_grid()), old_error_level, error_level)
                w = np.reshape(w, (S, self.cb.get_total_grid()))
                        
            # Keep the level of field value
            for i in range(S):
                # w[i] *= self.mask
                w[i] -= self.cb.integral(w[i])/self.cb.get_volume()
        
        # Print free energy as per chain expression
        print("Free energy per chain (for each chain type):")
        for p in range(self.molecules.get_n_polymer_types()):
            energy_total_per_chain = energy_total*self.molecules.get_polymer(p).get_alpha()/ \
                                                  self.molecules.get_polymer(p).get_volume_fraction()
            print("\tβF/n_%d : %12.7f" % (p+1, energy_total_per_chain))

        # Store phi and w
        self.phi = phi
        self.w = w

        # w_aux_1 = self.mpt1.to_aux_fields(w)
        # w_aux_4 = self.mpt4.to_aux_fields(w)
        # w_aux_5 = self.mpt5.to_aux_fields(w)

        # print("w_aux_1:\n", np.mean(w_aux_1[1]), np.std(w_aux_1[1]))
        # print("w_aux_4:\n", np.mean(w_aux_4[1]), np.std(w_aux_4[1]))
        # print("w_aux_5:\n", np.mean(w_aux_5[1]), np.std(w_aux_5[1]))

        # print("phi['A'][0:32]:", phi["A"][0:32])

    # def get_concentrations(self,):
    #     return self.phi
    
    # def get_fields(self,):
    #     w_dict = {}
    #     for idx, monomer_type in enumerate(self.monomer_types):
    #         w_dict[monomer_type] = self.w[idx,:]
    #     return w_dict