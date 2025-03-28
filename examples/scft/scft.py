import os
import time
import re
import numpy as np
import sys
import math
import json
import yaml
import copy
import itertools
import networkx as nx
import matplotlib.pyplot as plt
import scipy.io
from langevinfts import *

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

class SymmetricPolymerTheory:
    def __init__(self, monomer_types, chi_n):
        self.monomer_types = monomer_types
        S = len(self.monomer_types)
        
        self.matrix_q = np.ones((S,S))/S
        self.matrix_p = np.identity(S) - self.matrix_q
        
        # Compute eigenvalues and orthogonal matrix
        eigenvalues, matrix_o = self.compute_eigen_system(chi_n, self.matrix_p)

        # Construct chi_n matrix
        matrix_chi = np.zeros((S,S))
        for i in range(S):
            for j in range(i+1,S):
                monomer_pair = [self.monomer_types[i], self.monomer_types[j]]
                monomer_pair.sort()
                key = monomer_pair[0] + "," + monomer_pair[1]
                if key in chi_n:
                    matrix_chi[i,j] = chi_n[key]
                    matrix_chi[j,i] = chi_n[key]

        self.matrix_chi = matrix_chi
        self.vector_s = np.matmul(matrix_chi, np.ones(S))/S
        self.vector_large_s = np.matmul(np.transpose(matrix_o), self.vector_s)

        # Compute coefficients for Hamiltonian computation
        h_const, h_coef_mu1, h_coef_mu2 = self.compute_h_coef(chi_n, eigenvalues)

        # Matrix A and Inverse for converting between auxiliary fields and monomer chemical potential fields
        matrix_a = matrix_o.copy()
        matrix_a_inv = np.transpose(matrix_o).copy()/S

        # Check the inverse matrix
        error = np.std(np.matmul(matrix_a, matrix_a_inv) - np.identity(S))
        assert(np.isclose(error, 0.0)), \
            "Invalid inverse of matrix A. Perhaps matrix O is not orthogonal."

        self.h_const = h_const
        self.h_coef_mu1 = h_coef_mu1
        self.h_coef_mu2 = h_coef_mu2

        self.eigenvalues = eigenvalues
        self.matrix_o = matrix_o
        self.matrix_a = matrix_a
        self.matrix_a_inv = matrix_a_inv

        print("------------ Polymer Field Theory for Multimonomer ------------")
        # print("Projection matrix P:\n\t", str(self.matrix_p).replace("\n", "\n\t"))
        # print("Projection matrix Q:\n\t", str(self.matrix_q).replace("\n", "\n\t"))
        print("Eigenvalues:\n\t", self.eigenvalues)
        print("Eigenvectors [v1, v2, ...] :\n\t", str(self.matrix_o).replace("\n", "\n\t"))
        print("Mapping matrix A:\n\t", str(self.matrix_a).replace("\n", "\n\t"))
        # print("A*Inverse[A]:\n\t", str(np.matmul(self.matrix_a, self.matrix_a_inv)).replace("\n", "\n\t"))
        # print("P matrix for field residuals:\n\t", str(self.matrix_p).replace("\n", "\n\t"))

        # print("Real Fields: ",      self.aux_fields_real_idx)
        # print("Imaginary Fields: ", self.aux_fields_imag_idx)
        
        print("In Hamiltonian:")
        print("\treference energy: ", self.h_const)
        print("\tcoefficients of int of mu(r)/V: ", self.h_coef_mu1)
        print("\tcoefficients of int of mu(r)^2/V: ", self.h_coef_mu2)

    def to_aux_fields(self, w):
        return np.matmul(self.matrix_a_inv, w)

    def to_monomer_fields(self, w_aux):
        return np.matmul(self.matrix_a, w_aux)

    def compute_eigen_system(self, chi_n, matrix_p):
        S = matrix_p.shape[0]

        # Compute eigenvalues and eigenvectors
        matrix_chi = np.zeros((S,S))
        for i in range(S):
            for j in range(i+1,S):
                monomer_pair = [self.monomer_types[i], self.monomer_types[j]]
                monomer_pair.sort()
                key = monomer_pair[0] + "," + monomer_pair[1]
                if key in chi_n:
                    matrix_chi[i,j] = chi_n[key]
                    matrix_chi[j,i] = chi_n[key]
        projected_chin = np.matmul(matrix_p, np.matmul(matrix_chi, matrix_p))
        eigenvalues, eigenvectors = np.linalg.eigh(projected_chin)

        # Reordering eigenvalues and eigenvectors
        sorted_indexes = np.argsort(np.abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[sorted_indexes]
        eigenvectors = eigenvectors[:,sorted_indexes]

        # Set the last eigenvector to [1, 1, ..., 1]/√S
        eigenvectors[:,-1] = np.ones(S)/np.sqrt(S)
        
        # Make a orthogonal matrix using Gram-Schmidt
        eigen_val_0 = np.isclose(eigenvalues, 0.0, atol=1e-12)
        eigenvalues[eigen_val_0] = 0.0
        eigen_vec_0 = eigenvectors[:,eigen_val_0]
        for i in range(eigen_vec_0.shape[1]-2,-1,-1):
            vec_0 = eigen_vec_0[:,i].copy()
            for j in range(i+1, eigen_vec_0.shape[1]):
                eigen_vec_0[:,i] -= eigen_vec_0[:,j]*np.dot(vec_0,eigen_vec_0[:,j])
            eigen_vec_0[:,i] /= np.linalg.norm(eigen_vec_0[:,i])
        eigenvectors[:,eigen_val_0] = eigen_vec_0

        # Make the first element of each vector positive to restore the conventional AB polymer field theory
        for i in range(S):
            if eigenvectors[0,i] < 0.0:
                eigenvectors[:,i] *= -1.0

        # Multiply √S to eigenvectors
        eigenvectors *= np.sqrt(S)

        return eigenvalues, eigenvectors

    def compute_h_coef(self, chi_n, eigenvalues):
        S = len(self.monomer_types)

        # Compute vector X_iS
        vector_s = np.zeros(S-1)
        for i in range(S-1):
            monomer_pair = [self.monomer_types[i], self.monomer_types[S-1]]
            monomer_pair.sort()
            key = monomer_pair[0] + "," + monomer_pair[1]            
            vector_s[i] = chi_n[key]

        # Compute reference part of Hamiltonian
        h_const = 0.5*np.sum(self.vector_s)/S
        for i in range(S-1):
            if not np.isclose(eigenvalues[i], 0.0):
                h_const -= 0.5*self.vector_large_s[i]**2/eigenvalues[i]/S

        # Compute coefficients of integral of μ(r)/V
        h_coef_mu1 = np.zeros(S-1)
        for i in range(S-1):
            if not np.isclose(eigenvalues[i], 0.0):
                h_coef_mu1[i] = self.vector_large_s[i]/eigenvalues[i]

        # Compute coefficients of integral of μ(r)^2/V
        h_coef_mu2 = np.zeros(S-1)
        for i in range(S-1):
            if not np.isclose(eigenvalues[i], 0.0):
                h_coef_mu2[i] = -0.5/eigenvalues[i]*S

        return h_const, h_coef_mu1, h_coef_mu2

    # Compute total Hamiltonian
    def compute_hamiltonian(self, molecules, w_aux, total_partitions):
        S = len(self.monomer_types)

        # Compute Hamiltonian part that is related to fields
        hamiltonian_fields = -np.mean(w_aux[S-1])
        for i in range(S-1):
            hamiltonian_fields += self.h_coef_mu2[i]*np.mean(w_aux[i]**2)
            hamiltonian_fields += self.h_coef_mu1[i]*np.mean(w_aux[i])
        
        # Compute Hamiltonian part that total partition functions
        hamiltonian_partition = 0.0
        for p in range(molecules.get_n_polymer_types()):
            hamiltonian_partition -= molecules.get_polymer(p).get_volume_fraction()/ \
                            molecules.get_polymer(p).get_alpha() * \
                            np.log(total_partitions[p])

        return hamiltonian_partition + hamiltonian_fields #+ self.h_const

class SCFT:
    def __init__(self, params): #, phi_target=None, mask=None):

        # Segment length
        self.monomer_types = sorted(list(params["segment_lengths"].keys()))
        self.segment_lengths = copy.deepcopy(params["segment_lengths"])
        self.distinct_polymers = copy.deepcopy(params["distinct_polymers"])
        
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
            assert(monomer_pair[0] in self.segment_lengths), \
                f"Monomer type '{monomer_pair[0]}' is not in 'segment_lengths'."
            assert(monomer_pair[1] in self.segment_lengths), \
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
        self.mpt = SymmetricPolymerTheory(self.monomer_types, self.chi_n)
        
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
        matrix_chi_inv = np.linalg.inv(matrix_chi)
        self.matrix_p = np.identity(S) - np.matmul(np.ones((S,S)), matrix_chi_inv)/np.sum(matrix_chi_inv)
        # print(matrix_chi)
        # print(matrix_chin)

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
        assert(len(self.distinct_polymers) >= 1), \
            "There is no polymer chain."

        total_volume_fraction = 0.0
        for polymer in self.distinct_polymers:
            total_volume_fraction += polymer["volume_fraction"]
        assert(np.isclose(total_volume_fraction,1.0)), "The sum of volume fractions must be equal to 1."

        # Polymer chains
        for polymer_counter, polymer in enumerate(self.distinct_polymers):
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

        # Random copolymer chains
        self.random_fraction = {}
        for polymer in self.distinct_polymers:

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
                statistical_segment_length += self.segment_lengths[monomer_type]**2 * polymer["blocks"][0]["fraction"][monomer_type]
                total_random_fraction += polymer["blocks"][0]["fraction"][monomer_type]
            statistical_segment_length = np.sqrt(statistical_segment_length)

            assert(np.isclose(total_random_fraction, 1.0)), \
                "The sum of volume fractions of random copolymer must be equal to 1."

            random_type_string = polymer["blocks"][0]["type"]
            assert(not random_type_string in self.segment_lengths), \
                f"The name of random copolymer '{random_type_string}' is already used as a type in 'segment_lengths' or other random copolymer"

            # Add random copolymers
            polymer["block_monomer_types"] = [random_type_string]
            self.segment_lengths.update({random_type_string:statistical_segment_length})
            self.random_fraction[random_type_string] = polymer["blocks"][0]["fraction"]

        # Make a monomer color dictionary
        dict_color= {}
        colors = ["red", "blue", "green", "cyan", "magenta", "yellow"]
        for count, type in enumerate(self.segment_lengths.keys()):
            if count < len(colors):
                dict_color[type] = colors[count]
            else:
                dict_color[type] = np.random.rand(3,)
        print("Monomer color: ", dict_color)
            
        # Draw polymer chain architectures
        for idx, polymer in enumerate(self.distinct_polymers):
        
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
        molecules = factory.create_molecules_information(params["chain_model"], params["ds"], self.segment_lengths)

        # Add polymer chains
        for polymer in self.distinct_polymers:
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
            n_var = len(self.monomer_types)*np.prod(params["nx"]) + len(params["lx"])
        else :
            n_var = len(self.monomer_types)*np.prod(params["nx"])
            
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

        print("---------- Simulation Parameters ----------")
        print("Platform :", platform)
        print("Box Dimension: %d" % (cb.get_dim()))
        print("Nx:", cb.get_nx())
        print("Lx:", cb.get_lx())
        print("dx:", cb.get_dx())
        print("Volume: %f" % (cb.get_volume()))

        print("Chain model: %s" % (params["chain_model"]))
        print("Segment lengths:\n\t", list(self.segment_lengths.items()))
        print("Conformational asymmetry (epsilon): ")
        for monomer_pair in itertools.combinations(self.monomer_types,2):
            print("\t%s/%s: %f" % (monomer_pair[0], monomer_pair[1], self.segment_lengths[monomer_pair[0]]/self.segment_lengths[monomer_pair[1]]))

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

        #  Save internal variables
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
            w_input[random_polymer_name] = np.zeros(self.cb.get_n_grid(), dtype=np.float64)
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
        m_dic = {"initial_params": self.params,
            "dim":self.cb.get_dim(), "nx":self.cb.get_nx(), "lx":self.cb.get_lx(),
            "monomer_types":self.monomer_types, "chi_n":chi_n_mat, "chain_model":self.chain_model, "ds":self.ds,
            "eigenvalues": self.mpt.eigenvalues, "matrix_a": self.mpt.matrix_a, "matrix_a_inverse": self.mpt.matrix_a_inv}

        # Add w fields to the dictionary
        for i, name in enumerate(self.monomer_types):
            m_dic["w_" + name] = self.w[i]
        
        # Add concentrations to the dictionary
        for name in self.monomer_types:
            m_dic["phi_" + name] = self.phi[name]

        # if self.mask is not None:
        #     m_dic["mask"] = self.mask
            
        # if self.phi_target is not None:
        #     m_dic["phi_target"] = self.phi_target

        # phi_total = np.zeros(self.cb.get_n_grid())
        # for name in self.monomer_types:
        #     phi_total += self.phi[name]
        # print(np.reshape(phi_total, self.cb.get_nx())[0,0,:])

        # Convert numpy arrays to lists
        for data in m_dic:
            if type(m_dic[data]).__module__ == np.__name__  :
                m_dic[data] = m_dic[data].tolist()

        for data in m_dic["initial_params"]:
            if type(m_dic["initial_params"][data]).__module__ == np.__name__  :
                m_dic["initial_params"][data] = m_dic["initial_params"][data].tolist()

        # Get input file extension
        _, file_extension = os.path.splitext(path)

        # Save data in matlab format
        if file_extension == ".mat":
            scipy.io.savemat(path, m_dic, long_field_names=True, do_compression=True)

        # Save data in json format
        elif file_extension == ".json":
            with open(path, "w") as f:
                json.dump(m_dic, f, indent=2)

        # Save data in yaml format
        elif file_extension == ".yaml":
            with open(path, "w") as f:
                f.write(yaml.dump(m_dic, default_flow_style=None, width=90, sort_keys=False))
        else:
            raise("Invalid output file extension.")

    def run(self, initial_fields, q_init=None):

        # The number of components
        S = len(self.monomer_types)

        # Assign large initial value for the energy and error
        energy_total = 1.0e20
        error_level = 1.0e20

        # Reset optimizer
        self.field_optimizer.reset_count()
        
        print("---------- Run ----------")
        print("iteration, mass error, total_partitions, energy_total, error_level", end="")
        if (self.box_is_altering):
            print(", box size")
        else:
            print("")

        # Reshape initial fields
        w = np.zeros([S, self.cb.get_n_grid()], dtype=np.float64)
        
        for i in range(S):
            w[i,:] = np.reshape(initial_fields[self.monomer_types[i]],  self.cb.get_n_grid())

        # Keep the level of field value
        for i in range(S):
            w[i] -= self.cb.integral(w[i])/self.cb.get_volume()
            
        # Iteration begins here
        for scft_iter in range(1, self.max_iter+1):

            # Compute total concentration for each monomer type
            phi, _ = self.compute_concentrations(w)

            # # Scaling phi
            # for monomer_type in self.monomer_types:
            #     phi[monomer_type] *= self.phi_rescaling

            # Convert monomer fields to auxiliary fields
            w_aux = self.mpt.to_aux_fields(w)

            # Calculate the total energy
            # energy_total = - self.cb.integral(self.phi_target*w_exchange[S-1])/self.cb.get_volume()
            total_partitions = [self.solver.get_total_partition(p) for p in range(self.molecules.get_n_polymer_types())]
            energy_total = self.mpt.compute_hamiltonian(self.molecules, w_aux, total_partitions)

            # Calculate difference between current total density and target density
            phi_total = np.zeros(self.cb.get_n_grid())
            for i in range(S):
                phi_total += phi[self.monomer_types[i]]
            # phi_diff = phi_total-self.phi_target
            phi_diff = phi_total-1.0

            # Calculate self-consistency error
            w_diff = np.zeros([S, self.cb.get_n_grid()], dtype=np.float64) # array for output fields
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
                error_normal += self.cb.inner_product(w[i],w[i])
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
                print("] %15.9f %15.7E " % (energy_total, error_level), end=" ")
                print("[", ",".join(["%10.7f" % (x) for x in self.cb.get_lx()]), "]")
            else:
                print("%8d %12.3E " % (scft_iter, mass_error), end=" [ ")
                for p in range(self.molecules.get_n_polymer_types()):
                    print("%13.7E " % (self.solver.get_total_partition(p)), end=" ")
                print("] %15.9f %15.7E " % (energy_total, error_level))

            # Conditions to end the iteration
            if error_level < self.tolerance:
                break

            # Calculate new fields using simple and Anderson mixing
            if (self.box_is_altering):
                dlx = -stress_array
                am_current  = np.concatenate((np.reshape(w,      S*self.cb.get_n_grid()), self.cb.get_lx()))
                am_diff     = np.concatenate((np.reshape(w_diff, S*self.cb.get_n_grid()), self.scale_stress*dlx))
                am_new = self.field_optimizer.calculate_new_fields(am_current, am_diff, old_error_level, error_level)

                # Copy fields
                w = np.reshape(am_new[0:S*self.cb.get_n_grid()], (S, self.cb.get_n_grid()))

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
                np.reshape(w,      S*self.cb.get_n_grid()),
                np.reshape(w_diff, S*self.cb.get_n_grid()), old_error_level, error_level)
                w = np.reshape(w, (S, self.cb.get_n_grid()))
                        
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

    # def get_concentrations(self,):
    #     return self.phi
    
    # def get_fields(self,):
    #     w_dict = {}
    #     for idx, monomer_type in enumerate(self.monomer_types):
    #         w_dict[monomer_type] = self.w[idx,:]
    #     return w_dict