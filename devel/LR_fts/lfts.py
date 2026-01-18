import os
import time
import re
import pathlib
import copy
import numpy as np
import itertools
import networkx as nx
import matplotlib.pyplot as plt
from scipy.io import savemat, loadmat
from polymerfts import *

# OpenMP environment variables
os.environ["MKL_NUM_THREADS"] = "1"  # always 1
os.environ["OMP_STACKSIZE"] = "1G"

def calculate_sigma(langevin_nbar, langevin_dt, n_grids, volume):
        return np.sqrt(2*langevin_dt*n_grids/(volume*np.sqrt(langevin_nbar)))

# LR compressor
class LR:
    def __init__(self, nx, lx):
        self.nx = nx
        self.lx = lx

        # # arrays for exponential time differencing
        # space_kx, space_ky, space_kz = np.meshgrid(
        #     2*np.pi/lx[0]*np.concatenate([np.arange((nx[0]+1)//2), nx[0]//2-np.arange(nx[0]//2)]),
        #     2*np.pi/lx[1]*np.concatenate([np.arange((nx[1]+1)//2), nx[1]//2-np.arange(nx[1]//2)]),
        #     2*np.pi/lx[2]*np.arange(nx[2]//2+1), indexing='ij')
        # mag_k2 = (space_kx**2 + space_ky**2 + space_kz**2)/6
        # mag_k2[0,0,0] = 1.0e-5 # to prevent 'division by zero' error

        # self.j_k = 2*(mag_k2+np.exp(-mag_k2)-1.0)/mag_k2**2
        # self.j_k[0,0,0] = 1.0

        # # Discrete chain
        # ds = 0.01
        # n = 1.0/ds
        # expx = np.exp(-mag_k2/n)
        # expnx = np.exp(-mag_k2)
        
        # self.g_k = (n + 2*expx*( n*(1-expx)-1+expnx )/(1-expx)**2 )/n**2
        # self.g_k[0,0,0] = 1.0

        # self.j_k_array_shape = [I,I] + self.nx
        # self.j_k_array_shape[-1] = self.j_k_array_shape[-1]//2 + 1
        
        # self.j_k = np.zeros(self.j_k_array_shape)

        self.j_k_array_shape = nx.copy()
        self.j_k_array_shape[-1] = self.j_k_array_shape[-1]//2 + 1
        self.j_k = np.zeros(self.j_k_array_shape)

    def reset_count(self,):
        pass
        
    def calculate_new_fields(self, w_current, negative_h_deriv, old_error_level, error_level):
        nx = self.nx
        w_diff = np.zeros_like(w_current)
        negative_h_deriv_k = np.fft.rfftn(np.reshape(negative_h_deriv, nx))/np.prod(nx)
        w_diff_k = negative_h_deriv_k/self.j_k
        w_diff[0] = np.reshape(np.fft.irfftn(w_diff_k, nx), np.prod(nx))*np.prod(nx)

        # w_diff_k = np.zeros(self.j_k_array_shape[1:], dtype=np.complex128)
        # negative_h_deriv_k = np.zeros(self.j_k_array_shape[1:], dtype=np.complex128)
        # for i in range(self.I):
        #     negative_h_deriv_k[i] = np.fft.rfftn(np.reshape(negative_h_deriv[i], nx))/np.prod(nx)
        # if self.I == 1:
        #     w_diff_k[0] = negative_h_deriv_k[0]/self.j_k[0,0]
        # if self.I == 2:
        #     det = self.j_k[0,0]*self.j_k[1,1] - self.j_k[0,1]*self.j_k[1,0]
        #     w_diff_k[0] =  self.j_k[1,1]*negative_h_deriv_k[0] - self.j_k[0,1]*negative_h_deriv_k[1]
        #     w_diff_k[1] = -self.j_k[1,0]*negative_h_deriv_k[0] + self.j_k[0,0]*negative_h_deriv_k[1]
        #     w_diff_k /= det
        # for i in range(self.I):     
        #     w_diff[i] = np.reshape(np.fft.irfftn(w_diff_k[i], nx), np.prod(nx))*np.prod(nx)

        return w_current + w_diff

# LRAM compressor
class LRAM:
    def __init__(self, lr, am):
        self.lr = lr
        self.am = am

    def reset_count(self,):
        self.am.reset_count()
        
    def calculate_new_fields(self, w_current, negative_h_deriv, old_error_level, error_level):
        nx = self.lr.nx
        
        w_lr_old = w_current.copy()
        w_lr_new = np.reshape(self.lr.calculate_new_fields(w_lr_old, negative_h_deriv, old_error_level, error_level), [1, np.prod(nx)])
        w_diff = w_lr_new - w_lr_old
        w_new = np.reshape(self.am.calculate_new_fields(w_lr_new, w_diff, old_error_level, error_level), [1, np.prod(nx)])

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

        # Indices whose auxiliary fields are real
        self.aux_fields_real_idx = []
        # Indices whose auxiliary fields are imaginary including the pressure field
        self.aux_fields_imag_idx = []
        for i in range(S-1):
            # assert(not np.isclose(eigenvalues[i], 0.0)), \
            #     "One of eigenvalues is zero for given chiN values."
            if np.isclose(eigenvalues[i], 0.0):
                print("One of eigenvalues is zero for given chiN values.")
            elif eigenvalues[i] > 0:
                self.aux_fields_imag_idx.append(i)
            else:
                self.aux_fields_real_idx.append(i)
        self.aux_fields_imag_idx.append(S-1) # add pressure field

        # The numbers of real and imaginary fields, respectively
        self.R = len(self.aux_fields_real_idx)
        self.I = len(self.aux_fields_imag_idx)

        if self.I > 1:
            print("(Warning!) For a given χN interaction parameter set, at least one of the auxiliary fields is an imaginary field. ", end="")
            print("The field fluctuations would not be fully reflected. Run this simulation at your own risk.")

        # Compute coefficients for Hamiltonian computation
        h_const, h_coef_mu1, h_coef_mu2 = self.compute_h_coef(chi_n, eigenvalues)

        # Matrix A and Inverse for converting between auxiliary fields and monomer chemical potential fields
        matrix_a = matrix_o.copy()
        matrix_a_inv = np.transpose(matrix_o).copy()/S

        # Check the inverse matrix
        error = np.std(np.matmul(matrix_a, matrix_a_inv) - np.identity(S))
        assert(np.isclose(error, 0.0)), \
            "Invalid inverse of matrix A. Perhaps matrix O is not orthogonal."

        # Compute derivatives of Hamiltonian coefficients w.r.t. χN
        epsilon = 1e-5
        self.h_const_deriv_chin = {}
        self.h_coef_mu1_deriv_chin = {}
        self.h_coef_mu2_deriv_chin = {}
        for key in chi_n:
            
            chi_n_p = chi_n.copy()
            chi_n_n = chi_n.copy()
            
            chi_n_p[key] += epsilon
            chi_n_n[key] -= epsilon

            # Compute eigenvalues and orthogonal matrix
            eigenvalues_p, _ = self.compute_eigen_system(chi_n_p, self.matrix_p)
            eigenvalues_n, _ = self.compute_eigen_system(chi_n_n, self.matrix_p)
            
            # Compute coefficients for Hamiltonian computation
            h_const_p, h_coef_mu1_p, h_coef_mu2_p = self.compute_h_coef(chi_n_p, eigenvalues_p)
            h_const_n, h_coef_mu1_n, h_coef_mu2_n = self.compute_h_coef(chi_n_n, eigenvalues_n)
            
            # Compute derivatives using finite difference
            self.h_const_deriv_chin[key] = (h_const_p - h_const_n)/(2*epsilon)
            self.h_coef_mu1_deriv_chin[key] = (h_coef_mu1_p - h_coef_mu1_n)/(2*epsilon)
            self.h_coef_mu2_deriv_chin[key] = (h_coef_mu2_p - h_coef_mu2_n)/(2*epsilon)

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

        print("Real Fields: ",      self.aux_fields_real_idx)
        print("Imaginary Fields: ", self.aux_fields_imag_idx)
        
        print("In Hamiltonian:")
        print("\treference energy: ", self.h_const)
        print("\tcoefficients of int of mu(r)/V: ", self.h_coef_mu1)
        print("\tcoefficients of int of mu(r)^2/V: ", self.h_coef_mu2)
        print("\tdH_ref/dχN: ", self.h_const_deriv_chin)
        print("\td(coef of mu(r))/dχN: ", self.h_coef_mu1_deriv_chin)
        print("\td(coef of mu(r)^2)/dχN: ", self.h_coef_mu2_deriv_chin)

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

        return hamiltonian_partition + hamiltonian_fields + self.h_const

    # Compute functional derivatives of Hamiltonian w.r.t. fields of selected indices
    def compute_func_deriv(self, w_aux, phi, indices):
        S = len(self.monomer_types)
                
        elapsed_time = {}
        time_e_start = time.time()
        h_deriv = np.zeros([len(indices), w_aux.shape[1]], dtype=np.float64)
        for count, i in enumerate(indices):
            # Return dH/dw
            if i != S-1:
                h_deriv[count] += 2*self.h_coef_mu2[i]*w_aux[i]
                h_deriv[count] +=   self.h_coef_mu1[i]
                for j in range(S):
                    h_deriv[count] += self.matrix_a[j,i]*phi[self.monomer_types[j]]
            else:
                for j in range(S):
                    h_deriv[count] += phi[self.monomer_types[j]]
                h_deriv[count] -= 1.0

            # Change the sign for the imaginary fields
            if i in self.aux_fields_imag_idx:
                h_deriv[count] = -h_deriv[count]

        elapsed_time["h_deriv"] = time.time() - time_e_start
        
        return  h_deriv, elapsed_time

    # Compute dH/dχN
    def compute_h_deriv_chin(self, chi_n, w_aux):
        S = len(self.monomer_types)

        dH = {}
        for key in chi_n:
            dH[key] = self.h_const_deriv_chin[key]
            for i in range(S-1):
                dH[key] += self.h_coef_mu2_deriv_chin[key][i]*np.mean(w_aux[i]**2)
                dH[key] += self.h_coef_mu1_deriv_chin[key][i]*np.mean(w_aux[i])                            
        return dH

class LFTS:
    def __init__(self, params, random_seed=None):

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
        if "use_checkpointing" in params and platform == "cuda":
            factory = PlatformSelector.create_factory(platform, params["use_checkpointing"])
        else:
            factory = PlatformSelector.create_factory(platform, False)
        factory.display_info()

        # (C++ class) Computation box
        self.cb = factory.create_computation_box(params["nx"], params["lx"])

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

        # The numbers of real and imaginary fields, respectively
        S = len(self.monomer_types)
        R = len(self.mpt.aux_fields_real_idx)
        I = len(self.mpt.aux_fields_imag_idx)

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
            molecules.add_polymer(polymer["volume_fraction"], polymer["blocks_input"])

        # (C++ class) Propagator Computation Optimizer
        if "aggregate_propagator_computation" in params:
            propagator_computation_optimizer = factory.create_propagator_computation_optimizer(molecules, params["aggregate_propagator_computation"])
        else:
            propagator_computation_optimizer = factory.create_propagator_computation_optimizer(molecules, True)

        # (C++ class) Solver using Pseudo-spectral method
        self.solver = factory.create_propagator_computation(self.cb, molecules, propagator_computation_optimizer, "rqm4")

        # (C++ class) Fields Relaxation using Anderson Mixing
        if params["compressor"]["name"] == "am" or params["compressor"]["name"] == "lram":
            am = factory.create_anderson_mixing(
                len(self.mpt.aux_fields_imag_idx)*np.prod(params["nx"]),   # the number of variables
                params["compressor"]["max_hist"],                           # maximum number of history
                params["compressor"]["start_error"],                        # when switch to AM from simple mixing
                params["compressor"]["mix_min"],                            # minimum mixing rate of simple mixing
                params["compressor"]["mix_init"])                           # initial mixing rate of simple mixing

        # Fields Relaxation using Linear Reponse Method
        if params["compressor"]["name"] == "lr" or params["compressor"]["name"] == "lram":
            assert(I == 1), \
                f"Currently, LR methods are not working for imaginary-valued auxiliary fields."

            lr = LR(self.cb.get_nx(), self.cb.get_lx())
            w_aux_perturbed = np.zeros([S, self.cb.get_total_grid()], dtype=np.float64)
            w_aux_perturbed[S-1,0] = 1e-3 # add a small perturbation at the pressure field
            w_aux_perturbed_k = np.fft.rfftn(np.reshape(w_aux_perturbed[S-1], self.cb.get_nx()))/np.prod(self.cb.get_nx())

            phi_perturbed, _ = self.compute_concentrations(w_aux_perturbed)
            h_deriv_perturbed, _ = self.mpt.compute_func_deriv(w_aux_perturbed, phi_perturbed, self.mpt.aux_fields_imag_idx)
            h_deriv_perturbed_k = np.fft.rfftn(np.reshape(h_deriv_perturbed, self.cb.get_nx()))/np.prod(self.cb.get_nx())
            j_k_numeric = np.real(h_deriv_perturbed_k/w_aux_perturbed_k)
            lr.j_k = j_k_numeric.copy()
            # print(np.mean(j_k_numeric), np.std(j_k_numeric))
            # print(np.mean(self.lr.j_k), np.std(self.lr.j_k))

            # # Get homogeneous solution
            # w_aux_homo = np.zeros([S, self.cb.get_total_grid()], dtype=np.float64)
            # phi_zero, _ = self.compute_concentrations(w_aux_homo)
            # phi_sum = np.zeros((S-1))
            # for i in range(S-1):
            #     for j in range(S):
            #         phi_sum[i] += self.mpt.matrix_a[j,i]*np.mean(phi_zero[self.monomer_types[j]])
            #     w_aux_homo[i,:] -= phi_sum[i] + self.mpt.h_coef_mu1[i]
            #     w_aux_homo[i,:] /= 2*self.mpt.h_coef_mu2[i]
            # print(w_aux_homo[:,0])

            # # Find j_k function in LR
            # for count_j, j in enumerate(self.mpt.aux_fields_imag_idx):
            #     w_aux_perturbed = w_aux_homo.copy()
            #     w_aux_perturbed[j,0] = 1e-3 # a small perturbation
            #     w_aux_perturbed_k = np.fft.rfftn(np.reshape(w_aux_perturbed[j], self.cb.get_nx()))/np.prod(self.cb.get_nx())

            #     phi_perturbed, _ = self.compute_concentrations(w_aux_perturbed)
            #     h_deriv_perturbed, _ = self.mpt.compute_func_deriv(w_aux_perturbed, phi_perturbed, self.mpt.aux_fields_imag_idx)
                
            #     for count_i, i in enumerate(self.mpt.aux_fields_imag_idx):
            #         h_deriv_perturbed_k = np.fft.rfftn(np.reshape(h_deriv_perturbed[count_i], self.cb.get_nx()))/np.prod(self.cb.get_nx())
            #         j_k_numeric = np.real(h_deriv_perturbed_k/w_aux_perturbed_k)
            #         print("i,j:", i,j)
            #         # print(np.mean(h_deriv_perturbed_k), np.std(h_deriv_perturbed_k))
            #         # print(np.mean(w_aux_perturbed_k), np.std(w_aux_perturbed_k))
            #         print(np.mean(j_k_numeric), np.std(j_k_numeric))
            #         # print(np.mean(self.lr.j_k), np.std(self.lr.j_k))
            #         lr.j_k[count_i, count_j] = j_k_numeric.copy()
        
        if params["compressor"]["name"] == "am":
            self.compressor = am
        elif params["compressor"]["name"] == "lr":
            self.compressor = lr
        elif params["compressor"]["name"] == "lram":
            self.compressor = LRAM(lr, am)            

        # Standard deviation of normal noise of Langevin dynamics
        langevin_sigma = calculate_sigma(params["langevin"]["nbar"], params["langevin"]["dt"], np.prod(params["nx"]), np.prod(params["lx"]))

        # dH/dw_aux[i] is scaled by dt_scaling[i]
        self.dt_scaling = np.ones(S)
        for i in range(S-1):
            self.dt_scaling[i] = np.abs(self.mpt.eigenvalues[i])/np.max(np.abs(self.mpt.eigenvalues))

        # Set random generator
        if random_seed is None:         
            self.random_bg = np.random.PCG64()  # Set random bit generator
        else:
            self.random_bg = np.random.PCG64(random_seed)
        self.random = np.random.Generator(self.random_bg)
        
        print("---------- Simulation Parameters ----------")
        print("Platform :", platform)
        print("Box Dimension: %d" % (self.cb.get_dim()))
        print("Nx:", self.cb.get_nx())
        print("Lx:", self.cb.get_lx())
        print("dx:", self.cb.get_dx())
        print("Volume: %f" % (self.cb.get_volume()))

        print("Chain model: %s" % (params["chain_model"]))
        print("Segment lengths:\n\t", list(self.segment_lengths.items()))
        print("Conformational asymmetry (epsilon): ")
        for monomer_pair in itertools.combinations(self.monomer_types,2):
            print("\t%s/%s: %f" % (monomer_pair[0], monomer_pair[1], self.segment_lengths[monomer_pair[0]]/self.segment_lengths[monomer_pair[1]]))

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
        print("Scaling factor of delta tau N for each field: ", self.dt_scaling)
        print("Random Number Generator: ", self.random_bg.state)

        propagator_computation_optimizer.display_blocks()
        propagator_computation_optimizer.display_propagators()

        #  Save internal variables
        self.params = params
        self.chain_model = params["chain_model"]
        self.ds = params["ds"]
        self.langevin = params["langevin"].copy()
        self.langevin.update({"sigma":langevin_sigma})

        self.verbose_level = params["verbose_level"]
        self.saddle = params["saddle"].copy()
        self.recording = params["recording"].copy()

        self.molecules = molecules
        self.propagator_computation_optimizer = propagator_computation_optimizer

    def compute_concentrations(self, w_aux):
        S = len(self.monomer_types)
        elapsed_time = {}

        # Convert auxiliary fields to monomer fields
        w = self.mpt.to_monomer_fields(w_aux)

        # Make a dictionary for input fields 
        w_input = {}
        for i in range(S):
            w_input[self.monomer_types[i]] = w[i]
        for random_polymer_name, random_fraction in self.random_fraction.items():
            w_input[random_polymer_name] = np.zeros(self.cb.get_total_grid(), dtype=np.float64)
            for monomer_type, fraction in random_fraction.items():
                w_input[random_polymer_name] += w_input[monomer_type]*fraction

        # For the given fields, compute propagators
        time_solver_start = time.time()
        self.solver.compute_propagators(w_input)
        elapsed_time["solver"] = time.time() - time_solver_start

        # Compute concentrations for each monomer type
        time_phi_start = time.time()
        phi = {}
        self.solver.compute_concentrations()
        for monomer_type in self.monomer_types:
            phi[monomer_type] = self.solver.get_total_concentration(monomer_type)

        # Add random copolymer concentration to each monomer type
        for random_polymer_name, random_fraction in self.random_fraction.items():
            phi[random_polymer_name] = self.solver.get_total_concentration(random_polymer_name)
            for monomer_type, fraction in random_fraction.items():
                phi[monomer_type] += phi[random_polymer_name]*fraction
        elapsed_time["phi"] = time.time() - time_phi_start

        return phi, elapsed_time

    def save_simulation_data(self, path, w, phi, langevin_step, normal_noise_prev):

        # Make a dictionary for chi_n
        chi_n_mat = {}
        for key in self.chi_n:
            chi_n_mat[key] = self.chi_n[key]

        # Make dictionary for data
        mdic = {
            "initial_params": self.params,
            "dim":self.cb.get_dim(), "nx":self.cb.get_nx(), "lx":self.cb.get_lx(),
            "monomer_types":self.monomer_types, "chi_n":chi_n_mat, "chain_model":self.chain_model, "ds":self.ds,
            "dt":self.langevin["dt"], "nbar":self.langevin["nbar"], 
            "eigenvalues": self.mpt.eigenvalues,
            "aux_fields_real": self.mpt.aux_fields_real_idx,
            "aux_fields_imag": self.mpt.aux_fields_imag_idx,
            "matrix_a": self.mpt.matrix_a, "matrix_a_inverse": self.mpt.matrix_a_inv, 
            "langevin_step":langevin_step,
            "random_generator":self.random_bg.state["bit_generator"],
            "random_state_state":str(self.random_bg.state["state"]["state"]),
            "random_state_inc":str(self.random_bg.state["state"]["inc"]),
            "normal_noise_prev":normal_noise_prev}

        # Add w fields to the dictionary
        for i, name in enumerate(self.monomer_types):
            mdic["w_" + name] = w[i]
        
        # Add concentrations to the dictionary
        for name in self.monomer_types:
            mdic["phi_" + name] = phi[name]

        # Save data with matlab format
        savemat(path, mdic, long_field_names=True, do_compression=True)

    def continue_run(self, file_name):

        # Load_data
        load_data = loadmat(file_name, squeeze_me=True)
        
        # Check if load_data["langevin_step"] is a multiple of self.recording["sf_recording_period"]
        if load_data["langevin_step"] % self.recording["sf_recording_period"] != 0:
            print(f"(Warning!) 'langevin_step' of {file_name} is not a multiple of 'sf_recording_period'.")
            next_sf_langevin_step = (load_data["langevin_step"]//self.recording["sf_recording_period"] + 1)*self.recording["sf_recording_period"]
            print(f"The structure function will be correctly recorded after {next_sf_langevin_step}th langevin_step." )

        # Restore random state
        self.random_bg.state ={'bit_generator': 'PCG64',
            'state': {'state': int(load_data["random_state_state"]),
                      'inc':   int(load_data["random_state_inc"])},
                      'has_uint32': 0, 'uinteger': 0}
        print("Restored Random Number Generator: ", self.random_bg.state)

        # Make initial_fields
        initial_fields = {}
        for name in self.monomer_types:
            initial_fields[name] = np.array(load_data["w_" + name])

        # Run
        self.run(initial_fields=initial_fields,
            normal_noise_prev=load_data["normal_noise_prev"],
            start_langevin_step=load_data["langevin_step"]+1)

    def run(self, initial_fields, normal_noise_prev=None, start_langevin_step=None):

        print("---------- Run  ----------")

        # The number of components
        S = len(self.monomer_types)

        # The numbers of real and imaginary fields, respectively
        R = len(self.mpt.aux_fields_real_idx)
        I = len(self.mpt.aux_fields_imag_idx)

        # Simulation data directory
        pathlib.Path(self.recording["dir"]).mkdir(parents=True, exist_ok=True)

        # Reshape initial fields
        w = np.zeros([S, self.cb.get_total_grid()], dtype=np.float64)
        for i in range(S):
            w[i] = np.reshape(initial_fields[self.monomer_types[i]],  self.cb.get_total_grid())
            
        # Convert monomer chemical potential fields into auxiliary fields
        w_aux = self.mpt.to_aux_fields(w)

        # Find saddle point
        print("iterations, mass error, total partitions, Hamiltonian, incompressibility error (or saddle point error)")
        phi, _, _, _, = self.find_saddle_point(w_aux=w_aux)

        # Dictionary to record history of H and dH/dχN
        H_history = []
        dH_history = {}
        for key in self.chi_n:
            dH_history[key] = []

        # Arrays for structure function
        sf_average = {} # <u(k) phi(-k)>
        for monomer_id_pair in itertools.combinations_with_replacement(list(range(S)),2):
            sorted_pair = sorted(monomer_id_pair)
            type_pair = self.monomer_types[sorted_pair[0]] + "," + self.monomer_types[sorted_pair[1]]
            sf_average[type_pair] = np.zeros_like(np.fft.rfftn(np.reshape(w[0], self.cb.get_nx())), np.complex128)

        # Create an empty array for field update algorithm
        if normal_noise_prev is None :
            normal_noise_prev = np.zeros([R, self.cb.get_total_grid()], dtype=np.float64)
        else:
            normal_noise_prev = normal_noise_prev

        if start_langevin_step is None :
            start_langevin_step = 1

        # The number of times that 'find_saddle_point' has failed to find a saddle point
        saddle_fail_count = 0
        successive_fail_count = 0

        # Init timers
        total_saddle_iter = 0
        total_error_level = 0
        time_start = time.time()

        # Langevin iteration begins here
        for langevin_step in range(start_langevin_step, self.langevin["max_step"]+1):
            print("Langevin step: ", langevin_step)

            # Copy data for restoring
            w_aux_copy = w_aux.copy()
            phi_copy = phi.copy()

            # Compute functional derivatives of Hamiltonian w.r.t. real-valued fields 
            w_lambda, _ = self.mpt.compute_func_deriv(w_aux, phi, self.mpt.aux_fields_real_idx)

            # Update w_aux using Leimkuhler-Matthews method
            normal_noise_current = self.random.normal(0.0, self.langevin["sigma"], [R, self.cb.get_total_grid()])
            for count, i in enumerate(self.mpt.aux_fields_real_idx):
                scaling = self.dt_scaling[i]
                w_aux[i] += -w_lambda[count]*self.langevin["dt"]*scaling + 0.5*(normal_noise_prev[count] + normal_noise_current[count])*np.sqrt(scaling)

            # Swap two noise arrays
            normal_noise_prev, normal_noise_current = normal_noise_current, normal_noise_prev

            # Find saddle point of the pressure field
            phi, hamiltonian, saddle_iter, error_level = self.find_saddle_point(w_aux=w_aux)
            total_saddle_iter += saddle_iter
            total_error_level += error_level

            # If the tolerance of the saddle point was not met, regenerate Langevin random noise and continue
            if np.isnan(error_level) or error_level >= self.saddle["tolerance"]:
                if successive_fail_count < 5:                
                    print("The tolerance of the saddle point was not met. Langevin random noise is regenerated.")

                    # Restore w_aux and phi
                    w_aux = w_aux_copy
                    phi = phi_copy
                    
                    # Increment counts and continue
                    successive_fail_count += 1
                    saddle_fail_count += 1
                    continue
                else:
                    print("The tolerance of the saddle point was not met %d times in a row. Simulation is aborted." % (successive_fail_count))
                    break
            else:
                successive_fail_count = 0

            # Compute H and dH/dχN
            if langevin_step % self.recording["sf_computing_period"] == 0:
                H_history.append(hamiltonian)
                dH = self.mpt.compute_h_deriv_chin(self.chi_n, w_aux)
                for key in self.chi_n:
                    dH_history[key].append(dH[key])

            # Save H and dH/dχN
            if langevin_step % self.recording["sf_recording_period"] == 0:
                H_history = np.array(H_history)
                mdic = {"H_history": H_history}
                for key in self.chi_n:
                    dH_history[key] = np.array(dH_history[key])
                    monomer_pair = sorted(key.split(","))
                    mdic["dH_history_" + monomer_pair[0] + "_" + monomer_pair[1]] = dH_history[key]
                savemat(os.path.join(self.recording["dir"], "dH_%06d.mat" % (langevin_step)), mdic, long_field_names=True, do_compression=True)
                # Reset dictionary
                H_history = []
                for key in self.chi_n:
                    dH_history[key] = []
                    
            # Calculate structure function
            if langevin_step % self.recording["sf_computing_period"] == 0:
                # Perform Fourier transforms
                mu_fourier = {}
                phi_fourier = {}
                for i in range(S):
                    key = self.monomer_types[i]
                    phi_fourier[key] = np.fft.rfftn(np.reshape(phi[self.monomer_types[i]], self.cb.get_nx()))/self.cb.get_total_grid()
                    mu_fourier[key] = np.zeros_like(phi_fourier[key], np.complex128)
                    for k in range(S-1) :
                        mu_fourier[key] += np.fft.rfftn(np.reshape(w_aux[k], self.cb.get_nx()))*self.mpt.matrix_a_inv[k,i]/self.mpt.eigenvalues[k]/self.cb.get_total_grid()
                # Accumulate S_ij(K), assuming that <u(k)>*<phi(-k)> is zero
                for key in sf_average:
                    monomer_pair = sorted(key.split(","))
                    sf_average[key] += mu_fourier[monomer_pair[0]]* np.conj( phi_fourier[monomer_pair[1]])

            # Save structure function
            if langevin_step % self.recording["sf_recording_period"] == 0:
                # Make a dictionary for chi_n
                chi_n_mat = {}
                for key in self.chi_n:
                    monomer_pair = sorted(key.split(","))
                    chi_n_mat[monomer_pair[0] + "," + monomer_pair[1]] = self.chi_n[key]
                mdic = {"dim":self.cb.get_dim(), "nx":self.cb.get_nx(), "lx":self.cb.get_lx(),
                        "chi_n":chi_n_mat, "chain_model":self.chain_model, "ds":self.ds,
                        "dt":self.langevin["dt"], "nbar":self.langevin["nbar"], "initial_params":self.params}
                # Add structure functions to the dictionary
                for key in sf_average:
                    sf_average[key] *= self.recording["sf_computing_period"]/self.recording["sf_recording_period"]* \
                            self.cb.get_volume()*np.sqrt(self.langevin["nbar"])
                    monomer_pair = sorted(key.split(","))
                    mdic["structure_function_" + monomer_pair[0] + "_" + monomer_pair[1]] = sf_average[key]
                savemat(os.path.join(self.recording["dir"], "structure_function_%06d.mat" % (langevin_step)), mdic, long_field_names=True, do_compression=True)
                # Reset arrays
                for key in sf_average:
                    sf_average[key][:,:,:] = 0.0

            # Save simulation data
            if langevin_step % self.recording["recording_period"] == 0:
                w = self.mpt.to_monomer_fields(w_aux)
                self.save_simulation_data(
                    path=os.path.join(self.recording["dir"], "fields_%06d.mat" % (langevin_step)),
                    w=w, phi=phi, langevin_step=langevin_step, normal_noise_prev=normal_noise_prev)

        print( "The number of times that tolerance of saddle point was not met and Langevin random noise was regenerated: %d times" % 
            (saddle_fail_count))

        # Estimate execution time
        time_duration = time.time() - time_start

        print("total time: %f, time per step: %f" %
            (time_duration, time_duration/(langevin_step+1-start_langevin_step)) )
        
        print("Total iterations for saddle points: %d, Iterations per Langevin step: %f" %
            (total_saddle_iter, total_saddle_iter/(langevin_step+1-start_langevin_step)))

    def find_saddle_point(self, w_aux):

        # The number of components
        S = len(self.monomer_types)

        # The numbers of real and imaginary fields, respectively
        R = len(self.mpt.aux_fields_real_idx)
        I = len(self.mpt.aux_fields_imag_idx)

        # Assign large initial value for error
        error_level = 1e20

        # Reset compressor
        self.compressor.reset_count()

        # Saddle point iteration begins here
        for saddle_iter in range(1,self.saddle["max_iter"]+1):
            
            # Compute total concentrations with noised w_aux
            phi, _ = self.compute_concentrations(w_aux)

            # Compute functional derivatives of Hamiltonian w.r.t. imaginary fields 
            h_deriv, _ = self.mpt.compute_func_deriv(w_aux, phi, self.mpt.aux_fields_imag_idx)

            # Compute total error
            old_error_level = error_level
            error_level_array = np.std(h_deriv, axis=1)
            error_level = np.max(error_level_array)

            # Print iteration # and error levels
            if(self.verbose_level == 2 or self.verbose_level == 1 and
            (error_level < self.saddle["tolerance"] or saddle_iter == self.saddle["max_iter"])):
            
                # Calculate Hamiltonian
                total_partitions = [self.solver.get_total_partition(p) for p in range(self.molecules.get_n_polymer_types())]
                hamiltonian = self.mpt.compute_hamiltonian(self.molecules, w_aux, total_partitions)

                # Check the mass conservation
                mass_error = np.mean(h_deriv[I-1])
                print("%8d %12.3E " % (saddle_iter, mass_error), end=" [ ")
                for p in range(self.molecules.get_n_polymer_types()):
                    print("%13.7E " % (self.solver.get_total_partition(p)), end=" ")
                print("] %15.9f   [" % (hamiltonian), end="")
                for i in range(I):
                    print("%13.7E" % (error_level_array[i]), end=" ")
                print("]")

            # Conditions to end the iteration
            if error_level < self.saddle["tolerance"]:
                break

            # Scaling h_deriv
            for count, i in enumerate(self.mpt.aux_fields_imag_idx):
                h_deriv[count] *= self.dt_scaling[i]

            # Calculate new fields using compressor (AM, LR, LRAM)
            w_aux[self.mpt.aux_fields_imag_idx] = np.reshape(self.compressor.calculate_new_fields(w_aux[self.mpt.aux_fields_imag_idx], -h_deriv, old_error_level, error_level), [I, self.cb.get_total_grid()])

        # Set mean of pressure field to zero
        w_aux[S-1] -= np.mean(w_aux[S-1])

        return phi, hamiltonian, saddle_iter, error_level