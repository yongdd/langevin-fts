import os
import time
import pathlib
import numpy as np
import itertools
from scipy.io import savemat, loadmat
from langevinfts import *

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

        # (c++ class) Create a factory for given platform and chain_model
        if "reduce_gpu_memory_usage" in params and platform == "cuda":
            factory = PlatformSelector.create_factory(platform, params["chain_model"], params["reduce_gpu_memory_usage"])
        else:
            factory = PlatformSelector.create_factory(platform, params["chain_model"], False)
        factory.display_info()

        # (C++ class) Computation box
        cb = factory.create_computation_box(params["nx"], params["lx"])

        # Flory-Huggins parameters, chi*N
        self.chi_n = {}
        for pair_chi_n in params["chi_n"]:
            assert(pair_chi_n[0] in params["segment_lengths"]), \
                f"Monomer type '{pair_chi_n[0]}' is not in 'segment_lengths'."
            assert(pair_chi_n[1] in params["segment_lengths"]), \
                f"Monomer type '{pair_chi_n[1]}' is not in 'segment_lengths'."
            assert(len(set(pair_chi_n[0:2])) == 2), \
                "Do not add self interaction parameter, " + str(pair_chi_n[0:3]) + "."
            assert(not frozenset(pair_chi_n[0:2]) in self.chi_n), \
                f"There are duplicated chi N ({pair_chi_n[0:2]}) parameters."
            self.chi_n[frozenset(pair_chi_n[0:2])] = pair_chi_n[2]

        for monomer_pair in itertools.combinations(self.monomer_types, 2):
            if not frozenset(list(monomer_pair)) in self.chi_n:
                self.chi_n[frozenset(list(monomer_pair))] = 0.0

        # Exchange mapping matrix.
        # See paper *J. Chem. Phys.* **2014**, 141, 174103
        S = len(self.monomer_types)
            
        # Compute exchange mapping for given chiN set
        # Initialize following variables:
        #   self.exchange_eigenvalues,
        #   self.matrix_o,
        #   self.free_energy_coef
        #   self.vector_s,
        #   self.matrix_a,
        #   self.matrix_a_inv,
        self.initialize_exchange_mapping()
        
        # Indices whose exchange fields are real
        self.exchange_fields_real_idx = []
        # Indices whose exchange fields are imaginary including the pressure field
        self.exchange_fields_imag_idx = []
        for i in range(S-1):
            assert(not np.isclose(self.exchange_eigenvalues[i], 0.0)), \
                "One of eigenvalues is zero for given chiN values."
            if self.exchange_eigenvalues[i] > 0:
                self.exchange_fields_imag_idx.append(i)
            else:
                self.exchange_fields_real_idx.append(i)
        self.exchange_fields_imag_idx.append(S-1) # add pressure field

        # The numbers of real and imaginary fields, respectively
        self.R = len(self.exchange_fields_real_idx)
        self.I = len(self.exchange_fields_imag_idx)

        if self.I > 1:
            print("(Warning!) For a given chi N interaction parameter set, at least one of the exchange fields is an imaginary field. ", end="")
            print("The field fluctuations would not be fully reflected. Run this simulation at your own risk.")

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
                "The sum of volume fraction of random copolymer must be equal to 1."

            random_type_string = polymer["blocks"][0]["type"]
            assert(not random_type_string in params["segment_lengths"]), \
                f"The name of random copolymer '{random_type_string}' is already used as a type in 'segment_lengths' or other random copolymer"

            # Add random copolymers
            polymer["block_monomer_types"] = [random_type_string]
            params["segment_lengths"].update({random_type_string:statistical_segment_length})
            self.random_fraction[random_type_string] = polymer["blocks"][0]["fraction"]

        # (C++ class) Mixture box
        if "use_superposition" in params:
            mixture = factory.create_mixture(params["ds"], params["segment_lengths"], params["use_superposition"])
        else:
            mixture = factory.create_mixture(params["ds"], params["segment_lengths"], True)

        # Add polymer chains
        for polymer in params["distinct_polymers"]:
            # print(polymer["volume_fraction"], polymer["block_monomer_types"], polymer["block_lengths"], polymer["v"], polymer["u"])
            mixture.add_polymer(polymer["volume_fraction"], polymer["block_monomer_types"], polymer["block_lengths"], polymer["v"] ,polymer["u"])

        # (C++ class) Solver using Pseudo-spectral method
        pseudo = factory.create_pseudo(cb, mixture)

        # (C++ class) Fields Relaxation using Anderson Mixing
        am = factory.create_anderson_mixing(
            len(self.exchange_fields_imag_idx)*np.prod(params["nx"]),   # the number of variables
            params["am"]["max_hist"],                                   # maximum number of history
            params["am"]["start_error"],                                # when switch to AM from simple mixing
            params["am"]["mix_min"],                                    # minimum mixing rate of simple mixing
            params["am"]["mix_init"])                                   # initial mixing rate of simple mixing

        # Langevin Dynamics
        # standard deviation of normal noise
        langevin_sigma = calculate_sigma(params["langevin"]["nbar"], params["langevin"]["dt"], np.prod(params["nx"]), np.prod(params["lx"]))
        self.dt_scaling = np.ones(S)
        for i in range(len(self.exchange_eigenvalues)):
            self.dt_scaling[i] = np.abs(self.exchange_eigenvalues[i])/np.max(np.abs(self.exchange_eigenvalues))
        
        # Set random generator
        if random_seed == None:         
            self.random_bg = np.random.PCG64()  # Set random bit generator
        else:
            self.random_bg = np.random.PCG64(random_seed)
        self.random = np.random.Generator(self.random_bg)
        
        # -------------- print simulation parameters ------------
        print("---------- Simulation Parameters ----------")
        print("Platform :", platform)
        print("Statistical Segment Lengths:", params["segment_lengths"])
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

        print("chiN: ")
        for pair in self.chi_n:
            sorted_name_pair = sorted(list(pair)[0:2])
            print("\t%s, %s: %f" % (sorted_name_pair[0], sorted_name_pair[1], self.chi_n[pair]))

        print("Eigenvalues:\n\t", self.exchange_eigenvalues)
        print("Column eigenvectors:\n\t", str(self.matrix_o).replace("\n", "\n\t"))
        print("Vector chi_iS:\n\t", str(self.vector_s).replace("\n", "\n\t"))
        print("Mapping matrix A:\n\t", str(self.matrix_a).replace("\n", "\n\t"))
        print("Inverse of A:\n\t", str(self.matrix_a_inv).replace("\n", "\n\t"))
        print("A*Inverse[A]:\n\t", str(np.matmul(self.matrix_a, self.matrix_a_inv)).replace("\n", "\n\t"))
        print("Real Fields: ",      self.exchange_fields_real_idx)
        print("Imaginary Fields: ", self.exchange_fields_imag_idx)

        for p in range(mixture.get_n_polymers()):
            print("distinct_polymers[%d]:" % (p) )
            print("\tvolume fraction: %f, alpha: %f, N: %d" %
                (mixture.get_polymer(p).get_volume_fraction(),
                 mixture.get_polymer(p).get_alpha(),
                 mixture.get_polymer(p).get_n_segment_total()))

        print("Invariant Polymerization Index (N_Ref): %d" % (params["langevin"]["nbar"]))
        print("Langevin Sigma: %f" % (langevin_sigma))
        print("Scaling factor of delta tau N for each field: ", self.dt_scaling)
        print("Random Number Generator: ", self.random_bg.state)

        mixture.display_blocks()
        mixture.display_propagators()

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
        self.mixture = mixture
        self.pseudo = pseudo
        self.am = am

    def compute_eigenvalues(self, chi_n, chin_epsilon=None):
        S = len(self.monomer_types)
        matrix_chi = np.zeros((S,S))
        matrix_chin = np.zeros((S-1,S-1))
        for i in range(S):
            for j in range(i+1,S):
                key = frozenset([self.monomer_types[i], self.monomer_types[j]])
                if key in chi_n:
                    matrix_chi[i,j] = chi_n[key]
                    matrix_chi[j,i] = chi_n[key]
                    
                # Add very small number to compute derivatives
                if chin_epsilon:
                    if key in chin_epsilon:
                        matrix_chi[i,j] += chin_epsilon[key]
                        matrix_chi[j,i] += chin_epsilon[key]
        
        for i in range(S-1):
            for j in range(S-1):
                matrix_chin[i,j] = matrix_chi[i,j] - matrix_chi[i,S-1] - matrix_chi[j,S-1] # fix a typo in the paper
        
        return np.linalg.eig(matrix_chin)

    def compute_free_energy_coef(self, chin, exchange_eigenvalues, epsilon=1e-3):
        S = len(self.monomer_types)
        free_energy_coef = {}            
        for key in chin:
            exchange_eigenvalues_plus,  _ = self.compute_eigenvalues(chin, chin_epsilon = {key: epsilon})
            exchange_eigenvalues_minus, _ = self.compute_eigenvalues(chin, chin_epsilon = {key: -epsilon})
            exchange_eigenvalues_derivative = (exchange_eigenvalues_plus-exchange_eigenvalues_minus)/(2*epsilon)
            # Compute derivative of -1/(2Vd) w.r.t. chiN
            free_energy_coef[key] = 0.5*exchange_eigenvalues_derivative/(exchange_eigenvalues)**2
        return free_energy_coef

    def initialize_exchange_mapping(self,):
        S = len(self.monomer_types)
        # Compute vector X_iS
        vector_s = np.zeros(S-1)
        for i in range(S-1):
            key = frozenset([self.monomer_types[i], self.monomer_types[S-1]])
            vector_s[i] = self.chi_n[key]

        # Compute eigenvalues and orthogonal matrix
        exchange_eigenvalues, matrix_o = self.compute_eigenvalues(self.chi_n)

        # Matrix A and Inverse for converting between exchange fields and species chemical potential fields
        matrix_a = np.zeros((S,S))
        matrix_a_inv = np.zeros((S,S))
        matrix_a[0:S-1,0:S-1] = matrix_o[0:S-1,0:S-1]
        matrix_a[:,S-1] = 1
        matrix_a_inv[0:S-1,0:S-1] = np.transpose(matrix_o[0:S-1,0:S-1])
        for i in range(S-1):
            matrix_a_inv[i,S-1] =  -np.sum(matrix_o[:,i])
            matrix_a_inv[S-1,S-1] = 1

        # Compute coefficients for free energy computation
        free_energy_coef = self.compute_free_energy_coef(self.chi_n, exchange_eigenvalues, epsilon=1e-3)
        
        self.exchange_eigenvalues = exchange_eigenvalues
        self.matrix_o = matrix_o
        self.vector_s = vector_s
        self.matrix_a = matrix_a
        self.matrix_a_inv = matrix_a_inv
        self.free_energy_coef = free_energy_coef

    def compute_concentrations(self, w_exchange):
        
        S = len(self.monomer_types)
        elapsed_time = {}

        # Convert to species chemical potential fields
        w = np.matmul(self.matrix_a, w_exchange)

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
        self.pseudo.compute_statistics(w_input)
        elapsed_time["pseudo"] = time.time() - time_p_start

        # Compute total concentration for each monomer type
        phi = {}
        for monomer_type in self.monomer_types:
            phi[monomer_type] = self.pseudo.get_total_concentration(monomer_type)

        # Add random copolymer concentration to each monomer type
        for random_polymer_name, random_fraction in self.random_fraction.items():
            phi[random_polymer_name] = self.pseudo.get_total_concentration(random_polymer_name)
            for monomer_type, fraction in random_fraction.items():
                phi[monomer_type] += phi[random_polymer_name]*fraction
        
        return phi, elapsed_time

    # Compute functional derivatives of Hamiltonian w.r.t. real exchange fields 
    def compute_func_deriv_real(self, w_exchange, phi):
        
        S = len(self.monomer_types)
        R = self.R
        elapsed_time = {}

        time_e_start = time.time()        
        h_deriv = np.zeros([R, self.cb.get_n_grid()], dtype=np.float64)
        for count, i in enumerate(self.exchange_fields_real_idx):
            h_deriv[count] -= 1.0/self.exchange_eigenvalues[i]*w_exchange[i]
        for count, i in enumerate(self.exchange_fields_real_idx):
            for j in range(S-1):
                h_deriv[count] += 1.0/self.exchange_eigenvalues[i]*self.matrix_o[j,i]*self.vector_s[j]
                h_deriv[count] += self.matrix_o[j,i]*phi[self.monomer_types[j]]
        elapsed_time["energy"] = time.time() - time_e_start
        
        return  h_deriv, elapsed_time

    # Compute functional derivatives of Hamiltonian w.r.t. imaginary exchange fields 
    def compute_func_deriv_imag(self, w_exchange, phi):

        S = len(self.monomer_types)
        I = self.I
        elapsed_time = {}

        time_e_start = time.time()
        h_deriv = np.zeros([I, self.cb.get_n_grid()], dtype=np.float64)
        for count, i in enumerate(self.exchange_fields_imag_idx):
            if i != S-1:
                h_deriv[count] -= 1.0/self.exchange_eigenvalues[i]*w_exchange[i]
        for count, i in enumerate(self.exchange_fields_imag_idx):
            if i != S-1:
                for j in range(S-1):
                    h_deriv[count] += 1.0/self.exchange_eigenvalues[i]*self.matrix_o[j,i]*self.vector_s[j]
                    h_deriv[count] += self.matrix_o[j,i]*phi[self.monomer_types[j]]
        for i in range(S):
            h_deriv[I-1] += phi[self.monomer_types[i]]
        h_deriv[I-1] -= 1.0
        elapsed_time["energy"] = time.time() - time_e_start
        
        return  h_deriv, elapsed_time

    def save_simulation_data(self, path, w, phi, langevin_step, normal_noise_prev):
        
        # Make dictionary for w fields
        w_species = {}
        for i, name in enumerate(self.monomer_types):
            w_species[name] = w[i]

        # Make dictionary for chi_n
        chi_n_mat = {}
        for pair_chi_n in self.params["chi_n"]:
            sorted_name_pair = sorted(pair_chi_n[0:2])
            chi_n_mat[sorted_name_pair[0] + "," + sorted_name_pair[1]] = pair_chi_n[2]

        # Make dictionary for data
        mdic = {"dim":self.cb.get_dim(), "nx":self.cb.get_nx(), "lx":self.cb.get_lx(),
            "chi_n":chi_n_mat, "chain_model":self.chain_model, "ds":self.ds,
            "dt":self.langevin["dt"], "nbar":self.langevin["nbar"], "initial_params": self.params,
            "eigenvalues": self.exchange_eigenvalues,
            "exchange_fields_real": self.exchange_fields_real_idx, "exchange_fields_imag": self.exchange_fields_imag_idx,
            "matrix_a": self.matrix_a, "matrix_a_inverse": self.matrix_a_inv, 
            "langevin_step":langevin_step,
            "random_generator":self.random_bg.state["bit_generator"],
            "random_state_state":str(self.random_bg.state["state"]["state"]),
            "random_state_inc":str(self.random_bg.state["state"]["inc"]),
            "normal_noise_prev":normal_noise_prev, "monomer_types":self.monomer_types}

        # Add w fields to the dictionary
        for i, name in enumerate(self.monomer_types):
            mdic["w_" + name] = w[i]
        
        # Add concentrations to the dictionary
        for name in self.monomer_types:
            mdic["phi_" + name] = phi[name]

        # Save data with matlab format
        savemat(path, mdic, do_compression=True)

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

        # run
        self.run(initial_fields=initial_fields,
            normal_noise_prev=load_data["normal_noise_prev"],
            start_langevin_step=load_data["langevin_step"]+1)

    def run(self, initial_fields, normal_noise_prev=None, start_langevin_step=None):

        # The number of components
        S = len(self.monomer_types)

        # The numbers of real and imaginary fields, respectively
        R = len(self.exchange_fields_real_idx)
        I = len(self.exchange_fields_imag_idx)

        # Simulation data directory
        pathlib.Path(self.recording["dir"]).mkdir(parents=True, exist_ok=True)

        # Reshape initial fields
        w = np.zeros([S, self.cb.get_n_grid()], dtype=np.float64)
        for i in range(S):
            w[i] = np.reshape(initial_fields[self.monomer_types[i]],  self.cb.get_n_grid())
            
        # Exchange-mapped chemical potential fields
        w_exchange = np.matmul(self.matrix_a_inv, w)

        # Find saddle point 
        phi, _, _, = self.find_saddle_point(w_exchange=w_exchange)

        # Arrays for structure function
        sf_average = {} # <u(k) phi(-k)>
        for monomer_id_pair in itertools.combinations_with_replacement(list(range(S)),2):
            sorted_pair = sorted(monomer_id_pair)
            type_pair = self.monomer_types[sorted_pair[0]] + "," + self.monomer_types[sorted_pair[1]]
            sf_average[type_pair] = np.zeros_like(np.fft.rfftn(np.reshape(w[0], self.cb.get_nx())), np.complex128)

        # # Arrays for free energy computation
        # dF_dchiN = {}
        # for key in self.chi_n:
        #     dF_dchiN[key] = np.zeros(self.recording["sf_recording_period"], dtype=np.float64)

        # Create an empty array for field update algorithm
        if type(normal_noise_prev) == type(None) :
            normal_noise_prev = np.zeros([R, self.cb.get_n_grid()], dtype=np.float64)
        else:
            normal_noise_prev = normal_noise_prev

        if start_langevin_step == None :
            start_langevin_step = 1

        # Init timers
        total_saddle_iter = 0
        total_error_level = 0
        time_start = time.time()

        #------------------ run ----------------------
        print("iteration, mass error, total partitions, total energy, incompressibility error (or saddle point error)")
        print("---------- Run  ----------")
        for langevin_step in range(start_langevin_step, self.langevin["max_step"]+1):
            print("Langevin step: ", langevin_step)
            
            # Compute functional derivatives of Hamiltonian w.r.t. real exchange fields 
            w_lambda, _ = self.compute_func_deriv_real(w_exchange, phi)

            # Update w_exchange using Leimkuhler-Matthews method
            normal_noise_current = self.random.normal(0.0, self.langevin["sigma"], [R, self.cb.get_n_grid()])
            for count, i in enumerate(self.exchange_fields_real_idx):
                scaling = self.dt_scaling[i]
                w_exchange[i] += -w_lambda[count]*self.langevin["dt"]*scaling + 0.5*(normal_noise_prev[count] + normal_noise_current[count])*np.sqrt(scaling)

            # Swap two noise arrays
            normal_noise_prev, normal_noise_current = normal_noise_current, normal_noise_prev

            # Find saddle point of the pressure field
            phi, saddle_iter, error_level = self.find_saddle_point(w_exchange=w_exchange)
            total_saddle_iter += saddle_iter
            total_error_level += error_level
            
            # # # Update exchange mapping for given chiN set
            # # self.initialize_exchange_mapping()
            # # Compute dF/dchiN
            # for key in self.chi_n:
            #     idx = langevin_step % self.recording["sf_recording_period"] - 1
            #     for k in range(S-1):                   
            #         dF_dchiN[key][idx] += self.free_energy_coef[key][k]*np.mean(w_exchange[k]**2)
            # #         print(idx, dF_dchiN[key][idx], self.free_energy_coef[key][k]*np.mean(w_exchange[k]**2))
            # # print(np.mean(w_exchange[0]**2), np.mean(w_exchange[0]))
                    
            # Calculate structure function
            if langevin_step % self.recording["sf_computing_period"] == 0:
                # Perform Fourier transforms
                mu_fourier = {}
                phi_fourier = {}
                for i in range(S):
                    key = self.monomer_types[i]
                    phi_fourier[key] = np.fft.rfftn(np.reshape(phi[self.monomer_types[i]], self.cb.get_nx()))/self.cb.get_n_grid()
                    mu_fourier[key] = np.zeros_like(np.fft.rfftn(np.reshape(w_exchange[0], self.cb.get_nx())), np.complex128)
                    for k in range(S-1) :
                        mu_fourier[key] += np.fft.rfftn(np.reshape(w_exchange[k], self.cb.get_nx()))*self.matrix_a_inv[k,i]/self.exchange_eigenvalues[k]/self.cb.get_n_grid()
                # Accumulate S_ij(K) 
                for monomer_id_pair in itertools.combinations_with_replacement(list(range(S)),2):
                    sorted_pair = sorted(monomer_id_pair)
                    i = sorted_pair[0]
                    j = sorted_pair[1]
                    type_pair = self.monomer_types[i] + "," + self.monomer_types[j]

                    # Assuming that <u(k)>*<phi(-k)> is zero in the disordered phase
                    sf_average[type_pair] += mu_fourier[self.monomer_types[i]]* np.conj( phi_fourier[self.monomer_types[j]])

            # Save structure function
            if langevin_step % self.recording["sf_recording_period"] == 0:
                
                # Make a dictionary for chi_n
                chi_n_mat = {}
                for pair_chi_n in self.params["chi_n"]:
                    sorted_name_pair = sorted(pair_chi_n[0:2])
                    chi_n_mat[sorted_name_pair[0] + "," + sorted_name_pair[1]] = pair_chi_n[2]

                mdic = {"dim":self.cb.get_dim(), "nx":self.cb.get_nx(), "lx":self.cb.get_lx(),
                    "chi_n":chi_n_mat, "chain_model":self.chain_model, "ds":self.ds,
                    "dt":self.langevin["dt"], "nbar":self.langevin["nbar"], "initial_params":self.params}
                
                # Add structure functions to the dictionary
                for monomer_id_pair in itertools.combinations_with_replacement(list(range(S)),2):
                    sorted_pair = sorted(monomer_id_pair)
                    i = sorted_pair[0]
                    j = sorted_pair[1]
                    type_pair = self.monomer_types[i] + "," + self.monomer_types[j]
                    sf_average[type_pair] *= self.recording["sf_computing_period"]/self.recording["sf_recording_period"]* \
                            self.cb.get_volume()*np.sqrt(self.langevin["nbar"])
                    mdic["structure_function_" + self.monomer_types[i] + "_" + self.monomer_types[j]] = sf_average[type_pair]
                savemat(os.path.join(self.recording["dir"], "structure_function_%06d.mat" % (langevin_step)), mdic, do_compression=True)
                
                # Reset Arrays
                for monomer_id_pair in itertools.combinations_with_replacement(list(range(S)),2):
                    sorted_pair = sorted(monomer_id_pair)
                    i = sorted_pair[0]
                    j = sorted_pair[1]
                    type_pair = self.monomer_types[i] + "," + self.monomer_types[j]
                    sf_average[type_pair][:,:,:] = 0.0
                
                # for key in self.chi_n:
                #     dF_dchiN[key][:] = 0.0

            # Save simulation data
            if (langevin_step) % self.recording["recording_period"] == 0:
                w = np.matmul(self.matrix_a, w_exchange)
                self.save_simulation_data(
                    path=os.path.join(self.recording["dir"], "fields_%06d.mat" % (langevin_step)),
                    w=w, phi=phi, langevin_step=langevin_step, normal_noise_prev=normal_noise_prev)

        # Estimate execution time
        time_duration = time.time() - time_start
        return total_saddle_iter, \
                total_saddle_iter/(self.langevin["max_step"]+1-start_langevin_step), \
                time_duration/(self.langevin["max_step"]+1-start_langevin_step), \
                total_error_level/(self.langevin["max_step"]+1-start_langevin_step)

    def find_saddle_point(self, w_exchange):

        # The number of components
        S = len(self.monomer_types)

        # The numbers of real and imaginary fields respectively
        R = len(self.exchange_fields_real_idx)
        I = len(self.exchange_fields_imag_idx)
            
        # Assign large initial value for the energy and error
        energy_total = 1e20
        error_level = 1e20

        # Reset Anderson mixing module
        self.am.reset_count()

        # Concentration of each monomer
        phi = {}

        # Compute hamiltonian part that is only related to real-valued fields
        energy_total_real = 0.0
        for count, i in enumerate(self.exchange_fields_real_idx):
            energy_total_real -= 0.5/self.exchange_eigenvalues[i]*np.dot(w_exchange[i], w_exchange[i])/self.cb.get_n_grid()
        for count, i in enumerate(self.exchange_fields_real_idx):
            for j in range(S-1):
                energy_total_real += 1.0/self.exchange_eigenvalues[i]*self.matrix_o[j,i]*self.vector_s[j]*np.mean(w_exchange[i])
        
        # Compute hamiltonian part that is constant
        for i in range(S-1):
            energy_ref = 0.0
            for j in range(S-1):
                energy_ref += self.matrix_o[j,i]*self.vector_s[j]
            energy_total_real -= 0.5*energy_ref**2/self.exchange_eigenvalues[i]

        # Saddle point iteration begins here
        for saddle_iter in range(1,self.saddle["max_iter"]+1):
            
            # Compute total concentrations with noised w_exchange
            phi, _ = self.compute_concentrations(w_exchange)

            # Compute functional derivatives of Hamiltonian w.r.t. real exchange fields 
            h_deriv, _ = self.compute_func_deriv_imag(w_exchange, phi)

            # Compute total error
            old_error_level = error_level
            error_level_array = np.std(h_deriv, axis=1)
            error_level = np.max(error_level_array)

            # Print iteration # and error levels
            if(self.verbose_level == 2 or self.verbose_level == 1 and
            (error_level < self.saddle["tolerance"] or saddle_iter == self.saddle["max_iter"])):
            
                # Calculate the total energy
                energy_total = energy_total_real - np.mean(w_exchange[S-1])
                for count, i in enumerate(self.exchange_fields_imag_idx):
                    if i != S-1:
                        energy_total -= 0.5/self.exchange_eigenvalues[i]*np.dot(w_exchange[i], w_exchange[i])/self.cb.get_n_grid()
                for count, i in enumerate(self.exchange_fields_imag_idx):
                    if i != S-1:
                        for j in range(S-1):
                            energy_total += 1.0/self.exchange_eigenvalues[i]*self.matrix_o[j,i]*self.vector_s[j]*np.mean(w_exchange[i])
                for p in range(self.mixture.get_n_polymers()):
                    energy_total -= self.mixture.get_polymer(p).get_volume_fraction()/ \
                                    self.mixture.get_polymer(p).get_alpha() * \
                                    np.log(self.pseudo.get_total_partition(p))

                # Check the mass conservation
                mass_error = np.mean(h_deriv[I-1])
                print("%8d %12.3E " % (saddle_iter, mass_error), end=" [ ")
                for p in range(self.mixture.get_n_polymers()):
                    print("%13.7E " % (self.pseudo.get_total_partition(p)), end=" ")
                print("] %15.9f   [" % (energy_total), end="")
                for i in range(I):
                    print("%13.7E" % (error_level_array[i]), end=" ")
                print("]")

            # Conditions to end the iteration
            if error_level < self.saddle["tolerance"]:
                break

            # Scaling h_deriv
            for count, i in enumerate(self.exchange_fields_imag_idx):
                h_deriv[count] *= self.dt_scaling[i]

            # Calculate new fields using simple and Anderson mixing
            w_exchange[self.exchange_fields_imag_idx] = np.reshape(self.am.calculate_new_fields(w_exchange[self.exchange_fields_imag_idx], h_deriv, old_error_level, error_level), [I, self.cb.get_n_grid()])
        
        # Set mean of pressure field to zero
        w_exchange[S-1] -= np.mean(w_exchange[S-1])
        
        return phi, saddle_iter, error_level
