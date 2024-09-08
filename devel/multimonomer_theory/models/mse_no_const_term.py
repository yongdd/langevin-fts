import time
import numpy as np


class MPT_MSE_No_Const:
    def __init__(self, monomer_types, chi_n):
        self.monomer_types = monomer_types
        S = len(self.monomer_types)

        # Compute eigenvalues and orthogonal matrix
        eigenvalues, matrix_o = self.compute_eigen_system(chi_n)

        # Compute vector X_iS
        vector_s = np.zeros(S-1)
        for i in range(S-1):
            monomer_pair = [self.monomer_types[i], self.monomer_types[S-1]]
            monomer_pair.sort()
            key = monomer_pair[0] + "," + monomer_pair[1]            
            vector_s[i] = chi_n[key]
        self.vector_large_s = np.matmul(np.transpose(matrix_o), vector_s)

        self.o_large_s = np.zeros(S)
        self.o_large_s[0:S-1] = vector_s
        self.o_large_s = np.reshape(self.o_large_s, (S, 1))

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
        h_const, h_coef_mu1, h_coef_mu2 = self.compute_h_coef(eigenvalues)

        # Matrix A and Inverse for converting between auxiliary fields and monomer chemical potential fields
        matrix_a = np.zeros((S,S))
        matrix_a_inv = np.zeros((S,S))
        matrix_a[0:S-1,0:S-1] = matrix_o[0:S-1,0:S-1]
        matrix_a[:,S-1] = 1
        matrix_a_inv[0:S-1,0:S-1] = np.transpose(matrix_o[0:S-1,0:S-1])
        for i in range(S-1):
            matrix_a_inv[i,S-1] =  -np.sum(matrix_o[:,i])
            matrix_a_inv[S-1,S-1] = 1

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
            eigenvalues_p, _ = self.compute_eigen_system(chi_n_p)
            eigenvalues_n, _ = self.compute_eigen_system(chi_n_n)
            
            # Compute coefficients for Hamiltonian computation
            h_const_p, h_coef_mu1_p, h_coef_mu2_p = self.compute_h_coef(eigenvalues_p)
            h_const_n, h_coef_mu1_n, h_coef_mu2_n = self.compute_h_coef(eigenvalues_n)
            
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
        return np.matmul(self.matrix_a_inv, w-self.o_large_s)

    def to_monomer_fields(self, w_aux):
        return np.matmul(self.matrix_a, w_aux) + self.o_large_s

    def compute_eigen_system(self, chi_n):
        S = len(self.monomer_types)
        matrix_chi = np.zeros((S,S))
        matrix_chin = np.zeros((S-1,S-1))
        for i in range(S):
            for j in range(i+1,S):
                monomer_pair = [self.monomer_types[i], self.monomer_types[j]]
                monomer_pair.sort()
                key = monomer_pair[0] + "," + monomer_pair[1]
                if key in chi_n:
                    matrix_chi[i,j] = chi_n[key]
                    matrix_chi[j,i] = chi_n[key]
        
        for i in range(S-1):
            for j in range(S-1):
                matrix_chin[i,j] = matrix_chi[i,j] - matrix_chi[i,S-1] - matrix_chi[j,S-1] # fix a typo in the paper
        
        return np.linalg.eigh(matrix_chin)

    def compute_h_coef(self, eigenvalues):
        S = len(self.monomer_types)

        # Compute reference part of Hamiltonian
        h_const = 0.0

        # Compute coefficients of integral of μ(r)/V
        h_coef_mu1 = np.zeros(S-1)

        # Compute coefficients of integral of μ(r)^2/V
        h_coef_mu2 = np.zeros(S-1)
        for i in range(S-1):
            if not np.isclose(eigenvalues[i], 0.0):
                h_coef_mu2[i] = -0.5/eigenvalues[i]

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
