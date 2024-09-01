import time
import numpy as np

class Symmetric_Original:
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

        vector_large_s_prime = self.vector_large_s.copy()
        vector_large_s_prime[S-1] = 0.0
        self.o_large_s = np.reshape(np.matmul(matrix_o, vector_large_s_prime), (S, 1))/S

        # Indices whose eigen fields are real
        self.eigen_fields_real_idx = []
        # Indices whose eigen fields are imaginary including the pressure field
        self.eigen_fields_imag_idx = []
        for i in range(S-1):
            # assert(not np.isclose(eigenvalues[i], 0.0)), \
            #     "One of eigenvalues is zero for given chiN values."
            if np.isclose(eigenvalues[i], 0.0):
                print("One of eigenvalues is zero for given chiN values.")
            elif eigenvalues[i] > 0:
                self.eigen_fields_imag_idx.append(i)
            else:
                self.eigen_fields_real_idx.append(i)
        self.eigen_fields_imag_idx.append(S-1) # add pressure field

        # The numbers of real and imaginary fields, respectively
        self.R = len(self.eigen_fields_real_idx)
        self.I = len(self.eigen_fields_imag_idx)

        if self.I > 1:
            print("(Warning!) For a given χN interaction parameter set, at least one of the eigen fields is an imaginary field. ", end="")
            print("The field fluctuations would not be fully reflected. Run this simulation at your own risk.")

        # Compute coefficients for Hamiltonian computation
        h_const, h_coef_mu1, h_coef_mu2 = self.compute_h_coef(chi_n, eigenvalues)

        # Matrix A and Inverse for converting between eigen fields and species chemical potential fields
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

        # print("Projection matrix P:\n\t", str(self.matrix_p).replace("\n", "\n\t"))
        # print("Projection matrix Q:\n\t", str(self.matrix_q).replace("\n", "\n\t"))
        print("Eigenvalues:\n\t", self.eigenvalues)
        print("Eigenvectors [v1, v2, ...] :\n\t", str(self.matrix_o).replace("\n", "\n\t"))
        print("Mapping matrix A:\n\t", str(self.matrix_a).replace("\n", "\n\t"))
        # print("A*Inverse[A]:\n\t", str(np.matmul(self.matrix_a, self.matrix_a_inv)).replace("\n", "\n\t"))
        # print("P matrix for field residuals:\n\t", str(self.matrix_p).replace("\n", "\n\t"))

        print("Real Fields: ",      self.eigen_fields_real_idx)
        print("Imaginary Fields: ", self.eigen_fields_imag_idx)
        
        print("In Hamiltonian:")
        print("\treference energy: ", self.h_const)
        print("\tcoefficients of int of mu(r)/V: ", self.h_coef_mu1)
        print("\tcoefficients of int of mu(r)^2/V: ", self.h_coef_mu2)
        print("\tdH_ref/dχN: ", self.h_const_deriv_chin)
        print("\td(coef of mu(r))/dχN: ", self.h_coef_mu1_deriv_chin)
        print("\td(coef of mu(r)^2)/dχN: ", self.h_coef_mu2_deriv_chin)

    def to_eigen_fields(self, w):
        return np.matmul(self.matrix_a_inv, w-self.o_large_s)

    def to_monomer_fields(self, w_eigen):
        return np.matmul(self.matrix_a, w_eigen) + self.o_large_s

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

        # Make the first element of each vector positive to recover the conventional AB polymer field theory
        for i in range(S):
            if eigenvectors[0,i] < 0.0:
                eigenvectors[:,i] *= -1.0

        # Multiply √S to eigenvectors
        eigenvectors *= np.sqrt(S)

        return eigenvalues, eigenvectors

    def compute_h_coef(self, chi_n, eigenvalues):
        S = len(self.monomer_types)

        # Compute reference part of Hamiltonian
        h_const = 0.5*np.sum(self.vector_s)/S

        # Compute coefficients of integral of μ(r)/V
        h_coef_mu1 = np.zeros(S-1)

        # Compute coefficients of integral of μ(r)^2/V
        h_coef_mu2 = np.zeros(S-1)
        for i in range(S-1):
            if not np.isclose(eigenvalues[i], 0.0):
                h_coef_mu2[i] = -0.5/eigenvalues[i]*S

        return h_const, h_coef_mu1, h_coef_mu2

    # Compute total Hamiltonian
    def compute_hamiltonian(self, molecules, w_eigen, total_partitions):
        S = len(self.monomer_types)

        # Compute Hamiltonian part that is related to fields
        hamiltonian_fields = -np.mean(w_eigen[S-1])
        for i in range(S-1):
            hamiltonian_fields += self.h_coef_mu2[i]*np.mean(w_eigen[i]**2)
            hamiltonian_fields += self.h_coef_mu1[i]*np.mean(w_eigen[i])
        
        # Compute Hamiltonian part that total partition functions
        hamiltonian_partition = 0.0
        for p in range(molecules.get_n_polymer_types()):
            hamiltonian_partition -= molecules.get_polymer(p).get_volume_fraction()/ \
                            molecules.get_polymer(p).get_alpha() * \
                            np.log(total_partitions[p])

        return hamiltonian_partition + hamiltonian_fields + self.h_const

    # Compute functional derivatives of Hamiltonian w.r.t. exchange and pressure fields of selected indices
    def compute_func_deriv(self, w_eigen, phi, indices):
        S = len(self.monomer_types)
                
        elapsed_time = {}
        time_e_start = time.time()
        h_deriv = np.zeros([len(indices), w_eigen.shape[1]], dtype=np.float64)
        for count, i in enumerate(indices):
            # Exchange fields
            if i != S-1:
                h_deriv[count] += 2*self.h_coef_mu2[i]*w_eigen[i]
                h_deriv[count] +=   self.h_coef_mu1[i]
                for j in range(S):
                    h_deriv[count] += self.matrix_a[j,i]*phi[self.monomer_types[j]]
            # Pressure field
            else:
                for j in range(S):
                    h_deriv[count] -= phi[self.monomer_types[j]]
                h_deriv[count] += 1.0
        elapsed_time["h_deriv"] = time.time() - time_e_start
        
        return  h_deriv, elapsed_time

    # Compute dH/dχN
    def compute_h_deriv_chin(self, chi_n, w_eigen):
        S = len(self.monomer_types)

        dH = {}
        for key in chi_n:
            dH[key] = self.h_const_deriv_chin[key]
            for i in range(S-1):
                dH[key] += self.h_coef_mu2_deriv_chin[key][i]*np.mean(w_eigen[i]**2)
                dH[key] += self.h_coef_mu1_deriv_chin[key][i]*np.mean(w_eigen[i])                            
        return dH
