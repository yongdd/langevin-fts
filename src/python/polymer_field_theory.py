import numpy as np

class SymmetricPolymerTheory:
    def __init__(self, monomer_types, chi_n, zeta_n):
        self.monomer_types = monomer_types
        S = len(self.monomer_types)
        
        # Compute eigenvalues, orthogonal matrix, vector_s and vector_S
        eigenvalues, matrix_o, vector_s, vector_large_s = self.compute_eigen_system(monomer_types, chi_n, zeta_n)

        # Indices whose auxiliary fields are real
        self.aux_fields_real_idx = []
        # Indices whose auxiliary fields are imaginary including the pressure field
        self.aux_fields_imag_idx = []
        for i in range(S):
            # assert(not np.isclose(eigenvalues[i], 0.0)), \
            #     "One of eigenvalues is zero for given chiN values."
            if np.isclose(eigenvalues[i], 0.0):
                print("One of eigenvalues is zero for given chiN values.")
            elif eigenvalues[i] > 0:
                self.aux_fields_imag_idx.append(i)
            else:
                self.aux_fields_real_idx.append(i)
        
        if zeta_n is None:
            self.aux_fields_imag_idx.append(S-1) # add pressure field

        # The numbers of real and imaginary fields, respectively
        self.R = len(self.aux_fields_real_idx)
        self.I = len(self.aux_fields_imag_idx)

        if self.I > 1:
            print("(Warning!) For a given χN interaction parameter set, at least one of the auxiliary fields is an imaginary field. ", end="")
            print("The field fluctuations would not be fully reflected. Run this simulation at your own risk.")

        # Compute coefficients for Hamiltonian computation
        h_const, h_coef_mu1, h_coef_mu2 = self.compute_h_coef(eigenvalues, vector_s, vector_large_s, zeta_n)

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
            eigenvalues_p, _, vector_s_p, vector_large_s_p = self.compute_eigen_system(monomer_types, chi_n_p, zeta_n)
            eigenvalues_n, _, vector_s_n, vector_large_s_n = self.compute_eigen_system(monomer_types, chi_n_n, zeta_n)
            
            # Compute coefficients for Hamiltonian computation
            h_const_p, h_coef_mu1_p, h_coef_mu2_p = self.compute_h_coef(eigenvalues_p, vector_s_p, vector_large_s_p, zeta_n)
            h_const_n, h_coef_mu1_n, h_coef_mu2_n = self.compute_h_coef(eigenvalues_n, vector_s_n, vector_large_s_n, zeta_n)
            
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

        print("------------ Polymer Field Theory for Multimonomer System------------")
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

    def compute_eigen_system(self, monomer_types, chi_n, zeta_n):
        S = len(monomer_types)

        # Construct chi_n matrix
        matrix_chin = np.zeros((S,S))
        for i in range(S):
            for j in range(i+1,S):
                monomer_pair = [self.monomer_types[i], self.monomer_types[j]]
                monomer_pair.sort()
                key = monomer_pair[0] + "," + monomer_pair[1]
                if key in chi_n:
                    matrix_chin[i,j] = chi_n[key]
                    matrix_chin[j,i] = chi_n[key]

        # Incompressible model
        if zeta_n is None:
            matrix_q = np.ones((S,S))/S
            matrix_p = np.identity(S) - matrix_q

            # Compute eigenvalues and eigenvectors
            projected_chin = np.matmul(matrix_p, np.matmul(matrix_chin, matrix_p))
            eigenvalues, eigenvectors = np.linalg.eigh(projected_chin)

            # Reordering eigenvalues and eigenvectors
            sorted_indexes = np.argsort(np.abs(eigenvalues))[::-1]
            eigenvalues = eigenvalues[sorted_indexes]
            eigenvectors = eigenvectors[:,sorted_indexes]

            # Reordering eigenvalues and eigenvectors to move the unit vector to S-1
            eigen_val_0 = np.isclose(eigenvalues, 0.0, atol=1e-12)
            eigen_vec_0 = eigenvectors[:,eigen_val_0]
            sorted_indexes = np.argsort(np.std(eigen_vec_0, axis=0))[::-1]
            eigenvectors[:,eigen_val_0] = eigen_vec_0[:,sorted_indexes]

            # Set the last eigenvector to [1, 1, ..., 1]/√S
            eigenvectors[:,-1] = np.ones(S)/np.sqrt(S)

            # Construct vector_s and vector_S
            vector_s = np.matmul(matrix_chin, np.ones(S))/S
            vector_large_s = np.matmul(np.transpose(eigenvectors), vector_s)

        # Compressible model
        else:
            # Compute eigenvalues and eigenvectors
            u = zeta_n*np.ones((S,S)) + matrix_chin
            eigenvalues, eigenvectors = np.linalg.eigh(u)

            # Reordering eigenvalues and eigenvectors
            sorted_indexes = np.argsort(eigenvalues)[::]
            eigenvalues = eigenvalues[sorted_indexes]
            eigenvectors = eigenvectors[:,sorted_indexes]     

            # Construct vector_s and vector_S
            vector_s = np.zeros(S)
            vector_large_s = -zeta_n*np.matmul(np.transpose(eigenvectors), np.ones(S))*np.sqrt(S)
        
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

        return eigenvalues, eigenvectors, vector_s, vector_large_s

    def compute_h_coef(self, eigenvalues, vector_s, vector_large_s, zeta_n):
        S = len(self.monomer_types)
        
        # Compute reference part of Hamiltonian
        h_const = 0.0
        for i in range(S):
            if not np.isclose(eigenvalues[i], 0.0):
                h_const -= 0.5*vector_large_s[i]**2/eigenvalues[i]/S

        # Compute coefficients of integral of μ(r)/V
        h_coef_mu1 = np.zeros(S)
        for i in range(S):
            if not np.isclose(eigenvalues[i], 0.0):
                h_coef_mu1[i] = vector_large_s[i]/eigenvalues[i]

        # Compute coefficients of integral of μ(r)^2/V
        h_coef_mu2 = np.zeros(S)
        for i in range(S):
            if not np.isclose(eigenvalues[i], 0.0):
                h_coef_mu2[i] = -0.5/eigenvalues[i]*S

        if zeta_n is None:
            h_const += 0.5*np.sum(vector_s)/S
            h_coef_mu1[S-1] = -1.0
            h_coef_mu2[S-1] =  0.0
        else:
            h_const += 0.5*zeta_n

        return h_const, h_coef_mu1, h_coef_mu2

    # Compute total Hamiltonian
    def compute_hamiltonian(self, molecules, w_aux, total_partitions, include_const_term=False):
        S = len(self.monomer_types)

        # Compute Hamiltonian part that is related to fields
        hamiltonian_fields = 0.0
        for i in range(S):
            if not np.isclose(self.h_coef_mu2[i], 0.0): 
                hamiltonian_fields += self.h_coef_mu2[i]*np.mean(w_aux[i]**2)
            if not np.isclose(self.h_coef_mu1[i], 0.0): 
                hamiltonian_fields += self.h_coef_mu1[i]*np.mean(w_aux[i])
        
        # Compute Hamiltonian part that total partition functions
        hamiltonian_partition = 0.0
        for p in range(molecules.get_n_polymer_types()):
            hamiltonian_partition -= molecules.get_polymer(p).get_volume_fraction()/ \
                            molecules.get_polymer(p).get_alpha() * \
                            np.log(total_partitions[p])

        if include_const_term:
            return hamiltonian_partition + hamiltonian_fields + self.h_const
        else:
            return hamiltonian_partition + hamiltonian_fields
    
    # Compute functional derivatives of Hamiltonian w.r.t. fields of selected indices
    def compute_func_deriv(self, w_aux, phi, indices):
        S = len(self.monomer_types)

        h_deriv = np.zeros([len(indices), w_aux.shape[1]], dtype=type(w_aux[0,0]))
        for count, i in enumerate(indices):
            # Return dH/dw
            h_deriv[count] += 2*self.h_coef_mu2[i]*w_aux[i]
            h_deriv[count] +=   self.h_coef_mu1[i]
            for j in range(S):
                h_deriv[count] += self.matrix_a[j,i]*phi[self.monomer_types[j]]

            # Change the sign for the imaginary fields
            if i in self.aux_fields_imag_idx:
                h_deriv[count] = -h_deriv[count]

        return h_deriv

    # Compute dH/dχN
    def compute_h_deriv_chin(self, chi_n, w_aux):
        S = len(self.monomer_types)

        dH = {}
        for key in chi_n:
            dH[key] = self.h_const_deriv_chin[key]
            for i in range(S):
                dH[key] += self.h_coef_mu2_deriv_chin[key][i]*np.mean(w_aux[i]**2)
                dH[key] += self.h_coef_mu1_deriv_chin[key][i]*np.mean(w_aux[i])                   
        return dH
