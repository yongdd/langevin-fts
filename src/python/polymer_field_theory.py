"""Multi-monomer polymer field theory transformations.

This module implements the symmetric polymer field theory for systems with
multiple monomer types, performing eigenvalue decomposition of interaction
matrices to obtain auxiliary field representations.
"""

import numpy as np
from .validation import ValidationError

class SymmetricPolymerTheory:
    """Symmetric polymer field theory for multi-monomer systems.

    Handles field transformations between monomer potential fields and auxiliary
    fields obtained from eigenvalue decomposition of the interaction matrix.
    This enables efficient simulation of multi-monomer polymer systems in both
    SCFT and L-FTS calculations.

    The transformation separates auxiliary fields into real-valued (fluctuating
    in L-FTS) and imaginary-valued (compressed to saddle point) components
    based on eigenvalue signs.

    Parameters
    ----------
    monomer_types : list of str
        Sorted list of monomer type labels (e.g., ["A", "B", "C"]).
    chi_n : dict
        Flory-Huggins interaction parameters × N_Ref,
        {monomer_pair: value}. Example: {"A,B": 15.0, "B,C": 20.0}.
        Self-interactions should not be included.
    zeta_n : float or None
        Compressibility parameter × N_Ref. If None, incompressible model
        is used (constraint Σφ_i = 1). If float, compressible model with
        finite bulk modulus.

    Attributes
    ----------
    monomer_types : list of str
        Monomer type labels.
    eigenvalues : ndarray
        Eigenvalues of the interaction matrix, shape (M,).
    matrix_o : ndarray
        Orthogonal eigenvector matrix, shape (M, M).
    matrix_a : ndarray
        Transformation matrix from auxiliary to monomer fields, shape (M, M).
    matrix_a_inv : ndarray
        Inverse transformation matrix (monomer to auxiliary), shape (M, M).
    aux_fields_real_idx : list of int
        Indices of real-valued auxiliary fields (negative eigenvalues).
    aux_fields_imag_idx : list of int
        Indices of imaginary-valued auxiliary fields (positive eigenvalues
        plus pressure field for incompressible case).
    R : int
        Number of real auxiliary fields.
    I : int
        Number of imaginary auxiliary fields.
    h_const : float
        Constant term in Hamiltonian.
    h_coef_mu1 : ndarray
        Coefficients for ∫μ(r) term in Hamiltonian, shape (M,).
    h_coef_mu2 : ndarray
        Coefficients for ∫μ(r)² term in Hamiltonian, shape (M,).
    """
    def __init__(self, monomer_types, chi_n, zeta_n):
        self.monomer_types = monomer_types
        self.zeta_n = zeta_n
        M = len(self.monomer_types)
        
        # Compute eigenvalues, orthogonal matrix, vector_s and vector_S
        eigenvalues, matrix_o, vector_s, vector_large_s = self.compute_eigen_system(monomer_types, chi_n, zeta_n)

        # Indices whose auxiliary fields are real
        self.aux_fields_real_idx = []
        # Indices whose auxiliary fields are imaginary including the pressure field
        self.aux_fields_imag_idx = []
        # The incompressible model (zeta_n None) has exactly one zero eigenvalue, the
        # pressure mode appended below; any other zero means a degenerate chi N matrix
        n_zero = int(np.sum(np.isclose(eigenvalues, 0.0)))
        if n_zero > (1 if zeta_n is None else 0):
            print("(Warning!) The chi N matrix has a zero eigenvalue; the corresponding auxiliary field is dropped.")
        for i in range(M):
            if np.isclose(eigenvalues[i], 0.0):
                continue
            if eigenvalues[i] > 0:
                self.aux_fields_imag_idx.append(i)
            else:
                self.aux_fields_real_idx.append(i)
        
        if zeta_n is None:
            self.aux_fields_imag_idx.append(M-1) # add pressure field

        # The numbers of real and imaginary fields, respectively
        self.R = len(self.aux_fields_real_idx)
        self.I = len(self.aux_fields_imag_idx)

        if self.I > 1:
            print("(Warning!) For a given χN interaction parameter set, at least one of the auxiliary fields is an imaginary field. ", end="")
            print("The field fluctuations would not be fully reflected. Run this simulation at your own risk.")

        # Compute coefficients for Hamiltonian computation
        h_const, h_coef_mu1, h_coef_mu2 = self.compute_h_coef(eigenvalues, vector_s, vector_large_s, zeta_n)

        # Matrix A and Inverse for converting between auxiliary potential fields and monomer potential fields
        matrix_a = matrix_o.copy()
        matrix_a_inv = np.transpose(matrix_o).copy()/M

        # Check the inverse matrix
        error = np.std(np.matmul(matrix_a, matrix_a_inv) - np.identity(M))
        if not np.isclose(error, 0.0):
            raise ValidationError("Invalid inverse of matrix A. Perhaps matrix O is not orthogonal.")
        
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

        print("Real Fields: ",      self.aux_fields_real_idx)
        print("Imaginary Fields: ", self.aux_fields_imag_idx)
        
        print("In Hamiltonian:")
        print("\treference energy: ", self.h_const)
        print("\tcoefficients of int of mu(r)/V: ", self.h_coef_mu1)
        print("\tcoefficients of int of mu(r)^2/V: ", self.h_coef_mu2)

    def to_aux_fields(self, w):
        """Transform monomer potential fields to auxiliary fields.

        Performs the transformation w_aux = A^(-1) @ w, where A is the
        eigenvector matrix. This converts from the monomer field representation
        to the auxiliary field representation.

        Parameters
        ----------
        w : dict or ndarray
            Monomer potential fields. Either:

            - dict: {monomer_type: field_array}, each array length total_grid
            - ndarray: shape (M, total_grid) in sorted monomer_types order

        Returns
        -------
        w_aux : ndarray
            Auxiliary potential fields, shape (M, total_grid).
        """
        if isinstance(w, dict):
            # Make an array of w
            M = len(self.monomer_types)
            total_grid = len(w[next(iter(w))])
            w_temp = np.zeros((M, total_grid))
            for count, type in enumerate(self.monomer_types):
                w_temp[count,:] = w[type]

            return np.matmul(self.matrix_a_inv, w_temp)
        else:
            return np.matmul(self.matrix_a_inv, w)

    def to_monomer_fields(self, w_aux):
        """Transform auxiliary fields to monomer potential fields.

        Performs the transformation w = A @ w_aux, where A is the eigenvector
        matrix. This is the inverse of to_aux_fields.

        Parameters
        ----------
        w_aux : ndarray
            Auxiliary potential fields, shape (M, total_grid).

        Returns
        -------
        w : ndarray
            Monomer potential fields, shape (M, total_grid) in sorted
            monomer_types order.
        """
        return np.matmul(self.matrix_a, w_aux)

    def to_monomer_fields_dict(self, w_aux):
        """Transform auxiliary fields to monomer fields as dictionary.

        Same as to_monomer_fields but returns a dictionary instead of ndarray.

        Parameters
        ----------
        w_aux : ndarray
            Auxiliary potential fields, shape (M, total_grid).

        Returns
        -------
        w : dict
            Monomer potential fields, {monomer_type: field_array}.
        """
        w_temp = np.matmul(self.matrix_a, w_aux)
        w = {}
        for count, type in enumerate(self.monomer_types):
            w[type] = w_temp[count]
        return w

    def compute_eigen_system(self, monomer_types, chi_n, zeta_n):
        M = len(monomer_types)

        # Construct chi_n matrix
        matrix_chin = np.zeros((M,M))
        for i in range(M):
            for j in range(i+1,M):
                monomer_pair = [self.monomer_types[i], self.monomer_types[j]]
                monomer_pair.sort()
                key = monomer_pair[0] + "," + monomer_pair[1]
                if key in chi_n:
                    matrix_chin[i,j] = chi_n[key]
                    matrix_chin[j,i] = chi_n[key]

        # Incompressible model
        if zeta_n is None:
            matrix_q = np.ones((M,M))/M
            matrix_p = np.identity(M) - matrix_q

            # Compute eigenvalues and eigenvectors
            projected_chin = np.matmul(matrix_p, np.matmul(matrix_chin, matrix_p))
            eigenvalues, eigenvectors = np.linalg.eigh(projected_chin)

            # Reordering eigenvalues and eigenvectors
            sorted_indexes = np.argsort(np.abs(eigenvalues))[::-1]
            eigenvalues = eigenvalues[sorted_indexes]
            eigenvectors = eigenvectors[:,sorted_indexes]

            # Reordering eigenvalues and eigenvectors to move the unit vector to M-1
            eigen_val_0 = np.isclose(eigenvalues, 0.0, atol=1e-12)
            eigen_vec_0 = eigenvectors[:,eigen_val_0]
            sorted_indexes = np.argsort(np.std(eigen_vec_0, axis=0))[::-1]
            eigenvectors[:,eigen_val_0] = eigen_vec_0[:,sorted_indexes]

            # Set the last eigenvector to [1, 1, ..., 1]/√S
            eigenvectors[:,-1] = np.ones(M)/np.sqrt(M)

            # Construct vector_s (vector_S follows from the final eigenvectors below)
            vector_s = np.matmul(matrix_chin, np.ones(M))/M

        # Compressible model
        else:
            # Compute eigenvalues and eigenvectors
            u = zeta_n*np.ones((M,M)) + matrix_chin
            eigenvalues, eigenvectors = np.linalg.eigh(u)

            # Reordering eigenvalues and eigenvectors
            sorted_indexes = np.argsort(eigenvalues)[::]
            eigenvalues = eigenvalues[sorted_indexes]
            eigenvectors = eigenvectors[:,sorted_indexes]     

            # Construct vector_s (vector_S follows from the final eigenvectors below)
            vector_s = np.zeros(M)
        
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
        for i in range(M):
            if eigenvectors[0,i] < 0.0:
                eigenvectors[:,i] *= -1.0

        # Multiply √S to eigenvectors
        eigenvectors *= np.sqrt(M)

        # vector_S = Aᵀs (incompressible) or -ζ Aᵀ(1, ..., 1) (compressible) with the final
        # matrix A: it sets the linear coefficients S/λ, so it must carry the signs and the
        # √M scaling of the eigenvectors above
        if zeta_n is None:
            vector_large_s = np.matmul(np.transpose(eigenvectors), vector_s)
        else:
            vector_large_s = -zeta_n*np.matmul(np.transpose(eigenvectors), np.ones(M))

        return eigenvalues, eigenvectors, vector_s, vector_large_s

    def compute_h_coef(self, eigenvalues, vector_s, vector_large_s, zeta_n):
        M = len(self.monomer_types)
        
        # Compute reference part of Hamiltonian
        h_const = 0.0
        for i in range(M):
            if not np.isclose(eigenvalues[i], 0.0):
                h_const -= 0.5*vector_large_s[i]**2/eigenvalues[i]/M

        # Compute coefficients of integral of μ(r)/V
        h_coef_mu1 = np.zeros(M)
        for i in range(M):
            if not np.isclose(eigenvalues[i], 0.0):
                h_coef_mu1[i] = vector_large_s[i]/eigenvalues[i]

        # Compute coefficients of integral of μ(r)^2/V
        h_coef_mu2 = np.zeros(M)
        for i in range(M):
            if not np.isclose(eigenvalues[i], 0.0):
                h_coef_mu2[i] = -0.5/eigenvalues[i]*M

        if zeta_n is None:
            h_const += 0.5*np.sum(vector_s)/M
            h_coef_mu1[M-1] = -1.0
            h_coef_mu2[M-1] =  0.0
        else:
            h_const += 0.5*zeta_n

        return h_const, h_coef_mu1, h_coef_mu2

    def compute_hamiltonian(self, molecules, w_aux, total_partitions, cb, include_const_term=False):
        """Compute the field-theoretic Hamiltonian.

        Calculates the Hamiltonian from auxiliary fields and partition functions.
        Used in both SCFT (for free energy) and L-FTS (for sampling weights).

        Parameters
        ----------
        molecules : Molecules
            C++ Molecules object containing polymer specifications.
        w_aux : ndarray
            Auxiliary potential fields, shape (M, n_grid) where n_grid is
            total_grid (without space group) or n_reduced (with space group).
        total_partitions : list of float
            Single-chain partition functions Q_p for each polymer type p.
        cb : ComputationBox
            C++ ComputationBox object for computing volume integrals.
            Correctly handles space group symmetry via orbit_counts.
        include_const_term : bool, optional
            If True, include reference energy h_const in result (default: False).
            Set to True for L-FTS absolute energy, False for SCFT relative energy.

        Returns
        -------
        hamiltonian : float
            Field-theoretic Hamiltonian value (dimensionless).
        """
        M = len(self.monomer_types)

        # Compute Hamiltonian part that is related to fields
        # Use cb.mean() for correct handling of space group symmetry
        hamiltonian_fields = 0.0
        for i in range(M):
            if not np.isclose(self.h_coef_mu2[i], 0.0):
                hamiltonian_fields += self.h_coef_mu2[i]*cb.mean(w_aux[i]**2)
            if not np.isclose(self.h_coef_mu1[i], 0.0):
                hamiltonian_fields += self.h_coef_mu1[i]*cb.mean(w_aux[i])
        
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
    
    def compute_func_deriv(self, w_aux, phi, indices):
        """Compute functional derivatives of Hamiltonian w.r.t. auxiliary fields.

        Calculates δH/δw_aux for specified auxiliary field indices. Used for
        field optimization in SCFT and field compression in L-FTS.

        Parameters
        ----------
        w_aux : ndarray
            Auxiliary potential fields, shape (M, total_grid).
        phi : dict
            Monomer concentration fields, {monomer_type: concentration_array}.
        indices : list of int
            Indices of auxiliary fields to compute derivatives for.
            Typically aux_fields_imag_idx for compression in L-FTS.

        Returns
        -------
        h_deriv : ndarray
            Functional derivatives δH/δw_aux, shape (len(indices), total_grid).
            For imaginary fields, the sign is flipped (returns -δH/δw).
        """
        M = len(self.monomer_types)

        h_deriv = np.zeros([len(indices), w_aux.shape[1]], dtype=type(w_aux[0,0]))
        for count, i in enumerate(indices):
            # Return dH/dw
            h_deriv[count] += 2*self.h_coef_mu2[i]*w_aux[i]
            h_deriv[count] +=   self.h_coef_mu1[i]
            for j in range(M):
                h_deriv[count] += self.matrix_a[j,i]*phi[self.monomer_types[j]]

            # Change the sign for the imaginary fields
            if i in self.aux_fields_imag_idx:
                h_deriv[count] = -h_deriv[count]

        return h_deriv

    def compute_h_deriv_chin(self, chi_n, w_aux):
        """Compute derivatives of Hamiltonian w.r.t. χN parameters.

        Calculates dH/dχ_{ij}N for all interaction parameters. Used in L-FTS
        to estimate how free energy changes with interaction strength.

        The derivative is taken at fixed monomer fields w = A w_aux, where the
        polymer part -ln Q[w] does not depend on χ; the auxiliary fields
        Ω = A(χ)⁻¹ w and the coefficients of the field part change with χ. The
        field part is a function of w alone, so the result does not depend on how
        the eigenvectors are chosen (signs, degenerate eigenvalues). For three or
        more monomer types A rotates with χ, so this differs sample by sample from
        the derivative at fixed Ω; the averages are equal (the rotation generator
        is antisymmetric), and at a saddle point the two values are equal.

        Parameters
        ----------
        chi_n : dict
            Flory-Huggins interaction parameters, {monomer_pair: value}.
        w_aux : ndarray
            Current auxiliary potential fields, shape (M, total_grid).

        Returns
        -------
        dH : dict
            Derivatives {monomer_pair: dH/dχN_value}.
        """
        M = len(self.monomer_types)

        # Volume averages <w_i> and <w_i w_j> (no conjugate: H is holomorphic in complex fields)
        w = np.matmul(self.matrix_a, np.asarray(w_aux))
        w_mean = np.mean(w, axis=1)
        w_product_mean = np.matmul(w, np.transpose(w))/w.shape[1]

        def field_part(chi):
            eigenvalues, matrix_o, vector_s, vector_large_s = self.compute_eigen_system(self.monomer_types, chi, self.zeta_n)
            h_const, h_coef_mu1, h_coef_mu2 = self.compute_h_coef(eigenvalues, vector_s, vector_large_s, self.zeta_n)
            matrix_a_inv = np.transpose(matrix_o)/M
            omega_mean = np.matmul(matrix_a_inv, w_mean)
            omega_sq_mean = np.einsum("ki,ij,kj->k", matrix_a_inv, w_product_mean, matrix_a_inv)
            return h_const + np.sum(h_coef_mu2*omega_sq_mean + h_coef_mu1*omega_mean)

        # Central differences with Richardson extrapolation (the field part is smooth in χ;
        # a larger step keeps the round-off of the large compressible terms small)
        def central_difference(key, epsilon):
            chi_n_p = chi_n.copy()
            chi_n_n = chi_n.copy()
            chi_n_p[key] += epsilon
            chi_n_n[key] -= epsilon
            return (field_part(chi_n_p) - field_part(chi_n_n))/(2*epsilon)

        epsilon = 1e-3
        dH = {}
        for key in chi_n:
            dH[key] = (4*central_difference(key, epsilon/2) - central_difference(key, epsilon))/3
        return dH
