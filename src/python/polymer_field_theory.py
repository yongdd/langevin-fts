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

    The transformation separates auxiliary fields into real-valued (thermally
    fluctuating in L-FTS) and imaginary-valued (compressed to saddle point)
    components based on eigenvalue signs.

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
    h_const_deriv_chin : dict
        Derivatives of h_const w.r.t. χN parameters.
    h_coef_mu1_deriv_chin : dict
        Derivatives of h_coef_mu1 w.r.t. χN parameters.
    h_coef_mu2_deriv_chin : dict
        Derivatives of h_coef_mu2 w.r.t. χN parameters.

    Notes
    -----
    **Field Theory Formulation:**

    The multi-monomer Hamiltonian is:

    .. math::
        H = \\sum_{i<j} \\frac{\\chi_{ij} N}{V} \\int \\phi_i(\\mathbf{r}) \\phi_j(\\mathbf{r}) d\\mathbf{r}

    Eigenvalue decomposition transforms this to:

    .. math::
        H = \\sum_k \\frac{\\lambda_k}{V} \\int w_k(\\mathbf{r})^2 d\\mathbf{r} + \\text{const}

    where λ_k are eigenvalues and w_k are auxiliary fields.

    **Field Classification:**

    - **Real auxiliary fields** (λ < 0): Thermally fluctuate in L-FTS
    - **Imaginary auxiliary fields** (λ > 0): Compressed to saddle point
    - **Pressure field** (incompressible): Enforces Σφ_i = 1

    **Transformation Matrices:**

    Monomer ↔ auxiliary transformations:

    .. math::
        \\mathbf{w} = \\mathbf{A} \\mathbf{w}_{\\text{aux}}

        \\mathbf{w}_{\\text{aux}} = \\mathbf{A}^{-1} \\mathbf{w}

    where A is the eigenvector matrix scaled by √M.

    **Warning for Multiple Imaginary Fields:**

    If I > 1 (multiple positive eigenvalues), some field fluctuations
    cannot be sampled in L-FTS. This occurs for certain χN parameter sets
    and may affect accuracy.

    See Also
    --------
    SCFT : Uses this class for multi-monomer SCFT calculations.
    LFTS : Uses this class for multi-monomer L-FTS calculations.

    References
    ----------
    .. [1] Delaney Vigil, D. L., et al. "Multimonomer Field Theory of
           Conformationally Asymmetric Polymer Blends." Macromolecules 2025,
           58, 816.
    .. [2] Lee, W.-B., et al. "Fluctuation Effects in Ternary AB+A+B Polymeric
           Emulsions." Macromolecules 2013, 46, 8037.

    Examples
    --------
    **AB Diblock (2 monomer types):**

    >>> monomer_types = ["A", "B"]
    >>> chi_n = {"A,B": 15.0}
    >>> mpt = SymmetricPolymerTheory(monomer_types, chi_n, zeta_n=None)
    >>> print(f"Eigenvalues: {mpt.eigenvalues}")
    >>> print(f"Real fields: {mpt.aux_fields_real_idx}")
    >>> print(f"Imaginary fields: {mpt.aux_fields_imag_idx}")

    **ABC Triblock (3 monomer types):**

    >>> monomer_types = ["A", "B", "C"]
    >>> chi_n = {"A,B": 20.0, "B,C": 20.0, "A,C": 20.0}
    >>> mpt = SymmetricPolymerTheory(monomer_types, chi_n, zeta_n=None)
    >>> # Transform monomer fields to auxiliary fields
    >>> w_aux = mpt.to_aux_fields(w_monomer)
    >>> # Transform back
    >>> w_monomer_restored = mpt.to_monomer_fields(w_aux)

    **Field transformation in practice:**

    >>> # In SCFT/LFTS, monomer potential fields w are converted to auxiliary
    >>> w_dict = {"A": w_A_array, "B": w_B_array}
    >>> w_aux = mpt.to_aux_fields(w_dict)  # Returns ndarray shape (M, grid)
    >>>
    >>> # Compute Hamiltonian from auxiliary fields
    >>> H = mpt.compute_hamiltonian(molecules, w_aux, total_partitions, cb)
    """
    def __init__(self, monomer_types, chi_n, zeta_n):
        self.monomer_types = monomer_types
        M = len(self.monomer_types)
        
        # Compute eigenvalues, orthogonal matrix, vector_s and vector_S
        eigenvalues, matrix_o, vector_s, vector_large_s = self.compute_eigen_system(monomer_types, chi_n, zeta_n)

        # Indices whose auxiliary fields are real
        self.aux_fields_real_idx = []
        # Indices whose auxiliary fields are imaginary including the pressure field
        self.aux_fields_imag_idx = []
        for i in range(M):
            # assert(not np.isclose(eigenvalues[i], 0.0)), \
            #     "One of eigenvalues is zero for given chiN values."
            if np.isclose(eigenvalues[i], 0.0):
                print("One of eigenvalues is zero for given chiN values.")
            elif eigenvalues[i] > 0:
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

        Notes
        -----
        The transformation uses matrix_a_inv = O^T / M where O is the
        orthogonal eigenvector matrix and M is the number of monomer types.

        See Also
        --------
        to_monomer_fields : Inverse transformation (auxiliary → monomer).
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

        See Also
        --------
        to_aux_fields : Forward transformation (monomer → auxiliary).
        to_monomer_fields_dict : Same transformation returning dict.
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

        See Also
        --------
        to_monomer_fields : Same transformation returning ndarray.
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

            # Construct vector_s and vector_S
            vector_s = np.matmul(matrix_chin, np.ones(M))/M
            vector_large_s = np.matmul(np.transpose(eigenvectors), vector_s)

        # Compressible model
        else:
            # Compute eigenvalues and eigenvectors
            u = zeta_n*np.ones((M,M)) + matrix_chin
            eigenvalues, eigenvectors = np.linalg.eigh(u)

            # Reordering eigenvalues and eigenvectors
            sorted_indexes = np.argsort(eigenvalues)[::]
            eigenvalues = eigenvalues[sorted_indexes]
            eigenvectors = eigenvectors[:,sorted_indexes]     

            # Construct vector_s and vector_S
            vector_s = np.zeros(M)
            vector_large_s = -zeta_n*np.matmul(np.transpose(eigenvectors), np.ones(M))*np.sqrt(M)
        
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

        Notes
        -----
        **Hamiltonian Components:**

        .. math::
            H = H_{\\text{partition}} + H_{\\text{fields}} + H_{\\text{const}}

        where:

        - H_partition = -Σ_p (φ_p / α_p) ln(Q_p)
        - H_fields = Σ_i h_coef_mu2[i] ⟨w_aux[i]²⟩ + h_coef_mu1[i] ⟨w_aux[i]⟩
        - H_const = reference energy (only if include_const_term=True)

        **Usage:**

        - SCFT: Reports H without const term (relative free energy)
        - L-FTS: Uses H with const term for Boltzmann weighting

        See Also
        --------
        compute_func_deriv : Compute δH/δw for field updates.
        compute_h_deriv_chin : Compute dH/dχN for parameter sweeps.
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

        Notes
        -----
        **Functional Derivative Formula:**

        For auxiliary field i:

        .. math::
            \\frac{\\delta H}{\\delta w_i} = 2 h_{\\mu2,i} w_i + h_{\\mu1,i} + \\sum_j A_{ji} \\phi_j

        where A is the transformation matrix and φ_j are monomer concentrations.

        **Sign Convention:**

        - Real fields (λ < 0): Returns +δH/δw
        - Imaginary fields (λ > 0): Returns -δH/δw

        This convention ensures that field updates move toward the saddle point.

        **Usage:**

        - SCFT: Computes self-consistency error from δH/δw
        - L-FTS compression: Finds saddle point where δH/δw_imag = 0

        See Also
        --------
        compute_hamiltonian : Compute H value.
        LFTS.find_saddle_point : Uses this for field compression.
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

        Notes
        -----
        **Derivative Calculation:**

        Uses finite difference of Hamiltonian coefficients computed during
        initialization:

        .. math::
            \\frac{dH}{d\\chi_{ij}N} = \\frac{dH_{\\text{const}}}{d\\chi_{ij}N} +
            \\sum_k \\left[ \\frac{dh_{\\mu2,k}}{d\\chi_{ij}N} \\langle w_k^2 \\rangle +
            \\frac{dh_{\\mu1,k}}{d\\chi_{ij}N} \\langle w_k \\rangle \\right]

        **Applications:**

        - Parameter sensitivity analysis
        - Estimating phase boundaries (dH/dχN changes sign at transitions)
        - Optimizing χN values to target specific morphologies

        **Accuracy:**

        Derivatives are computed by finite difference with ε = 10^-5 during
        initialization. For higher accuracy, reduce epsilon in __init__.

        See Also
        --------
        compute_hamiltonian : Hamiltonian calculation.
        LFTS.run : Records dH/dχN history during simulation.
        """
        M = len(self.monomer_types)

        dH = {}
        for key in chi_n:
            dH[key] = self.h_const_deriv_chin[key]
            for i in range(M):
                dH[key] += self.h_coef_mu2_deriv_chin[key][i]*np.mean(w_aux[i]**2)
                dH[key] += self.h_coef_mu1_deriv_chin[key][i]*np.mean(w_aux[i])                   
        return dH
