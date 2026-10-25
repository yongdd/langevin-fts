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
    matrix_chi : ndarray
        Symmetric χN matrix in the order of monomer_types, zero diagonal, shape (M, M).
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
        # Pair keys in sorted order ("B,A" -> "A,B"), as the χN matrix looks them up
        chi_n = {",".join(sorted(key.split(","))): value for key, value in chi_n.items()}

        # χN matrix; pairs missing from chi_n do not interact
        matrix_chi = np.zeros((M,M))
        for i in range(M):
            for j in range(i+1,M):
                monomer_pair = [self.monomer_types[i], self.monomer_types[j]]
                monomer_pair.sort()
                key = monomer_pair[0] + "," + monomer_pair[1]
                if key in chi_n:
                    matrix_chi[i,j] = chi_n[key]
                    matrix_chi[j,i] = chi_n[key]
        self.matrix_chi = matrix_chi

        # Compute eigenvalues, orthogonal matrix, vector_s and vector_S
        eigenvalues, matrix_o, vector_s, vector_large_s = self.compute_eigen_system(matrix_chi, zeta_n)

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

        # Role of each auxiliary field for the χN derivatives (see _field_coefficients_derivative)
        self._roles = []
        for i in range(M):
            if zeta_n is None and i == M-1:
                self._roles.append("pressure")
            elif np.isclose(eigenvalues[i], 0.0):
                self._roles.append("dropped")
            else:
                self._roles.append("imaginary" if eigenvalues[i] > 0 else "real")
        self.aux_fields_dropped_idx = [i for i, role in enumerate(self._roles) if role == "dropped"]
        self._vector_s = vector_s
        self._vector_large_s = vector_large_s
        if self.R > 1:
            lead = self.aux_fields_real_idx[0]
            gap = min(abs(eigenvalues[j] - eigenvalues[lead]) for j in self.aux_fields_real_idx[1:])
            if gap < 1e-2*abs(eigenvalues[lead]):
                print("(Warning!) The leading real field has (nearly) the same eigenvalue as another real field. ", end="")
                print("Its eigenvector is then not well defined, and an order parameter built from it (WTMD) changes abruptly with χN.")

        # χN derivatives of the field part at fixed auxiliary fields (see compute_h_deriv_chin);
        # with WTMD the leading real field follows its own eigenvector
        self.chi_n = {key: float(value) for key, value in chi_n.items()}
        self.h_deriv_chin_coefficients = {key: self._field_coefficients_derivative(key, False) for key in chi_n}
        if self.R > 1:
            self._h_deriv_chin_coefficients_wtmd = {key: self._field_coefficients_derivative(key, True) for key in chi_n}
        else:
            self._h_deriv_chin_coefficients_wtmd = self.h_deriv_chin_coefficients

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

    def compute_eigen_system(self, matrix_chi, zeta_n):
        M = matrix_chi.shape[0]

        # Incompressible model
        if zeta_n is None:
            matrix_q = np.ones((M,M))/M
            matrix_p = np.identity(M) - matrix_q

            # Compute eigenvalues and eigenvectors
            projected_chin = np.matmul(matrix_p, np.matmul(matrix_chi, matrix_p))
            eigenvalues, eigenvectors = np.linalg.eigh(projected_chin)

            # Reordering eigenvalues and eigenvectors
            sorted_indexes = np.argsort(np.abs(eigenvalues))[::-1]
            eigenvalues = eigenvalues[sorted_indexes]
            eigenvectors = eigenvectors[:,sorted_indexes]

            # Reordering eigenvalues and eigenvectors to move the unit vector to M-1 (zero
            # eigenvalues: |λ| <= 1e-8, the tolerance used for the roles of the fields)
            eigen_val_0 = np.isclose(eigenvalues, 0.0)
            eigen_vec_0 = eigenvectors[:,eigen_val_0]
            sorted_indexes = np.argsort(np.std(eigen_vec_0, axis=0))[::-1]
            eigenvectors[:,eigen_val_0] = eigen_vec_0[:,sorted_indexes]

            # Set the last eigenvector to [1, 1, ..., 1]/√S
            eigenvectors[:,-1] = np.ones(M)/np.sqrt(M)

            # The other eigenvectors lie in the plane orthogonal to it. eigh mixes a little of the
            # uniform vector (about ε|χN|/|λ|) into an eigenvector whose eigenvalue is close to the
            # pressure's 0; then remove it and make the set orthonormal again, in the order of |λ|
            # (when none holds such a part, the eigenvectors are left as they are)
            if np.max(np.abs(np.matmul(eigenvectors[:,-1], eigenvectors[:,:-1])), initial=0.0) > 1e-12:
                for i in range(M-1):
                    vector = eigenvectors[:,i] - eigenvectors[:,-1]*np.dot(eigenvectors[:,-1], eigenvectors[:,i])
                    for j in range(i):
                        vector -= eigenvectors[:,j]*np.dot(eigenvectors[:,j], vector)
                    eigenvectors[:,i] = vector/np.linalg.norm(vector)

            # Construct vector_s (vector_S follows from the final eigenvectors below)
            vector_s = np.matmul(matrix_chi, np.ones(M))/M

        # Compressible model
        else:
            # Compute eigenvalues and eigenvectors
            u = zeta_n*np.ones((M,M)) + matrix_chi
            eigenvalues, eigenvectors = np.linalg.eigh(u)

            # Reordering eigenvalues and eigenvectors
            sorted_indexes = np.argsort(eigenvalues)[::]
            eigenvalues = eigenvalues[sorted_indexes]
            eigenvectors = eigenvectors[:,sorted_indexes]     

            # Construct vector_s (vector_S follows from the final eigenvectors below)
            vector_s = np.zeros(M)
        
        # Make a orthogonal matrix using Gram-Schmidt
        eigen_val_0 = np.isclose(eigenvalues, 0.0)
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

    def pin_dropped_fields(self, w_aux):
        """Fix absent auxiliary modes to their uniform physical potential S/M.

        A zero eigenvalue removes a fluctuating field, but its linear interaction
        can still supply a uniform monomer potential. Initial fields must not
        leave an arbitrary spatial potential frozen in such a mode.
        """
        for i in self.aux_fields_dropped_idx:
            w_aux[i] = self._vector_large_s[i] / len(self.monomer_types)

    def compute_hamiltonian(self, molecules, w_aux, total_partitions, cb, include_const_term=False, phi=None):
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
        phi : dict, optional
            Monomer concentrations. Needed for nonuniform dropped modes or
            random copolymers with a dropped mode. For uniform dropped modes
            the canonical mean concentrations can be obtained from the chains.

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
            reference = self.h_const
            dropped = [i for i, role in enumerate(self._roles) if role == "dropped"]
            if dropped:
                if phi is None:
                    phi_mean = np.zeros(M)
                    for p in range(molecules.get_n_polymer_types()):
                        polymer = molecules.get_polymer(p)
                        for block in polymer.get_blocks():
                            if block.monomer_type not in self.monomer_types:
                                raise ValueError("Pass phi to compute_hamiltonian for random copolymers with dropped fields.")
                            phi_mean[self.monomer_types.index(block.monomer_type)] += (
                                polymer.get_volume_fraction() * block.contour_length / polymer.get_alpha())
                    for s in range(molecules.get_n_solvent_types()):
                        fraction, name = molecules.get_solvent(s)
                        phi_mean[self.monomer_types.index(name)] += fraction
                for i in dropped:
                    # Restores the linear interaction lost on dropping lambda=0,
                    # and removes any uniform potential already included through Q.
                    offset = self._vector_large_s[i]/M - w_aux[i]
                    if phi is None:
                        # Allow the small residual left in a dropped mode by a
                        # finite-tolerance SCFT solve (default tolerance 1e-8).
                        if not np.allclose(w_aux[i], cb.mean(w_aux[i]), rtol=0, atol=1e-8):
                            raise ValueError("Pass phi to compute_hamiltonian for nonuniform dropped fields.")
                        reference += cb.mean(offset) * np.dot(self.matrix_a[:, i], phi_mean)
                    else:
                        projected_phi = sum(self.matrix_a[j, i]*phi[name]
                                            for j, name in enumerate(self.monomer_types))
                        reference += cb.mean(offset * projected_phi)
            return hamiltonian_partition + hamiltonian_fields + reference
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

    def _field_coefficients_derivative(self, key, follow_lead):
        """d/dχN[key] of the field part at fixed auxiliary fields Ω (see compute_h_deriv_chin).

        First-order perturbation of U a_k = λ_k a_k (U = PχP, or χ + ζ11ᵀ for the
        compressible model; |a_k|² = M). The fields follow a frame Ã(χ) with dÃ = A K, K
        antisymmetric. Between fields of different roles (real, imaginary, dropped) K is the
        motion of the eigenvectors, K_jk = a_jᵀ dU a_k / (M (λ_k - λ_j)); it moves the
        subspaces on which the real and the imaginary fields live, and no denominator
        vanishes. Within a role the frame is a free choice (it changes single values of
        dH/dχN, not their average) and K = 0. With follow_lead (WTMD) the leading real field
        follows its own eigenvector, K_jk = a_jᵀ dU a_k / (M (λ_k - λ_j)) also with the other
        real fields, so that an order parameter built from it stays fixed (K = 0 where its
        eigenvalue is degenerate and the eigenvector undefined).
        In this frame Λ = ÃᵀUÃ/M stays block diagonal, the field part is
        -(M/2) Ωᵀ Λ⁻¹ Ω + (Λ⁻¹ S)·Ω - Sᵀ Λ⁻¹ S/(2M) + const over the fields with λ ≠ 0, with
        S = Ãᵀ s, and dΛ = [Λ, K] + Aᵀ dU A/M, dS = -K S + Aᵀ ds.

        Fields dropped at a zero eigenvalue (other than the pressure field) have no field
        (their rows of w_aux are not used); their mean-field part (1/2) yᵀ Λ_d y/M + S_d·y/M,
        y = A_dᵀ φ, has the derivative (1/2) yᵀ dΛ_d y/M + dS_d·y/M + S_d·dy/M with
        dy_d = Σ_j K_jd y_j, which gives coefficients of <φ_i φ_j> and <φ_j>.

        Returns (d_h_const, d_Q2, d_q1, d_A, d_phi2, d_phi1): dH/dχN = d_h_const
        + Σ d_Q2 ∘ <ΩΩᵀ> + d_q1·<Ω> + Σ d_A ∘ <φΩᵀ> + Σ d_phi2 ∘ <φφᵀ> + d_phi1·<φ>.
        """
        M = len(self.monomer_types)
        A = self.matrix_a
        lam = self.eigenvalues
        roles = self._roles

        # dχ/dχN[key], dU and ds
        i, j = [self.monomer_types.index(t) for t in key.split(",")]
        d_chi = np.zeros((M, M))
        d_chi[i, j] = d_chi[j, i] = 1.0
        if self.zeta_n is None:
            matrix_p = np.identity(M) - np.ones((M, M))/M
            d_u = np.matmul(matrix_p, np.matmul(d_chi, matrix_p))
            d_s = np.matmul(d_chi, np.ones(M))/M
            d_const = 0.5*np.sum(d_chi)/M**2
        else:
            d_u = d_chi
            d_s = np.zeros(M)
            d_const = 0.0
        coupling = np.matmul(np.transpose(A), np.matmul(d_u, A))/M      # a_jᵀ dU a_k / M

        # Frame generator K
        K = np.zeros((M, M))
        lead = self.aux_fields_real_idx[0] if (follow_lead and self.R > 0) else None
        for a in range(M):
            for b in range(M):
                if a == b or "pressure" in (roles[a], roles[b]):
                    continue
                degenerate = abs(lam[a] - lam[b]) <= 1e-8*max(abs(lam[a]), abs(lam[b]), 1.0)
                if roles[a] != roles[b] or (lead in (a, b) and not degenerate):
                    K[a, b] = coupling[a, b]/(lam[b] - lam[a])

        d_lam = np.matmul(np.diag(lam), K) - np.matmul(K, np.diag(lam)) + coupling
        S = self._vector_large_s
        d_S = -np.matmul(K, S) + np.matmul(np.transpose(A), d_s)

        # Fields with λ ≠ 0 (kept), their coefficients and h_const
        kept = [k for k in range(M) if roles[k] in ("real", "imaginary")]
        inv = np.zeros(M)
        inv[kept] = 1.0/lam[kept]
        d_Q2 = np.zeros((M, M))
        d_Q2[np.ix_(kept, kept)] = 0.5*M*inv[kept][:, None]*d_lam[np.ix_(kept, kept)]*inv[kept][None, :]
        lam_inv_S = inv*S
        d_q1 = np.zeros(M)
        d_q1[kept] = inv[kept]*(d_S[kept] - np.matmul(d_lam[np.ix_(kept, kept)], lam_inv_S[kept]))
        d_h_const = -np.dot(d_S[kept], lam_inv_S[kept])/M \
            + 0.5*np.dot(lam_inv_S[kept], np.matmul(d_lam[np.ix_(kept, kept)], lam_inv_S[kept]))/M + d_const

        # Dropped fields: (1/2) yᵀ dΛ_d y/M + dS_d·y/M + S_d·dy/M with y = A_dᵀ φ and
        # dy = (A K)[:, d]ᵀ φ; the rotation of their own columns enters only through S_d·dy
        dropped = [k for k in range(M) if roles[k] == "dropped"]
        A_d = A[:, dropped]
        d_A = np.matmul(A, K)
        d_phi2 = np.matmul(A_d, np.matmul(d_lam[np.ix_(dropped, dropped)], np.transpose(A_d)))/(2*M)
        d_phi1 = (np.matmul(A_d, d_S[dropped]) + np.matmul(d_A[:, dropped], S[dropped]))/M
        d_A[:, dropped] = 0.0

        return d_h_const, d_Q2, d_q1, d_A, d_phi2, d_phi1

    def compute_h_deriv_chin(self, chi_n=None, w_aux=None, phi=None, wtmd=False):
        """Compute derivatives of Hamiltonian w.r.t. χN parameters.

        Calculates dH/dχ_{ij}N for all interaction parameters. Used in L-FTS
        to estimate how free energy changes with interaction strength.

        The derivative is taken at fixed auxiliary fields Ω, the integration variables:
        the coefficients of the field part change with χ, and for three or more monomer
        types so does the mapping w = A(χ) Ω, which changes the polymer part -ln Q[w] by
        <Σ_j φ_j (∂A/∂χ Ω)_j>. Averaged over the real fields with the imaginary fields at
        their saddle point (L-FTS), this is the χ derivative of the free energy of that
        approximation; with every field fluctuating (CL-FTS) it is the derivative of the
        full free energy. At a saddle point it equals the derivative at fixed monomer fields.
        The derivatives are analytic (first-order perturbation of the eigenvectors, see
        _field_coefficients_derivative), so nearly equal eigenvalues need no special care.
        The frame of the auxiliary fields moves only between real, imaginary and dropped
        fields; with wtmd=True the leading real field (the WTMD order parameter) also follows
        its own eigenvector, which keeps the order parameter fixed but makes single values
        noisy (variance growing as 1/gap) when its eigenvalue nears another real one.
        The χ-dependent normalization of the Hubbard-Stratonovich transformation, a
        constant independent of the fields, is not included (as for AB systems).

        A field dropped at a zero eigenvalue contributes the mean-field term
        (1/2)(dλ/dχ)<y²>/M + (dS/dχ)<y>/M with y = a·φ, computed from the concentrations
        of the fields. In a fluctuating simulation the exact term holds the correlation of
        the microscopic densities, which also has an intrachain part of order 1/C that no
        field represents; that part is missing.

        Parameters
        ----------
        chi_n : dict, optional
            Flory-Huggins interaction parameters, {monomer_pair: value}. The derivatives always
            use the χN of this object; when given, chi_n must equal it. Without it, pass w_aux
            and phi by keyword.
        w_aux : ndarray
            Current auxiliary potential fields, shape (M, total_grid).
        phi : dict, optional
            Concentrations {monomer_type: array} of the fields (smeared, as in the forces,
            when smearing is used). Needed when A depends on χ (three or more monomer types)
            or a field is dropped (e.g. AB at χN = 0).
        wtmd : bool
            The leading real field follows its own eigenvector (for WTMD statistics).

        Returns
        -------
        dH : dict
            Derivatives {monomer_pair: dH/dχN_value}.
        """
        if chi_n is not None and not isinstance(chi_n, dict):
            raise TypeError("compute_h_deriv_chin: chi_n must be a dict {monomer_pair: value}; pass the "
                            "fields by keyword, compute_h_deriv_chin(w_aux=w_aux, phi=phi).")
        if w_aux is None:
            raise TypeError("compute_h_deriv_chin needs the auxiliary fields w_aux.")
        if chi_n is not None and {",".join(sorted(key.split(","))): float(value) for key, value in chi_n.items()} != self.chi_n:
            raise ValueError("compute_h_deriv_chin needs the chi N of this SymmetricPolymerTheory.")

        # Volume averages <Ω_k>, <Ω_k Ω_l>, <φ_j Ω_k>, <φ_j> and <φ_i φ_j> (no conjugate: H is
        # holomorphic in complex fields)
        w_aux = np.asarray(w_aux)
        n_grid = w_aux.shape[1]
        omega_mean = np.mean(w_aux, axis=1)
        omega_product_mean = np.matmul(w_aux, np.transpose(w_aux))/n_grid
        if phi is None:
            return self.h_deriv_chin_from_moments(omega_mean, omega_product_mean, wtmd=wtmd)
        phi_array = np.array([phi[monomer_type] for monomer_type in self.monomer_types])
        phi_omega_mean = np.matmul(phi_array, np.transpose(w_aux))/n_grid
        phi_mean = np.mean(phi_array, axis=1)
        phi_product_mean = np.matmul(phi_array, np.transpose(phi_array))/n_grid
        return self.h_deriv_chin_from_moments(omega_mean, omega_product_mean, phi_omega_mean, phi_mean, phi_product_mean, wtmd)

    def h_deriv_chin_from_moments(self, omega_mean, omega_product_mean, phi_omega_mean=None, phi_mean=None, phi_product_mean=None,
                                  wtmd=False):
        """dH/dχN from the averages <Ω_k>, <Ω_k Ω_l>, <φ_j Ω_k>, <φ_j> and <φ_i φ_j> (see
        compute_h_deriv_chin); the concentration moments may be omitted when their
        coefficients vanish."""
        dH = {}
        coefficients = self._h_deriv_chin_coefficients_wtmd if wtmd else self.h_deriv_chin_coefficients
        for key, (d_h_const, d_Q2, d_q1, d_A, d_phi2, d_phi1) in coefficients.items():
            value = d_h_const + np.sum(d_Q2*omega_product_mean) + np.dot(d_q1, omega_mean)
            for coefficient, moment in ((d_A, phi_omega_mean), (d_phi2, phi_product_mean), (d_phi1, phi_mean)):
                if moment is not None:
                    value += np.sum(coefficient*moment)
                elif np.max(np.abs(coefficient), initial=0.0) > 1e-10:
                    raise ValueError("For this chi N the auxiliary-field mapping depends on chi N or a field "
                                     "is dropped; compute_h_deriv_chin needs the concentrations phi.")
            dH[key] = value
        return dH
