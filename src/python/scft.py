import os
import time
import logging
import numpy as np
import sys
import json
import yaml
import copy
import itertools
from pathlib import Path
from typing import Dict, Any, Optional, Callable, Union
import scipy.io

from . import _core
from .polymer_field_theory import SymmetricPolymerTheory
from .space_group import SpaceGroup
from .propagator_solver import PropagatorSolver
from .validation import validate_scft_params, ValidationError
from .config import load_config
from .result import SCFTResult, IterationInfo
from .utils import (
    create_monomer_color_dict,
    draw_polymer_architecture,
    parse_chi_n,
    validate_polymers,
    process_polymer_blocks,
    process_random_copolymer,
    DEFAULT_MONOMER_COLORS,
    configure_logging,
)
from .smearing import Smearing

# Module-level logger
logger = logging.getLogger(__name__)

# OpenMP environment variables
os.environ["MKL_NUM_THREADS"] = "1"  # always 1
os.environ["OMP_STACKSIZE"] = "1G"

# For ADAM optimizer, see https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
class Adam:
    """ADAM optimizer for SCFT field optimization.

    Implements the ADAM (Adaptive Moment Estimation) optimization algorithm
    for finding saddle points in self-consistent field theory calculations.
    This is an alternative to Anderson Mixing for field optimization.

    The algorithm maintains first and second moment estimates of gradients
    and uses them to adapt the learning rate for each parameter. This can
    provide more stable convergence for some systems.

    Parameters
    ----------
    total_grid : int
        Total number of grid points (size of field arrays to optimize).
    lr : float, optional
        Initial learning rate, γ (default: 1e-2). Controls step size.
    b1 : float, optional
        Exponential decay rate for first moment estimates, β₁ (default: 0.9).
        Should be in [0, 1).
    b2 : float, optional
        Exponential decay rate for second moment estimates, β₂ (default: 0.999).
        Should be in [0, 1).
    eps : float, optional
        Small constant for numerical stability, ε (default: 1e-8).
        Prevents division by zero.
    gamma : float, optional
        Learning rate decay factor (default: 1.0).
        Learning rate at iteration T is lr * γ^(T-1).

    Attributes
    ----------
    total_grid : int
        Size of field arrays.
    lr : float
        Base learning rate.
    b1, b2 : float
        Moment decay rates.
    eps : float
        Numerical stability constant.
    gamma : float
        Learning rate decay factor.
    count : int
        Current iteration number.
    m : ndarray
        First moment estimates (moving average of gradients).
    v : ndarray
        Second moment estimates (moving average of squared gradients).

    See Also
    --------
    SCFT : Main SCFT class that uses this optimizer.

    Notes
    -----
    This implementation follows the ADAM algorithm from [1]_. The update rule is:

    .. math::
        m_t = β_1 m_{t-1} + (1-β_1) g_t

        v_t = β_2 v_{t-1} + (1-β_2) g_t^2

        \\hat{m}_t = m_t / (1 - β_1^t)

        \\hat{v}_t = v_t / (1 - β_2^t)

        w_t = w_{t-1} + γ_t \\hat{m}_t / (\\sqrt{\\hat{v}_t} + ε)

    where g_t is the gradient (field difference) at iteration t.

    References
    ----------
    .. [1] Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic
           optimization. ICLR 2015.

    .. [2] https://pytorch.org/docs/stable/generated/torch.optim.Adam.html

    Examples
    --------
    >>> # Create ADAM optimizer for 32x32x32 grid
    >>> optimizer = Adam(total_grid=32*32*32, lr=1e-2)
    >>>
    >>> # In SCFT iteration loop
    >>> w_new = optimizer.calculate_new_fields(w_current, w_diff,
    ...                                        old_error, error_level)
    """
    def __init__(self, total_grid,
                    lr = 1e-2,       # initial learning rate, γ
                    b1 = 0.9,        # β1
                    b2 = 0.999,      # β2
                    eps = 1e-8,      # epsilon, small number to prevent dividing by zero
                    gamma = 1.0,     # learning rate at Tth iteration is lr*γ^(T-1)
                    ):
        self.total_grid = total_grid
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.gamma = gamma
        self.count = 1
        
        self.m = np.zeros(total_grid, dtype=np.float64) # first moment
        self.v = np.zeros(total_grid, dtype=np.float64) # second moment
        
    def reset_count(self,):
        """Reset optimizer state for a new optimization run.

        Resets the iteration counter to 1 and clears the moment estimates.
        This should be called when starting a new SCFT run with different
        initial conditions.

        Notes
        -----
        This method reinitializes:
        - Iteration counter (count) to 1
        - First moment estimate (m) to zeros
        - Second moment estimate (v) to zeros
        """
        self.count = 1
        self.m[:] = 0.0
        self.v[:] = 0.0        
        
    def calculate_new_fields(self, w_current, w_diff, old_error_level, error_level):
        """Compute updated fields using ADAM optimization algorithm.

        Performs one step of ADAM optimization to update the field values
        based on the current fields and their gradients (field differences).

        Parameters
        ----------
        w_current : ndarray
            Current field values, shape (total_grid,) or (M, grid_size).
        w_diff : ndarray
            Field gradients (self-consistency error), same shape as w_current.
        old_error_level : float
            Error level from previous iteration (not used in ADAM, kept for
            compatibility with Anderson Mixing interface).
        error_level : float
            Current error level (not used in ADAM, kept for compatibility).

        Returns
        -------
        w_new : ndarray
            Updated field values, same shape as w_current.

        Notes
        -----
        The ADAM update includes:

        1. Decay learning rate: lr_t = lr * γ^(t-1)
        2. Update biased first moment: m_t = β₁ m_{t-1} + (1-β₁) g_t
        3. Update biased second moment: v_t = β₂ v_{t-1} + (1-β₂) g_t²
        4. Bias correction: m̂_t = m_t/(1-β₁^t), v̂_t = v_t/(1-β₂^t)
        5. Update fields: w_t = w_{t-1} + lr_t * m̂_t / (√v̂_t + ε)

        The bias correction in steps 4 compensates for the initialization
        of moment estimates at zero, particularly important in early iterations.
        """
        lr = self.lr*self.gamma**(self.count-1)

        self.m = self.b1*self.m + (1.0-self.b1)*w_diff
        self.v = self.b2*self.v + (1.0-self.b2)*w_diff**2
        m_hat = self.m/(1.0-self.b1**self.count)
        v_hat = self.v/(1.0-self.b2**self.count)

        w_new = w_current + lr*m_hat/(np.sqrt(v_hat) + self.eps)

        self.count += 1
        return w_new

class SCFT:
    """Self-Consistent Field Theory solver for polymer systems.

    This class implements SCFT calculations for arbitrary polymer architectures
    including linear, branched, and random copolymers with multiple monomer types.
    It supports both Anderson Mixing and ADAM optimization algorithms for finding
    saddle points of the field-theoretic Hamiltonian.

    The implementation uses a platform-independent interface to computational backends
    (CUDA GPU or CPU with Intel MKL), automatic propagator computation optimization
    for complex polymer architectures, and optional space group symmetry constraints.

    Parameters
    ----------
    params : dict
        Simulation parameters dictionary containing:

        **Grid and Box Parameters:**

        - nx : list of int
            Number of grid points in each dimension, e.g., [32, 32, 32] for 3D.
        - lx : list of float
            Box dimensions [a, b, c] following crystallographic convention.
            Units: a_Ref * N_Ref^(1/2).
        - angles : list of float, optional
            Unit cell angles [alpha, beta, gamma] in degrees (default: [90, 90, 90]).
            Following crystallographic convention:

            - alpha = angle between axes b and c (indices 1,2)
            - beta = angle between axes a and c (indices 0,2)
            - gamma = angle between axes a and b (indices 0,1)

            Common systems:

            - Orthorhombic: [90, 90, 90] (default)
            - Hexagonal: [90, 90, 120] with a = b
            - Monoclinic: [90, beta, 90] with beta != 90
            - Triclinic: all angles can vary

        - axis_ordering : list of str, optional
            Specifies the ordering of axes in lx and angles arrays.
            Default: ["a", "b", "c"] (standard crystallographic convention).
            Example: ["b", "a", "c"] means lx = [b, a, c] and angles are reordered
            accordingly. The code internally converts to [a, b, c] ordering.

        - box_is_altering : bool
            If True, optimize box size along with fields to minimize stress.

        **Chain Model Parameters:**

        - chain_model : {'discrete', 'continuous'}
            Chain propagation model. 'discrete' for discrete Gaussian chains,
            'continuous' for continuous Gaussian chains (pseudo-spectral method).
        - ds : float
            Contour step size, typically 1/N_Ref where N_Ref is the reference
            polymerization index.

        **Monomer and Interaction Parameters:**

        - segment_lengths : dict
            Relative statistical segment lengths, {monomer_type: length}.
            Example: {"A": 1.0, "B": 1.0} for symmetric AB diblock.
        - chi_n : dict
            Flory-Huggins interaction parameters × N_Ref,
            {monomer_pair: chi*N_Ref}. Example: {"A,B": 15.0}.
            Self-interactions (e.g., "A,A") should not be included.

        **Polymer Architecture:**

        - distinct_polymers : list of dict
            List of polymer chain specifications. Each dict contains:

            - volume_fraction : float
                Volume fraction of this polymer species.
            - blocks : list of dict
                Block definitions. Each block has:

                - type : str
                    Monomer type label.
                - length : float
                    Block length (contour fraction).
                - v, u : int (optional)
                    Vertex indices for branched polymers.

        **Platform Selection:**

        - platform : {'cuda', 'cpu-mkl'}, optional
            Computational backend. Auto-selected if not specified:
            1D → cpu-mkl, 2D/3D → cuda (if available).
        - reduce_memory : bool, optional
            If True, store only propagator checkpoints instead of full histories,
            recomputing propagators as needed (default: False).
            Reduces memory usage but increases computation time by 2-4x.
        - numerical_method : {'rqm4', 'etdrk4', 'cn-adi2', 'cn-adi4-lr'}, optional
            Numerical algorithm for propagator computation (default: 'rqm4'):

            - 'rqm4': Pseudo-spectral with 4th-order Richardson extrapolation
            - 'etdrk4': Pseudo-spectral with ETDRK4 exponential integrator
            - 'cn-adi2': Real-space with 2nd-order Crank-Nicolson ADI
            - 'cn-adi4-lr': Real-space with 4th-order CN-ADI (Local Richardson extrapolation)

        **Optimization Parameters:**

        - optimizer : dict
            Optimizer configuration:

            For Anderson Mixing (name='am'):

            - name : str
                'am' for Anderson Mixing.
            - max_hist : int
                Maximum number of history vectors (default: 20).
            - start_error : float
                Error threshold to switch from simple mixing to AM (default: 1e-1).
            - mix_min : float
                Minimum mixing parameter (default: 0.1).
            - mix_init : float
                Initial mixing parameter (default: 0.1).

            For ADAM (name='adam'):

            - name : str
                'adam' for ADAM optimizer.
            - lr : float
                Learning rate (default: 1e-2).
            - gamma : float
                Learning rate decay (default: 1.0, no decay).

        **Convergence Parameters:**

        - max_iter : int, optional
            Maximum number of iterations (default: 2000).
        - tolerance : float, optional
            Convergence tolerance for error_level (default: 1e-8).

        **Advanced Options:**

        - crystal_system : str, optional
            Crystal system for box optimization constraints. Determines which
            lattice parameters can vary independently during box relaxation.
            Options:

            - 'Orthorhombic' (default): a, b, c independent; all angles = 90°
            - 'Tetragonal': a = b ≠ c; all angles = 90°
            - 'Cubic': a = b = c; all angles = 90°
            - 'Hexagonal': a = b ≠ c; alpha = beta = 90°, gamma = 120°
            - 'Monoclinic': a, b, c independent; alpha = gamma = 90°, beta varies
            - 'Triclinic': all lengths and angles can vary independently

        - space_group : dict, optional
            Space group symmetry constraints (beta feature):

            - symbol : str
                Hermann-Mauguin symbol (e.g., 'Ia-3d' for gyroid).
            - number : int, optional
                Hall number for space group.

        - scale_stress : float, optional
            Scaling factor for stress in box relaxation (default: 1.0).
        - aggregate_propagator_computation : bool, optional
            Enable propagator aggregation optimization (default: True).

    Attributes
    ----------
    monomer_types : list of str
        Sorted list of monomer type labels.
    segment_lengths : dict
        Statistical segment lengths for each monomer type.
    chi_n : dict
        Flory-Huggins interaction parameters (sorted pairs).
    distinct_polymers : list
        Polymer specifications with block information.
    cb : ComputationBox
        Computation box managing grid, FFT, and integrals.
    molecules : Molecules
        C++ Molecules object containing polymer chain definitions.
    solver : PropagatorComputation
        C++ propagator solver object.
    field_optimizer : AndersonMixing or Adam
        Field optimization algorithm instance.
    mpt : SymmetricPolymerTheory
        Multi-monomer field theory transformations.
    matrix_chi : ndarray
        Interaction parameter matrix (M×M).
    matrix_p : ndarray
        Projection matrix for field residuals (M×M).
    phi : dict
        Final monomer concentration fields {monomer_type: array}.
    w : ndarray
        Final potential fields, shape (M, total_grid).
    free_energy : float
        Final free energy per unit volume.
    error_level : float
        Final convergence error.
    iter : int
        Number of iterations performed.

    Raises
    ------
    AssertionError
        If parameter validation fails (e.g., volume fractions don't sum to 1,
        invalid monomer types in chi_n, etc.).
    ValueError
        If invalid space group or optimizer name is specified.

    See Also
    --------
    LFTS : Langevin field-theoretic simulation class.
    SymmetricPolymerTheory : Multi-monomer field theory transformations.
    Adam : ADAM optimizer for field updates.

    Notes
    -----
    **Units and Conventions:**

    - Length unit: a_Ref * N_Ref^(1/2) where a_Ref is reference statistical
      segment length, N_Ref is reference polymerization index.
    - Fields w are "per reference chain" potentials. Multiply by ds for
      "per segment" potentials.
    - Free energy is reported as βF/V (dimensionless free energy per volume).

    **Computational Details:**

    - Pseudo-spectral methods: RQM4 (4th-order Richardson) or ETDRK4 (exponential integrator).
    - Real-space methods: CN-ADI2 (2nd-order) or CN-ADI4 (4th-order Crank-Nicolson ADI).
    - Propagators use dynamic programming to avoid redundant calculations for
      branched polymers (see [1]_).
    - Anderson Mixing accelerates SCFT convergence by mixing field history.
    - ADAM can be more stable than Anderson Mixing for some systems.

    **Random Copolymers:**

    Define random copolymers by adding "fraction" dict to a single-block polymer:

    .. code-block:: python

        {
            "volume_fraction": 0.5,
            "blocks": [{
                "type": "Random_AB",
                "length": 1.0,
                "fraction": {"A": 0.3, "B": 0.7}
            }]
        }

    **Performance:**

    - 1D simulations: cpu-mkl is recommended
    - 2D/3D simulations: cuda provides 10-20x speedup over cpu-mkl
    - Memory usage: ~500MB for 64³ grid (standard), ~50MB (reduce_memory)

    References
    ----------
    .. [1] Kang, H., et al. "Efficient Computation of Chain Propagators for
           Polymers with Complex Architectures." J. Chem. Theory Comput. 2025,
           21, 3676.
    .. [2] Delaney Vigil, D. L., et al. "Multimonomer Field Theory of
           Conformationally Asymmetric Polymer Blends." Macromolecules 2025,
           58, 816.

    Examples
    --------
    **AB Diblock Copolymer Lamellar Phase:**

    >>> import numpy as np
    >>> from polymerfts import scft
    >>>
    >>> # Define parameters
    >>> params = {
    ...     "nx": [32, 32, 32],
    ...     "lx": [4.36, 4.36, 4.36],
    ...     "chain_model": "discrete",
    ...     "ds": 1/90,
    ...     "segment_lengths": {"A": 1.0, "B": 1.0},
    ...     "chi_n": {"A,B": 15.0},
    ...     "distinct_polymers": [{
    ...         "volume_fraction": 1.0,
    ...         "blocks": [
    ...             {"type": "A", "length": 0.5},
    ...             {"type": "B", "length": 0.5}
    ...         ]
    ...     }],
    ...     "optimizer": {
    ...         "name": "am",
    ...         "max_hist": 20,
    ...         "start_error": 1e-1,
    ...         "mix_min": 0.1,
    ...         "mix_init": 0.1
    ...     },
    ...     "max_iter": 2000,
    ...     "tolerance": 1e-8,
    ...     "box_is_altering": False
    ... }
    >>>
    >>> # Initialize SCFT solver
    >>> calc = scft.SCFT(params)
    >>>
    >>> # Prepare initial fields (small random perturbation)
    >>> nx_total = np.prod(params["nx"])
    >>> w_init = {
    ...     "A": 0.01 * np.random.normal(size=nx_total),
    ...     "B": 0.01 * np.random.normal(size=nx_total)
    ... }
    >>>
    >>> # Run SCFT iteration
    >>> calc.run(initial_fields=w_init)
    >>>
    >>> # Save results
    >>> calc.save_results("scft_lamellar.mat")
    >>>
    >>> # Access final fields and concentrations
    >>> print(f"Free energy: {calc.free_energy:.6f}")
    >>> print(f"Error level: {calc.error_level:.3e}")
    >>> phi_A = calc.phi["A"]  # Concentration field for monomer A
    >>> w_A = calc.w[0]        # Potential field for monomer A

    **ABC Triblock Copolymer:**

    >>> params_abc = {
    ...     "nx": [64, 64, 64],
    ...     "lx": [8.0, 8.0, 8.0],
    ...     "chain_model": "continuous",
    ...     "ds": 1/100,
    ...     "segment_lengths": {"A": 1.0, "B": 1.0, "C": 1.0},
    ...     "chi_n": {"A,B": 20.0, "B,C": 20.0, "A,C": 20.0},
    ...     "distinct_polymers": [{
    ...         "volume_fraction": 1.0,
    ...         "blocks": [
    ...             {"type": "A", "length": 0.33},
    ...             {"type": "B", "length": 0.34},
    ...             {"type": "C", "length": 0.33}
    ...         ]
    ...     }],
    ...     "optimizer": {"name": "adam", "lr": 1e-2, "gamma": 0.999},
    ...     "box_is_altering": False
    ... }
    >>> calc_abc = scft.SCFT(params_abc)

    For complete working examples, see:

    - examples/scft/Lamella3D.py - Lamellar phase
    - examples/scft/Gyroid.py - Gyroid phase with box relaxation
    - examples/scft/ABC_Triblock_Sphere3D.py - Spherical phase
    - examples/scft/phases/ - Various morphologies
    """
    def __init__(self, params): #, phi_target=None, mask=None):

        # Validate input parameters
        validate_scft_params(params)

        # Segment length
        self.monomer_types = sorted(list(params["segment_lengths"].keys()))
        self.segment_lengths = copy.deepcopy(params["segment_lengths"])
        self.distinct_polymers = copy.deepcopy(params["distinct_polymers"])
        
        if len(self.monomer_types) != len(set(self.monomer_types)):
            raise ValidationError("There are duplicated monomer_types")

        # Choose platform among [cuda, cpu-mkl]
        avail_platforms = _core.PlatformSelector.avail_platforms()
        if "platform" in params:
            platform = params["platform"]
        elif "cpu-mkl" in avail_platforms and len(params["nx"]) == 1: # for 1D simulation, use CPU
            platform = "cpu-mkl"
        elif "cuda" in avail_platforms: # If cuda is available, use GPU
            platform = "cuda"
        else:
            platform = avail_platforms[0]

        # Get checkpointing option
        reduce_memory = params.get("reduce_memory", False)

        # Get boundary conditions
        bc = params.get("bc", None)
        if bc is None:
            # Default to periodic boundaries
            bc = ["periodic"] * (2 * len(params["nx"]))

        # Flory-Huggins parameters, χN
        self.chi_n = parse_chi_n(params["chi_n"], self.segment_lengths)

        # Add missing monomer pair combinations with zero interaction
        for monomer_pair in itertools.combinations(self.monomer_types, 2):
            monomer_pair = list(monomer_pair)
            monomer_pair.sort()
            sorted_monomer_pair = monomer_pair[0] + "," + monomer_pair[1]
            if sorted_monomer_pair not in self.chi_n:
                self.chi_n[sorted_monomer_pair] = 0.0
        
        # Multimonomer polymer field theory
        self.mpt = SymmetricPolymerTheory(self.monomer_types, self.chi_n, zeta_n=None)

        # Matrix for field residuals.
        # See *J. Chem. Phys.* **2017**, 146, 244902
        M = len(self.monomer_types)
        matrix_chi = np.zeros((M,M))
        for i in range(M):
            for j in range(i+1,M):
                key = self.monomer_types[i] + "," + self.monomer_types[j]
                if key in self.chi_n:
                    matrix_chi[i,j] = self.chi_n[key]
                    matrix_chi[j,i] = self.chi_n[key]
        
        self.matrix_chi = matrix_chi
        matrix_chi_inv = np.linalg.inv(matrix_chi)
        self.matrix_p = np.identity(M) - np.matmul(np.ones((M,M)), matrix_chi_inv)/np.sum(matrix_chi_inv)
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

        # Validate polymer chain definitions
        validate_polymers(self.distinct_polymers)

        # Process polymer chain blocks
        for polymer in self.distinct_polymers:
            blocks_input = process_polymer_blocks(polymer)
            polymer["blocks_input"] = blocks_input

        # Process random copolymer chains
        self.random_fraction = {}
        for polymer in self.distinct_polymers:
            result = process_random_copolymer(polymer, self.segment_lengths)
            if result is not None:
                random_type_string, statistical_segment_length, fraction_dict = result
                polymer["block_monomer_types"] = [random_type_string]
                self.segment_lengths[random_type_string] = statistical_segment_length
                self.random_fraction[random_type_string] = fraction_dict

        # Make a monomer color dictionary
        dict_color = create_monomer_color_dict(self.segment_lengths)
        logger.debug(f"Monomer color: {dict_color}")

        # Draw polymer chain architectures
        for idx, polymer in enumerate(self.distinct_polymers):
            draw_polymer_architecture(polymer, idx, params["ds"], dict_color)

        # Get angles if provided
        angles = params.get("angles", None)

        # Handle axis ordering (convert user ordering to standard [a, b, c])
        axis_ordering = params.get("axis_ordering", None)
        self.axis_ordering = axis_ordering  # Store for reference

        if axis_ordering is not None:
            dim = len(params["nx"])
            if len(axis_ordering) != dim:
                raise ValueError(f"axis_ordering length ({len(axis_ordering)}) must match dimension ({dim})")

            # Validate axis labels
            valid_labels = ["a", "b", "c"][:dim]
            if sorted(axis_ordering) != valid_labels:
                raise ValueError(f"axis_ordering must contain exactly {valid_labels}, got {axis_ordering}")

            # Compute permutation from user ordering to standard [a, b, c]
            # e.g., ["b", "a", "c"] -> permutation [1, 0, 2] means:
            #   user[0] (b) goes to standard[1]
            #   user[1] (a) goes to standard[0]
            #   user[2] (c) goes to standard[2]
            perm = [axis_ordering.index(label) for label in valid_labels]
            self._axis_perm = perm  # perm[i] = index in user array for standard axis i

            # Reorder nx and lx from user ordering to standard [a, b, c]
            nx_reordered = [params["nx"][perm[i]] for i in range(dim)]
            lx_reordered = [params["lx"][perm[i]] for i in range(dim)]

            # Reorder angles if provided (3D only)
            # angles = [alpha, beta, gamma] where:
            #   alpha = angle between b,c (axes 1,2)
            #   beta = angle between a,c (axes 0,2)
            #   gamma = angle between a,b (axes 0,1)
            # When axes are permuted, angles must be remapped
            if angles is not None and dim == 3:
                # Build mapping: which original angle corresponds to which new angle
                # After permutation, the angle between new axes i,j is the angle
                # that was between original axes perm[i], perm[j]
                def angle_index(i, j):
                    """Return angle array index for angle between axes i and j"""
                    if (i, j) == (1, 2) or (i, j) == (2, 1):
                        return 0  # alpha
                    elif (i, j) == (0, 2) or (i, j) == (2, 0):
                        return 1  # beta
                    else:  # (0, 1) or (1, 0)
                        return 2  # gamma

                angles_reordered = [0.0, 0.0, 0.0]
                # New alpha (angle between new b,c) = old angle between perm[1], perm[2]
                angles_reordered[0] = angles[angle_index(perm[1], perm[2])]
                # New beta (angle between new a,c) = old angle between perm[0], perm[2]
                angles_reordered[1] = angles[angle_index(perm[0], perm[2])]
                # New gamma (angle between new a,b) = old angle between perm[0], perm[1]
                angles_reordered[2] = angles[angle_index(perm[0], perm[1])]
                angles = angles_reordered

            # Update params for PropagatorSolver creation
            params = copy.deepcopy(params)
            params["nx"] = nx_reordered
            params["lx"] = lx_reordered

            print(f"Axis ordering: {axis_ordering} -> reordered to [a, b, c]")
            print(f"  nx: {params['nx']}, lx: {params['lx']}")
            if angles is not None:
                print(f"  angles: {angles}")
        else:
            self._axis_perm = None

        # Get numerical method
        numerical_method = params.get("numerical_method", "rqm4")

        # Create PropagatorSolver instance
        self.prop_solver = PropagatorSolver(
            nx=params["nx"],
            lx=params["lx"],
            ds=params["ds"],
            bond_lengths=self.segment_lengths,
            bc=bc,
            chain_model=params["chain_model"],
            numerical_method=numerical_method,
            platform=platform,
            reduce_memory=reduce_memory,
        )

        # Set angles if provided (after initialization, before adding polymers)
        if angles is not None:
            # Re-create with angles by accessing internal factory
            self.prop_solver._computation_box = self.prop_solver._factory.create_computation_box(
                params["nx"], params["lx"], angles=angles, bc=bc
            )

        # Add polymer chains
        for polymer in self.distinct_polymers:
            if "initial_conditions" in polymer:
                self.prop_solver.add_polymer(polymer["volume_fraction"], polymer["blocks_input"], polymer["initial_conditions"])
            else:
                self.prop_solver.add_polymer(polymer["volume_fraction"], polymer["blocks_input"])

        # Initialize the solver (creates propagator optimizer and propagator computation)
        self.prop_solver._initialize_solver()

        # Set aggregate_propagator_computation option
        if "aggregate_propagator_computation" in params:
            aggregate = params["aggregate_propagator_computation"]
            # Recreate propagator optimizer with the specified option
            self.prop_solver._propagator_optimizer = self.prop_solver._factory.create_propagator_computation_optimizer(
                self.prop_solver._molecules, aggregate
            )
            # Recreate propagator computation with the new optimizer
            self.prop_solver._propagator_computation = self.prop_solver._factory.create_propagator_computation(
                self.prop_solver._computation_box, self.prop_solver._molecules, self.prop_solver._propagator_optimizer,
                self.prop_solver.numerical_method
            )

        # Display factory info
        self.prop_solver._factory.display_info()

        # Expose computation box as public attribute (documented in class docstring)
        self.cb = self.prop_solver._computation_box

        # Initialize smearing (must be after self.cb is set)
        self.smearing = Smearing(
            self.cb.get_nx(),
            self.cb.get_lx(),
            params.get("smearing", None)
        )

        # Scaling factor for stress when the fields and box size are simultaneously computed
        if "scale_stress" in params:
            self.scale_stress = params["scale_stress"]
        else:
            self.scale_stress = 1

        # Space group symmetry operations
        if "space_group" in params:
            # Create a space group object
            if "number" in params["space_group"]:
                self.sg = SpaceGroup(params["nx"], params["space_group"]["symbol"], params["space_group"]["number"])
            else:
                self.sg = SpaceGroup(params["nx"], params["space_group"]["symbol"])
                
            if not self.sg.crystal_system in ["Orthorhombic", "Tetragonal", "Cubic", "Hexagonal", "Trigonal"]:
                raise ValueError("The crystal system of the space group must be Orthorhombic, Tetragonal, Cubic, Hexagonal, or Trigonal. " +
                    "The current crystal system is " + self.sg.crystal_system + ".")

            if self.sg.crystal_system == "Orthorhombic":
                self.lx_reduced_indices = [0, 1, 2]
                self.lx_full_indices = [0, 1, 2]
            elif self.sg.crystal_system == "Tetragonal":
                # Standard tetragonal: a = b ≠ c (4-fold axis along z)
                self.lx_reduced_indices = [0, 2]  # a and c as independent
                self.lx_full_indices = [0, 0, 1]  # Map: reduced[0]->a, reduced[0]->b, reduced[1]->c
            elif self.sg.crystal_system == "Cubic":
                self.lx_reduced_indices = [0]
                self.lx_full_indices = [0, 0, 0]
            elif self.sg.crystal_system == "Hexagonal" or self.sg.crystal_system == "Trigonal":
                # Hexagonal: a = b ≠ c, α = β = 90°, γ = 120°
                self.lx_reduced_indices = [0, 2]  # Only a and c can vary independently
                self.lx_full_indices = [0, 0, 1]  # Map: reduced[0]->a, reduced[0]->b, reduced[1]->c

            # For space groups, angles are fixed by the crystal system
            self.angles_reduced_indices = []
            self.angles_full_indices = []

            # Total number of variables to be adjusted to minimize the Hamiltonian
            # Using reduced basis from space group symmetry
            n_reduced = len(self.sg.irreducible_mesh)
            if params["box_is_altering"]:
                n_var = len(self.monomer_types)*n_reduced + len(self.sg.lattice_parameters)
            else :
                n_var = len(self.monomer_types)*n_reduced

            # # Set symmetry operations to the solver
            # solver.set_symmetry_operations(sg.get_symmetry_operations())
        else:
            self.sg = None

            # Crystal system determines which lattice parameters can be optimized
            # Default: all box lengths can vary, angles are fixed at their initial values
            dim = len(params["nx"])

            # Auto-detect crystal system from angles for 2D non-orthogonal systems
            if "crystal_system" not in params and dim == 2 and "angles" in params:
                # For 2D, angles can be [gamma] or [alpha, beta, gamma]
                angles = params["angles"]
                gamma = angles[0] if len(angles) == 1 else angles[2]
                if abs(gamma - 120.0) < 1e-6:
                    # Hexagonal 2D: gamma = 120°, a = b
                    crystal_system = "Hexagonal2D"
                elif abs(gamma - 90.0) > 1e-6:
                    # General oblique 2D: gamma ≠ 90°
                    crystal_system = "Oblique2D"
                else:
                    crystal_system = "Orthorhombic"
            else:
                crystal_system = params.get("crystal_system", "Orthorhombic")

            self.crystal_system = crystal_system

            # Non-orthogonal 3D crystal systems require 3D
            if crystal_system in ["Monoclinic", "Triclinic", "Hexagonal"] and dim != 3:
                raise ValueError(f"Crystal system '{crystal_system}' requires 3D simulation (nx must have 3 elements).")

            if crystal_system == "Hexagonal2D":
                # 2D hexagonal: γ = 120°, a = b (single lattice parameter)
                self.lx_reduced_indices = [0]  # Only a (b = a enforced)
                self.lx_full_indices = [0, 0]  # a, a
                self.angles_reduced_indices = []  # γ fixed at 120°
                self.angles_full_indices = []
            elif crystal_system == "Oblique2D":
                # 2D oblique: γ ≠ 90°, a and b can vary independently, γ is optimized
                self.lx_reduced_indices = [0, 1]  # a and b
                self.lx_full_indices = [0, 1]  # a, b
                self.angles_reduced_indices = [2]  # γ is at index 2 in full angles array
                self.angles_full_indices = [2]  # Maps to γ (third angle)
            elif crystal_system == "Orthorhombic":
                # α = β = γ = 90°, a, b, c can all vary
                self.lx_reduced_indices = list(range(len(params["lx"])))
                self.lx_full_indices = list(range(len(params["lx"])))
                self.angles_reduced_indices = []  # No angles to optimize
                self.angles_full_indices = []
            elif crystal_system == "Tetragonal":
                # α = β = γ = 90°, a = b ≠ c
                self.lx_reduced_indices = [0, 2]  # a and c
                self.lx_full_indices = [0, 0, 1]  # a, a, c
                self.angles_reduced_indices = []
                self.angles_full_indices = []
            elif crystal_system == "Cubic":
                # α = β = γ = 90°, a = b = c
                self.lx_reduced_indices = [0]
                self.lx_full_indices = [0, 0, 0]
                self.angles_reduced_indices = []
                self.angles_full_indices = []
            elif crystal_system == "Hexagonal":
                # α = β = 90°, γ = 120°, a = b ≠ c
                self.lx_reduced_indices = [0, 2]  # a and c
                self.lx_full_indices = [0, 0, 1]  # a, a, c
                self.angles_reduced_indices = []  # γ fixed at 120°
                self.angles_full_indices = []
            elif crystal_system == "Monoclinic":
                # α = γ = 90°, β ≠ 90°, a, b, c can all vary
                # β is the angle between a and c axes (index 1 in [α, β, γ])
                self.lx_reduced_indices = [0, 1, 2]  # a, b, c
                self.lx_full_indices = [0, 1, 2]
                self.angles_reduced_indices = [1]  # Only β can vary
                self.angles_full_indices = [1]  # Maps to β
            elif crystal_system == "Triclinic":
                # All angles and lengths can vary
                self.lx_reduced_indices = [0, 1, 2]
                self.lx_full_indices = [0, 1, 2]
                self.angles_reduced_indices = [0, 1, 2]  # α, β, γ
                self.angles_full_indices = [0, 1, 2]
            else:
                raise ValueError(f"Unknown crystal system: {crystal_system}. " +
                    "Choose from: Orthorhombic, Tetragonal, Cubic, Hexagonal, Monoclinic, Triclinic.")

            # Total number of variables to be adjusted to minimize the Hamiltonian
            n_angles = len(self.angles_reduced_indices)
            if params["box_is_altering"]:
                n_var = len(self.monomer_types)*np.prod(params["nx"]) + len(self.lx_reduced_indices) + n_angles
            else :
                n_var = len(self.monomer_types)*np.prod(params["nx"])
            
        # Select an optimizer among 'Anderson Mixing' and 'ADAM' for finding saddle point
        # (C++ class) Anderson Mixing method for finding saddle point
        if params["optimizer"]["name"] == "am":
            self.field_optimizer = self.prop_solver.create_anderson_mixing(n_var,
                params["optimizer"]["max_hist"],     # maximum number of history
                params["optimizer"]["start_error"],  # when switch to AM from simple mixing
                params["optimizer"]["mix_min"],      # minimum mixing rate of simple mixing
                params["optimizer"]["mix_init"])     # initial mixing rate of simple mixing

        # (Python class) ADAM optimizer for finding saddle point
        elif params["optimizer"]["name"] == "adam":
            self.field_optimizer = Adam(total_grid = n_var,
                lr = params["optimizer"]["lr"],
                gamma = params["optimizer"]["gamma"])
        else:
            raise ValueError(f"Invalid optimizer name: {params['optimizer']['name']}. Choose among 'am' and 'adam'.")

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
        print("Box Dimension: %d" % (self.prop_solver.dim))
        print("Nx:", self.prop_solver.nx)
        print("Lx [a, b, c]:", self.prop_solver.get_lx())
        angles_deg = list(self.prop_solver.get_angles_degrees())
        print("Angles [alpha, beta, gamma]:", angles_deg)
        print("dx:", self.prop_solver.get_dx())
        print("Volume: %f" % (self.prop_solver.get_volume()))

        print("Chain model: %s" % (params["chain_model"]))
        print("Segment lengths:\n\t", list(self.segment_lengths.items()))
        print("Conformational asymmetry (epsilon): ")
        for monomer_pair in itertools.combinations(self.monomer_types,2):
            print("\t%s/%s: %f" % (monomer_pair[0], monomer_pair[1], self.segment_lengths[monomer_pair[0]]/self.segment_lengths[monomer_pair[1]]))

        print("χN: ")
        for key in self.chi_n:
            print("\t%s: %f" % (key, self.chi_n[key]))

        print("P matrix for field residuals:\n\t", str(self.matrix_p).replace("\n", "\n\t"))

        self.prop_solver._molecules.display_architectures()
        self.prop_solver._propagator_optimizer.display_statistics()

        #  Save internal variables
        self.params = params
        self.chain_model = params["chain_model"]
        self.ds = params["ds"]
        self.box_is_altering = params["box_is_altering"]

        # Stress computation interval (compute stress every N iterations)
        # Options: integer (1, 2, 3, ...) or "adaptive"
        # - Integer: compute stress every N iterations
        # - "adaptive": automatically adjust based on error level
        #   (more frequent when error high, less frequent when converging)
        # Default is 3 for ~15% speedup with minimal impact on convergence
        self.stress_interval = params.get("stress_interval", 3)

        self.max_iter = max_iter
        self.tolerance = tolerance

        # For context manager
        self._is_open = True

    # =========================================================================
    # Class Methods
    # =========================================================================

    @classmethod
    def from_file(cls, path: Union[str, Path], **overrides) -> "SCFT":
        """Create SCFT instance from configuration file.

        Loads parameters from YAML or JSON configuration file and
        creates an SCFT instance. Supports parameter overrides.

        Parameters
        ----------
        path : str or Path
            Path to configuration file (.yaml, .yml, or .json).
        **overrides : Any
            Parameter values to override from the file.

        Returns
        -------
        SCFT
            Configured SCFT instance.

        Examples
        --------
        >>> calc = SCFT.from_file("simulation.yaml")
        >>> calc.run(initial_fields)

        >>> # Override specific parameters
        >>> calc = SCFT.from_file("base.yaml", max_iter=500, tolerance=1e-7)
        """
        params = load_config(path)
        params.update(overrides)
        return cls(params)

    # =========================================================================
    # Context Manager Protocol
    # =========================================================================

    def __enter__(self) -> "SCFT":
        """Enter context manager.

        Returns
        -------
        SCFT
            Self for use in with statement.

        Examples
        --------
        >>> with SCFT(params) as calc:
        ...     calc.run(initial_fields)
        ...     # Resources automatically cleaned up on exit
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Exit context manager, cleaning up resources.

        Parameters
        ----------
        exc_type : type or None
            Exception type if an exception occurred.
        exc_val : Exception or None
            Exception value if an exception occurred.
        exc_tb : traceback or None
            Exception traceback if an exception occurred.

        Returns
        -------
        bool
            False to propagate any exceptions.
        """
        self.close()
        return False

    def close(self) -> None:
        """Release resources held by the SCFT instance.

        Call this method when done with the SCFT instance if not
        using the context manager. Safe to call multiple times.
        """
        if self._is_open:
            # Release propagator solver resources
            if hasattr(self, 'prop_solver'):
                # PropagatorSolver cleanup if needed
                pass
            self._is_open = False

    def compute_concentrations(self, w):
        """Compute monomer concentration fields for given potential fields.

        Solves the modified diffusion equations to compute chain propagators,
        then integrates over chain contours to obtain monomer concentration
        distributions. Handles both block copolymers and random copolymers.

        Parameters
        ----------
        w : ndarray
            Potential fields, shape (M, total_grid) where M is the number
            of monomer types. Each row contains the field for one monomer type.

        Returns
        -------
        phi : dict
            Monomer concentration fields, {monomer_type: concentration_array}.
            Each array has length total_grid.
        elapsed_time : dict
            Timing information:

            - 'pseudo' : float
                Time (seconds) for propagator computation.
            - 'phi' : float
                Time (seconds) for concentration integration.

        Notes
        -----
        **Propagator Computation:**

        For each polymer chain, solves the modified diffusion equation:

        .. math::
            \\frac{\\partial q}{\\partial s} = \\frac{a^2}{6} \\nabla^2 q - w q

        where q(r,s) is the chain propagator, a is the statistical segment
        length, w(r) is the potential field, and s is the contour variable.

        **Concentration Calculation:**

        Monomer concentration at position r is computed by integrating
        propagators over chain contours:

        .. math::
            \\phi_A(\\mathbf{r}) = \\frac{V}{Q} \\int_0^{\\alpha_A} ds \\, q(\\mathbf{r},s) q^\\dagger(\\mathbf{r},s)

        where Q is the single-chain partition function, V is volume, and
        q†(r,s) is the complementary propagator.

        **Random Copolymers:**

        For random copolymers, the effective field is computed as:

        .. math::
            w_{\\text{random}} = \\sum_i f_i w_i

        where f_i is the fraction of monomer type i in the random copolymer.
        Concentrations are then distributed to individual monomer types
        according to these fractions.

        **Performance:**

        - Propagator computation dominates runtime (~95% of total).
        - CUDA: ~0.5s for 32³ grid, 100 segments.
        - CPU-MKL: ~5s for 32³ grid, 100 segments.

        See Also
        --------
        run : Main SCFT iteration loop that calls this method.
        SCFT.solver.compute_propagators : C++ method for propagator computation.
        """
        M = len(self.monomer_types)
        elapsed_time = {}

        # Make a dictionary for input fields
        w_input = {}
        for i in range(M):
            w_input[self.monomer_types[i]] = w[i]
        for random_polymer_name, random_fraction in self.random_fraction.items():
            w_input[random_polymer_name] = np.zeros(self.prop_solver.n_grid, dtype=np.float64)
            for monomer_type, fraction in random_fraction.items():
                w_input[random_polymer_name] += w_input[monomer_type]*fraction

        # Apply smearing to fields before computing propagators
        w_input_for_propagator = self.smearing.apply_to_dict(w_input)

        # For the given fields, compute the polymer statistics
        time_p_start = time.time()
        self.prop_solver.compute_propagators(w_input_for_propagator)
        self.prop_solver.compute_concentrations()
        elapsed_time["pseudo"] = time.time() - time_p_start

        # Compute total concentration for each monomer type
        phi = {}
        time_phi_start = time.time()
        for monomer_type in self.monomer_types:
            phi[monomer_type] = self.prop_solver.get_concentration(monomer_type)
        elapsed_time["phi"] = time.time() - time_phi_start

        # Add random copolymer concentration to each monomer type
        for random_polymer_name, random_fraction in self.random_fraction.items():
            phi[random_polymer_name] = self.prop_solver.get_concentration(random_polymer_name)
            for monomer_type, fraction in random_fraction.items():
                phi[monomer_type] += phi[random_polymer_name]*fraction

        return phi, elapsed_time

    def save_results(self, path):
        """Save SCFT results to file in MATLAB, JSON, or YAML format.

        Exports the computed fields, concentrations, and simulation parameters
        to a file. The format is determined by the file extension (.mat, .json,
        or .yaml).

        Parameters
        ----------
        path : str
            Output file path. Extension determines format:

            - '.mat' : MATLAB format (scipy.io.savemat)
            - '.json' : JSON format (human-readable, larger files)
            - '.yaml' : YAML format (human-readable, structured)

        Raises
        ------
        ValueError
            If file extension is not .mat, .json, or .yaml.

        Notes
        -----
        **Saved Data Structure:**

        The output file contains a dictionary with these keys:

        - initial_params : dict
            Original params dict passed to __init__.
        - dim, nx, lx : int, list, list
            Grid dimension, grid points, box sizes.
        - monomer_types : list of str
            Monomer type labels.
        - chi_n : dict
            Flory-Huggins interaction parameters.
        - chain_model : str
            'discrete' or 'continuous'.
        - ds : float
            Contour step size.
        - eigenvalues : ndarray
            Eigenvalues of interaction matrix (from field theory).
        - matrix_a, matrix_a_inverse : ndarray
            Eigenvector matrices for field transformations.
        - w_{monomer_type} : ndarray
            Potential field for each monomer type.
        - phi_{monomer_type} : ndarray
            Concentration field for each monomer type.
        - space_group_symbol, space_group_hall_number : str, int (optional)
            Space group information if symmetry constraints were used.

        **File Formats:**

        *MATLAB (.mat):*
        - Binary format, compact, fast I/O
        - Compatible with MATLAB and Python (scipy.io)
        - Recommended for large datasets

        *JSON (.json):*
        - Text format, human-readable
        - Easy to inspect and parse
        - Larger file size than .mat

        *YAML (.yaml):*
        - Text format, most human-readable
        - Preserves structure well
        - Good for configuration files

        **Loading Results:**

        .. code-block:: python

            # MATLAB format
            import scipy.io
            data = scipy.io.loadmat("scft_results.mat")
            w_A = data["w_A"]
            phi_A = data["phi_A"]

            # JSON format
            import json
            with open("scft_results.json", "r") as f:
                data = json.load(f)

            # YAML format
            import yaml
            with open("scft_results.yaml", "r") as f:
                data = yaml.safe_load(f)

        See Also
        --------
        run : Main SCFT iteration that produces results to save.

        Examples
        --------
        >>> # After running SCFT
        >>> calc.run(initial_fields=w_init)
        >>>
        >>> # Save in different formats
        >>> calc.save_results("lamellar.mat")    # MATLAB format
        >>> calc.save_results("lamellar.json")   # JSON format
        >>> calc.save_results("lamellar.yaml")   # YAML format
        """
        # Make a dictionary for chi_n
        chi_n_mat = {}
        for key in self.chi_n:
            chi_n_mat[key] = self.chi_n[key]

        # Make a dictionary for data
        m_dic = {"initial_params": self.params,
            "dim":self.prop_solver.dim, "nx":self.prop_solver.nx, "lx":self.prop_solver.get_lx(),
            "angles":list(self.prop_solver.get_angles_degrees()),
            "monomer_types":self.monomer_types, "chi_n":chi_n_mat, "chain_model":self.chain_model, "ds":self.ds,
            "eigenvalues": self.mpt.eigenvalues, "matrix_a": self.mpt.matrix_a, "matrix_a_inverse": self.mpt.matrix_a_inv}

        if self.sg is not None:
            m_dic["space_group_symbol"] = self.sg.spacegroup_symbol
            m_dic["space_group_hall_number"] = self.sg.hall_number

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

        # phi_total = np.zeros(self.cb.get_total_grid())
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
            raise ValueError("Invalid output file extension. Use .mat, .json, or .yaml.")

    def run(
        self,
        initial_fields: Dict[str, np.ndarray],
        q_init: Optional[Dict] = None,
        on_iteration: Optional[Callable[[IterationInfo], Optional[bool]]] = None,
        return_result: bool = False
    ) -> Optional[SCFTResult]:
        """Run SCFT iteration to find saddle point of the Hamiltonian.

        Iteratively solves the self-consistent field equations until convergence
        or maximum iterations reached. At each iteration:

        1. Computes monomer concentrations φ for current fields w
        2. Calculates self-consistency error and free energy
        3. Updates fields using optimizer (Anderson Mixing or ADAM)
        4. Optionally adjusts box size to minimize stress (if box_is_altering=True)

        The iteration continues until the error_level falls below tolerance or
        max_iter is reached.

        Parameters
        ----------
        initial_fields : dict
            Initial potential fields, {monomer_type: field_array}.
            Each field_array should have shape matching nx (can be flattened or
            shaped). Typical initialization: small random perturbations around zero.
        q_init : dict, optional
            Initial propagator values for grafted polymers (default: None).
            Format: {propagator_key: value_array}. Rarely used; for advanced
            applications like brush layers.
        on_iteration : callable, optional
            Callback function called after each iteration.
            Signature: on_iteration(info: IterationInfo) -> Optional[bool]
            If callback returns False, iteration stops early.
            Use for monitoring, checkpointing, or custom stopping criteria.
        return_result : bool, optional
            If True, return an SCFTResult object (default: False).
            For backward compatibility, results are also stored as attributes.

        Attributes Set
        --------------
        phi : dict
            Final monomer concentration fields {monomer_type: array}.
        w : ndarray
            Final potential fields, shape (M, total_grid).
        free_energy : float
            Final free energy per unit volume, βF/V.
        error_level : float
            Final convergence error (L2 norm of field residuals).
        iter : int
            Number of iterations performed.

        Notes
        -----
        **SCFT Equations:**

        The self-consistent field equations to be solved are:

        .. math::
            w_A(\\mathbf{r}) = \\sum_B \\chi_{AB} N_{\\text{ref}} \\phi_B(\\mathbf{r}) + \\xi(\\mathbf{r})

            \\sum_A \\phi_A(\\mathbf{r}) = 1

        where χ_AB is the Flory-Huggins parameter, N_ref is the reference
        polymerization index, and ξ(r) is a Lagrange multiplier field enforcing
        incompressibility.

        **Convergence Criterion:**

        Iteration stops when:

        .. math::
            \\text{error\\_level} = \\sqrt{\\frac{\\sum_i \\langle (w_{\\text{new},i} - w_{\\text{old},i})^2 \\rangle}{1 + \\sum_i \\langle w_{\\text{old},i}^2 \\rangle}} < \\text{tolerance}

        where <·> denotes spatial averaging and i ranges over monomer types.

        **Free Energy:**

        The free energy per volume is computed from the field-theoretic Hamiltonian:

        .. math::
            \\frac{\\beta F}{V} = -\\sum_p \\frac{\\phi_p}{\\alpha_p} \\ln Q_p + \\mathbf{w}^T \\mathbf{A}^{-1} \\mathbf{w}

        where p indexes polymer types, φ_p is volume fraction, α_p is relative
        chain length, Q_p is the single-chain partition function, and A is related
        to the interaction matrix.

        **Box Relaxation:**

        If box_is_altering=True, the code simultaneously optimizes box dimensions
        to minimize stress:

        .. math::
            \\sigma_{ij} = -\\frac{\\partial (\\beta F/V)}{\\partial L_{ij}}

        Box size updates are restricted to |ΔL_x / L_x| < 0.1 per iteration for
        stability.

        **Optimization Algorithms:**

        *Anderson Mixing:*
        - Accelerates convergence by mixing history of previous iterations
        - More robust for most polymer systems
        - Parameters: mix_min, mix_init, max_hist, start_error

        *ADAM:*
        - Adaptive learning rate optimization
        - Can be more stable for difficult cases
        - Parameters: lr (learning rate), gamma (decay factor)

        **Performance Tips:**

        - Start with small mix_min/mix_init (0.01-0.1) for stability
        - Use good initial guess (e.g., from similar parameters) for faster convergence
        - For box relaxation, start with fixed box, then restart with box_is_altering=True
        - CUDA is 10-20x faster than CPU for 2D/3D systems

        **Common Issues:**

        *Non-convergence:*
        - Reduce mixing parameters (mix_min, mix_init)
        - Try different optimizer (AM ↔ ADAM)
        - Improve initial guess
        - Check for phase coexistence (try different box sizes)

        *Negative concentrations:*
        - Reduce ds (contour step size)
        - Check chi_n parameters are reasonable
        - Indicates numerical instability

        *Slow convergence:*
        - Increase mixing parameters (if stable)
        - Use better initial guess
        - For symmetric systems, use space group symmetry

        See Also
        --------
        compute_concentrations : Compute φ for given w.
        save_results : Save converged results to file.
        Adam : ADAM optimizer details.

        Examples
        --------
        **Basic SCFT run:**

        >>> import numpy as np
        >>> from polymerfts import scft
        >>>
        >>> # Initialize (params defined earlier)
        >>> calc = scft.SCFT(params)
        >>>
        >>> # Random initial fields
        >>> nx_total = np.prod(params["nx"])
        >>> w_init = {
        ...     "A": 0.01 * np.random.normal(size=nx_total),
        ...     "B": 0.01 * np.random.normal(size=nx_total)
        ... }
        >>>
        >>> # Run SCFT
        >>> calc.run(initial_fields=w_init)
        >>>
        >>> # Check convergence
        >>> if calc.error_level < calc.tolerance:
        ...     print(f"Converged in {calc.iter} iterations")
        ...     print(f"Free energy: {calc.free_energy:.6f}")
        ... else:
        ...     print(f"Not converged after {calc.max_iter} iterations")

        **Restarting from previous results:**

        >>> # Load previous results
        >>> import scipy.io
        >>> data = scipy.io.loadmat("previous_run.mat")
        >>>
        >>> # Use previous fields as initial guess
        >>> w_restart = {
        ...     "A": data["w_A"].flatten(),
        ...     "B": data["w_B"].flatten()
        ... }
        >>>
        >>> # Run with new parameters
        >>> calc_new = scft.SCFT(params_new)
        >>> calc_new.run(initial_fields=w_restart)

        **Box relaxation workflow:**

        >>> # Step 1: Fixed box SCFT
        >>> params["box_is_altering"] = False
        >>> calc1 = scft.SCFT(params)
        >>> calc1.run(initial_fields=w_init)
        >>>
        >>> # Step 2: Box relaxation from converged state
        >>> params["box_is_altering"] = True
        >>> calc2 = scft.SCFT(params)
        >>> w_converged = {m: calc1.w[i] for i, m in enumerate(calc1.monomer_types)}
        >>> calc2.run(initial_fields=w_converged)
        >>> print(f"Optimized box: {calc2.prop_solver.get_lx()}")
        """

        # The number of components
        M = len(self.monomer_types)

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
        w = np.zeros([M, self.prop_solver.n_grid], dtype=np.float64)

        for i in range(M):
            w[i,:] = np.reshape(initial_fields[self.monomer_types[i]], self.prop_solver.n_grid)

        # Symmetrize initial fields if space group is set
        if self.sg is not None:
            w = self.sg.symmetrize(w)

        # Keep the level of field value
        for i in range(M):
            w[i] -= self.prop_solver.integral(w[i])/self.prop_solver.get_volume()
            
        # Iteration begins here
        for iter in range(1, self.max_iter+1):

            # Compute total concentration for each monomer type
            phi, _ = self.compute_concentrations(w)

            # # Scaling phi
            # for monomer_type in self.monomer_types:
            #     phi[monomer_type] *= self.phi_rescaling

            # Convert monomer fields to auxiliary fields
            w_aux = self.mpt.to_aux_fields(w)

            # Calculate the total energy
            total_partitions = [self.prop_solver.get_partition_function(p) for p in range(self.prop_solver.get_n_polymer_types())]
            energy_total = self.mpt.compute_hamiltonian(self.prop_solver._molecules, w_aux, total_partitions, include_const_term=False)

            # Calculate difference between current total density and target density
            phi_total = sum(phi[m] for m in self.monomer_types)
            phi_diff = phi_total - 1.0

            # Calculate self-consistency error (vectorized)
            # phi_array[i] = phi[monomer_types[i]], shape (M, n_grid)
            phi_array = np.array([phi[m] for m in self.monomer_types])
            w_diff = self.matrix_chi @ phi_array - self.matrix_p @ w

            # Keep the level of functional derivatives
            for i in range(M):
                w_diff[i] -= self.prop_solver.integral(w_diff[i])/self.prop_solver.get_volume()

            # error_level measures the "relative distance" between the input and output fields
            old_error_level = error_level
            error_level = 0.0
            error_normal = 1.0  # add 1.0 to prevent divergence
            for i in range(M):
                error_level += self.prop_solver.inner_product(w_diff[i],w_diff[i])
                error_normal += self.prop_solver.inner_product(w[i],w[i])
            error_level = np.sqrt(error_level/error_normal)

            # Print iteration # and error levels and check the mass conservation
            mass_error = self.prop_solver.integral(phi_diff)/self.prop_solver.get_volume()
            
            if (self.box_is_altering):
                # Calculate stress (only every stress_interval iterations to save time)
                # stress_interval > 1 can significantly speed up computation since
                # stress computation takes ~30% of iteration time
                #
                # Adaptive mode: compute stress more frequently when error is high,
                # less frequently when error is low (box is nearly converged)
                if self.stress_interval == "adaptive":
                    if error_level > 1e-2:
                        effective_stress_interval = 1
                    elif error_level > 1e-4:
                        effective_stress_interval = 2
                    else:
                        effective_stress_interval = 4
                else:
                    effective_stress_interval = self.stress_interval

                compute_stress_this_iter = (iter == 1) or (iter % effective_stress_interval == 0)

                if compute_stress_this_iter:
                    self.prop_solver.compute_stress()
                    stress_array = np.array(self.prop_solver.get_stress())
                    cached_stress_array = stress_array.copy()
                else:
                    # Reuse cached stress from previous computation
                    stress_array = cached_stress_array

                dim = self.prop_solver.dim

                # Only include stress components that are being optimized
                # Stress array layout differs by dimension:
                # 3D: [σ_xx, σ_yy, σ_zz, σ_xy, σ_xz, σ_yz]
                # 2D: [σ_xx, σ_yy, σ_xy, 0, 0, 0]
                # For 1D/2D, only include stress components for existing dimensions
                diagonal_stress = stress_array[0:dim]
                # For non-orthogonal systems, include off-diagonal stress if angles are being optimized
                angle_stress_indices = []
                if hasattr(self, 'angles_reduced_indices') and len(self.angles_reduced_indices) > 0:
                    # Mapping from angle index to stress array index
                    # 3D: α→σ_yz(5), β→σ_xz(4), γ→σ_xy(3)
                    # 2D: γ→σ_xy(2) (only γ matters in 2D)
                    if dim == 3:
                        angle_stress_map = {0: 5, 1: 4, 2: 3}
                    else:  # dim == 2
                        angle_stress_map = {2: 2}  # γ→σ_xy at index 2 for 2D
                    for i in self.angles_reduced_indices:
                        if i in angle_stress_map:
                            angle_stress_indices.append(angle_stress_map[i])
                off_diagonal_stress = stress_array[angle_stress_indices] if angle_stress_indices else np.array([])
                relevant_stress = np.concatenate([diagonal_stress, off_diagonal_stress])
                error_level += np.sqrt(np.sum(relevant_stress**2))

                print("%8d %12.3E " %
                (iter, mass_error), end=" [ ")
                for Q in total_partitions:
                    print("%13.7E " % (Q), end=" ")
                print("] %15.9f %15.7E " % (energy_total, error_level), end=" ")
                print("lx=[", ",".join(["%9.6f" % (x) for x in self.prop_solver.get_lx()]), "]", end="")
                # Also print angles if optimizing non-orthogonal system
                if hasattr(self, 'angles_reduced_indices') and len(self.angles_reduced_indices) > 0:
                    print(" angles=[", ",".join(["%7.2f" % (x) for x in self.prop_solver.get_angles_degrees()]), "]", end="")
                print()
            else:
                print("%8d %12.3E " % (iter, mass_error), end=" [ ")
                for Q in total_partitions:
                    print("%13.7E " % (Q), end=" ")
                print("] %15.9f %15.7E " % (energy_total, error_level))

            # Call progress callback if provided
            if on_iteration is not None:
                info = IterationInfo(
                    iteration=iter,
                    error_level=error_level,
                    energy=energy_total,
                    mass_error=mass_error,
                    partition_functions=total_partitions,
                    lx=list(self.prop_solver.get_lx()) if self.box_is_altering else None,
                    angles=list(self.prop_solver.get_angles_degrees()) if (
                        self.box_is_altering and hasattr(self, 'angles_reduced_indices')
                        and len(self.angles_reduced_indices) > 0
                    ) else None
                )
                callback_result = on_iteration(info)
                if callback_result is False:
                    logger.info(f"Iteration stopped by callback at iteration {iter}")
                    break

            # Conditions to end the iteration
            if error_level < self.tolerance:
                break

            # Convert to reduced basis for space group (if set)
            if self.sg is None:
                w_reduced = w
                w_diff_reduced = w_diff
            else:
                # Convert to reduced basis for Anderson mixing
                w_reduced = self.sg.to_reduced_basis(w)
                w_diff_reduced = self.sg.to_reduced_basis(w_diff)

                # Verify recovery: check that from_reduced_basis(to_reduced_basis(w)) ~ w
                w_recovered = self.sg.from_reduced_basis(w_reduced)
                recovery_error = np.max(np.abs(w - w_recovered))
                if recovery_error > 1e-10:
                    print(f"WARNING: Reduced basis recovery error = {recovery_error:.2e}")

            # Calculate new fields using simple and Anderson mixing
            if self.box_is_altering:
                # Stress components:
                # [0,1,2] = σ_xx, σ_yy, σ_zz (diagonal) - for box lengths
                # [3,4,5] = σ_xy, σ_xz, σ_yz (off-diagonal) - for angles
                dlx_full = -stress_array[0:3]  # Diagonal stress for lengths

                # For crystal systems with constraints (a=b or a=b=c), average the constrained stress components
                if hasattr(self, 'crystal_system'):
                    if self.crystal_system == "Hexagonal2D":
                        # 2D hexagonal: a = b, average σ_xx and σ_yy for the combined parameter
                        dlx_reduced = np.array([0.5 * (dlx_full[0] + dlx_full[1])])
                    elif self.crystal_system == "Tetragonal" or (self.crystal_system == "Hexagonal" and len(self.lx_reduced_indices) == 2):
                        # Tetragonal/3D Hexagonal: a = b ≠ c, average σ_xx and σ_yy for a, keep σ_zz for c
                        dlx_reduced = np.array([0.5 * (dlx_full[0] + dlx_full[1]), dlx_full[2]])
                    elif self.crystal_system == "Cubic":
                        # Cubic: a = b = c, average all three components
                        dlx_reduced = np.array([np.mean(dlx_full)])
                    else:
                        # No constraints: use stress directly for reduced indices
                        dlx_reduced = dlx_full[self.lx_reduced_indices]
                else:
                    dlx_reduced = dlx_full[self.lx_reduced_indices]

                # Map off-diagonal stress to angle gradients
                # For Monoclinic: σ_xz (index 4) drives β angle
                # For Triclinic: σ_xy→γ, σ_xz→β, σ_yz→α
                # Stress array layout differs by dimension:
                # 3D: [σ_xx, σ_yy, σ_zz, σ_xy, σ_xz, σ_yz]
                # 2D: [σ_xx, σ_yy, σ_xy, 0, 0, 0]
                # Note: Sign is POSITIVE (not negative like for lengths) because:
                # - Metric g_ij contains cos(γ) for off-diagonal terms
                # - d(cos(γ))/dγ = -sin(γ), which flips the sign
                # - So dγ = +σ_xy (not -σ_xy) for gradient descent
                if dim == 3:
                    angle_stress_map = {0: 5, 1: 4, 2: 3}
                else:  # dim == 2
                    angle_stress_map = {2: 2}  # γ→σ_xy at index 2 for 2D
                dangle = np.array([stress_array[angle_stress_map[i]] for i in self.angles_reduced_indices if i in angle_stress_map])

                # Current values
                current_lx = np.array(self.prop_solver.get_lx())[self.lx_reduced_indices]
                current_angles = np.array(self.prop_solver.get_angles_degrees())[self.angles_reduced_indices] if len(self.angles_reduced_indices) > 0 else np.array([])

                # Build optimization vectors
                am_current = np.concatenate((w_reduced.flatten(), current_lx, current_angles))
                am_diff = np.concatenate((w_diff_reduced.flatten(),
                                         self.scale_stress*dlx_reduced,
                                         self.scale_stress*dangle))
                am_new = self.field_optimizer.calculate_new_fields(am_current, am_diff, old_error_level, error_level)

                # Copy fields
                w_reduced = np.reshape(am_new[0:len(w_reduced.flatten())], (M, -1))

                # Extract new box lengths
                n_lx = len(self.lx_reduced_indices)
                n_angles = len(self.angles_reduced_indices)

                # Set box size
                # Restricting |dLx| to be less than 10 % of Lx
                old_lx = current_lx
                new_lx = np.array(am_new[-(n_lx + n_angles):-(n_angles)] if n_angles > 0 else am_new[-n_lx:])
                new_dlx = np.clip((new_lx-old_lx)/old_lx, -0.1, 0.1)
                new_lx = (1 + new_dlx)*old_lx

                # Set box angles (if optimizing angles)
                if n_angles > 0:
                    old_angles = current_angles
                    new_angles = np.array(am_new[-n_angles:])
                    # Restrict angle change to 5 degrees per step
                    new_dangle = np.clip(new_angles - old_angles, -5.0, 5.0)
                    new_angles = old_angles + new_dangle

                    # Build full angles array (in degrees)
                    full_angles = list(self.prop_solver.get_angles_degrees())
                    for i, idx in enumerate(self.angles_full_indices):
                        full_angles[idx] = new_angles[i]

                    # Update box with both lengths and angles
                    self.prop_solver.set_lattice_parameters(list(np.array(new_lx)[self.lx_full_indices]), full_angles)
                else:
                    self.prop_solver.set_lx(list(np.array(new_lx)[self.lx_full_indices]))

                # Update bond parameters using new lx
                self.prop_solver.update_laplacian_operator()
            else:
                w_reduced = self.field_optimizer.calculate_new_fields(w_reduced.flatten(), w_diff_reduced.flatten(), old_error_level, error_level)
                w_reduced = np.reshape(w_reduced, (M, -1))

            # Convert back from reduced basis to full grid
            if self.sg is None:
                w = w_reduced
            else:
                w = self.sg.from_reduced_basis(w_reduced)
            
            # Keep the level of field value
            for i in range(M):
                w[i] -= self.prop_solver.integral(w[i])/self.prop_solver.get_volume()

        # Print free energy as per chain expression
        print("Free energy per chain (for each chain type):")
        for p in range(self.prop_solver.get_n_polymer_types()):
            energy_total_per_chain = energy_total*self.prop_solver.get_polymer(p).get_alpha()/ \
                                                  self.prop_solver.get_polymer(p).get_volume_fraction()
            print("\tβF/n_%d : %12.7f" % (p, energy_total_per_chain))

        # Store phi and w
        self.phi = phi
        self.w = w

        # Store free energy
        self.free_energy = energy_total
        self.error_level = error_level
        self.iter = iter

        # Return SCFTResult if requested
        if return_result:
            return SCFTResult(
                converged=error_level < self.tolerance,
                free_energy=energy_total,
                error_level=error_level,
                iterations=iter,
                phi=phi,
                w=w,
                partition_functions=total_partitions,
                lx=list(self.prop_solver.get_lx()),
                nx=list(self.prop_solver.nx),
                monomer_types=self.monomer_types,
                params=self.params,
                stress=stress_array if self.box_is_altering else None,
                angles=list(self.prop_solver.get_angles_degrees()) if (
                    hasattr(self, 'angles_reduced_indices') and len(self.angles_reduced_indices) > 0
                ) else None
            )
        return None