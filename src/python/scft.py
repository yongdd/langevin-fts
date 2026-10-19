import os
import time
import numpy as np
import json
import yaml
import copy
import itertools
from pathlib import Path
from typing import Dict, Any, Optional, Callable, Union
import scipy.io

from dataclasses import dataclass
from typing import List

from . import _core
from .polymer_field_theory import SymmetricPolymerTheory
from .propagator_solver import PropagatorSolver
from .validation import validate_system_params, ValidationError
from .utils import (
    load_config,
    parse_chi_n,
    validate_polymers,
    process_polymer_blocks,
    process_random_copolymer,
    draw_polymer_architectures,
)
from .smearing import Smearing

# OpenMP environment variables
os.environ.setdefault("OMP_NUM_THREADS", "1")  # respect user override
os.environ.setdefault("OMP_STACKSIZE", "1G")

@dataclass
class SCFTResult:
    """Result object for SCFT simulations.

    Encapsulates all outputs from an SCFT simulation run, providing
    a clean interface for accessing results.

    Attributes
    ----------
    converged : bool
        Whether the simulation converged within tolerance.
    free_energy : float
        Final free energy per unit volume (dimensionless).
    error_level : float
        Final convergence error (L2 norm of field residuals).
    iterations : int
        Number of iterations performed.
    phi : dict
        Monomer concentration fields {monomer_type: array}.
    w : np.ndarray
        Potential fields, shape (M, total_grid).
    partition_functions : list
        Single-chain partition functions for each polymer type.
    stress : np.ndarray or None
        Stress tensor components (if box_is_altering was True).
    lx : list
        Final box dimensions.
    nx : list
        Grid dimensions.
    monomer_types : list
        List of monomer type labels.
    params : dict
        Original simulation parameters.
    """
    converged: bool
    free_energy: float
    error_level: float
    iterations: int
    phi: Dict[str, np.ndarray]
    w: np.ndarray
    partition_functions: List[float]
    lx: List[float]
    nx: List[int]
    monomer_types: List[str]
    params: Dict[str, Any]
    stress: Optional[np.ndarray] = None
    angles: Optional[List[float]] = None

    def get_concentration(self, monomer_type: str) -> np.ndarray:
        """Get concentration field for a monomer type.

        Parameters
        ----------
        monomer_type : str
            Monomer type label (e.g., "A", "B").

        Returns
        -------
        np.ndarray
            Concentration field for the specified monomer.

        Raises
        ------
        KeyError
            If monomer type not found.
        """
        return self.phi[monomer_type]

@dataclass
class IterationInfo:
    """Information about a single SCFT iteration.

    Used for progress callbacks to provide iteration details.

    Attributes
    ----------
    iteration : int
        Current iteration number.
    error_level : float
        Current convergence error.
    energy : float
        Current free energy.
    mass_error : float
        Mass conservation error.
    partition_functions : list
        Partition functions for each polymer.
    lx : list or None
        Current box dimensions (if box_is_altering).
    angles : list or None
        Current lattice angles (if non-orthogonal).
    """
    iteration: int
    error_level: float
    energy: float
    mass_error: float
    partition_functions: List[float]
    lx: Optional[List[float]] = None
    angles: Optional[List[float]] = None


# Which lattice parameters may vary during box relaxation (box_is_altering=True):
#   lx_reduced : indices of the independent box lengths among [a, b, c]
#   lx_full    : for each of a, b, c, the independent length it follows
#   angles     : indices of the free angles among [alpha, beta, gamma]
# "Orthorhombic" frees every box length in any dimension and is handled in the constructor.
# To add a crystal system, add one row here (and, if it needs averaged stress
# components like a = b, one case in SCFT._update_fields_and_box).
CRYSTAL_SYSTEMS = {
    "Orthorhombic": {"lx_reduced": None,      "lx_full": None,      "angles": []},
    "Tetragonal":   {"lx_reduced": [0, 2],    "lx_full": [0, 0, 1], "angles": []},         # a = b != c
    "Cubic":        {"lx_reduced": [0],       "lx_full": [0, 0, 0], "angles": []},         # a = b = c
    "Hexagonal":    {"lx_reduced": [0, 2],    "lx_full": [0, 0, 1], "angles": []},         # a = b != c, gamma = 120
    "Trigonal":     {"lx_reduced": [0, 2],    "lx_full": [0, 0, 1], "angles": []},         # hexagonal axes (space group only)
    "Monoclinic":   {"lx_reduced": [0, 1, 2], "lx_full": [0, 1, 2], "angles": [1]},        # beta free
    "Triclinic":    {"lx_reduced": [0, 1, 2], "lx_full": [0, 1, 2], "angles": [0, 1, 2]},  # all free
    "Hexagonal2D":  {"lx_reduced": [0],       "lx_full": [0, 0],    "angles": []},         # a = b, gamma = 120
    "Oblique2D":    {"lx_reduced": [0, 1],    "lx_full": [0, 1],    "angles": [2]},        # gamma free
}

class SCFT:
    """Self-consistent field theory (SCFT) of polymer melts and solutions.

    Finds the saddle point of the field-theoretic Hamiltonian with Anderson
    mixing for arbitrary polymer architectures (linear, branched, random
    copolymers, mixtures), optionally relaxing the box (and the angles of
    non-orthogonal cells) and using space-group symmetry.

    Parameters
    ----------
    params : dict
        Simulation parameters; every key is documented in
        docs/references/Parameters.md. Required: nx, lx, chain_model, ds,
        segment_lengths, chi_n, distinct_polymers, optimizer, box_is_altering.

    Attributes
    ----------
    prop_solver : PropagatorSolver
        Propagator computation (C++ objects inside).
    cb : ComputationBox
        Grid, box and integrals (C++).
    mpt : SymmetricPolymerTheory
        Monomer <-> auxiliary field transformations.
    phi, w, free_energy, error_level, iter
        Set by run(): concentrations {type: array}, fields (M, n_grid),
        free energy per volume, final error and iteration count.

    Examples
    --------
    >>> calc = SCFT(params)                       # params: see docs/references/Parameters.md
    >>> calc.run(initial_fields={"A": w_A, "B": w_B})
    >>> calc.save_results("fields.mat")
    >>> calc.free_energy, calc.phi["A"], calc.w[0]

    Complete scripts: examples/scft/ (Lamella3D.py, Gyroid.py, phases/).
    """
    @staticmethod
    def _apply_axis_ordering(params):
        """Reorder nx/lx/angles in ``params`` (in place) from the user's axis ordering
        to the standard [a, b, c]. Returns (axis_ordering, perm); both None when the
        option is absent. perm[i] is the position of standard axis i in the user's arrays.
        """
        axis_ordering = params.get("axis_ordering", None)
        if axis_ordering is None:
            return None, None
        dim = len(params["nx"])
        labels = ["a", "b", "c"][:dim]
        if len(axis_ordering) != dim:
            raise ValueError(f"axis_ordering length ({len(axis_ordering)}) must match dimension ({dim})")
        if sorted(axis_ordering) != labels:
            raise ValueError(f"axis_ordering must contain exactly {labels}, got {axis_ordering}")
        perm = [axis_ordering.index(label) for label in labels]
        params["nx"] = [params["nx"][perm[i]] for i in range(dim)]
        params["lx"] = [params["lx"][perm[i]] for i in range(dim)]
        angles = params.get("angles", None)
        if angles is not None and dim == 3:
            # angles = [alpha, beta, gamma] are the angles between axes (b,c), (a,c), (a,b);
            # the new angle between axes i and j is the old angle between perm[i] and perm[j]
            angle_index = {(1, 2): 0, (0, 2): 1, (0, 1): 2}
            params["angles"] = [angles[angle_index[tuple(sorted((perm[i], perm[j])))]]
                                for (i, j) in ((1, 2), (0, 2), (0, 1))]
        print(f"Axis ordering: {axis_ordering} -> reordered to [a, b, c]")
        print(f"  nx: {params['nx']}, lx: {params['lx']}")
        if params.get("angles", None) is not None:
            print(f"  angles: {params['angles']}")
        return axis_ordering, perm

    def __init__(self, params):

        # Validate input parameters
        validate_system_params(params)
        for key in ("optimizer", "box_is_altering"):
            if key not in params:
                raise ValidationError(f"Missing required parameter for SCFT: '{key}'")
        params = copy.deepcopy(params)
        params.pop("space_group_basis", None)

        # Handle axis ordering (convert user ordering to standard [a, b, c]);
        # this rewrites nx/lx/angles in `params`.
        self.axis_ordering, self._axis_perm = self._apply_axis_ordering(params)

        # Monomer types and statistical segment lengths
        self.monomer_types = sorted(list(params["segment_lengths"].keys()))
        self.segment_lengths = copy.deepcopy(params["segment_lengths"])
        self.distinct_polymers = copy.deepcopy(params["distinct_polymers"])

        # Flory-Huggins parameters chi N; pairs the user did not give do not interact
        self.chi_n = parse_chi_n(params["chi_n"], self.segment_lengths)
        for pair in itertools.combinations(self.monomer_types, 2):
            key = ",".join(sorted(pair))
            if key not in self.chi_n:
                self.chi_n[key] = 0.0

        # Compressibility parameter (None: incompressible)
        self.zeta_n = params.get("zeta_n", None)

        # Polymer chains: block connectivity, then random copolymers (one effective monomer type each)
        validate_polymers(self.distinct_polymers)
        for polymer in self.distinct_polymers:
            polymer["blocks_input"] = process_polymer_blocks(polymer)
        self.random_fraction = {}
        for polymer in self.distinct_polymers:
            result = process_random_copolymer(polymer, self.segment_lengths)
            if result is not None:
                random_type, effective_segment_length, fraction = result
                polymer["block_monomer_types"] = [random_type]
                self.segment_lengths[random_type] = effective_segment_length
                self.random_fraction[random_type] = fraction

        # Draw the chain architectures (polymer_<id>.png in the current directory)
        draw_polymer_architectures(self.distinct_polymers, params["ds"], self.segment_lengths)

        # Boundary conditions (default: periodic) and lattice angles [alpha, beta, gamma] in degrees
        bc = params.get("bc", None)
        if bc is None:
            bc = ["periodic"] * (2 * len(params["nx"]))
        angles = params.get("angles", None)

        # Multimonomer polymer field theory
        self.mpt = SymmetricPolymerTheory(self.monomer_types, self.chi_n, zeta_n=self.zeta_n)

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

        # Create space group object early (before PropagatorSolver) if specified
        if params.get("space_group", None) is not None:
            if "number" in params["space_group"]:
                self.sg = _core.SpaceGroup(params["nx"], params["space_group"]["symbol"], params["space_group"]["number"])
            else:
                self.sg = _core.SpaceGroup(params["nx"], params["space_group"]["symbol"])
            # Auto-enable physical basis when possible (m3 → pmmm → z-mirror).
            # NOTE: reuse the (possibly axis_ordering-reordered) local `angles`;
            # missing angles means an orthogonal cell for classification, while
            # `angles` itself stays None so the computation box is not
            # needlessly recreated below.
            try:
                alpha, beta, gamma = angles if angles is not None else (90.0, 90.0, 90.0)
            except Exception:
                alpha, beta, gamma = 90.0, 90.0, 90.0
            tol = 1e-8
            is_orthogonal = (abs(alpha - 90.0) < tol and abs(beta - 90.0) < tol and abs(gamma - 90.0) < tol)
            z_axis_orthogonal = (abs(alpha - 90.0) < tol and abs(beta - 90.0) < tol)

            if is_orthogonal:
                # Fall back down the basis hierarchy when a basis is not
                # applicable to this group/grid (the C++ side throws
                # RuntimeError). Other exception types indicate real bugs
                # and must propagate.
                try:
                    self.sg.enable_m3_physical_basis()
                except RuntimeError:
                    try:
                        self.sg.enable_pmmm_physical_basis()
                    except RuntimeError:
                        pass  # keep the irreducible (full-group) basis
            elif z_axis_orthogonal:
                # Hexagonal/trigonal (hexagonal-axes) space groups use a hybrid
                # orbit convention (node-centered x,y; cell-centered z), making
                # all in-plane symmetry maps exact on the grid. The z-mirror
                # physical basis (cell-centered z fold) remains valid.
                try:
                    self.sg.enable_z_mirror_physical_basis()
                except RuntimeError:
                    pass  # keep the irreducible (full-group) basis

        else:
            self.sg = None

        # Propagator solver: the C++ computation box, molecules and solver live inside it
        self.prop_solver = PropagatorSolver(
            nx=params["nx"], lx=params["lx"], ds=params["ds"],
            bond_lengths=self.segment_lengths, bc=bc,
            chain_model=params["chain_model"],
            numerical_method=params.get("numerical_method", None),   # None: "rqm4"
            platform=params.get("platform", "auto"),   # auto: CPU for 1D, CUDA for 2D/3D when available
            reduce_memory=params.get("reduce_memory", False),
            space_group=self.sg,
            angles=angles,
            aggregate_propagator_computation=params.get("aggregate_propagator_computation", True),
        )
        for polymer in self.distinct_polymers:
            self.prop_solver.add_polymer(polymer["volume_fraction"], polymer["blocks_input"],
                                         polymer.get("initial_conditions", None))

        # Computation box as a public attribute (this creates the C++ objects)
        self.cb = self.prop_solver.computation_box

        # Display factory info
        self.prop_solver.display_platform_info()

        # Initialize smearing (must be after self.cb is set)
        if params.get("smearing", None) is not None and self.sg is not None:
            # Smearing operates on full-grid fields, but with a space group the
            # fields live in the reduced basis; Smearing.apply would reshape a
            # reduced-basis array to the full grid and crash (or corrupt data).
            raise NotImplementedError(
                "'smearing' is not supported together with 'space_group'. "
                "Disable one of the two options.")
        self.smearing = Smearing(
            self.cb.get_nx(),
            self.cb.get_lx(),
            params.get("smearing", None)
        )

        # Scaling factor for stress when the fields and box size are simultaneously computed
        self.scale_stress = params.get("scale_stress", 1)

        # Which lattice parameters may vary during box relaxation (table CRYSTAL_SYSTEMS)
        dim = len(params["nx"])
        if self.sg is not None:
            # The space group fixes the crystal system; angles never vary
            crystal_system = self.sg.get_crystal_system()
            if not crystal_system in ["Orthorhombic", "Tetragonal", "Cubic", "Hexagonal", "Trigonal"]:
                raise ValueError("The crystal system of the space group must be Orthorhombic, Tetragonal, Cubic, Hexagonal, or Trigonal. " +
                    "The current crystal system is " + crystal_system + ".")
            n_field_var = len(self.monomer_types) * self.sg.get_n_reduced_basis_full()
        else:
            # The user chooses; a 2D cell without "crystal_system" is classified from gamma
            if params.get("crystal_system", None) is None and dim == 2 and "angles" in params:
                gamma = params["angles"][0] if len(params["angles"]) == 1 else params["angles"][2]
                if abs(gamma - 120.0) < 1e-6:
                    crystal_system = "Hexagonal2D"
                elif abs(gamma - 90.0) > 1e-6:
                    crystal_system = "Oblique2D"
                else:
                    crystal_system = "Orthorhombic"
            else:
                crystal_system = params.get("crystal_system", None) or "Orthorhombic"
            self.crystal_system = crystal_system

            if crystal_system in ["Monoclinic", "Triclinic", "Hexagonal"] and dim != 3:
                raise ValueError(f"Crystal system '{crystal_system}' requires 3D simulation (nx must have 3 elements).")
            if crystal_system not in CRYSTAL_SYSTEMS or crystal_system == "Trigonal":
                raise ValueError(f"Unknown crystal system: {crystal_system}. " +
                    "Choose from: Orthorhombic, Tetragonal, Cubic, Hexagonal, Monoclinic, Triclinic.")
            n_field_var = len(self.monomer_types) * np.prod(params["nx"])

        row = CRYSTAL_SYSTEMS[crystal_system]
        if crystal_system == "Orthorhombic":    # every box length is free, in any dimension
            self.lx_reduced_indices = list(range(dim))
            self.lx_full_indices = list(range(dim))
        else:
            self.lx_reduced_indices = list(row["lx_reduced"])
            self.lx_full_indices = list(row["lx_full"])
        self.angles_reduced_indices = [] if self.sg is not None else list(row["angles"])
        self.angles_full_indices = list(self.angles_reduced_indices)

        # Number of variables the optimizer adjusts: the fields, plus the free box lengths and angles
        if params["box_is_altering"]:
            n_var = n_field_var + len(self.lx_reduced_indices) + len(self.angles_reduced_indices)
        else:
            n_var = n_field_var

        # Enable off-diagonal (cross-term) stress computation when angle
        # optimization is active. Without this explicit opt-in, a run started
        # at exactly 90-degree angles would compute zero angle derivatives
        # (the orthogonal-box stress fast path skips the V12/V13/V23 cross
        # terms) and the angles would never move away from 90 degrees.
        if len(self.angles_reduced_indices) > 0:
            self.prop_solver.set_force_off_diagonal_stress(True)

        # Anderson mixing (C++ class) for the saddle-point iteration
        optimizer = params["optimizer"]
        if optimizer["name"] != "am":
            raise ValueError(f"Invalid optimizer name: {optimizer['name']}. Only 'am' (Anderson mixing) is available.")
        self.field_optimizer = self.prop_solver.create_anderson_mixing(n_var,
            optimizer["max_hist"],     # maximum number of history
            optimizer["start_error"],  # when switch to AM from simple mixing
            optimizer["mix_min"],      # minimum mixing rate of simple mixing
            optimizer["mix_init"])     # initial mixing rate of simple mixing

        print("---------- Simulation Parameters ----------")
        print("Platform :", self.prop_solver.platform)
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

        if self.zeta_n is not None:
            print("zeta_n: %f" % self.zeta_n)
        print("P matrix for field residuals:\n\t", str(self.matrix_p).replace("\n", "\n\t"))

        self.prop_solver.display_architectures()
        self.prop_solver.display_statistics()

        #  Save internal variables
        self.params = params
        self.chain_model = params["chain_model"]
        self.ds = params["ds"]
        self.box_is_altering = params["box_is_altering"]

        # Stress computation interval: an integer (every N iterations) or
        # "adaptive" (more often while the error is high, see run()).
        self.stress_interval = params.get("stress_interval", 1)

        self.max_iter = params.get("max_iter", 2000)       # maximum number of iterations
        self.tolerance = params.get("tolerance", 1e-8)     # stop when the self-consistency error is below this

        # FTS-consistent energy convention: pin the field means to the saddle point
        # so the SCFT free energy matches the field-theoretic Hamiltonian used in
        # L-FTS/CL-FTS (F_dis = -chi_N*(f-1/2)^2 for the disordered phase).
        # Off: mean = 0 convention (F_dis = 0 for the disordered phase).
        self.fts_consistent_energy = params.get("fts_consistent_energy", False)

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
        """
        params = load_config(path)
        params.update(overrides)
        return cls(params)

    def compute_concentrations(self, w, q_init=None):
        """Compute monomer concentration fields for given potential fields.

        NOTE: unlike LFTS/CLFTS.compute_concentrations (which take AUXILIARY
        fields), this method takes MONOMER potential fields directly.

        Solves the modified diffusion equations to compute chain propagators,
        then integrates over chain contours to obtain monomer concentration
        distributions. Handles both block copolymers and random copolymers.
        q_init optionally supplies initial propagator values for grafted
        polymers (passed through to PropagatorSolver.compute_propagators).

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
        """
        M = len(self.monomer_types)
        elapsed_time = {}

        # Field optimizer operates on irreducible grid when space group is set.
        if self.sg is not None:
            w = self.sg.irreducible_to_reduced(w)

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
        self.prop_solver.compute_propagators(w_input_for_propagator, q_init=q_init)
        self.prop_solver.compute_concentrations()
        elapsed_time["pseudo"] = time.time() - time_p_start

        # Partition-consistency guardrail (checked once, on the first call,
        # like LFTS): run() warns when "partition_ok" is below 0.5.
        if not getattr(self, "_partition_checked", False):
            self._partition_checked = True
            elapsed_time["partition_ok"] = \
                1.0 if self.prop_solver.check_total_partition() else 0.0

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

        # Convert concentrations back to irreducible grid for optimizer.
        if self.sg is not None:
            phi = {name: self.sg.reduced_to_irreducible(phi[name]) for name in phi}

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
            m_dic["space_group_symbol"] = self.sg.get_spacegroup_symbol()
            m_dic["space_group_hall_number"] = self.sg.get_hall_number()

        def _field_to_full(field: np.ndarray) -> np.ndarray:
            """Convert irreducible/reduced basis field to full grid if space group is set."""
            if self.sg is None:
                return field

            n_reduced = self.sg.get_n_reduced_basis()
            n_irreducible = self.sg.get_n_reduced_basis_full()
            field_use = field
            if n_reduced != n_irreducible and field.size == n_irreducible:
                field_use = self.sg.irreducible_to_reduced(field)
            return self.sg.from_reduced_basis(field_use)

        # Add w fields to the dictionary (expand to full grid if space group is set)
        for i, name in enumerate(self.monomer_types):
            m_dic["w_" + name] = _field_to_full(self.w[i])

        # Add concentrations to the dictionary (expand to full grid if space group is set)
        for name in self.monomer_types:
            m_dic["phi_" + name] = _field_to_full(self.phi[name])

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

    # =========================================================================
    # run() helpers (used more than once, or long enough to stand on their own)
    # =========================================================================

    def _pin_field_means(self, w, chi_phi_mean, xi=0.0):
        """Normalize the field means in place (mean() works in the reduced basis
        when a space group is set).

        fts_consistent_energy: pin the means to the saddle point,
        mean(w_i) = sum_j chi_ij mean(phi_j) + xi. For incompressible systems only the
        P-projected mean is pinned; mean(W+) is a gauge freedom that P annihilates.
        Otherwise (mean = 0 convention) incompressible fields are shifted to zero
        mean and compressible fields are left as they are.
        """
        M = len(self.monomer_types)
        if self.fts_consistent_energy:
            if self.zeta_n is None:
                p_target = self.matrix_p @ chi_phi_mean
                w_mean = np.array([self.prop_solver.mean(w[i]) for i in range(M)])
                p_current = self.matrix_p @ w_mean
                shift = p_target - p_current
                for i in range(M):
                    w[i] += shift[i]
            else:
                for i in range(M):
                    w[i] += chi_phi_mean[i] + xi - self.prop_solver.mean(w[i])
        elif self.zeta_n is None:
            for i in range(M):
                w[i] -= self.prop_solver.mean(w[i])

    def _update_fields_and_box(self, w, w_diff, stress_array, old_error_level, error_level):
        """One optimizer step on the concatenated (fields, box lengths, angles)
        vector. Applies the new lattice parameters to the solver and returns the
        new fields. Length changes are limited to 10 % and angle changes to
        5 degrees per step.
        """
        M = len(self.monomer_types)
        dim = self.prop_solver.dim

        # -dH/dL for the independent box lengths; lengths tied by the crystal
        # system (a = b, or a = b = c) use the averaged derivative
        dlx_full = -stress_array[0:3]
        crystal_system = getattr(self, 'crystal_system', None)   # not set when a space group fixes the system
        if crystal_system == "Hexagonal2D":
            dlx_reduced = np.array([0.5 * (dlx_full[0] + dlx_full[1])])
        elif crystal_system == "Tetragonal" or (crystal_system == "Hexagonal" and len(self.lx_reduced_indices) == 2):
            dlx_reduced = np.array([0.5 * (dlx_full[0] + dlx_full[1]), dlx_full[2]])
        elif crystal_system == "Cubic":
            dlx_reduced = np.array([np.mean(dlx_full)])
        else:
            dlx_reduced = dlx_full[self.lx_reduced_indices]

        # Stress array layout: 3D [dH/dL1, dH/dL2, dH/dL3, dH/dgamma, dH/dbeta, dH/dalpha],
        # 2D [dH/dL1, dH/dL2, dH/dgamma, 0, 0, 0]. get_stress() returns +dH/d(parameter)
        # for angles as well as lengths (the d(cos gamma)/d gamma = -sin gamma factor is
        # already applied in the C++ compute_stress post-processing), so gradient
        # descent uses the same negative sign as for the box lengths.
        angle_stress_map = {0: 5, 1: 4, 2: 3} if dim == 3 else {2: 2}
        dangle = -np.array([stress_array[angle_stress_map[i]] for i in self.angles_reduced_indices if i in angle_stress_map])

        current_lx = np.array(self.prop_solver.get_lx())[self.lx_reduced_indices]
        current_angles = np.array(self.prop_solver.get_angles_degrees())[self.angles_reduced_indices] if len(self.angles_reduced_indices) > 0 else np.array([])

        am_current = np.concatenate((w.flatten(), current_lx, current_angles))
        am_diff = np.concatenate((w_diff.flatten(),
                                 self.scale_stress*dlx_reduced,
                                 self.scale_stress*dangle))
        am_new = self.field_optimizer.calculate_new_fields(am_current, am_diff, old_error_level, error_level)

        w = np.reshape(am_new[0:len(w.flatten())], (M, -1))

        n_lx = len(self.lx_reduced_indices)
        n_angles = len(self.angles_reduced_indices)

        # Restricting |dLx| to be less than 10 % of Lx
        old_lx = current_lx
        new_lx = np.array(am_new[-(n_lx + n_angles):-(n_angles)] if n_angles > 0 else am_new[-n_lx:])
        new_dlx = np.clip((new_lx-old_lx)/old_lx, -0.1, 0.1)
        new_lx = (1 + new_dlx)*old_lx

        if n_angles > 0:
            old_angles = current_angles
            new_angles = np.array(am_new[-n_angles:])
            new_dangle = np.clip(new_angles - old_angles, -5.0, 5.0)
            new_angles = old_angles + new_dangle

            full_angles = list(self.prop_solver.get_angles_degrees())
            for i, idx in enumerate(self.angles_full_indices):
                full_angles[idx] = new_angles[i]
            self.prop_solver.set_lattice_parameters(list(np.array(new_lx)[self.lx_full_indices]), full_angles)
        else:
            self.prop_solver.set_lx(list(np.array(new_lx)[self.lx_full_indices]))

        # Update bond parameters using new lx
        self.prop_solver.update_laplacian_operator()
        return w

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
        3. Updates fields using optimizer (Anderson mixing)
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
        """

        M = len(self.monomer_types)

        # Assign large initial value for the energy and error
        energy_total = 1.0e20
        error_level = 1.0e20

        # Reset optimizer
        self.field_optimizer.reset_count()

        print("---------- Run ----------")
        print("iteration, mass error, total_partitions, energy_total, error_level", end="")
        print(", box size" if self.box_is_altering else "")

        # ---- 1. Initial fields -> working array of shape (M, n_grid) ----
        # With a space group the fields are symmetrized and converted to the irreducible basis.
        if self.sg is not None:
            w_full = np.zeros([M, self.prop_solver.total_grid], dtype=np.float64)
            for i in range(M):
                w_full[i,:] = np.reshape(initial_fields[self.monomer_types[i]], self.prop_solver.total_grid)
            w_full = self.sg.symmetrize(w_full)
            w = self.sg.to_reduced_basis(w_full)
            try:
                w = self.sg.reduced_to_irreducible(w)
            except Exception:
                pass
        else:
            w = np.zeros([M, self.prop_solver.n_grid], dtype=np.float64)
            for i in range(M):
                w[i,:] = np.reshape(initial_fields[self.monomer_types[i]], self.prop_solver.n_grid)

        # Normalize the initial field means. With fts_consistent_energy the target is
        # chi N . mean(phi) from the chain composition: mean(phi_i) = sum_p f_p sum_{blocks of type i} length
        if self.fts_consistent_energy:
            phi_mean = np.zeros(M)
            for polymer in self.distinct_polymers:
                for block in polymer["blocks"]:
                    phi_mean[self.monomer_types.index(block["type"])] += polymer["volume_fraction"] * block["length"]
            self._pin_field_means(w, self.matrix_chi @ phi_mean)
        else:
            self._pin_field_means(w, None)

        # ---- 2. Iteration ----
        # Stress array layout: 3D [dH/dL1, dH/dL2, dH/dL3, dH/dgamma, dH/dbeta, dH/dalpha],
        # 2D [dH/dL1, dH/dL2, dH/dgamma, 0, 0, 0]; the angle index in [alpha, beta, gamma]
        # maps to the stress index through angle_stress_map
        dim = self.prop_solver.dim
        angle_stress_map = {0: 5, 1: 4, 2: 3} if dim == 3 else {2: 2}
        angle_stress_indices = [angle_stress_map[i] for i in self.angles_reduced_indices if i in angle_stress_map]
        stress_array = None
        time_iter_start = time.time()
        for iter in range(1, self.max_iter+1):

            # Concentrations for the current fields
            phi, runtime_info = self.compute_concentrations(w, q_init=q_init)
            if runtime_info.get("partition_ok", 1.0) < 0.5:
                print(f"Warning: partition consistency check reported a mismatch at iteration {iter}.")

            # Total energy from the auxiliary fields and the partition functions
            w_aux = self.mpt.to_aux_fields(w)
            total_partitions = [self.prop_solver.get_partition_function(p) for p in range(self.prop_solver.get_n_polymer_types())]
            if self.sg is not None and self.sg.get_n_reduced_basis_full() != self.sg.get_n_reduced_basis():
                w_aux_energy = self.sg.irreducible_to_reduced(w_aux)
            else:
                w_aux_energy = w_aux
            energy_total = self.mpt.compute_hamiltonian(
                self.prop_solver.molecules,
                w_aux_energy,
                total_partitions,
                self.cb,
                include_const_term=False
            )

            # Deviation of the total density from 1
            phi_total = sum(phi[m] for m in self.monomer_types)
            phi_diff = phi_total - 1.0

            # Self-consistency residual, phi_array[i] = phi[monomer_types[i]], shape (M, n_grid)
            phi_smeared = self.smearing.apply_to_dict(phi)
            phi_array = np.array([phi_smeared[m] for m in self.monomer_types])
            w_diff = self.matrix_chi @ phi_array - self.matrix_p @ w

            # Keep the level of functional derivatives
            for i in range(M):
                w_diff[i] -= self.prop_solver.mean(w_diff[i])

            # Error level: relative distance between the input and output fields,
            # sqrt( sum_i <w_diff_i^2> / (1 + sum_i <w_i^2>) )
            old_error_level = error_level
            error_level = 0.0
            error_normal = 1.0  # add 1.0 to prevent divergence
            for i in range(M):
                error_level += self.prop_solver.inner_product(w_diff[i], w_diff[i])
                error_normal += self.prop_solver.inner_product(w[i], w[i])
            error_level = np.sqrt(error_level/error_normal)
            mass_error = self.prop_solver.mean(phi_diff)

            if self.box_is_altering:
                # The stress costs ~30 % of an iteration: compute it every stress_interval
                # iterations ("adaptive": every 1, 2 or 4 as the error drops) and reuse it in between
                if self.stress_interval == "adaptive":
                    stress_interval = 1 if error_level > 1e-2 else (2 if error_level > 1e-4 else 4)
                else:
                    stress_interval = self.stress_interval
                if (iter == 1) or (iter % stress_interval == 0):
                    self.prop_solver.compute_stress()
                    stress_array = np.array(self.prop_solver.get_stress())
                # The stress components being optimized (box lengths of the existing
                # dimensions and, if active, the angles) count towards the error
                relevant_stress = np.concatenate([stress_array[0:dim],
                                                  stress_array[angle_stress_indices] if angle_stress_indices else np.array([])])
                error_level += np.sqrt(np.sum(relevant_stress**2))

            # Print this iteration
            print("%8d %12.3E " % (iter, mass_error), end=" [ ")
            for Q in total_partitions:
                print("%13.7E " % (Q), end=" ")
            if self.box_is_altering:
                print("] %15.9f %15.7E " % (energy_total, error_level), end=" ")
                print("lx=[", ",".join(["%10.7f" % (x) for x in self.prop_solver.get_lx()]), "]", end="")
                if len(self.angles_reduced_indices) > 0:
                    print(" angles=[", ",".join(["%7.2f" % (x) for x in self.prop_solver.get_angles_degrees()]), "]", end="")
                print()
            else:
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
                        self.box_is_altering and len(self.angles_reduced_indices) > 0
                    ) else None
                )
                callback_result = on_iteration(info)
                if callback_result is False:
                    print(f"Iteration stopped by callback at iteration {iter}")
                    break

            # Conditions to end the iteration
            if error_level < self.tolerance:
                break

            # Calculate new fields using simple and Anderson mixing
            # Note: w and w_diff are in irreducible basis when space group is set
            if self.box_is_altering:
                w = self._update_fields_and_box(w, w_diff, stress_array, old_error_level, error_level)
            else:
                w = self.field_optimizer.calculate_new_fields(w.flatten(), w_diff.flatten(), old_error_level, error_level)
                w = np.reshape(w, (M, -1))

            # Normalize field means
            if self.fts_consistent_energy:
                # mean(φ_i) = f_i exactly by chain normalization, so the target is constant
                chi_phi_mean = np.array([self.prop_solver.mean((self.matrix_chi @ phi_array)[i]) for i in range(M)])
                xi = 0.0 if self.zeta_n is None else self.zeta_n * (self.prop_solver.mean(phi_total) - 1.0)
                self._pin_field_means(w, chi_phi_mean, xi)
            else:
                self._pin_field_means(w, None)

            # Irreducible grid remains valid without extra symmetrization.

        # Print iteration timing
        time_iter_total = time.time() - time_iter_start
        print("Iteration time: %.3f seconds (%.4f sec/iter, %d iterations)" % (time_iter_total, time_iter_total/iter, iter))

        # Print free energy as per chain expression
        print("Free energy per chain (for each chain type):")
        for p in range(self.prop_solver.get_n_polymer_types()):
            energy_total_per_chain = energy_total*self.prop_solver.get_polymer(p).get_alpha()/ \
                                                  self.prop_solver.get_polymer(p).get_volume_fraction()
            print("\tβF/n_%d : %12.7f" % (p, energy_total_per_chain))

        # Final partition consistency check (only once at the end)
        if not self.prop_solver.check_total_partition():
            print("Warning: Final partition consistency check failed.")

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
                    len(self.angles_reduced_indices) > 0
                ) else None
            )
        return None
