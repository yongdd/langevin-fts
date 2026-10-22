import os
import time
import numpy as np
import json
import yaml
import copy
import itertools
from pathlib import Path
from typing import Dict, Any, Optional, Callable, Union, List
import scipy.io

from dataclasses import dataclass

from . import _core
from .polymer_field_theory import SymmetricPolymerTheory
from .propagator_solver import PropagatorSolver
from .validation import validate_system_params, ValidationError
from .utils import (
    load_config,
    parse_chi_n,
    process_polymer_blocks,
    process_random_copolymer,
    draw_polymer_architectures,
    simulation_metadata,
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
        Free energy relative to the homogeneous state at the same composition,
        in units of rho_0 V / N. Zero for a disordered bulk state.
    error_level : float
        Final convergence error (L2 norm of field residuals).
    iterations : int
        Number of iterations performed.
    phi : dict
        Monomer concentration fields {monomer_type: array} on the full grid
        (expanded from the reduced basis when a space group is used).
    w : dict
        Potential fields {monomer_type: array} on the full grid. They can be
        passed back to SCFT.run as initial_fields, for example to continue a
        parameter sweep from the previous solution.
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
    free_energy_homogeneous : float
        Full homogeneous Flory-Huggins value at the same composition.
    free_energy_absolute : float
        Total canonical Helmholtz free energy, beta F_total / (rho_0 V / N),
        including ideal mixing entropy. Equals free_energy + free_energy_homogeneous.
    chemical_potentials : list or None
        beta mu_p per polymer molecule, with mean(xi) = 0 for incompressible
        systems. Not available for grafted chains or supplied initial propagators.
    grand_potential : float or None
        free_energy_absolute - sum_p chemical_potentials[p] * phi_p / alpha_p, in the
        same pressure convention. Use coexistence_quantities() for phase comparisons.
    polymer_lengths, polymer_volume_fractions : list or None
        Relative chain lengths alpha_p and polymer volume fractions phi_p.
    """
    converged: bool
    free_energy: float
    error_level: float
    iterations: int
    phi: Dict[str, np.ndarray]
    w: Dict[str, np.ndarray]
    partition_functions: List[float]
    lx: List[float]
    nx: List[int]
    monomer_types: List[str]
    params: Dict[str, Any]
    stress: Optional[np.ndarray] = None
    angles: Optional[List[float]] = None
    free_energy_homogeneous: Optional[float] = None
    chemical_potentials: Optional[List[float]] = None
    grand_potential: Optional[float] = None
    polymer_lengths: Optional[List[float]] = None
    polymer_volume_fractions: Optional[List[float]] = None
    free_energy_absolute: Optional[float] = None

    def coexistence_quantities(self, reference_polymer=0):
        """Return exchange chemical potentials and the reduced grand potential.

        Requires a converged, incompressible, periodic bulk solution. The
        reference polymer's chemical potential is set to zero by a pressure shift.
        """
        from .phase_equilibrium import coexistence_quantities
        return coexistence_quantities(self, reference_polymer)

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
    phi, w, free_energy, free_energy_absolute, error_level, iter
        Set by run(): concentrations {type: array}, fields (M, n_grid),
        relative and full CE free energies, final error and iteration count.

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
        """Reorder nx/lx/bc/angles in ``params`` (in place) from the user's axis ordering
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
        bc = params.get("bc", None)
        if bc is not None and len(bc) == 2 * dim:
            # bc = [low, high] of each axis in the order of nx
            params["bc"] = [bc[2 * perm[i] + side] for i in range(dim) for side in (0, 1)]
        angles = params.get("angles", None)
        if angles is not None and dim == 3:
            # angles = [alpha, beta, gamma] are the angles between axes (b,c), (a,c), (a,b);
            # the new angle between axes i and j is the old angle between perm[i] and perm[j]
            angle_index = {(1, 2): 0, (0, 2): 1, (0, 1): 2}
            params["angles"] = [angles[angle_index[tuple(sorted((perm[i], perm[j])))]]
                                for (i, j) in ((1, 2), (0, 2), (0, 1))]
        print(f"Axis ordering: {axis_ordering} -> reordered to [a, b, c]")
        print(f"  nx: {params['nx']}, lx: {params['lx']}")
        if params.get("bc", None) is not None:
            print(f"  bc: {params['bc']}")
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

        # Handle axis ordering (convert user ordering to standard [a, b, c]); this
        # rewrites nx/lx/bc/angles in `params`, so the key is dropped from the copy that
        # save_results() writes as "initial_params" (it must not be applied twice)
        self.axis_ordering, self._axis_perm = self._apply_axis_ordering(params)
        params.pop("axis_ordering", None)

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
        if self.zeta_n is not None and self.zeta_n < 0:
            raise ValidationError(f"zeta_n must not be negative, got {self.zeta_n}.")

        # Polymer chains: block connectivity, then random copolymers (one effective monomer type each)
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
        if angles is not None and len(params["nx"]) == 3 and len(angles) != 3:
            raise ValidationError("angles must be [alpha, beta, gamma] (three entries) for a 3D box.")
        if angles is not None and len(params["nx"]) < 3:
            dim = len(params["nx"])
            if len(angles) not in (0, 1, 3):
                raise ValidationError("angles must have 0, 1 or 3 entries for a 1D or 2D box.")
            fixed = angles if dim == 1 else (angles[:2] if len(angles) == 3 else [])
            if not np.all(np.isclose(fixed, 90.0, rtol=0.0, atol=1e-10)):
                raise ValidationError("A 1D box requires all angles to be 90 degrees; "
                                      "a 2D box requires alpha and beta to be 90 degrees.")

        # Multimonomer polymer field theory
        self.mpt = SymmetricPolymerTheory(self.monomer_types, self.chi_n, zeta_n=self.zeta_n)

        # χN matrix
        M = len(self.monomer_types)
        matrix_chi = np.zeros((M,M))
        for i in range(M):
            for j in range(i+1,M):
                key = self.monomer_types[i] + "," + self.monomer_types[j]
                if key in self.chi_n:
                    matrix_chi[i,j] = self.chi_n[key]
                    matrix_chi[j,i] = self.chi_n[key]
        
        self.matrix_chi = matrix_chi
        # Incompressible SCFT needs at least two monomer types. A singular χN (a zero pair in a
        # three-monomer system, chemically identical monomers) is fine, and so is 1ᵀ(χN)⁻¹1 = 0,
        # where the auxiliary field of a zero eigenvalue of the projected χN has a linear term:
        # it acts as a uniform field, and the free energy is computed from the monomer fields
        if self.zeta_n is None and M < 2:
            raise ValidationError("Incompressible SCFT needs at least two monomer types.")

        # Field residual (see the run loop): the composition part is projected with
        # P⊥ = I - eeᵀ/M, and the pressure part is scaled by κ = max(max|λ|, 1)/2 with λ
        # the eigenvalues of P⊥ χN P⊥, the size of the composition-pressure coupling
        # (χN/2 for AB)
        self.matrix_q_perp = np.identity(M) - np.ones((M, M))/M
        projected_chi = self.matrix_q_perp @ matrix_chi @ self.matrix_q_perp
        self.pressure_kappa = max(np.max(np.abs(np.linalg.eigvalsh(projected_chi))), 1.0)/2

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
                # (the m3 basis needs the recursive 3m CrysFFT: even grid, nz >= 16
                # and a compatible translation parity)
                try:
                    if not self.sg.m3_crysfft_feasible():
                        raise RuntimeError("recursive 3m CrysFFT is not feasible on this grid")
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

            # The cell must belong to the crystal system of the space group. Otherwise
            # the symmetry operations are not isometries of the box and the run would
            # silently converge to a constrained, unphysical state.
            crystal_system_sg = self.sg.get_crystal_system()
            lx_sg = [float(v) for v in params["lx"]]
            angles_sg = [float(v) for v in angles] if angles is not None else [90.0, 90.0, 90.0]
            same = lambda a, b: abs(a - b) <= 1e-6 * max(abs(a), abs(b), 1.0)
            cell_ok, need = True, ""
            if crystal_system_sg == "Cubic":
                cell_ok = same(lx_sg[0], lx_sg[1]) and same(lx_sg[0], lx_sg[2]) and is_orthogonal
                need = "lx[0] = lx[1] = lx[2] and angles [90, 90, 90]"
            elif crystal_system_sg == "Tetragonal":
                cell_ok = same(lx_sg[0], lx_sg[1]) and is_orthogonal
                need = "lx[0] = lx[1] and angles [90, 90, 90]"
            elif crystal_system_sg == "Orthorhombic":
                cell_ok = is_orthogonal
                need = "angles [90, 90, 90]"
            elif crystal_system_sg == "Hexagonal":
                cell_ok = same(lx_sg[0], lx_sg[1]) and z_axis_orthogonal and same(angles_sg[2], 120.0)
                need = "lx[0] = lx[1] and angles [90, 90, 120]"
            elif crystal_system_sg == "Trigonal":
                # the axes setting is a property of the group (P and hexagonal-axes R
                # settings: z-block-diagonal rotations; rhombohedral-axes R settings:
                # 3-fold about [111]), not of the cell
                if self.sg.uses_hexagonal_axes():
                    cell_ok = same(lx_sg[0], lx_sg[1]) and z_axis_orthogonal and same(angles_sg[2], 120.0)
                    need = "lx[0] = lx[1] and angles [90, 90, 120] (hexagonal-axes setting)"
                else:
                    cell_ok = (same(lx_sg[0], lx_sg[1]) and same(lx_sg[0], lx_sg[2])
                               and same(angles_sg[0], angles_sg[1]) and same(angles_sg[0], angles_sg[2]))
                    need = "lx[0] = lx[1] = lx[2] and alpha = beta = gamma (rhombohedral-axes setting)"
                    if cell_ok and params.get("box_is_altering", False):
                        # a = b = c and alpha = beta = gamma would have to vary together
                        raise ValidationError("Box relaxation is not supported for rhombohedral-axes settings; "
                                              "use the hexagonal-axes setting of the space group.")
            if not cell_ok:
                raise ValidationError(f"The cell does not match the {crystal_system_sg.lower()} crystal system of "
                                      f"space group {params['space_group']['symbol']}: it needs {need}. "
                                      f"Got lx = {lx_sg}, angles = {angles_sg}.")

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
            params.get("smearing", None),
            angles=self.prop_solver.get_angles_degrees(),
            bc=self.prop_solver.bc
        )

        # Scaling factor for stress when the fields and box size are simultaneously computed
        self.scale_stress = params.get("scale_stress", 1)

        # Which lattice parameters may vary during box relaxation (table CRYSTAL_SYSTEMS)
        dim = len(params["nx"])
        if self.sg is not None:
            # The space group fixes the crystal system; angles never vary
            crystal_system = self.sg.get_crystal_system()
            if crystal_system not in ["Orthorhombic", "Tetragonal", "Cubic", "Hexagonal", "Trigonal"]:
                raise ValueError("The crystal system of the space group must be Orthorhombic, Tetragonal, Cubic, Hexagonal, or Trigonal. " +
                    "The current crystal system is " + crystal_system + ".")
            n_field_var = len(self.monomer_types) * self.sg.get_n_reduced_basis_full()
        else:
            # The user chooses; a 2D cell without "crystal_system" is classified from gamma
            if params.get("crystal_system", None) is None and dim == 2 and angles is not None and len(angles) > 0:
                gamma = angles[0] if len(angles) == 1 else angles[2]
                if abs(gamma - 120.0) < 1e-6:
                    crystal_system = "Hexagonal2D"
                elif abs(gamma - 90.0) > 1e-6:
                    crystal_system = "Oblique2D"
                else:
                    crystal_system = "Orthorhombic"
            else:
                crystal_system = params.get("crystal_system", None) or "Orthorhombic"

            valid_systems = {1: ["Orthorhombic"],
                             2: ["Orthorhombic", "Hexagonal2D", "Oblique2D"],
                             3: ["Orthorhombic", "Tetragonal", "Cubic", "Hexagonal", "Monoclinic", "Triclinic"]}[dim]
            if crystal_system not in valid_systems:
                raise ValueError(f"Crystal system '{crystal_system}' is not valid for a {dim}D box. " +
                    "Choose from: " + ", ".join(valid_systems) + ".")
            n_field_var = len(self.monomer_types) * np.prod(params["nx"])
        self.crystal_system = crystal_system

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
        if n_var > _core.MAX_GRID_ARRAY:
            raise ValidationError(
                f"Anderson mixing would adjust {n_var} variables ({len(self.monomer_types)} fields of "
                f"{n_field_var // len(self.monomer_types)} values); at most {_core.MAX_GRID_ARRAY} are supported "
                "(32-bit indices). Use fewer grid points or a space group.")
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
        print("Pressure scale of the field residual, κ: %f" % self.pressure_kappa)

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
        if self.max_iter < 1:
            raise ValidationError(f"'max_iter' must be at least 1, got {self.max_iter}")
        self.tolerance = params.get("tolerance", 1e-8)     # stop when the self-consistency error is below this

        # The fields of an incompressible run have zero means (the free energy, computed from
        # the monomer fields, does not depend on them); fts_consistent_energy, which pinned
        # them to the saddle point, has no effect any more
        if "fts_consistent_energy" in params:
            print("Warning: 'fts_consistent_energy' has no effect: the free energy does not depend on the "
                  "field means, and incompressible fields have zero means.")

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

        # Partition consistency check (forward vs backward propagators), once per run
        if not getattr(self, "_partition_checked", False):
            self._partition_checked = True
            if not self.prop_solver.check_total_partition():
                raise ValidationError("Partition consistency check failed (forward/backward mismatch).")

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

    def _to_standard_axes(self, field):
        """Transpose a full-grid array given in the user's axis order (axis_ordering) to [a, b, c]."""
        user_nx = [0] * len(self._axis_perm)
        for i, axis in enumerate(self._axis_perm):
            user_nx[axis] = self.prop_solver.nx[i]
        return np.ravel(np.transpose(np.reshape(np.asarray(field), user_nx), self._axis_perm))

    def _to_full_grid(self, field: np.ndarray) -> np.ndarray:
        """Expand a field of the working basis to the full grid (a copy without a space group)."""
        if self.sg is None:
            return np.array(field, dtype=np.float64)
        # self.w and self.phi are stored in the irreducible basis. With a physical
        # basis the reduced ordering differs even when the sizes coincide (mmm
        # orthorhombic groups with the m3 basis), so the conversion is gated on
        # the basis, not on the sizes.
        if self.sg.physical_basis_name() != "irreducible":
            field = self.sg.irreducible_to_reduced(field)
        return self.sg.from_reduced_basis(field)

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
        metadata = simulation_metadata(self)
        metadata.update(execution_status="converged" if self.error_level < self.tolerance else "not_converged",
                        iterations=self.iter)
        m_dic["simulation_metadata_json"] = json.dumps(metadata)

        m_dic.update(free_energy=self.free_energy,
                     free_energy_absolute=self.free_energy_absolute,
                     free_energy_homogeneous=self.free_energy_homogeneous,
                     partition_functions=self.partition_functions,
                     polymer_lengths=self.polymer_lengths,
                     polymer_volume_fractions=self.polymer_volume_fractions)
        if self.chemical_potentials is not None:
            m_dic["chemical_potentials"] = self.chemical_potentials
            m_dic["grand_potential"] = self.grand_potential

        if self.sg is not None:
            m_dic["space_group_symbol"] = self.sg.get_spacegroup_symbol()
            m_dic["space_group_hall_number"] = self.sg.get_hall_number()

        # Add w fields to the dictionary (expand to full grid if space group is set)
        for i, name in enumerate(self.monomer_types):
            m_dic["w_" + name] = self._to_full_grid(self.w[i])

        # Add concentrations to the dictionary (expand to full grid if space group is set)
        for name in self.monomer_types:
            m_dic["phi_" + name] = self._to_full_grid(self.phi[name])

        # Convert nested parameters too, without modifying self.params.
        def plain(value):
            if isinstance(value, np.ndarray):
                return plain(value.tolist())
            if isinstance(value, np.generic):
                return plain(value.item())
            if isinstance(value, dict):
                return {plain(key): plain(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [plain(item) for item in value]
            return value

        m_dic = plain(m_dic)

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
                f.write(yaml.safe_dump(m_dic, default_flow_style=None, width=90, sort_keys=False))
        else:
            raise ValueError("Invalid output file extension. Use .mat, .json, or .yaml.")

    # =========================================================================
    # run() helpers (used more than once, or long enough to stand on their own)
    # =========================================================================

    def mean_concentrations(self):
        """Volume-averaged concentration of each monomer type from the chain compositions.

        mean(phi_i) = sum_p (f_p / alpha_p) sum_{blocks b of p} length_b * (fraction of i in b),
        which is what the C++ normalization of the concentrations gives.
        """
        phi_mean = np.zeros(len(self.monomer_types))
        for p, polymer in enumerate(self.distinct_polymers):
            alpha = self.prop_solver.get_polymer(p).get_alpha()
            for block in polymer["blocks"]:
                weight = polymer["volume_fraction"] * block["length"] / alpha
                fractions = self.random_fraction.get(block["type"], {block["type"]: 1.0})
                for monomer_type, fraction in fractions.items():
                    phi_mean[self.monomer_types.index(monomer_type)] += weight * fraction
        return phi_mean

    def _pin_field_means(self, w, chi_phi_mean, xi=0.0):
        """Normalize the field means in place (mean() works in the reduced basis
        when a space group is set).

        Compressible systems have no gauge freedom: the means are always pinned to the
        saddle point, mean(w_i) = sum_j chi_ij mean(phi_j) + xi with
        xi = zeta (mean(sum_j phi_j) - 1) (the residual leaves the means out).
        Incompressible fields are shifted to zero mean (the uniform modes do not change the
        concentrations, and the free energy does not depend on them).
        """
        M = len(self.monomer_types)
        if self.zeta_n is not None:
            for i in range(M):
                w[i] += chi_phi_mean[i] + xi - self.prop_solver.mean(w[i])
        else:
            for i in range(M):
                w[i] -= self.prop_solver.mean(w[i])

    def compute_chemical_potentials(self, w=None, partition_functions=None):
        """Restore the saddle-point field means and return beta mu per chain.

        w and partition_functions must describe the same CE state; by default
        use the final fields and Q saved by run(). No propagators are recomputed.
        Incompressible: mean(xi) = 0. Compressible: mean(xi) = zeta_n (sum(phi)-1).
        """
        if w is None:
            w = self.w
        if isinstance(w, dict):
            w = [w[m] for m in self.monomer_types]
        if partition_functions is None:
            partition_functions = self.partition_functions
        phi_mean = self.mean_concentrations()
        target = self.matrix_chi @ phi_mean
        if self.zeta_n is not None:
            target += self.zeta_n * (np.sum(phi_mean) - 1.0)
        shift = target - np.array([self.prop_solver.mean(field) for field in w])
        mu = []
        for p, polymer in enumerate(self.distinct_polymers):
            chain = self.prop_solver.get_polymer(p)
            concentration = chain.get_volume_fraction() / chain.get_alpha()
            if concentration == 0.0:
                mu.append(float("-inf"))
                continue
            field_shift = 0.0
            for block in polymer["blocks"]:
                fractions = self.random_fraction.get(block["type"], {block["type"]: 1.0})
                for monomer, fraction in fractions.items():
                    field_shift += block["length"] * fraction * shift[self.monomer_types.index(monomer)]
            mu.append(float(np.log(concentration) - np.log(partition_functions[p]) + field_shift))
        return mu

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
        crystal_system = self.crystal_system
        if crystal_system == "Hexagonal2D":
            dlx_reduced = np.array([0.5 * (dlx_full[0] + dlx_full[1])])
        elif crystal_system in ("Tetragonal", "Hexagonal", "Trigonal") and len(self.lx_reduced_indices) == 2:
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

        # Angles enter the optimizer in radians, the unit of their residual -dH/dθ
        am_current = np.concatenate((w.flatten(), current_lx, np.deg2rad(current_angles)))
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
            new_angles = np.rad2deg(np.array(am_new[-n_angles:]))
            new_dangle = np.clip(new_angles - old_angles, -5.0, 5.0)
            new_angles = old_angles + new_dangle

            full_angles = list(self.prop_solver.get_angles_degrees())
            for i, idx in enumerate(self.angles_full_indices):
                full_angles[idx] = new_angles[i]
            self.prop_solver.set_lattice_parameters(list(np.array(new_lx)[self.lx_full_indices]), full_angles)
        else:
            self.prop_solver.set_lx(list(np.array(new_lx)[self.lx_full_indices]))

        # The smearing kernel Γ(k) is a function of the physical wavenumber
        self.smearing.set_box(self.prop_solver.get_lx(), self.prop_solver.get_angles_degrees())
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
            If True, return an SCFTResult object with the fields on the full
            grid (default: False). The results are also stored as attributes.

        Attributes Set
        --------------
        phi : dict
            Final monomer concentration fields {monomer_type: array}.
        w : ndarray
            Final potential fields, shape (M, total_grid).
        free_energy : float
            Free energy relative to the homogeneous state at the same composition.
        free_energy_absolute : float
            Full canonical free energy, beta F_total / (rho_0 V / N).
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
        # With axis_ordering the arrays are given in the axis order of nx and lx and are
        # transposed to the standard [a, b, c] order (the results are in that order)
        if self._axis_perm is not None:
            initial_fields = {m: self._to_standard_axes(f) for m, f in initial_fields.items()}
            if q_init is not None:
                q_init = {label: self._to_standard_axes(f) for label, f in q_init.items()}

        # With a space group the fields are symmetrized and converted to the irreducible basis.
        if self.sg is not None:
            w_full = np.zeros([M, self.prop_solver.total_grid], dtype=np.float64)
            for i in range(M):
                w_full[i,:] = np.reshape(initial_fields[self.monomer_types[i]], self.prop_solver.total_grid)
            w_full = self.sg.symmetrize(w_full)
            w = self.sg.to_reduced_basis(w_full)
            w = self.sg.reduced_to_irreducible(w)
        else:
            w = np.zeros([M, self.prop_solver.n_grid], dtype=np.float64)
            for i in range(M):
                w[i,:] = np.reshape(initial_fields[self.monomer_types[i]], self.prop_solver.n_grid)

        # Normalize the initial field means. In a compressible model the target is
        # chi N . mean(phi) + xi from the chain compositions (see mean_concentrations)
        if self.zeta_n is not None:
            phi_mean = self.mean_concentrations()
            xi = 0.0 if self.zeta_n is None else self.zeta_n * (np.sum(phi_mean) - 1.0)
            self._pin_field_means(w, self.matrix_chi @ phi_mean, xi)
        else:
            self._pin_field_means(w, None)

        # Check the partition consistency on the first field of this run
        self._partition_checked = False

        # ---- 2. Iteration ----
        # Stress array layout: 3D [dH/dL1, dH/dL2, dH/dL3, dH/dgamma, dH/dbeta, dH/dalpha],
        # 2D [dH/dL1, dH/dL2, dH/dgamma, 0, 0, 0]; the angle index in [alpha, beta, gamma]
        # maps to the stress index through angle_stress_map
        dim = self.prop_solver.dim
        angle_stress_map = {0: 5, 1: 4, 2: 3} if dim == 3 else {2: 2}
        angle_stress_indices = [angle_stress_map[i] for i in self.angles_reduced_indices if i in angle_stress_map]
        stress_array = None
        stress_indices = list(range(dim)) + angle_stress_indices

        def current_stress(w, phi):
            self.prop_solver.compute_stress()
            stress = np.array(self.prop_solver.get_stress())
            if self.smearing.enabled:
                # Include the box dependence of the smearing kernel Γ(k).
                smearing_stress = self.smearing.stress(w, [phi[m] for m in self.monomer_types])
                if self.smearing.orthogonal and not angle_stress_indices:
                    smearing_stress[dim:] = 0.0
                stress += smearing_stress[:len(stress)]
            return stress

        # Ideal mixing term is constant during a CE run. Define c log(c) = 0 at c = 0.
        self.polymer_lengths = [self.prop_solver.get_polymer(p).get_alpha()
                                for p in range(self.prop_solver.get_n_polymer_types())]
        self.polymer_volume_fractions = [self.prop_solver.get_polymer(p).get_volume_fraction()
                                         for p in range(self.prop_solver.get_n_polymer_types())]
        chain_densities = np.array(self.polymer_volume_fractions) / self.polymer_lengths
        present = chain_densities > 0.0
        ideal_energy = float(np.sum(chain_densities[present] * (np.log(chain_densities[present]) - 1.0)))
        phi_mean = self.mean_concentrations()
        self.free_energy_homogeneous = ideal_energy + 0.5*float(phi_mean @ self.matrix_chi @ phi_mean)

        time_iter_start = time.time()
        for iter in range(1, self.max_iter+1):

            # Concentrations for the current fields
            phi, runtime_info = self.compute_concentrations(w, q_init=q_init)

            # Deviation of the total density from 1
            phi_total = sum(phi[m] for m in self.monomer_types)
            phi_diff = phi_total - 1.0

            # Concentrations the fields couple to (smeared with smearing), phi_array[i] =
            # phi[monomer_types[i]], shape (M, n_grid)
            phi_smeared = self.smearing.apply_to_dict(phi)
            phi_array = np.array([phi_smeared[m] for m in self.monomer_types])

            # Full CE free energy: Σ_p c_p[ln(c_p/Q_p)-1] + <½ φᵀ χN φ - w·φ>
            # (+ compressibility penalty). Uniform field shifts cancel between ln Q and w·φ.
            # At the saddle point F - ideal_energy is the Hamiltonian recorded by L-FTS.
            total_partitions = [self.prop_solver.get_partition_function(p) for p in range(self.prop_solver.get_n_polymer_types())]
            energy_total = ideal_energy
            for p, concentration in enumerate(chain_densities):
                if concentration > 0.0:
                    energy_total -= concentration*np.log(total_partitions[p])
            for i in range(M):
                for j in range(i+1, M):
                    if self.matrix_chi[i, j] != 0.0:
                        energy_total += self.matrix_chi[i, j]*self.prop_solver.mean(phi_array[i]*phi_array[j])
                energy_total -= self.prop_solver.mean(w[i]*phi_array[i])
            if self.zeta_n is not None:
                energy_total += 0.5*self.zeta_n*self.prop_solver.mean((np.sum(phi_array, axis=0) - 1.0)**2)
            # Keep the reported iteration energy relative to the homogeneous state.
            # Thermodynamic transforms and comparisons across compositions use the full F.
            energy_absolute = energy_total
            energy_total -= self.free_energy_homogeneous

            # The SCFT equations are w = χφ + ξe with Σφ = 1 (incompressible), or w = χφ + ζ(Σφ - 1)e
            # (compressible), e = (1, ..., 1). Composition part: P⊥(χφ - w), the residual of the
            # exchange fields of the symmetric theory. Pressure part along e: κ(Σφ - 1), which raises
            # the pressure where the density is too high; κ = max(max|λ|, 1)/2 follows the coupling
            # of the composition and the pressure (for AB κ = χN/2). Compressible: the stiff ζ term
            # lies along e, so that part of E = χφ + ζ(Σφ - 1)e - w is scaled by κ/(κ + ζ), which
            # gives the incompressible residual for ζ → ∞ and leaves E unscaled for ζ → 0
            e_column = np.ones((M, 1))
            if self.zeta_n is None:
                w_diff = self.matrix_q_perp @ (self.matrix_chi @ phi_array - w) \
                    + self.pressure_kappa * e_column * (np.sum(phi_array, axis=0) - 1.0)[None, :]
            else:
                residual = self.matrix_chi @ phi_array + self.zeta_n * (np.sum(phi_array, axis=0) - 1.0) - w
                composition = self.matrix_q_perp @ residual
                w_diff = composition + (self.pressure_kappa / (self.pressure_kappa + self.zeta_n)) * (residual - composition)

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
            field_error_level = error_level
            mass_error = self.prop_solver.mean(phi_diff)

            if self.box_is_altering:
                # The stress costs ~30 % of an iteration: compute it every stress_interval
                # iterations ("adaptive": every 1, 2 or 4 as the error drops) and reuse it in between
                if self.stress_interval == "adaptive":
                    stress_interval = 1 if error_level > 1e-2 else (2 if error_level > 1e-4 else 4)
                else:
                    stress_interval = self.stress_interval
                stress_is_current = (iter == 1) or (iter % stress_interval == 0) or iter == self.max_iter
                if stress_is_current:
                    stress_array = current_stress(w, phi)
                # The stress components being optimized (box lengths of the existing
                # dimensions and, if active, the angles) count towards the error
                error_level += np.linalg.norm(stress_array[stress_indices])
                # A cached stress can guide updates, but cannot certify convergence.
                if not stress_is_current and error_level < self.tolerance:
                    stress_array = current_stress(w, phi)
                    stress_is_current = True
                    error_level = field_error_level + np.linalg.norm(stress_array[stress_indices])

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
                    if self.box_is_altering and not stress_is_current:
                        stress_array = current_stress(w, phi)
                        error_level = field_error_level + np.linalg.norm(stress_array[stress_indices])
                    print(f"Iteration stopped by callback at iteration {iter}")
                    break

            # Conditions to end the iteration
            if error_level < self.tolerance or iter == self.max_iter:
                break

            # Calculate new fields using simple and Anderson mixing
            # Note: w and w_diff are in irreducible basis when space group is set
            if self.box_is_altering:
                w = self._update_fields_and_box(w, w_diff, stress_array, old_error_level, error_level)
            else:
                w = self.field_optimizer.calculate_new_fields(w.flatten(), w_diff.flatten(), old_error_level, error_level)
                w = np.reshape(w, (M, -1))

            # Normalize field means
            if self.zeta_n is not None:
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

        print("Free energy relative to the homogeneous state: %.9f" % energy_total)
        print("Full canonical free energy βF/(ρ₀V/N): %.9f" % energy_absolute)
        print("Homogeneous Flory-Huggins free energy: %.9f" % self.free_energy_homogeneous)
        print("Relative free energy divided by the number of chains of each type:")
        for p, concentration in enumerate(chain_densities):
            if concentration > 0.0:
                print("\tβΔF/n_%d : %12.7f" % (p, energy_total / concentration))

        # Final partition consistency check (only once at the end)
        if not self.prop_solver.check_total_partition():
            print("Warning: Final partition consistency check failed.")

        # Store phi and w
        self.phi = phi
        self.w = w

        # Store free energy
        self.free_energy = energy_total
        self.free_energy_absolute = energy_absolute
        self.error_level = error_level
        self.iter = iter
        self.partition_functions = list(total_partitions)
        self.chemical_potentials = None
        self.grand_potential = None
        if q_init is None and not any(p.get("initial_conditions") for p in self.distinct_polymers):
            self.chemical_potentials = self.compute_chemical_potentials()
            self.grand_potential = float(energy_absolute - np.dot(chain_densities[present],
                                                                 np.array(self.chemical_potentials)[present]))
            print("Chemical potentials βμ (mean ξ = 0 for incompressible systems):", self.chemical_potentials)
            print("Grand potential in the same pressure convention: %.9f" % self.grand_potential)

        # Return SCFTResult if requested
        if return_result:
            return SCFTResult(
                converged=error_level < self.tolerance,
                free_energy=energy_total,
                error_level=error_level,
                iterations=iter,
                phi={name: self._to_full_grid(phi[name]) for name in self.monomer_types},
                w={name: self._to_full_grid(w[i]) for i, name in enumerate(self.monomer_types)},
                partition_functions=total_partitions,
                lx=list(self.prop_solver.get_lx()),
                nx=list(self.prop_solver.nx),
                monomer_types=self.monomer_types,
                params=self.params,
                stress=stress_array if self.box_is_altering else None,
                angles=list(self.prop_solver.get_angles_degrees()) if (
                    len(self.angles_reduced_indices) > 0
                ) else None,
                free_energy_homogeneous=self.free_energy_homogeneous,
                free_energy_absolute=self.free_energy_absolute,
                chemical_potentials=self.chemical_potentials,
                grand_potential=self.grand_potential,
                polymer_lengths=list(self.polymer_lengths),
                polymer_volume_fractions=list(self.polymer_volume_fractions)
            )
        return None
