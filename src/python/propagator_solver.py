"""PropagatorSolver: chain propagators, partition functions and concentrations
for a given set of potential fields, without touching the C++ factory API.
Used directly in the tutorials and by SCFT, LFTS and CLFTS.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
from numpy.typing import NDArray

from . import _core


class PropagatorSolver:
    """Propagator computation for a polymer system on a grid.

    Create the solver, add the polymers with add_polymer(), then call
    compute_propagators(w) and read get_propagator(), get_partition_function()
    and get_concentration(). The C++ objects are created on first use.

    Parameters
    ----------
    nx, lx : list
        Grid points and box lengths per dimension (1D, 2D or 3D).
    ds : float
        Contour step (1/N).
    bond_lengths : dict
        Statistical segment length per monomer type, e.g. {"A": 1.0, "B": 1.0}.
    bc : list of str
        Boundary conditions, two per dimension in the order
        [x_low, x_high, y_low, y_high, z_low, z_high]: "periodic",
        "reflecting" or "absorbing".
    chain_model : str
        "continuous" or "discrete".
    numerical_method : str, optional
        "rqm4" (default), "rk2" or "cn-adi2"; continuous chains only.
    platform : str, optional
        "cuda", "cpu-mkl", "cpu-fftw" or "auto" (CPU for 1D, CUDA for 2D/3D).
    reduce_memory : bool
        Store propagator checkpoints only (less memory, 2-4x slower).
    mask : ndarray, optional
        Accessible-volume mask on the full grid (1 inside, 0 impenetrable).
    space_group : SpaceGroup, optional
        Space group for the reduced-basis computation.
    angles : list of float, optional
        Lattice angles [alpha, beta, gamma] in degrees.
    aggregate_propagator_computation : bool
        Let the optimizer merge equivalent propagators (default True).
    field_type : str
        "real" or "complex" fields.

    Examples
    --------
    >>> solver = PropagatorSolver(nx=[64], lx=[4.0], ds=0.01, bond_lengths={"A": 1.0},
    ...                           bc=["reflecting", "reflecting"], chain_model="continuous",
    ...                           platform="cpu-mkl")
    >>> solver.add_polymer(1.0, [["A", 1.0, 0, 1]])
    >>> solver.compute_propagators({"A": np.zeros(64)})
    >>> q = solver.get_propagator(polymer=0, v=0, u=1, step=50)
    >>> Q = solver.get_partition_function(polymer=0)
    """

    # Map numerical methods to solver types
    _PSEUDO_METHODS = {"rqm4", "rk2"}
    _REALSPACE_METHODS = {"cn-adi2"}

    def __init__(
        self,
        nx: List[int],
        lx: List[float],
        ds: float,
        bond_lengths: Dict[str, float],
        bc: List[str],
        chain_model: str,
        numerical_method: Optional[str] = None,
        platform: str = "auto",
        reduce_memory: bool = False,
        mask: Optional[NDArray[np.floating]] = None,
        space_group: Optional[Any] = None,
        angles: Optional[List[float]] = None,
        aggregate_propagator_computation: bool = True,
        field_type: str = "real"
    ) -> None:
        """Initialize the propagator solver.

        ``angles`` (lattice angles in degrees), ``aggregate_propagator_computation``
        (let the optimizer merge equivalent propagators) and ``field_type``
        (``"real"`` or ``"complex"`` fields) are optional. After construction,
        add the polymers with :meth:`add_polymer`; the C++ objects are created
        on first use.
        """

        # Store grid parameters
        self.nx = list(nx)
        self.lx = list(lx)
        self.dim = len(nx)
        self._total_grid = int(np.prod(nx))

        # Validate dimensions
        if len(lx) != self.dim:
            raise ValueError(f"nx and lx must have same length: {len(nx)} != {len(lx)}")

        # Store boundary conditions
        self.bc = list(bc)

        # Validate boundary conditions
        expected_bc_len = 2 * self.dim
        if len(self.bc) != expected_bc_len:
            raise ValueError(
                f"bc must have {expected_bc_len} elements for {self.dim}D: got {len(self.bc)}"
            )

        # Store other parameters
        self.ds = ds
        self.chain_model = chain_model.lower()
        self.bond_lengths = dict(bond_lengths)

        self.method, self.numerical_method = self._resolve_numerical_method(self.chain_model, numerical_method)

        self.platform = self._auto_platform(self.dim) if platform == "auto" else platform

        self._check_boundary_conditions()

        # Store checkpointing option
        self.reduce_memory = reduce_memory
        # Lattice angles, optimizer aggregation and field type
        self.angles = list(angles) if angles is not None else None
        self.aggregate_propagator_computation = bool(aggregate_propagator_computation)
        self.field_type = field_type.lower()
        if self.field_type not in ("real", "complex"):
            raise ValueError(f"field_type must be 'real' or 'complex', got '{field_type}'")

        # Store and validate mask.
        # Compare against the full grid size: the n_grid property cannot be
        # used here because it touches _propagator_computation, which is not
        # created until after polymers are added (masks are always supplied
        # on the full grid regardless of any space group).
        if mask is not None:
            if np.size(mask) != self._total_grid:
                raise ValueError(
                    f"mask has wrong size: {np.size(mask)} != {self._total_grid}"
                )
            self.mask = np.asarray(mask)
        else:
            self.mask = None

        # Create factory
        self._factory = _core.PlatformSelector.create_factory(
            self.platform, self.reduce_memory, self.field_type
        )

        # Create molecules
        self._molecules = self._factory.create_molecules_information(
            self.chain_model, self.ds, self.bond_lengths
        )

        # Track polymers added
        self._polymers_added = False

        # Solver will be created after polymers are added
        self._propagator_computation = None
        self._computation_box = None
        self._propagator_optimizer = None
        self._fields_set = False
        if space_group is not None:
            activated = space_group.physical_basis_name()
            if activated == "irreducible":
                print("Space-group basis: irreducible (physical basis not available)")
            else:
                print(f"Space-group basis: {activated}-physical")
        self._space_group = space_group

    # ------------------------------------------------------------------
    # Constructor helpers
    # ------------------------------------------------------------------

    @classmethod
    def _resolve_numerical_method(cls, chain_model, numerical_method):
        """Return (method, numerical_method) for the chain model.

        The discrete chain model has its own bond-convolution solver and ignores
        ``numerical_method``; continuous chains default to RQM4.
        """
        if chain_model == "discrete":
            if numerical_method is not None:
                print(f"Note: numerical_method '{numerical_method}' is ignored for discrete chain model.")
            return "pseudospectral", "rqm4"   # placeholder, not used for discrete chains
        numerical_method = "rqm4" if numerical_method is None else numerical_method.lower()
        if numerical_method in cls._PSEUDO_METHODS:
            return "pseudospectral", numerical_method
        if numerical_method in cls._REALSPACE_METHODS:
            return "realspace", numerical_method
        raise ValueError(
            f"Unknown numerical_method '{numerical_method}'. "
            f"Valid options: {cls._PSEUDO_METHODS | cls._REALSPACE_METHODS}"
        )

    @staticmethod
    def _auto_platform(dim):
        """Platform for ``platform="auto"``: CPU for 1D (GPU overhead outweighs
        the gain), CUDA for 2D/3D when available, otherwise the first CPU
        backend of this build.
        """
        avail = _core.PlatformSelector.avail_platforms()
        cpu_platforms = [p for p in avail if p.startswith("cpu")]
        if dim == 1 and cpu_platforms:
            return cpu_platforms[0]
        if "cuda" in avail:
            return "cuda"
        if cpu_platforms:
            return cpu_platforms[0]
        raise RuntimeError("No computational platform available in this build.")

    def _check_boundary_conditions(self):
        """Reject boundary conditions the pseudo-spectral transform cannot handle.

        Reflecting (DCT) and absorbing (DST) boundaries are supported on both
        platforms, verified against each other to machine precision in 1D/2D/3D
        for continuous and discrete chains. Mixing periodic with non-periodic
        directions is not supported by the spectral transform, so it fails
        loudly instead of silently switching solvers.
        """
        bc_lower = [b.lower() for b in self.bc]
        self._has_non_periodic_bc = any(b in ["reflecting", "absorbing"] for b in bc_lower)
        has_periodic = any(b == "periodic" for b in bc_lower)
        if self.method == "pseudospectral" and has_periodic and self._has_non_periodic_bc:
            if self.chain_model == "discrete":
                raise NotImplementedError(
                    "Mixing periodic and non-periodic boundary conditions is not "
                    "supported by the pseudo-spectral transform, and the discrete "
                    "chain model has no real-space solver. Use uniform boundary "
                    "conditions (all-periodic, all-reflecting, or all-absorbing)."
                )
            raise ValueError(
                "Mixing periodic and non-periodic boundary conditions is not "
                "supported by the pseudo-spectral method "
                f"(requested numerical_method='{self.numerical_method}'). "
                "Either set numerical_method='cn-adi2' to use the real-space "
                "solver (continuous chains only), or use uniform boundary "
                "conditions in all directions."
            )

    # -------------------- Access to the underlying C++ objects --------------------

    @property
    def computation_box(self) -> Any:
        """The C++ ComputationBox (grid, box, integrals)."""
        self._initialize_solver()
        return self._computation_box

    @property
    def molecules(self) -> Any:
        """The C++ Molecules object holding the polymer definitions."""
        return self._molecules

    @property
    def propagator_optimizer(self) -> Any:
        """The C++ PropagatorComputationOptimizer."""
        self._initialize_solver()
        return self._propagator_optimizer

    @property
    def propagator_computation(self) -> Any:
        """The C++ PropagatorComputation (the propagator solver itself)."""
        self._initialize_solver()
        return self._propagator_computation

    @property
    def space_group(self) -> Optional[Any]:
        """The space group in use, or None."""
        return self._space_group

    def display_platform_info(self) -> None:
        """Print the platform and device information of the backend."""
        self._factory.display_info()

    def display_architectures(self) -> None:
        """Print the polymer architectures."""
        self._molecules.display_architectures()

    def display_statistics(self) -> None:
        """Print the propagator computation statistics of the optimizer."""
        self.propagator_optimizer.display_statistics()

    def set_force_off_diagonal_stress(self, flag: bool) -> None:
        """Force computation of the off-diagonal stress components."""
        self.propagator_computation.set_force_off_diagonal_stress(flag)

    def add_polymer(
        self,
        volume_fraction: float,
        blocks: List[List[Union[str, float, int]]],
        grafting_points: Optional[Dict[int, str]] = None
    ) -> None:
        """Add a polymer species to the system.

        Parameters
        ----------
        volume_fraction : float
            Volume fraction of this polymer species.
        blocks : list of lists
            Block specifications. Each block is [monomer_type, length, v, u]
            where v and u are node indices defining the block connectivity.

            For a simple linear chain A-B diblock:
                blocks = [["A", 0.5, 0, 1], ["B", 0.5, 1, 2]]

            For a homopolymer:
                blocks = [["A", 1.0, 0, 1]]
        grafting_points : dict, optional
            Custom initial conditions for specific nodes.
            Keys are node indices (int), values are string labels for q_init.
            Example: {0: "G"} means node 0 uses q_init["G"] as initial condition.
        """
        if self._propagator_computation is not None:
            raise RuntimeError(
                "Cannot add polymers after solver has been initialized. "
                "Add all polymers before calling compute_propagators()."
            )

        if grafting_points is None:
            self._molecules.add_polymer(volume_fraction, blocks)
        else:
            self._molecules.add_polymer(volume_fraction, blocks, grafting_points)
        self._polymers_added = True

    def _initialize_solver(self) -> None:
        """Initialize the solver (called lazily when needed)."""
        if self._propagator_computation is not None:
            return

        if not self._polymers_added:
            raise RuntimeError(
                "No polymers added. Call add_polymer() before using the solver."
            )

        # Create propagator computation optimizer
        self._propagator_optimizer = self._factory.create_propagator_computation_optimizer(
            self._molecules, self.aggregate_propagator_computation
        )

        # Create computation box if not already set (e.g., by external code setting angles)
        if self._computation_box is None:
            box_kwargs = dict(nx=self.nx, lx=self.lx, bc=self.bc)
            if self.angles is not None:
                box_kwargs["angles"] = self.angles
            if self.mask is not None:
                box_kwargs["mask"] = self.mask
            self._computation_box = self._factory.create_computation_box(**box_kwargs)

        # Create solver (with the space group when one is set)
        self._propagator_computation = self._factory.create_propagator_computation(
            self._computation_box, self._molecules, self._propagator_optimizer,
            self.numerical_method, self._space_group
        )

    def compute_propagators(
        self,
        w_fields: Dict[str, NDArray[np.floating]],
        q_init: Optional[Dict[str, NDArray[np.floating]]] = None
    ) -> None:
        """Compute all chain propagators for the given potential fields.

        This must be called after adding polymers. It computes all chain
        propagators using the specified potential fields.

        Parameters
        ----------
        w_fields : dict
            Potential fields for each monomer type.
            Keys are monomer type strings (e.g., "A", "B").
            Values are numpy arrays of shape matching the grid.
        q_init : dict, optional
            Custom initial conditions for grafting points.
            Keys are string labels matching those in grafting_points.
            Values are numpy arrays in the same basis as w_fields (n_grid
            entries: the reduced basis when a space group is set; convert a
            full-grid array with to_reduced_basis).
            Example: {"G": delta_function_array}
        """
        self._initialize_solver()

        # Validate field shapes
        for key, field in w_fields.items():
            if np.size(field) != self.n_grid:
                raise ValueError(
                    f"Field '{key}' has wrong size: {np.size(field)} != {self.n_grid}"
                )

        # Validate q_init shapes if provided
        if q_init is not None:
            for key, field in q_init.items():
                if np.size(field) != self.n_grid:
                    raise ValueError(
                        f"q_init '{key}' has wrong size: {np.size(field)} != {self.n_grid}"
                    )
            self._propagator_computation.compute_propagators(w_fields, q_init=q_init)
        else:
            self._propagator_computation.compute_propagators(w_fields)
        self._fields_set = True

    def get_propagator(self, polymer: int, v: int, u: int, step: int) -> NDArray[np.floating]:
        """Get the chain propagator at a specific contour position.

        Parameters
        ----------
        polymer : int
            Polymer index (0-based).
        v : int
            Starting node index for the block.
        u : int
            Ending node index for the block.
        step : int
            Contour step index (0 to n_segments).

        Returns
        -------
        numpy.ndarray
            Propagator field q(r, s) at the specified contour position.
        """
        if not self._fields_set:
            raise RuntimeError(
                "Propagators not computed. Call compute_propagators() first."
            )

        return self._propagator_computation.get_chain_propagator(polymer, v, u, step)

    def get_partition_function(self, polymer: int) -> float:
        """Get the single-chain partition function.

        Parameters
        ----------
        polymer : int
            Polymer index (0-based).

        Returns
        -------
        float
            Partition function Q for the specified polymer.
        """
        if not self._fields_set:
            raise RuntimeError(
                "Propagators not computed. Call compute_propagators() first."
            )

        return self._propagator_computation.get_total_partition(polymer)

    def check_total_partition(self) -> bool:
        """Check forward/backward partition function consistency.

        Returns
        -------
        bool
            True when the solver reports consistent total partitions.
        """
        if not self._fields_set:
            raise RuntimeError(
                "Propagators not computed. Call compute_propagators() first."
            )
        return bool(self._propagator_computation.check_total_partition())

    def compute_concentrations(self) -> None:
        """Compute ensemble-averaged concentrations.

        Must be called after compute_propagators(). Computes the segment density
        for each monomer type.
        """
        if not self._fields_set:
            raise RuntimeError(
                "Propagators not computed. Call compute_propagators() first."
            )

        self._propagator_computation.compute_concentrations()

    def get_concentration(self, monomer_type: str) -> NDArray[np.floating]:
        """Get the total concentration for a monomer type.

        Must be called after compute_concentrations().

        Parameters
        ----------
        monomer_type : str
            Monomer type (e.g., "A", "B").

        Returns
        -------
        numpy.ndarray
            Concentration field phi(r) for the specified monomer type.
        """
        if not self._fields_set:
            raise RuntimeError(
                "Propagators not computed. Call compute_propagators() first."
            )

        return self._propagator_computation.get_total_concentration(monomer_type)

    def get_grid_points(self) -> Tuple[NDArray[np.floating], ...]:
        """Get the coordinates of grid points.

        Returns
        -------
        tuple of numpy.ndarray
            Coordinate arrays for each dimension. For 1D returns (x,),
            for 2D returns (x, y), for 3D returns (x, y, z).
            Each array has shape matching the grid.
        """
        coords = []
        for d in range(self.dim):
            dx = self.lx[d] / self.nx[d]
            coord_1d = (np.arange(self.nx[d]) + 0.5) * dx

            # Create full grid array
            shape = [1] * self.dim
            shape[d] = self.nx[d]
            coord = coord_1d.reshape(shape)

            # Broadcast to full grid
            full_shape = self.nx
            coord = np.broadcast_to(coord, full_shape)
            coords.append(coord.copy())

        return tuple(coords)

    def gaussian_initial_condition(
        self,
        center: Optional[List[float]] = None,
        sigma: float = 0.4
    ) -> NDArray[np.floating]:
        """Create a Gaussian initial condition for the propagator.

        Parameters
        ----------
        center : list of float, optional
            Center of the Gaussian in each dimension.
            Default: center of the box.
        sigma : float, optional
            Standard deviation of the Gaussian. Default: 0.4

        Returns
        -------
        numpy.ndarray
            Gaussian field with unit peak amplitude (NOT normalized to unit
            integral; divide by `np.sum(q) * solver.get_dv()` if a normalized
            distribution is needed).
        """
        if center is None:
            center = [L / 2 for L in self.lx]

        coords = self.get_grid_points()
        r_squared = sum(
            (coord - c)**2 for coord, c in zip(coords, center)
        )

        return np.exp(-r_squared / (2 * sigma**2))

    def get_dv(self) -> float:
        """Get the volume element (grid cell volume).

        This is useful for normalizing delta-function initial conditions.

        Returns
        -------
        float
            Volume of each grid cell: dv = (Lx/Nx) * (Ly/Ny) * ...
        """
        dv = 1.0
        for i in range(self.dim):
            dv *= self.lx[i] / self.nx[i]
        return dv

    @property
    def n_grid(self):
        """Number of grid points for field operations.

        Returns n_reduced when space group is set on the propagator
        computation, otherwise returns total_grid.

        Returns
        -------
        int
            Number of grid points (n_reduced or total_grid).
        """
        if self._propagator_computation is not None:
            return self._propagator_computation.get_cb().get_n_basis()
        return self._total_grid

    @property
    def total_grid(self):
        """Total number of grid points (always full grid, ignoring space group).

        Returns
        -------
        int
            Total number of grid points.
        """
        return self._total_grid

    @property
    def info(self):
        """Get a summary of the solver configuration.

        Returns
        -------
        str
            Human-readable summary of the solver settings.
        """
        lines = [
            f"PropagatorSolver Configuration:",
            f"  Dimensions: {self.dim}D",
            f"  Grid: {self.nx}",
            f"  Box size: {self.lx}",
            f"  Boundary conditions: {self.bc}",
            f"  Chain model: {self.chain_model}",
            f"  Contour step (ds): {self.ds}",
            f"  Numerical method: {self.numerical_method}",
            f"  Platform: {self.platform}",
            f"  Monomer types: {list(self.bond_lengths.keys())}",
        ]
        return "\n".join(lines)

    def __repr__(self):
        return (
            f"PropagatorSolver(nx={self.nx}, lx={self.lx}, "
            f"numerical_method='{self.numerical_method}', platform='{self.platform}')"
        )

    # -------------------- Box operations --------------------

    def get_volume(self) -> float:
        """Get the total volume of the simulation box.

        Returns
        -------
        float
            Volume of the simulation box.
        """
        self._initialize_solver()
        return self._computation_box.get_volume()

    def integral(self, field: NDArray[np.floating]) -> float:
        """Compute the volume integral of a field.

        Parameters
        ----------
        field : numpy.ndarray
            Field to integrate.

        Returns
        -------
        float
            Integral of the field over the box volume.
        """
        self._initialize_solver()
        field_use = self._maybe_to_reduced(field)
        return self._computation_box.integral(field_use)

    def mean(self, field: NDArray[np.floating]) -> float:
        """Compute the spatial average (mean) of a field.

        Parameters
        ----------
        field : numpy.ndarray
            Field to average.

        Returns
        -------
        float
            Spatial average of the field (integral / volume).
        """
        self._initialize_solver()
        field_use = self._maybe_to_reduced(field)
        return self._computation_box.mean(field_use)

    def inner_product(
        self,
        field1: NDArray[np.floating],
        field2: NDArray[np.floating]
    ) -> float:
        """Compute the inner product of two fields.

        Parameters
        ----------
        field1 : numpy.ndarray
            First field.
        field2 : numpy.ndarray
            Second field.

        Returns
        -------
        float
            Inner product of the two fields.
        """
        self._initialize_solver()
        field1_use = self._maybe_to_reduced(field1)
        field2_use = self._maybe_to_reduced(field2)
        return self._computation_box.inner_product(field1_use, field2_use)

    def multi_inner_product(
        self,
        n_fields: int,
        fields1: NDArray[np.floating],
        fields2: NDArray[np.floating]
    ) -> float:
        """Compute the inner product of multiple field pairs.

        Parameters
        ----------
        n_fields : int
            Number of field pairs.
        fields1 : numpy.ndarray
            First set of fields (n_fields * n_grid).
        fields2 : numpy.ndarray
            Second set of fields (n_fields * n_grid).

        Returns
        -------
        float
            Sum of inner products.
        """
        self._initialize_solver()
        fields1_use = self._maybe_to_reduced_multi(n_fields, fields1)
        fields2_use = self._maybe_to_reduced_multi(n_fields, fields2)
        return self._computation_box.multi_inner_product(n_fields, fields1_use, fields2_use)

    def zero_mean(self, field: NDArray[np.floating]) -> None:
        """Subtract the spatial mean from a field in-place.

        Parameters
        ----------
        field : numpy.ndarray
            Field to modify.
        """
        self._initialize_solver()
        if self._space_group is None:
            self._computation_box.zero_mean(field)
            return

        n_irreducible = self._space_group.get_n_reduced_basis_full()
        if self._physical_basis_active() and field.size == n_irreducible:
            # Compute mean in reduced basis, apply shift in irreducible basis.
            field_reduced = self._space_group.irreducible_to_reduced(field)
            mean_val = self._computation_box.mean(field_reduced)
            field -= mean_val
        else:
            self._computation_box.zero_mean(field)

    def _physical_basis_active(self) -> bool:
        """True when a physical basis (m3, Pmmm or z-mirror) is enabled.

        The reduced basis then differs from the irreducible basis in ordering even
        when the two have the same size (mmm orthorhombic groups with the m3
        basis), so the conversion must never be skipped on a size comparison.
        """
        return self._space_group is not None and self._space_group.physical_basis_name() != "irreducible"

    def _maybe_to_reduced(self, field: NDArray[np.floating]) -> NDArray[np.floating]:
        """Convert irreducible-basis field to reduced basis when physical basis is active."""
        if self._space_group is None:
            return field
        n_irreducible = self._space_group.get_n_reduced_basis_full()
        if self._physical_basis_active() and field.size == n_irreducible:
            return self._space_group.irreducible_to_reduced(field)
        return field

    def _maybe_to_reduced_multi(
        self,
        n_fields: int,
        fields: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Convert multiple irreducible-basis fields to reduced basis if needed."""
        if self._space_group is None:
            return fields
        n_reduced = self._space_group.get_n_reduced_basis()
        n_irreducible = self._space_group.get_n_reduced_basis_full()
        if self._physical_basis_active() and fields.size == n_fields * n_irreducible:
            reshaped = fields.reshape((n_fields, n_irreducible))
            reduced = self._space_group.irreducible_to_reduced(reshaped)
            return reduced.reshape((n_fields * n_reduced,))
        return fields

    def set_lx(self, lx: List[float]) -> None:
        """Update the box dimensions.

        Parameters
        ----------
        lx : list of float
            New box dimensions.
        """
        self._initialize_solver()
        self._computation_box.set_lx(lx)
        self.lx = list(lx)

    def set_lattice_parameters(self, lx: List[float], angles: List[float]) -> None:
        """Update the lattice parameters (box dimensions and angles).

        Parameters
        ----------
        lx : list of float
            New box dimensions.
        angles : list of float
            New lattice angles in degrees [alpha, beta, gamma].
        """
        self._initialize_solver()
        self._computation_box.set_lattice_parameters(lx, angles)
        self.lx = list(lx)

    def get_lx(self) -> List[float]:
        """Get current box dimensions."""
        self._initialize_solver()
        return self._computation_box.get_lx()

    def get_dx(self) -> List[float]:
        """Get grid spacing in each dimension."""
        self._initialize_solver()
        return self._computation_box.get_dx()

    def get_angles_degrees(self) -> List[float]:
        """Get lattice angles in degrees."""
        self._initialize_solver()
        return self._computation_box.get_angles_degrees()

    # -------------------- Stress computation --------------------

    def compute_stress(self) -> None:
        """Compute the stress tensor for box relaxation.

        Must be called after compute_propagators().
        """
        if not self._fields_set:
            raise RuntimeError(
                "Propagators not computed. Call compute_propagators() first."
            )
        self._propagator_computation.compute_stress()

    def get_stress(self) -> List[float]:
        """Get the computed stress tensor.

        Must be called after compute_stress().

        Returns
        -------
        list of float
            Lattice-parameter derivatives of the free energy:
            3D: [dH/dL1, dH/dL2, dH/dL3, dH/d(gamma), dH/d(beta), dH/d(alpha)]
            2D: [dH/dL1, dH/dL2, dH/d(gamma), 0, 0, 0]; 1D: only index 0.
        """
        if not self._fields_set:
            raise RuntimeError(
                "Propagators not computed. Call compute_propagators() first."
            )
        return self._propagator_computation.get_stress()

    def update_laplacian_operator(self) -> None:
        """Update the Laplacian operator after changing box dimensions.

        Must be called after set_lattice_parameters() when using pseudospectral method.
        """
        self._initialize_solver()
        self._propagator_computation.update_laplacian_operator()

    # -------------------- Space Group / Reduced Basis --------------------

    def to_reduced_basis(self, field: NDArray[np.floating]) -> NDArray[np.floating]:
        """Convert full grid field to reduced basis.

        Parameters
        ----------
        field : numpy.ndarray
            Field on full grid (n_grid,).

        Returns
        -------
        numpy.ndarray
            Field on reduced basis (n_reduced,).
        """
        if self._space_group is None:
            raise RuntimeError(
                "Space group not set. Pass space_group= to PropagatorSolver()."
            )
        field_2d = np.reshape(field, (1, -1))
        return self._space_group.to_reduced_basis(field_2d)[0]

    def from_reduced_basis(self, field_reduced: NDArray[np.floating]) -> NDArray[np.floating]:
        """Convert reduced basis field to full grid.

        Parameters
        ----------
        field_reduced : numpy.ndarray
            Field on reduced basis (n_reduced,).

        Returns
        -------
        numpy.ndarray
            Field on full grid (n_grid,).
        """
        if self._space_group is None:
            raise RuntimeError(
                "Space group not set. Pass space_group= to PropagatorSolver()."
            )
        field_2d = np.reshape(field_reduced, (1, -1))
        return self._space_group.from_reduced_basis(field_2d)[0]

    # -------------------- Molecule information --------------------

    def get_n_polymer_types(self) -> int:
        """Get the number of polymer types.

        Returns
        -------
        int
            Number of distinct polymer species.
        """
        return self._molecules.get_n_polymer_types()

    def get_polymer(self, polymer_id: int) -> Any:
        """Get polymer object for a specific polymer type.

        Parameters
        ----------
        polymer_id : int
            Polymer index (0-based).

        Returns
        -------
        Polymer
            Polymer object with methods like get_volume_fraction(), get_alpha(), etc.
        """
        return self._molecules.get_polymer(polymer_id)

    def get_model_name(self) -> str:
        """Get the chain model name.

        Returns
        -------
        str
            Chain model ("Continuous" or "Discrete").
        """
        return self._molecules.get_model_name()

    # -------------------- Anderson Mixing factory --------------------

    def create_anderson_mixing(
        self,
        n_var: int,
        max_hist: int,
        start_error: float,
        mix_min: float,
        mix_init: float
    ) -> Any:
        """Create an Anderson Mixing optimizer.

        Parameters
        ----------
        n_var : int
            Number of variables to optimize.
        max_hist : int
            Maximum number of history vectors.
        start_error : float
            Error threshold to switch from simple mixing to AM.
        mix_min : float
            Minimum mixing parameter.
        mix_init : float
            Initial mixing parameter.

        Returns
        -------
        AndersonMixing
            Anderson Mixing optimizer instance.
        """
        return self._factory.create_anderson_mixing(
            n_var, max_hist, start_error, mix_min, mix_init
        )
