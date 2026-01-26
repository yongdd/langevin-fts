"""
PropagatorSolver - A simple interface for polymer propagator calculations.

This module provides a user-friendly wrapper around the low-level factory-based
API, hiding implementation details from users who are not familiar with
programming patterns like abstract factories.

Example usage:

    from polymerfts import PropagatorSolver
    import numpy as np

    # Create a solver for 1D problem with reflecting boundaries
    solver = PropagatorSolver(
        nx=[64], lx=[4.0],
        ds=0.01,
        bond_lengths={"A": 1.0},
        bc=["reflecting", "reflecting"],
        chain_model="continuous",
        numerical_method="rqm4",  # optional, defaults to "rqm4"
        platform="cpu-fftw"
    )

    # For discrete chain model, numerical_method is not needed
    # (discrete chain has its own solver)
    solver_discrete = PropagatorSolver(
        nx=[32, 32, 32], lx=[4.0, 4.0, 4.0],
        ds=0.01,
        bond_lengths={"A": 1.0, "B": 1.0},
        bc=["periodic"] * 6,
        chain_model="discrete"
    )

    # Add a homopolymer
    solver.add_polymer(volume_fraction=1.0, blocks=[["A", 1.0, 0, 1]])

    # Compute all propagators
    solver.compute_propagators({"A": np.zeros(64)})

    # Get propagator at specific contour position
    q = solver.get_propagator(polymer=0, v=0, u=1, step=50)

    # Get partition function
    Q = solver.get_partition_function(polymer=0)

    # Compute and get concentrations
    phi_A = solver.get_concentration("A")
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
from numpy.typing import NDArray

from . import _core


class PropagatorSolver:
    """
    High-level interface for polymer propagator calculations.

    This class provides a simplified API for computing chain propagators
    using either pseudo-spectral (FFT/DCT/DST) or real-space (Crank-Nicolson)
    methods. It automatically handles platform selection and hides the
    factory pattern from users.

    Parameters
    ----------
    nx : list of int
        Number of grid points in each dimension. Length determines dimensionality.
    lx : list of float
        Box size in each dimension.
    ds : float
        Contour step size for chain discretization (e.g., 0.01 for N=100 segments).
    bond_lengths : dict
        Statistical segment lengths for each monomer type, e.g., {"A": 1.0, "B": 1.0}.
    bc : list of str
        Boundary conditions. Format depends on dimensionality:
        - 1D: [x_low, x_high]
        - 2D: [x_low, x_high, y_low, y_high]
        - 3D: [x_low, x_high, y_low, y_high, z_low, z_high]
        Options: "periodic", "reflecting", "absorbing"
    chain_model : str
        Chain model type: "continuous" or "discrete".
    numerical_method : str, optional
        Numerical algorithm for propagator computation.
        Only applicable for continuous chain model (default: 'rqm4'):
        - "rqm4": Pseudo-spectral with 4th-order Richardson extrapolation
        - "rk2": Pseudo-spectral with 2nd-order operator splitting
        - "etdrk4": Pseudo-spectral with ETDRK4 exponential integrator
        - "cn-adi2": Real-space with 2nd-order Crank-Nicolson ADI
        - "cn-adi4-lr": Real-space with 4th-order CN-ADI
        Note: Discrete chain model has its own solver; this parameter is ignored.
    platform : str
        Computational platform: "cpu-fftw", "cpu-fftw", or "cuda".
    reduce_memory : bool
        If True, store only propagator checkpoints instead of full histories,
        recomputing propagators as needed. Reduces memory usage but increases
        computation time by 2-4x.
    mask : numpy.ndarray, optional
        Mask array defining accessible (1) and impenetrable (0) regions.
        After each propagation step, the propagator is multiplied by the mask.
        Useful for simulating polymers around nanoparticles or in confined geometries.

    Attributes
    ----------
    nx : list of int
        Grid dimensions
    lx : list of float
        Box dimensions
    dim : int
        Number of spatial dimensions
    n_grid : int
        Total number of grid points

    Examples
    --------
    Simple 1D propagator with reflecting boundaries (continuous chain):

    >>> solver = PropagatorSolver(
    ...     nx=[64], lx=[4.0],
    ...     ds=0.01,
    ...     bond_lengths={"A": 1.0},
    ...     bc=["reflecting", "reflecting"],
    ...     chain_model="continuous",
    ...     numerical_method="rqm4",
    ...     platform="cpu-fftw"
    ... )
    >>> solver.add_polymer(1.0, [["A", 1.0, 0, 1]])
    >>> solver.compute_propagators({"A": np.zeros(64)})
    >>> q = solver.get_propagator(polymer=0, v=0, u=1, step=50)
    >>> Q = solver.get_partition_function(polymer=0)

    Discrete chain model (numerical_method not needed):

    >>> solver = PropagatorSolver(
    ...     nx=[32, 32, 32], lx=[4.0, 4.0, 4.0],
    ...     ds=0.01,
    ...     bond_lengths={"A": 1.0, "B": 1.0},
    ...     bc=["periodic"] * 6,
    ...     chain_model="discrete"
    ... )

    2D thin film with real-space CN-ADI4 method (continuous only):

    >>> solver = PropagatorSolver(
    ...     nx=[32, 24], lx=[4.0, 3.0],
    ...     ds=0.01,
    ...     bond_lengths={"A": 1.0},
    ...     bc=["reflecting", "reflecting", "absorbing", "absorbing"],
    ...     chain_model="continuous",
    ...     numerical_method="cn-adi4-lr",
    ...     platform="cuda"
    ... )
    """

    # Map numerical methods to solver types
    _PSEUDO_METHODS = {"rqm4", "rk2", "etdrk4"}
    _REALSPACE_METHODS = {"cn-adi2", "cn-adi4-lr"}

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
        space_group: Optional[Any] = None
    ) -> None:
        """Initialize the propagator solver."""

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

        # Handle numerical method based on chain model
        if self.chain_model == "discrete":
            # Discrete chain model has its own solver, numerical_method is not used
            if numerical_method is not None:
                print(f"Note: numerical_method '{numerical_method}' is ignored for discrete chain model.")
            self.method = "pseudospectral"
            self.numerical_method = "rqm4"  # Placeholder, not actually used for discrete
        else:
            # Continuous chain model: validate and set numerical method
            if numerical_method is None:
                numerical_method = "rqm4"
            else:
                numerical_method = numerical_method.lower()

            if numerical_method in self._PSEUDO_METHODS:
                self.method = "pseudospectral"
            elif numerical_method in self._REALSPACE_METHODS:
                self.method = "realspace"
            else:
                raise ValueError(
                    f"Unknown numerical_method '{numerical_method}'. "
                    f"Valid options: {self._PSEUDO_METHODS | self._REALSPACE_METHODS}"
                )
            self.numerical_method = numerical_method

        # Determine platform
        if platform == "auto":
            # Use CUDA for 2D/3D, CPU for 1D
            if self.dim >= 2:
                platform = "cuda"
            else:
                platform = "cpu-fftw"
        self.platform = platform

        # Check if non-periodic BC is used with CUDA pseudospectral
        self._has_non_periodic_bc = any(
            b.lower() in ["reflecting", "absorbing"] for b in self.bc
        )
        if (self._has_non_periodic_bc and
            self.platform == "cuda" and
            self.method == "pseudospectral"):
            # CUDA pseudospectral only supports periodic BC, fall back to realspace
            print("Note: CUDA pseudo-spectral only supports periodic BC. "
                  "Switching to real-space CN-ADI2 method.")
            self.method = "realspace"
            self.numerical_method = "cn-adi2"

        # Store checkpointing option
        self.reduce_memory = reduce_memory

        # Store and validate mask
        if mask is not None:
            if np.size(mask) != self.n_grid:
                raise ValueError(
                    f"mask has wrong size: {np.size(mask)} != {self.n_grid}"
                )
            self.mask = np.asarray(mask)
        else:
            self.mask = None

        # Create factory
        self._factory = _core.PlatformSelector.create_factory(
            self.platform, self.reduce_memory
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
        self._space_group = space_group

    def add_polymer(
        self,
        volume_fraction: float,
        blocks: List[List[Union[str, float, int]]],
        grafting_points: Optional[Dict[int, str]] = None
    ) -> None:
        """
        Add a polymer species to the system.

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

        Examples
        --------
        Add an AB diblock copolymer:

        >>> solver.add_polymer(1.0, [["A", 0.5, 0, 1], ["B", 0.5, 1, 2]])

        Add a homopolymer:

        >>> solver.add_polymer(1.0, [["A", 1.0, 0, 1]])

        Add a homopolymer with grafting point at node 0:

        >>> solver.add_polymer(1.0, [["A", 1.0, 0, 1]], grafting_points={0: "G"})
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
            self._molecules, True
        )

        # Create computation box if not already set (e.g., by external code setting angles)
        if self._computation_box is None:
            if self.mask is not None:
                self._computation_box = self._factory.create_computation_box(
                    nx=self.nx, lx=self.lx, bc=self.bc, mask=self.mask
                )
            else:
                self._computation_box = self._factory.create_computation_box(
                    nx=self.nx, lx=self.lx, bc=self.bc
                )

        # Create solver with space_group if provided
        # Handle both C++ SpaceGroup objects and Python wrappers with _cpp_sg attribute
        if self._space_group is None:
            cpp_space_group = None
        elif hasattr(self._space_group, '_cpp_sg'):
            cpp_space_group = self._space_group._cpp_sg
        else:
            # Already a C++ SpaceGroup object
            cpp_space_group = self._space_group
        self._propagator_computation = self._factory.create_propagator_computation(
            self._computation_box, self._molecules, self._propagator_optimizer,
            self.numerical_method, cpp_space_group
        )

    def compute_propagators(
        self,
        w_fields: Dict[str, NDArray[np.floating]],
        q_init: Optional[Dict[str, NDArray[np.floating]]] = None
    ) -> None:
        """
        Compute all chain propagators for the given potential fields.

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
            Values are numpy arrays of shape matching the grid.
            Example: {"G": delta_function_array}

        Examples
        --------
        Compute propagators with zero fields:

        >>> solver.compute_propagators({"A": np.zeros(64)})

        Compute propagators for AB diblock:

        >>> solver.compute_propagators({"A": w_A, "B": w_B})

        Compute propagators with custom initial condition:

        >>> solver.compute_propagators({"A": w_A}, q_init={"G": q_init_array})
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
        """
        Get the chain propagator at a specific contour position.

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

        Examples
        --------
        Get propagator at step 50 for a homopolymer:

        >>> q = solver.get_propagator(polymer=0, v=0, u=1, step=50)
        """
        if not self._fields_set:
            raise RuntimeError(
                "Propagators not computed. Call compute_propagators() first."
            )

        return self._propagator_computation.get_chain_propagator(polymer, v, u, step)

    def get_partition_function(self, polymer: int) -> float:
        """
        Get the single-chain partition function.

        Parameters
        ----------
        polymer : int
            Polymer index (0-based).

        Returns
        -------
        float
            Partition function Q for the specified polymer.

        Examples
        --------
        >>> Q = solver.get_partition_function(polymer=0)
        """
        if not self._fields_set:
            raise RuntimeError(
                "Propagators not computed. Call compute_propagators() first."
            )

        return self._propagator_computation.get_total_partition(polymer)

    def compute_concentrations(self) -> None:
        """
        Compute ensemble-averaged concentrations.

        Must be called after compute_propagators(). Computes the segment density
        for each monomer type.

        Examples
        --------
        >>> solver.compute_propagators({"A": w_A, "B": w_B})
        >>> solver.compute_concentrations()
        >>> phi_A = solver.get_concentration("A")
        """
        if not self._fields_set:
            raise RuntimeError(
                "Propagators not computed. Call compute_propagators() first."
            )

        self._propagator_computation.compute_concentrations()

    def get_concentration(self, monomer_type: str) -> NDArray[np.floating]:
        """
        Get the total concentration for a monomer type.

        Must be called after compute_concentrations().

        Parameters
        ----------
        monomer_type : str
            Monomer type (e.g., "A", "B").

        Returns
        -------
        numpy.ndarray
            Concentration field phi(r) for the specified monomer type.

        Examples
        --------
        >>> solver.compute_concentrations()
        >>> phi_A = solver.get_concentration("A")
        """
        if not self._fields_set:
            raise RuntimeError(
                "Propagators not computed. Call compute_propagators() first."
            )

        return self._propagator_computation.get_total_concentration(monomer_type)

    def get_grid_points(self) -> Tuple[NDArray[np.floating], ...]:
        """
        Get the coordinates of grid points.

        Returns
        -------
        tuple of numpy.ndarray
            Coordinate arrays for each dimension. For 1D returns (x,),
            for 2D returns (x, y), for 3D returns (x, y, z).
            Each array has shape matching the grid.

        Examples
        --------
        Get coordinates for a 2D grid:

        >>> x, y = solver.get_grid_points()
        >>> r_squared = x**2 + y**2
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
        """
        Create a Gaussian initial condition for the propagator.

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
            Gaussian field normalized for use as initial propagator.

        Examples
        --------
        Create centered Gaussian:

        >>> q_init = solver.gaussian_initial_condition()

        Create off-center Gaussian:

        >>> q_init = solver.gaussian_initial_condition(center=[1.0, 2.0])
        """
        if center is None:
            center = [L / 2 for L in self.lx]

        coords = self.get_grid_points()
        r_squared = sum(
            (coord - c)**2 for coord, c in zip(coords, center)
        )

        return np.exp(-r_squared / (2 * sigma**2))

    def get_dv(self) -> float:
        """
        Get the volume element (grid cell volume).

        This is useful for normalizing delta-function initial conditions.

        Returns
        -------
        float
            Volume of each grid cell: dv = (Lx/Nx) * (Ly/Ny) * ...

        Examples
        --------
        Create normalized delta function at origin:

        >>> dv = solver.get_dv()
        >>> q_init = np.zeros(solver.nx)
        >>> q_init[0, 0, 0] = 1.0 / dv  # Normalized delta function
        """
        dv = 1.0
        for i in range(self.dim):
            dv *= self.lx[i] / self.nx[i]
        return dv

    @property
    def n_grid(self):
        """
        Number of grid points for field operations.

        Returns n_irreducible when space group is set on the propagator
        computation, otherwise returns total_grid.

        Returns
        -------
        int
            Number of grid points (n_irreducible or total_grid).
        """
        if self._propagator_computation is not None:
            return self._propagator_computation.get_cb().get_n_basis()
        return self._total_grid

    @property
    def total_grid(self):
        """
        Total number of grid points (always full grid, ignoring space group).

        Returns
        -------
        int
            Total number of grid points.
        """
        return self._total_grid

    @property
    def info(self):
        """
        Get a summary of the solver configuration.

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
        """
        Get the total volume of the simulation box.

        Returns
        -------
        float
            Volume of the simulation box.
        """
        self._initialize_solver()
        return self._computation_box.get_volume()

    def integral(self, field: NDArray[np.floating]) -> float:
        """
        Compute the volume integral of a field.

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
        return self._computation_box.integral(field)

    def inner_product(
        self,
        field1: NDArray[np.floating],
        field2: NDArray[np.floating]
    ) -> float:
        """
        Compute the inner product of two fields.

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
        return self._computation_box.inner_product(field1, field2)

    def multi_inner_product(
        self,
        n_fields: int,
        fields1: NDArray[np.floating],
        fields2: NDArray[np.floating]
    ) -> float:
        """
        Compute the inner product of multiple field pairs.

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
        return self._computation_box.multi_inner_product(n_fields, fields1, fields2)

    def zero_mean(self, field: NDArray[np.floating]) -> None:
        """
        Subtract the spatial mean from a field in-place.

        Parameters
        ----------
        field : numpy.ndarray
            Field to modify.
        """
        self._initialize_solver()
        self._computation_box.zero_mean(field)

    def set_lx(self, lx: List[float]) -> None:
        """
        Update the box dimensions.

        Parameters
        ----------
        lx : list of float
            New box dimensions.
        """
        self._initialize_solver()
        self._computation_box.set_lx(lx)
        self.lx = list(lx)

    def set_lattice_parameters(self, lx: List[float], angles: List[float]) -> None:
        """
        Update the lattice parameters (box dimensions and angles).

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
        """
        Compute the stress tensor for box relaxation.

        Must be called after compute_propagators().
        """
        if not self._fields_set:
            raise RuntimeError(
                "Propagators not computed. Call compute_propagators() first."
            )
        self._propagator_computation.compute_stress()

    def get_stress(self) -> List[float]:
        """
        Get the computed stress tensor.

        Must be called after compute_stress().

        Returns
        -------
        list of float
            Stress components [sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_xz, sigma_yz].
        """
        if not self._fields_set:
            raise RuntimeError(
                "Propagators not computed. Call compute_propagators() first."
            )
        return self._propagator_computation.get_stress()

    def update_laplacian_operator(self) -> None:
        """
        Update the Laplacian operator after changing box dimensions.

        Must be called after set_lattice_parameters() when using pseudospectral method.
        """
        self._initialize_solver()
        self._propagator_computation.update_laplacian_operator()

    # -------------------- Space Group / Reduced Basis --------------------

    def set_space_group(self, space_group) -> None:
        """
        Set space group for reduced basis representation.

        .. deprecated::
            Pass space_group to the constructor instead.
            This method will be removed in a future version.

        Parameters
        ----------
        space_group : SpaceGroup
            SpaceGroup object from polymerfts.space_group module.
        """
        if self._propagator_computation is not None:
            raise RuntimeError(
                "Cannot set space_group after solver initialization. "
                "Pass space_group to the PropagatorSolver constructor instead."
            )
        self._space_group = space_group

    def to_reduced_basis(self, field: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Convert full grid field to reduced basis.

        Parameters
        ----------
        field : numpy.ndarray
            Field on full grid (n_grid,).

        Returns
        -------
        numpy.ndarray
            Field on reduced basis (n_irreducible,).
        """
        if self._space_group is None:
            raise RuntimeError(
                "Space group not set. Call set_space_group() first."
            )
        field_2d = np.reshape(field, (1, -1))
        return self._space_group.to_reduced_basis(field_2d)[0]

    def from_reduced_basis(self, field_reduced: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Convert reduced basis field to full grid.

        Parameters
        ----------
        field_reduced : numpy.ndarray
            Field on reduced basis (n_irreducible,).

        Returns
        -------
        numpy.ndarray
            Field on full grid (n_grid,).
        """
        if self._space_group is None:
            raise RuntimeError(
                "Space group not set. Call set_space_group() first."
            )
        field_2d = np.reshape(field_reduced, (1, -1))
        return self._space_group.from_reduced_basis(field_2d)[0]

    # -------------------- Molecule information --------------------

    def get_n_polymer_types(self) -> int:
        """
        Get the number of polymer types.

        Returns
        -------
        int
            Number of distinct polymer species.
        """
        return self._molecules.get_n_polymer_types()

    def get_polymer(self, polymer_id: int) -> Any:
        """
        Get polymer object for a specific polymer type.

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
        """
        Get the chain model name.

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
        """
        Create an Anderson Mixing optimizer.

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
