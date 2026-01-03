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
        bc=["reflecting", "reflecting"],
        ds=0.01,
        bond_lengths={"A": 1.0}
    )

    # Add a homopolymer
    solver.add_polymer(volume_fraction=1.0, blocks=[["A", 1.0, 0, 1]])

    # Initialize with zero potential
    solver.set_fields({"A": np.zeros(64)})

    # Evolve propagator
    q = np.exp(-((x - 2.0)**2) / 0.32)  # Gaussian initial condition
    for step in range(50):
        q = solver.advance(q, "A")
"""

import numpy as np
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
    bc : list of str, optional
        Boundary conditions. Format depends on dimensionality:
        - 1D: [x_low, x_high]
        - 2D: [x_low, x_high, y_low, y_high]
        - 3D: [x_low, x_high, y_low, y_high, z_low, z_high]
        Options: "periodic", "reflecting", "absorbing"
        Default: all periodic
    ds : float, optional
        Contour step size for chain discretization. Default: 0.01
    bond_lengths : dict, optional
        Statistical segment lengths for each monomer type.
        Default: {"A": 1.0}
    chain_model : str, optional
        Chain model type: "continuous" or "discrete". Default: "continuous"
    method : str, optional
        Numerical method: "pseudospectral" or "realspace". Default: "pseudospectral"
    platform : str, optional
        Computational platform: "auto", "cpu-mkl", or "cuda".
        "auto" selects cuda for 2D/3D, cpu-mkl for 1D. Default: "auto"

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
    Simple 1D propagator with reflecting boundaries:

    >>> solver = PropagatorSolver(
    ...     nx=[64], lx=[4.0],
    ...     bc=["reflecting", "reflecting"],
    ...     ds=0.01
    ... )
    >>> solver.add_polymer(1.0, [["A", 1.0, 0, 1]])
    >>> solver.set_fields({"A": np.zeros(64)})
    >>> q_out = solver.advance(q_in, "A")

    2D thin film with mixed boundaries:

    >>> solver = PropagatorSolver(
    ...     nx=[32, 24], lx=[4.0, 3.0],
    ...     bc=["reflecting", "reflecting", "absorbing", "absorbing"],
    ...     ds=0.01,
    ...     platform="cuda"
    ... )
    """

    def __init__(
        self,
        nx,
        lx,
        bc=None,
        ds=0.01,
        bond_lengths=None,
        chain_model="continuous",
        method="pseudospectral",
        platform="auto"
    ):
        """Initialize the propagator solver."""

        # Store grid parameters
        self.nx = list(nx)
        self.lx = list(lx)
        self.dim = len(nx)
        self.n_grid = int(np.prod(nx))

        # Validate dimensions
        if len(lx) != self.dim:
            raise ValueError(f"nx and lx must have same length: {len(nx)} != {len(lx)}")

        # Default boundary conditions (all periodic)
        if bc is None:
            bc = ["periodic"] * (2 * self.dim)
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
        self.method = method.lower()

        # Default bond lengths
        if bond_lengths is None:
            bond_lengths = {"A": 1.0}
        self.bond_lengths = dict(bond_lengths)

        # Determine platform
        if platform == "auto":
            # Use CUDA for 2D/3D, CPU for 1D
            if self.dim >= 2:
                platform = "cuda"
            else:
                platform = "cpu-mkl"
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
                  "Switching to real-space method.")
            self.method = "realspace"

        # Create factory
        self._factory = _core.PlatformSelector.create_factory(self.platform, False)

        # Create molecules
        self._molecules = self._factory.create_molecules_information(
            self.chain_model, self.ds, self.bond_lengths
        )

        # Track polymers added
        self._polymers_added = False

        # Solver will be created after polymers are added
        self._solver = None
        self._cb = None
        self._prop_opt = None
        self._fields_set = False

    def add_polymer(self, volume_fraction, blocks):
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

        Examples
        --------
        Add an AB diblock copolymer:

        >>> solver.add_polymer(1.0, [["A", 0.5, 0, 1], ["B", 0.5, 1, 2]])

        Add a homopolymer:

        >>> solver.add_polymer(1.0, [["A", 1.0, 0, 1]])
        """
        if self._solver is not None:
            raise RuntimeError(
                "Cannot add polymers after solver has been initialized. "
                "Add all polymers before calling set_fields() or advance()."
            )

        self._molecules.add_polymer(volume_fraction, blocks)
        self._polymers_added = True

    def _initialize_solver(self):
        """Initialize the solver (called lazily when needed)."""
        if self._solver is not None:
            return

        if not self._polymers_added:
            raise RuntimeError(
                "No polymers added. Call add_polymer() before using the solver."
            )

        # Create propagator computation optimizer
        self._prop_opt = self._factory.create_propagator_computation_optimizer(
            self._molecules, True
        )

        # Create computation box
        self._cb = self._factory.create_computation_box(
            nx=self.nx, lx=self.lx, bc=self.bc
        )

        # Create solver based on method
        if self.method == "pseudospectral":
            self._solver = self._factory.create_pseudospectral_solver(
                self._cb, self._molecules, self._prop_opt
            )
        elif self.method == "realspace":
            self._solver = self._factory.create_realspace_solver(
                self._cb, self._molecules, self._prop_opt
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def set_fields(self, w_fields):
        """
        Set the potential fields for propagator calculation.

        This must be called before advancing propagators. The fields
        determine the Boltzmann weights exp(-w*ds) used in the propagator
        evolution.

        Parameters
        ----------
        w_fields : dict
            Potential fields for each monomer type.
            Keys are monomer type strings (e.g., "A", "B").
            Values are numpy arrays of shape matching the grid.

        Examples
        --------
        Set zero fields for a single monomer type:

        >>> solver.set_fields({"A": np.zeros(64)})

        Set fields for AB diblock:

        >>> solver.set_fields({"A": w_A, "B": w_B})
        """
        self._initialize_solver()

        # Validate field shapes
        for key, field in w_fields.items():
            if np.size(field) != self.n_grid:
                raise ValueError(
                    f"Field '{key}' has wrong size: {np.size(field)} != {self.n_grid}"
                )

        self._solver.compute_propagators(w_fields)
        self._fields_set = True

    def advance(self, q_in, monomer_type):
        """
        Advance the propagator by one contour step.

        This evolves the propagator q(r, s) to q(r, s+ds) using the
        diffusion equation with the potential field set by set_fields().

        Parameters
        ----------
        q_in : numpy.ndarray
            Input propagator field.
        monomer_type : str
            Monomer type for this segment (e.g., "A").

        Returns
        -------
        numpy.ndarray
            Output propagator after one step.

        Examples
        --------
        Evolve propagator for 50 steps:

        >>> q = initial_condition
        >>> for step in range(50):
        ...     q = solver.advance(q, "A")
        """
        if not self._fields_set:
            raise RuntimeError(
                "Fields not set. Call set_fields() before advance()."
            )

        q_in = np.asarray(q_in, dtype=np.float64)
        if q_in.size != self.n_grid:
            raise ValueError(
                f"q_in has wrong size: {q_in.size} != {self.n_grid}"
            )

        return self._solver.advance_propagator_single_segment(
            q_in.ravel(), monomer_type
        )

    def propagate(self, q_init, monomer_type, n_steps):
        """
        Propagate for multiple contour steps.

        Convenience method that calls advance() repeatedly.

        Parameters
        ----------
        q_init : numpy.ndarray
            Initial propagator field.
        monomer_type : str
            Monomer type for all segments.
        n_steps : int
            Number of contour steps to take.

        Returns
        -------
        numpy.ndarray
            Final propagator after n_steps.

        Examples
        --------
        Propagate for the full chain length:

        >>> q_final = solver.propagate(q_init, "A", n_steps=100)
        """
        q = np.asarray(q_init, dtype=np.float64).ravel()
        for _ in range(n_steps):
            q = self.advance(q, monomer_type)
        return q

    def get_grid_points(self):
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

    def gaussian_initial_condition(self, center=None, sigma=0.4):
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
            f"  Method: {self.method}",
            f"  Platform: {self.platform}",
            f"  Monomer types: {list(self.bond_lengths.keys())}",
        ]
        return "\n".join(lines)

    def __repr__(self):
        return (
            f"PropagatorSolver(nx={self.nx}, lx={self.lx}, "
            f"method='{self.method}', platform='{self.platform}')"
        )
