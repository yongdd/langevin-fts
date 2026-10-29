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

    @staticmethod
    def numerical_methods():
        """Names/capabilities from the same registry used by native factories."""
        return _core.numerical_methods()

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
        (``"real"`` or ``"complex"`` fields) are optional. The computation box is
        created here; after construction, add the polymers with
        :meth:`add_polymer`; the propagator computation is created on first use.
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

        self.field_type = field_type.lower()
        if self.field_type not in ("real", "complex"):
            raise ValueError(f"field_type must be 'real' or 'complex', got '{field_type}'")

        self.platform = self._auto_platform() if platform == "auto" else platform

        self._check_boundary_conditions()

        # Store checkpointing option
        self.reduce_memory = reduce_memory
        # Lattice angles and optimizer aggregation
        self.angles = list(angles) if angles is not None else None
        self.aggregate_propagator_computation = bool(aggregate_propagator_computation)

        info = self.numerical_methods()[self.numerical_method]
        if self.field_type == "complex" and not info["supports_complex"]:
            raise ValueError(f"{self.numerical_method} does not support complex fields")

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

        # The box does not depend on the polymers; the computation is created after they are added
        box_kwargs = dict(nx=self.nx, lx=self.lx, bc=self.bc)
        if self.angles is not None:
            box_kwargs["angles"] = self.angles
        if self.mask is not None:
            box_kwargs["mask"] = self.mask
        self._computation_box = self._factory.create_computation_box(**box_kwargs)
        # Early form of the factory check, with the orthogonality tolerance of the box
        if info["requires_orthogonal_cell"] and not self._computation_box.is_orthogonal():
            raise ValueError(f"{self.numerical_method} requires an orthogonal computation box")
        self._propagator_computation = None
        self._propagator_optimizer = None
        self._fourier = None
        if space_group is not None:
            activated = space_group.physical_basis_name()
            size = space_group.get_n_reduced_basis()
            if activated == "irreducible":
                print(f"Space-group basis: irreducible, {size} points (physical basis not available)")
            else:
                print(f"Space-group basis: {activated}-physical, {size} points")
        self._space_group = space_group

    # ------------------------------------------------------------------
    # Constructor helpers
    # ------------------------------------------------------------------

    @classmethod
    def _resolve_numerical_method(cls, chain_model, numerical_method):
        """Return (method, numerical_method) as the native factories resolve them.

        The discrete chain model has its own bond-convolution solver and ignores
        ``numerical_method``; continuous chains default to RQM4.
        """
        if chain_model == "discrete" and numerical_method is not None:
            print(f"Note: numerical_method '{numerical_method}' is ignored for discrete chain model.")
        try:
            name = _core.resolve_numerical_method(chain_model, (numerical_method or "").lower())
        except RuntimeError as exc:
            raise ValueError(str(exc)) from None
        return cls.numerical_methods()[name]["solver_type"], name

    def _auto_platform(self):
        """Platform for ``platform="auto"``: CPU for 1D (GPU overhead outweighs
        the gain), CUDA for 2D/3D when available, otherwise the first CPU
        backend of this build. Inputs that only the CPU backends run stay on
        the CPU: the CUDA pseudo-spectral solvers need even sizes of at most 6142
        points along absorbing axes (the sine transform stages a line of nx+2
        values in the 48 KB of shared memory of a block), and periodic boxes for
        complex fields of continuous chains.
        """
        avail = _core.PlatformSelector.avail_platforms()
        cpu_platforms = [p for p in avail if p.startswith("cpu")]
        if self.dim == 1 and cpu_platforms:
            return cpu_platforms[0]
        if "cuda" in avail:
            bc = [b.lower() for b in self.bc]
            cpu_only = self.numerical_methods()[self.numerical_method]["solver_type"] == "pseudospectral" and (
                any("absorbing" in bc[2*d:2*d+2] and (self.nx[d] % 2 or self.nx[d] > 6142) for d in range(self.dim))
                or (self.field_type == "complex" and self.chain_model == "continuous"
                    and any(b != "periodic" for b in bc)))
            if not (cpu_only and cpu_platforms):
                return "cuda"
        if cpu_platforms:
            return cpu_platforms[0]
        raise RuntimeError("No computational platform available in this build.")

    def _check_boundary_conditions(self):
        """Reject boundary conditions the pseudo-spectral transform cannot handle.

        Reflecting (DCT) and absorbing (DST) boundaries are supported on both
        platforms, verified against each other to machine precision in 1D/2D/3D
        for continuous and discrete chains. CUDA needs even sizes along absorbing
        axes and, for continuous chains, real fields (see _auto_platform). Mixing
        periodic with non-periodic directions is not supported by the spectral
        transform, so it fails loudly instead of silently switching solvers.
        """
        bc_lower = [b.lower() for b in self.bc]
        self._has_non_periodic_bc = any(b in ["reflecting", "absorbing"] for b in bc_lower)
        has_periodic = any(b == "periodic" for b in bc_lower)
        if not self.numerical_methods()[self.numerical_method]["supports_mixed_boundaries"] and has_periodic and self._has_non_periodic_bc:
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
        """The C++ ComputationBox (grid, box, integrals), with the computation's basis."""
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

    @property
    def fourier(self) -> Any:
        """Fourier transforms and Fourier multipliers of full-grid fields on this platform.

        ``forward(fields)`` and ``backward(coefficients)`` transform one field
        (shape ``(n_grid,)``) or several (``(n_fields, n_grid)``); ``backward`` includes
        the normalization. ``set_multiplier(M)`` stores a multiplier, on
        the GPU with CUDA, and returns its handle: ``M`` has the coefficient shape
        (one function applied to each field) or ``(n_out, n_in)`` followed by it.
        ``apply([(handle, fields), ...])`` returns ``backward(Σ M · forward(fields))``.

        Periodic boxes use the FFT: the half spectrum of ``numpy.fft.rfftn`` for
        real fields, the full ``fftn`` spectrum for complex fields. Boxes with walls
        use the cosine series Σ_j f_j cos(πm(j+½)/N) along every axis, half of
        ``scipy.fft.dctn(f, type=2)`` per axis. Real fields need real arrays (complex
        ones are rejected), and their multipliers act as real operators: on the
        planes of the half spectrum that hold both k and -k (last index 0 and n/2)
        ``M`` is replaced by (M(k) + conj M(-k))/2, so an odd derivative i·k loses
        its Nyquist mode; ``backward`` treats its coefficients the same way.

        ``k_squared()`` gives |k|² in the coefficient shape for the current box; a
        stored multiplier keeps its values, so store it again after the box
        changes (``set_lx`` and ``set_lattice_parameters`` need the polymers to be
        added). See ``examples/tools/FourierTransform.py``.
        """
        if self._fourier is None:
            periodic = [self.bc[2 * d].lower() == "periodic" and self.bc[2 * d + 1].lower() == "periodic"
                        for d in range(self.dim)]
            if any(periodic) and not all(periodic):
                raise ValueError("The Fourier transform cannot mix periodic and wall axes.")
            axes = ["periodic" if p else "reflecting" for p in periodic]
            self._fourier = self._factory.create_fourier_transform(self._computation_box, axes)
        return self._fourier

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
        # The native Molecules rejects a polymer once the computation exists
        if grafting_points is None:
            self._molecules.add_polymer(volume_fraction, blocks)
        else:
            self._molecules.add_polymer(volume_fraction, blocks, grafting_points)

    def _computed(self) -> Any:
        """The native computation, which rejects results that were not computed."""
        if self._propagator_computation is None:
            raise RuntimeError("Propagators not computed. Call compute_propagators() first.")
        return self._propagator_computation

    def _initialize_solver(self) -> None:
        """Initialize the solver (called lazily when needed)."""
        if self._propagator_computation is not None:
            return

        if self._molecules.get_n_polymer_types() == 0:
            raise RuntimeError(
                "No polymers added. Call add_polymer() before using the solver."
            )

        # Create propagator computation optimizer
        self._propagator_optimizer = self._factory.create_propagator_computation_optimizer(
            self._molecules, self.aggregate_propagator_computation
        )

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
        self._check_real(*w_fields.values(), *(q_init or {}).values())
        # The native binding checks the sizes of w_fields and q_init against n_grid
        if q_init is not None:
            self._propagator_computation.compute_propagators(w_fields, q_init=q_init)
        else:
            self._propagator_computation.compute_propagators(w_fields)

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
        return self._computed().get_chain_propagator(polymer, v, u, step)

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
        return self._computed().get_total_partition(polymer)

    def check_total_partition(self, tolerance: float = _core.PARTITION_TOLERANCE) -> bool:
        """Check forward/backward partition function consistency.

        Parameters
        ----------
        tolerance : float
            Largest accepted relative error of :meth:`partition_errors`.

        Returns
        -------
        bool
            True when every polymer's error is finite and at most ``tolerance``.
        """
        return bool(self._computed().check_total_partition(tolerance))

    def partition_errors(self) -> List[float]:
        """Relative partition function error of each polymer.

        Q must be the same at every contour point. For a positive canonical
        weight, deviations are normalized by the largest magnitude of the
        reference and contour partitions; Q = 0 is invalid. With zero canonical
        weight (grand-canonical species), the scale is the largest integral of
        the absolute contour integrand, so cancellation at Q = 0 is regular.
        The discrete-chain integrand includes its inverse Boltzmann weight.
        Non-finite partitions or scales give an infinite error. Nothing is printed.
        """
        return list(self._computed().compute_partition_errors())

    def compute_concentrations(self) -> None:
        """Compute ensemble-averaged concentrations.

        Must be called after compute_propagators(). Computes the segment density
        for each monomer type.
        """
        self._computed().compute_concentrations()

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
        return self._computed().get_total_concentration(monomer_type)

    def get_concentration_gce(self, fugacity: float, polymer: int, monomer_type: str) -> NDArray[np.floating]:
        """Concentration of one monomer type from one polymer in the grand-canonical ensemble.

        fugacity = exp(βμ_p) in the convention βμ_p = ln(φ_p/α_p) - ln Q_p, so the chains number
        fugacity Q_p per reference chain; the result is fugacity times the unnormalized segment
        density (independent of the volume fraction given to add_polymer). Must be called after
        compute_concentrations().
        """
        return self._computed().get_total_concentration_gce(fugacity, polymer, monomer_type)

    def get_grid_points(self) -> Tuple[NDArray[np.floating], ...]:
        """Get the Cartesian coordinates of the grid points.

        The points sit at the fractional coordinates (i + 1/2)/n along the lattice
        vectors of the current box (a along x, b in the xy plane), so the angles
        and later box changes are included. Hexagonal-axes space groups index x
        and y from the nodes, i/n (docs/theory/SpaceGroup.md).

        Returns
        -------
        tuple of numpy.ndarray
            Coordinate arrays for each dimension. For 1D returns (x,),
            for 2D returns (x, y), for 3D returns (x, y, z).
            Each array has shape matching the grid.
        """
        offsets = [0.5] * self.dim
        if self._space_group is not None and self._space_group.uses_hexagonal_axes():
            offsets[0] = offsets[1] = 0.0
        vectors = np.array(self._computation_box.get_lattice_vectors())[:self.dim, :self.dim]
        fractions = np.meshgrid(*[(np.arange(n) + offset) / n for n, offset in zip(self.nx, offsets)],
                                indexing="ij")
        return tuple(sum(fractions[d] * vectors[d, k] for d in range(self.dim)) for k in range(self.dim))

    def gaussian_initial_condition(
        self,
        center: Optional[List[float]] = None,
        sigma: float = 0.4
    ) -> NDArray[np.floating]:
        """Create a Gaussian initial condition for the propagator.

        Parameters
        ----------
        center : list of float, optional
            Cartesian center of the Gaussian.
            Default: center of the cell (half the sum of the lattice vectors).
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
            vectors = np.array(self._computation_box.get_lattice_vectors())[:self.dim, :self.dim]
            center = 0.5 * vectors.sum(axis=0)

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
            Volume of each grid cell, V / (Nx Ny Nz) of the current box (the
            cell volume includes the angles; a mask leaves it unchanged).
        """
        return self._computation_box.get_volume() / self._total_grid

    @property
    def n_grid(self):
        """Number of grid points for field operations.

        Returns n_reduced when a space group is given, also before the first
        computation creates the native solver, otherwise returns total_grid.

        Returns
        -------
        int
            Number of grid points (n_reduced or total_grid).
        """
        if self._propagator_computation is not None:
            return self._propagator_computation.get_cb().get_n_basis()
        # The computation will use the group's current basis (frozen once it is created)
        if self._space_group is not None:
            return self._space_group.get_n_reduced_basis()
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

    def integral(self, field: NDArray[np.floating], *, basis: str = "active") -> float:
        """Compute the volume integral of a field.

        Parameters
        ----------
        field : numpy.ndarray
            Field to integrate.
        basis : {"active", "irreducible"}
            Input representation. The default is the active reduced basis used
            by get_concentration() and to_reduced_basis().

        Returns
        -------
        float
            Integral of the field over the box volume.
        """
        self._initialize_solver()
        self._check_real(field)
        field_use = self._to_active_basis(field, basis)
        return self._computation_box.integral(field_use)

    def mean(self, field: NDArray[np.floating], *, basis: str = "active") -> float:
        """Compute the spatial average (mean) of a field.

        Parameters
        ----------
        field : numpy.ndarray
            Field to average.
        basis : {"active", "irreducible"}
            Input representation, as for integral(). Equal array sizes do not
            imply equal ordering of the two bases.

        Returns
        -------
        float
            Spatial average of the field (integral / volume).
        """
        self._initialize_solver()
        self._check_real(field)
        field_use = self._to_active_basis(field, basis)
        return self._computation_box.mean(field_use)

    def inner_product(
        self,
        field1: NDArray[np.floating],
        field2: NDArray[np.floating],
        *,
        basis: str = "active"
    ) -> float:
        """Compute the inner product of two fields.

        Parameters
        ----------
        field1 : numpy.ndarray
            First field.
        field2 : numpy.ndarray
            Second field.
        basis : {"active", "irreducible"}
            Representation of both inputs, as for integral().

        Returns
        -------
        float
            Inner product of the two fields.
        """
        self._initialize_solver()
        self._check_real(field1, field2)
        field1_use = self._to_active_basis(field1, basis)
        field2_use = self._to_active_basis(field2, basis)
        return self._computation_box.inner_product(field1_use, field2_use)

    def multi_inner_product(
        self,
        n_fields: int,
        fields1: NDArray[np.floating],
        fields2: NDArray[np.floating],
        *,
        basis: str = "active"
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
        basis : {"active", "irreducible"}
            Representation of both inputs, as for integral().

        Returns
        -------
        float
            Sum of inner products.
        """
        self._initialize_solver()
        self._check_real(fields1, fields2)
        fields1_use = self._to_active_basis_multi(n_fields, fields1, basis)
        fields2_use = self._to_active_basis_multi(n_fields, fields2, basis)
        return self._computation_box.multi_inner_product(n_fields, fields1_use, fields2_use)

    def zero_mean(self, field: NDArray[np.floating], *, basis: str = "active") -> None:
        """Subtract the spatial mean from a field in-place.

        With a mask the mean over the accessible volume is subtracted (at every
        point), so that mean() of the result is zero.

        Parameters
        ----------
        field : numpy.ndarray
            Field to modify.
        basis : {"active", "irreducible"}
            Input representation, as for integral().
        """
        self._check_real(field)
        self._initialize_solver()
        field_reduced = self._to_active_basis(field, basis)
        if field_reduced is field:
            self._computation_box.zero_mean(field)
            return

        # Compute in the active basis and shift the caller's irreducible array.
        # As in ComputationBox.zero_mean, normalize over accessible volume.
        mean_val = self._computation_box.mean(field_reduced)
        if self.mask is not None:
            mean_val *= self._computation_box.get_volume() / self._computation_box.get_accessible_volume()
        field -= mean_val

    def _physical_basis_active(self) -> bool:
        """True when a physical basis (m3, Pmmm or z-mirror) is enabled.

        It may differ from the irreducible basis in ordering even when their
        sizes agree. Callers therefore specify the representation explicitly.
        """
        return self._space_group is not None and self._space_group.physical_basis_name() != "irreducible"

    def _check_real(self, *fields) -> None:
        """Reject complex arrays for real fields: the conversion would drop their imaginary parts."""
        if self.field_type == "real" and any(np.iscomplexobj(field) for field in fields):
            raise ValueError('This solver has real fields: got a complex array. Complex fields need a '
                             'solver created with field_type="complex".')

    def _to_active_basis(self, field: NDArray[np.floating], basis: str) -> NDArray[np.floating]:
        """Convert only an explicitly declared irreducible input."""
        if basis not in ("active", "irreducible"):
            raise ValueError("basis must be 'active' or 'irreducible'")
        if basis == "irreducible" and self._physical_basis_active():
            return self._space_group.irreducible_to_reduced(field)
        return field

    def _to_active_basis_multi(
        self,
        n_fields: int,
        fields: NDArray[np.floating],
        basis: str
    ) -> NDArray[np.floating]:
        """Convert multiple irreducible-basis fields to reduced basis if needed."""
        if basis not in ("active", "irreducible"):
            raise ValueError("basis must be 'active' or 'irreducible'")
        if basis == "irreducible" and self._physical_basis_active():
            n_reduced = self._space_group.get_n_reduced_basis()
            n_irreducible = self._space_group.get_n_reduced_basis_full()
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
        self.update_laplacian_operator()

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
        self.update_laplacian_operator()

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
        self._computed().compute_stress()

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
        return self._computed().get_stress()

    def get_stress_gce(self, fugacities: List[float]) -> List[float]:
        """Lattice-parameter derivatives -Σ_p z_p dQ_p/dθ of the grand-canonical term -Σ_p z_p Q_p of H,
        with the chains weighted by the fugacities z_p (get_stress gives the derivatives of
        -Σ_p (φ_p/α_p) ln Q_p). Must be called after compute_stress()."""
        return self._computed().get_stress_gce([float(z) for z in fugacities])

    def update_laplacian_operator(self) -> None:
        """Update the Laplacian operator after changing box dimensions.

        The box setters call this automatically; it remains available after
        changing the underlying C++ computation box directly.
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
        self._check_real(field)
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
        self._check_real(field_reduced)
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
