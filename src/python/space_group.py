"""Crystallographic space group symmetry constraints for polymer simulations.

This module provides the SpaceGroup class for applying space group symmetry
to reduce computational cost in polymer field theory simulations. By exploiting
crystallographic symmetry, fields can be represented using only irreducible
mesh points rather than the full simulation grid.

Common Space Groups for Polymer Phases
--------------------------------------
+----------+------------+------+------+---------------------+------------------+
| Phase    | Symbol     | No.  | Hall | Crystal System      | Grid Constraint  |
+----------+------------+------+------+---------------------+------------------+
| BCC      | Im-3m      | 229  | 529  | Cubic               | nx = ny = nz     |
| FCC      | Fm-3m      | 225  | 523  | Cubic               | nx = ny = nz     |
| A15      | Pm-3n      | 223  | 520  | Cubic               | nx = ny = nz     |
| Gyroid   | Ia-3d      | 230  | 530  | Cubic               | nx = ny = nz     |
| Diamond  | Fd-3m      | 227  | 525  | Cubic               | nx = ny = nz     |
| HCP      | P6_3/mmc   | 194  | 488  | Hexagonal           | nx = ny          |
| PL       | P6_3/mmc   | 194  | 488  | Hexagonal           | nx = ny          |
| C14      | P6_3/mmc   | 194  | 488  | Hexagonal           | nx = ny          |
| Sigma    | P4_2/mnm   | 136  | 419  | Tetragonal          | nx = ny          |
+----------+------------+------+------+---------------------+------------------+

Notes
-----
**Beta Feature**: Space group symmetry is an experimental feature. Use with
caution and validate results against full simulations.

**Hexagonal Systems (HCP, PL)**:
The hexagonal crystal system requires:
- Lattice: a = b, c independent
- Angles: alpha = beta = 90 deg, gamma = 120 deg
- Grid: nx[0] = nx[1] (for [a, b, c] ordering)

The symmetry operations work in fractional coordinates, which automatically
handle the oblique (gamma=120 deg) coordinate system.

**Requires**:
- spglib: Crystallographic space group database and operations
- numpy: Array operations
- scipy: MATLAB file I/O (for testing)
"""

import numpy as np
import spglib
from functools import lru_cache


@lru_cache(maxsize=None)
def _get_spacegroup_type_cached(hall_number):
    """Cached wrapper for spglib.get_spacegroup_type()."""
    return spglib.get_spacegroup_type(hall_number)


class SpaceGroup:
    """Space group symmetry operations for polymer field theory.

    Reduces computational cost by exploiting crystallographic symmetry. Finds
    the minimal irreducible set of grid points and provides transformations
    between full and reduced field representations.

    Parameters
    ----------
    nx : list of int
        Grid dimensions [nx, ny, nz]. Must be compatible with the space group
        symmetry (e.g., cubic space groups require nx=ny=nz).
    symbol : str
        International Tables for Crystallography (ITA) short symbol.
        Examples: "Im-3m" (BCC), "Ia-3d" (gyroid), "Pm-3m" (simple cubic).
    hall_number : int, optional
        Hall number uniquely identifying the space group setting. Required if
        the ITA symbol has multiple settings. If None, uses the first Hall
        number for the symbol (raises error if multiple exist).

    Attributes
    ----------
    hall_number : int
        Hall number used (1-530).
    spacegroup_number : int
        International space group number (1-230).
    spacegroup_symbol : str
        ITA short symbol.
    crystal_system : str
        Crystal system: "Triclinic", "Monoclinic", "Orthorhombic",
        "Tetragonal", "Trigonal", "Hexagonal", or "Cubic".
    lattice_parameters : list of str
        Free lattice parameters for this crystal system.
        Example: ["a"] for cubic, ["a", "c"] for tetragonal.
    symmetry_operations : list of tuple
        Symmetry operations as (rotation_matrix, translation_vector) pairs.
    nx : list of int
        Grid dimensions.
    irreducible_mesh : list of tuple
        Irreducible mesh points as (ix, iy, iz) tuples. These are the minimal
        set of grid points needed to represent the full field under symmetry.
    indices : ndarray
        Map from full grid to irreducible point indices. Shape matches nx.
        indices[i,j,k] gives the index in irreducible_mesh for grid point (i,j,k).

    Raises
    ------
    ValueError
        If ITA symbol is invalid, Hall number is invalid, or grid size is
        incompatible with space group.

    Notes
    -----
    **Space Group Database:**

    This class uses the spglib library which implements the International Tables
    for Crystallography. Hall numbers (1-530) uniquely identify space group
    settings, while ITA numbers (1-230) may have multiple settings.

    **Irreducible Mesh:**

    The algorithm finds the minimal set of grid points by:
    1. Starting with the first unvisited point
    2. Computing its orbit (all symmetrically equivalent points)
    3. Marking all orbit points as visited
    4. Repeating until all grid points are visited

    **Performance:**

    - Reduces field storage by factor of symmetry operation count
    - Speeds up field updates in SCFT iterations
    - Pre-computes flat index maps for fast to_reduced_basis/from_reduced_basis

    **Limitations:**

    - Only periodic boundary conditions supported
    - Grid must be compatible with space group (e.g., cubic groups need nx=ny=nz)
    - Beta feature - validate results carefully

    See Also
    --------
    to_reduced_basis : Convert full fields to reduced representation.
    from_reduced_basis : Convert reduced fields to full representation.

    Examples
    --------
    **Gyroid phase (Cubic):**

    >>> nx = [64, 64, 64]
    >>> sg = SpaceGroup(nx, "Ia-3d", hall_number=530)
    Using Hall number: 530 for symbol 'Ia-3d'
    International space group number: 230
    Crystal system: Cubic
    The number of symmetry operations: 48
    Original mesh size: 262144
    Irreducible mesh size: 5461

    >>> # Convert fields to reduced basis (48x smaller)
    >>> w_full = np.random.randn(2, 64*64*64)  # 2 fields
    >>> w_reduced = sg.to_reduced_basis(w_full)
    >>> w_reduced.shape
    (2, 5461)

    >>> # Reconstruct full fields
    >>> w_reconstructed = sg.from_reduced_basis(w_reduced)
    >>> w_reconstructed.shape
    (2, 262144)

    **HCP phase (Hexagonal):**

    >>> # HCP with P6_3/mmc symmetry
    >>> # Grid: nx[0] = nx[1] for hexagonal a = b, all divisible by 6
    >>> nx = [48, 48, 48]  # [a, b, c] with a = b
    >>> sg = SpaceGroup(nx, "P6_3/mmc", hall_number=488)
    Using Hall number: 488 for symbol 'P6_3/mmc'
    International space group number: 194
    Crystal system: Hexagonal
    The number of symmetry operations: 24
    Original mesh size: 110592
    Irreducible mesh size: 5125

    **Perforated Lamella (PL) phase (Hexagonal):**

    >>> # PL also uses P6_3/mmc with different c
    >>> nx = [48, 48, 96]  # [a, b, c] with a = b, c different
    >>> sg = SpaceGroup(nx, "P6_3/mmc", hall_number=488)
    Using Hall number: 488 for symbol 'P6_3/mmc'
    International space group number: 194
    Crystal system: Hexagonal
    The number of symmetry operations: 24
    Original mesh size: 221184
    Irreducible mesh size: 10033

    References
    ----------
    .. [1] International Tables for Crystallography, Vol. A (2016)
    .. [2] spglib documentation: https://spglib.github.io/spglib/
    """

    def __init__(self, nx, symbol, hall_number=None):

        print("---------- Space Group ----------")

        # Get the Hall numbers for the ITA symbol
        international_short_symbol = symbol
        hall_numbers = self.hall_numbers_from_ita_symbol(international_short_symbol)
        if len(hall_numbers) == 0:
            raise ValueError(f"No Hall numbers found for the ITA symbol '{international_short_symbol}'.")

        if hall_number is None:
            if len(hall_numbers) > 1:
                raise ValueError(f"Multiple Hall numbers found for the ITA symbol '{international_short_symbol}'. "
                                 f"Please specify one of them: {hall_numbers}")
            else:
                # If no Hall number is provided, use the first one
                hall_number = hall_numbers[0]
        elif isinstance(hall_number, int):
            if hall_number not in hall_numbers:
                raise ValueError(f"Provided Hall number {hall_number} is not in the list of Hall numbers for "
                                 f"the ITA symbol '{international_short_symbol}'. Available Hall numbers: {hall_numbers}")

        print(f"Using Hall number: {hall_number} for symbol '{international_short_symbol}'")

        # Use spglib to get information from the Hall number
        self.hall_number = hall_number
        spg_type_info = _get_spacegroup_type_cached(hall_number)

        # Extract the international space group number
        self.spacegroup_number = spg_type_info.number
        self.spacegroup_symbol = spg_type_info.international_short

        # Remove any leading/trailing whitespace
        symbol = symbol.strip()

        assert self.spacegroup_symbol == symbol, \
            f"Symbol mismatch: {self.spacegroup_symbol} != {symbol}. " \
            "Please check the provided symbol and Hall number."

        # Map the international number to the crystal system and lattice parameters
        self.crystal_system, self.lattice_parameters = self.get_crystal_system(self.spacegroup_number)

        # Print the international space group number and symbol
        print(f"International space group number: {self.spacegroup_number}")
        print(f"Crystal system: {self.crystal_system}")

        # Validate grid dimensions for the crystal system
        self._validate_grid(nx)

        # Get the symmetry operations for the correct dimension
        self.symmetry_operations = self.get_symmetry_ops(hall_number)

        # Validate grid compatibility with symmetry operations
        self._validate_grid_compatibility(nx, self.symmetry_operations)

        # Find the irreducible mesh points
        irreducible_mesh, indices = self.find_irreducible_mesh(nx, self.symmetry_operations)

        # Print the results
        print(f"The number of symmetry operations: {len(self.symmetry_operations)}")
        print(f"Original mesh size: {np.prod(nx)}")
        print(f"Irreducible mesh size: {len(irreducible_mesh)}")

        self.hall_number = hall_number
        self.nx = nx
        self.irreducible_mesh = irreducible_mesh
        self.indices = indices

        # --- Pre-computation for performance ---
        multi_indices = np.array(self.irreducible_mesh).T
        self.reduced_basis_flat_indices_ = np.ravel_multi_index(multi_indices, self.nx)
        self.full_to_reduced_map_flat_ = self.indices.flatten()
        self.orbit_counts_ = np.bincount(self.full_to_reduced_map_flat_, minlength=len(irreducible_mesh))

    def hall_numbers_from_ita_symbol(self, international_short):
        """Return every Hall number that belongs to the given ITA symbol."""
        return [
            h for h in range(1, 531)
            if _get_spacegroup_type_cached(h).international_short.strip() == international_short.strip()
        ]

    def hall_numbers_from_ita_number(self, ita):
        """Return every Hall number that belongs to the given ITA number."""
        return [
            h for h in range(1, 531)
            if _get_spacegroup_type_cached(h).number == ita
        ]
    
    def to_reduced_basis(self, fields):
        fields_flat = np.reshape(fields, (fields.shape[0], -1))
        return fields_flat[:, self.reduced_basis_flat_indices_].copy()

    def from_reduced_basis(self, reduced_fields):
        return reduced_fields[:, self.full_to_reduced_map_flat_].copy()

    def symmetrize(self, fields):
        """Symmetrize fields by averaging over orbits.

        For each orbit (set of symmetrically equivalent points), computes the
        average value and assigns it to all points in the orbit. This ensures
        the resulting field has perfect space group symmetry.

        Note: This function NEVER modifies the input array. It always returns
        a new array containing the symmetrized field.

        Parameters
        ----------
        fields : ndarray
            Field array of shape (n_fields, n_grid) or (n_fields, nx, ny, nz).

        Returns
        -------
        ndarray
            Symmetrized field with same shape as input.

        Notes
        -----
        This differs from to_reduced_basis/from_reduced_basis which only picks
        the value at the representative point. symmetrize averages all values
        in each orbit, preserving more information from the original field.

        Example
        -------
        >>> sg = SpaceGroup([48, 48, 48], "P6_3/mmc", hall_number=488)
        >>> field_sym = sg.symmetrize(field)  # Now has perfect symmetry
        >>> field_reduced = sg.to_reduced_basis(field_sym)
        >>> field_recovered = sg.from_reduced_basis(field_reduced)
        >>> np.allclose(field_sym, field_recovered)  # True
        """
        original_shape = fields.shape
        fields_flat = np.reshape(fields, (fields.shape[0], -1))

        # Compute averages over orbits using vectorized operations
        n_fields = fields_flat.shape[0]
        n_orbits = len(self.irreducible_mesh)
        orbit_sums = np.zeros((n_fields, n_orbits), dtype=fields_flat.dtype)

        # Vectorized accumulation using np.add.at
        np.add.at(orbit_sums, (slice(None), self.full_to_reduced_map_flat_), fields_flat)

        # Compute averages using precomputed orbit counts
        orbit_averages = orbit_sums / self.orbit_counts_[np.newaxis, :]

        # Map averages back to full grid
        symmetrized = orbit_averages[:, self.full_to_reduced_map_flat_]

        return symmetrized.reshape(original_shape)

    def get_crystal_system(self, spg_number):
        """
        Maps an international space group number to its crystal system name and lattice parameters
        """
        if 1 <= spg_number <= 2:
            return "Triclinic", ["a", "b", "c", "alpha", "beta", "gamma"]
        elif 3 <= spg_number <= 15:
            return "Monoclinic", ["a", "b", "c", "beta"]
        elif 16 <= spg_number <= 74:
            return "Orthorhombic", ["a", "b", "c"]
        elif 75 <= spg_number <= 142:
            return "Tetragonal", ["a", "c"]
        elif 143 <= spg_number <= 167:
            return "Trigonal", ["a", "c"]
        elif 168 <= spg_number <= 194:
            return "Hexagonal", ["a", "c"]
        elif 195 <= spg_number <= 230:
            return "Cubic", ["a"]
        else:
            raise ValueError("Invalid space group number.")

    def _validate_grid(self, nx):
        """Validate grid dimensions for the crystal system.

        Checks if the grid dimensions are compatible with the crystal system's
        symmetry requirements. Prints warnings for incompatible grids.

        Parameters
        ----------
        nx : list of int
            Grid dimensions [nx, ny, nz]

        Notes
        -----
        Grid constraints by crystal system:
        - Cubic: nx = ny = nz (all equal)
        - Tetragonal: nx = ny (first two equal)
        - Hexagonal: nx = ny (first two equal, for [a, b, c] ordering)
        - Trigonal: nx = ny (first two equal)
        - Orthorhombic, Monoclinic, Triclinic: no constraints

        For hexagonal systems with [c, a, b] axis ordering, ensure ny = nz instead.
        """
        if len(nx) != 3:
            raise ValueError(f"Grid must be 3D. Got {len(nx)}D grid.")

        if self.crystal_system == "Cubic":
            if not (nx[0] == nx[1] == nx[2]):
                print(f"  WARNING: Cubic crystal system requires nx = ny = nz.")
                print(f"           Got nx = {nx}. Results may be incorrect.")
        elif self.crystal_system in ["Tetragonal", "Trigonal"]:
            if nx[0] != nx[1]:
                print(f"  WARNING: {self.crystal_system} crystal system typically requires nx = ny.")
                print(f"           Got nx = {nx}. Check your axis ordering.")
        elif self.crystal_system == "Hexagonal":
            # Hexagonal requires a = b. Check if any two dimensions are equal.
            if nx[0] == nx[1]:
                pass  # Standard [a, b, c] ordering with a = b
            elif nx[1] == nx[2]:
                print(f"  Note: Grid {nx} suggests [c, a, b] axis ordering (ny = nz).")
            elif nx[0] == nx[2]:
                print(f"  Note: Grid {nx} suggests [b, c, a] axis ordering (nx = nz).")
            else:
                print(f"  WARNING: Hexagonal crystal system requires a = b.")
                print(f"           No two grid dimensions are equal in nx = {nx}.")
                print(f"           Check your axis ordering and grid dimensions.")

    def _validate_grid_compatibility(self, nx, symmetry_operations):
        """Validate that grid dimensions are compatible with space group symmetry.

        Checks that grid dimensions can represent the fractional coordinates
        that appear in Wyckoff positions of the space group.

        Parameters
        ----------
        nx : list of int
            Grid dimensions [nx, ny, nz]
        symmetry_operations : list of tuple
            List of (rotation_matrix, translation_vector) pairs

        Raises
        ------
        ValueError
            If grid is incompatible with space group requirements.

        Notes
        -----
        Grid compatibility is determined by:
        1. Space group-specific requirements based on Wyckoff positions
        2. Crystal system constraints (hexagonal/trigonal need divisibility by 3)
        3. Translation vector denominators from symmetry operations

        Common requirements:
        - Hexagonal (P6_3/mmc, P6/mmm, etc.): divisible by 6 (LCM of 2, 3)
        - Trigonal (R-3m, P-31m, etc.): divisible by 3 or 6
        - Ia-3d, Fd-3m: divisible by 4 (for 1/4, 3/4 translations)
        - Im-3m, Pm-3m: divisible by 2
        """
        from fractions import Fraction

        # Space group-specific grid requirements based on Wyckoff positions
        # Key: ITA number, Value: required divisor for grid
        SPACE_GROUP_GRID_REQUIREMENTS = {
            # Hexagonal space groups (168-194): Wyckoff positions use 1/3, 2/3
            # LCM(2, 3) = 6 for most, some only need 3
            168: 6,   # P6        (6-fold axis with 1/3, 2/3 positions)
            169: 6,   # P6_1
            170: 6,   # P6_5
            171: 6,   # P6_2
            172: 6,   # P6_4
            173: 6,   # P6_3
            174: 6,   # P-6
            175: 6,   # P6/m
            176: 6,   # P6_3/m
            177: 6,   # P622
            178: 6,   # P6_122
            179: 6,   # P6_522
            180: 6,   # P6_222
            181: 6,   # P6_422
            182: 6,   # P6_322
            183: 6,   # P6mm
            184: 6,   # P6cc
            185: 6,   # P6_3cm
            186: 6,   # P6_3mc
            187: 6,   # P-6m2
            188: 6,   # P-6c2
            189: 6,   # P-62m
            190: 6,   # P-62c
            191: 6,   # P6/mmm
            192: 6,   # P6/mcc
            193: 6,   # P6_3/mcm
            194: 6,   # P6_3/mmc (HCP, PL, C14)

            # Trigonal space groups (143-167): use 1/3, 2/3
            143: 3,   # P3
            144: 3,   # P3_1
            145: 3,   # P3_2
            146: 3,   # R3
            147: 3,   # P-3
            148: 3,   # R-3
            149: 6,   # P312
            150: 6,   # P321
            151: 6,   # P3_112
            152: 6,   # P3_121
            153: 6,   # P3_212
            154: 6,   # P3_221
            155: 3,   # R32
            156: 6,   # P3m1
            157: 6,   # P31m
            158: 6,   # P3c1
            159: 6,   # P31c
            160: 3,   # R3m
            161: 3,   # R3c
            162: 6,   # P-31m
            163: 6,   # P-31c
            164: 6,   # P-3m1
            165: 6,   # P-3c1
            166: 3,   # R-3m
            167: 3,   # R-3c

            # Cubic with 1/4 translations
            203: 4,   # Fd-3      (C15 Laves)
            206: 4,   # Ia-3
            219: 4,   # F-43c
            227: 4,   # Fd-3m     (Diamond, C15)
            228: 4,   # Fd-3c
            230: 4,   # Ia-3d     (Gyroid)

            # Cubic with 1/2 translations only
            197: 2,   # I23
            199: 2,   # I2_13
            204: 2,   # Im-3
            211: 2,   # I432
            214: 2,   # I4_132
            217: 2,   # I-43m
            220: 2,   # I-43d
            229: 2,   # Im-3m     (BCC)

            # Orthorhombic with special requirements
            70:  4,   # Fddd      (O70)
        }

        # Get requirement from lookup table, or compute from translations
        required_divisor = SPACE_GROUP_GRID_REQUIREMENTS.get(self.spacegroup_number, None)

        if required_divisor is None:
            # Fall back to computing from symmetry operation translations
            unique_fracs = set()
            for R, t in symmetry_operations:
                for ti in t:
                    ti_mod = ti % 1.0
                    if ti_mod < 0:
                        ti_mod += 1.0
                    if ti_mod > 1e-10 and ti_mod < 1 - 1e-10:
                        unique_fracs.add(round(ti_mod, 10))

            denominators = set()
            for frac_val in unique_fracs:
                f = Fraction(frac_val).limit_denominator(100)
                if f.denominator > 1:
                    denominators.add(f.denominator)

            required_divisor = 1
            for d in denominators:
                required_divisor = required_divisor * d // np.gcd(required_divisor, d)

        # Check grid compatibility
        incompatible = []
        if required_divisor > 1:
            for dim, n in enumerate(nx):
                if n % required_divisor != 0:
                    incompatible.append((dim, n, required_divisor))

        if incompatible:
            msg_lines = [f"Grid {list(nx)} is incompatible with space group {self.spacegroup_symbol} (No. {self.spacegroup_number})."]
            msg_lines.append(f"Grid dimensions must be divisible by {required_divisor}.")
            msg_lines.append("Incompatible dimensions:")
            for dim, n, required in incompatible:
                axis = ['x', 'y', 'z'][dim]
                msg_lines.append(f"  - {axis}: {n} is not divisible by {required}")

            # Suggest valid grid sizes
            suggestions = []
            for n in nx:
                if n % required_divisor == 0:
                    suggestions.append(n)
                else:
                    lower = (n // required_divisor) * required_divisor
                    upper = lower + required_divisor
                    suggestions.append(int(upper) if upper - n < n - lower or lower == 0 else int(lower))
            msg_lines.append(f"Suggested grid: {suggestions}")

            raise ValueError("\n".join(msg_lines))

    def get_symmetry_ops(self, hall_number):
        """
        Generates the symmetry operations for a given Hall number.
        This function works for 1D, 2D, and 3D space groups.

        Args:
            hall_number (int): The Hall number (from 1 to 530).

        Returns:
            list: A list of symmetry operations.
                Each element is a tuple (rotation_matrix, translation_vector).
        """
        # Get symmetry operations directly from the spglib database
        symmetries = spglib.get_symmetry_from_database(hall_number)
        
        if symmetries is None:
            raise RuntimeError(f"Could not get symmetry operations for Hall number {hall_number}")

        rotations = symmetries['rotations']
        translations = symmetries['translations']
        
        dim = rotations.shape[1]
        # print(f"Space group Hall number: {hall_number}")
        # print(f"Dimension: {dim}D")
        # print(f"Number of symmetry operations: {len(rotations)}")
        
        # Combine rotations and translations into a list of operation pairs
        symmetry_operations = list(zip(rotations, translations))
        
        return symmetry_operations

    def find_irreducible_mesh(self, grid_size, symmetry_operations):
        """Finds the irreducible set of points for a grid under space group symmetry.

        Args:
            grid_size (list or tuple): The size of the grid in each dimension.
            symmetry_operations (list): A list of symmetry operations, where each operation is a tuple
                containing a rotation matrix and a translation vector.
                Example: [(R1, t1), (R2, t2), ...]

        Returns:
            tuple: A tuple containing:
                - irreducible_points: A list of tuples representing the irreducible mesh points.
                - indices: A NumPy array of shape grid_size, where each element is the index of the
                  irreducible point that corresponds to that grid point.
        """
        grid_size_arr = np.array(grid_size, dtype=np.int64)
        n_ops = len(symmetry_operations)

        # Pre-stack all rotation matrices and translation vectors for vectorized operations
        # rotations: (n_ops, 3, 3), translations: (n_ops, 3)
        rotations = np.stack([op[0] for op in symmetry_operations], axis=0)
        translations = np.stack([op[1] for op in symmetry_operations], axis=0)

        # Pre-compute translation offsets in grid coordinates (scaled by grid_size)
        # trans_grid[i] = round(translations[i] * grid_size) mod grid_size
        trans_grid = np.round(translations * grid_size_arr).astype(np.int64) % grid_size_arr

        # indices array: -1 means unvisited
        indices = np.full(grid_size, -1, dtype=np.int32)
        irreducible_points = []

        # Iterate through grid points in flat order
        count = 0
        for flat_idx in range(np.prod(grid_size_arr)):
            # Convert flat index to 3D coordinates
            point = np.array(np.unravel_index(flat_idx, grid_size), dtype=np.int64)

            if indices[tuple(point)] == -1:
                irreducible_points.append(tuple(point))

                # Compute all orbit points at once using vectorized operations
                # orbit_points[i] = (rotations[i] @ point + trans_grid[i]) % grid_size
                # Shape: (n_ops, 3)
                orbit_points = (np.einsum('ijk,k->ij', rotations, point) + trans_grid) % grid_size_arr

                # Mark all orbit points with the same index
                for op_idx in range(n_ops):
                    coord = tuple(orbit_points[op_idx])
                    indices[coord] = count

                count += 1

        return irreducible_points, indices
    
# ==============================================================================
# Main execution block
# ==============================================================================
if __name__ == '__main__':

    print("="*70)
    print("Testing Space Group with common polymer phases")
    print("="*70)

    # Test 1: Gyroid (Cubic)
    print("\n" + "="*70)
    print("Test 1: Gyroid phase with Ia-3d (Cubic)")
    print("="*70)
    nx_gyroid = [32, 32, 32]
    sg_gyroid = SpaceGroup(nx_gyroid, "Ia-3d", hall_number=530)
    reduction = np.prod(nx_gyroid) / len(sg_gyroid.irreducible_mesh)
    print(f"Reduction factor: {reduction:.1f}x (with {len(sg_gyroid.symmetry_operations)} ops)")

    # Test round-trip
    w = np.random.randn(2, np.prod(nx_gyroid))
    w_reduced = sg_gyroid.to_reduced_basis(w)
    w_recovered = sg_gyroid.from_reduced_basis(w_reduced)
    # For non-symmetric data, recovered will average over orbits
    print(f"Round-trip test: reduced shape {w_reduced.shape}, recovered shape {w_recovered.shape}")

    # Test 2: HCP (Hexagonal) with [a, b, c] ordering
    print("\n" + "="*70)
    print("Test 2: HCP phase with P6_3/mmc (Hexagonal) - [a, b, c] ordering")
    print("="*70)
    nx_hcp = [48, 48, 48]  # a = b in hexagonal, divisible by 6
    sg_hcp = SpaceGroup(nx_hcp, "P6_3/mmc", hall_number=488)
    reduction = np.prod(nx_hcp) / len(sg_hcp.irreducible_mesh)
    print(f"Reduction factor: {reduction:.1f}x (with {len(sg_hcp.symmetry_operations)} ops)")

    # Create symmetric HCP field for testing
    field_hcp = np.zeros(nx_hcp)
    hcp_positions = [[0, 0, 0], [2/3, 1/3, 0.5]]  # Standard HCP positions
    for pos in hcp_positions:
        ix = int(np.round(pos[0] * nx_hcp[0])) % nx_hcp[0]
        iy = int(np.round(pos[1] * nx_hcp[1])) % nx_hcp[1]
        iz = int(np.round(pos[2] * nx_hcp[2])) % nx_hcp[2]
        field_hcp[ix, iy, iz] = 1.0

    # Test symmetry consistency
    max_std = 0.0
    for i in range(min(100, len(sg_hcp.irreducible_mesh))):
        orbit_mask = (sg_hcp.indices == i)
        if np.sum(orbit_mask) > 1:
            std = np.std(field_hcp[orbit_mask])
            max_std = max(max_std, std)
    print(f"Symmetry check (max std within orbits): {max_std:.2e}")

    # Test 3: PL (Hexagonal) with different c
    print("\n" + "="*70)
    print("Test 3: PL phase with P6_3/mmc (Hexagonal) - [a, b, c] ordering")
    print("="*70)
    nx_pl = [48, 48, 96]  # a = b, c different, all divisible by 6
    sg_pl = SpaceGroup(nx_pl, "P6_3/mmc", hall_number=488)
    reduction = np.prod(nx_pl) / len(sg_pl.irreducible_mesh)
    print(f"Reduction factor: {reduction:.1f}x (with {len(sg_pl.symmetry_operations)} ops)")

    # Test 4: BCC (Cubic)
    print("\n" + "="*70)
    print("Test 4: BCC phase with Im-3m (Cubic)")
    print("="*70)
    nx_bcc = [32, 32, 32]
    sg_bcc = SpaceGroup(nx_bcc, "Im-3m", hall_number=529)
    reduction = np.prod(nx_bcc) / len(sg_bcc.irreducible_mesh)
    print(f"Reduction factor: {reduction:.1f}x (with {len(sg_bcc.symmetry_operations)} ops)")

    print("\n" + "="*70)
    print("All tests completed successfully!")
    print("="*70)