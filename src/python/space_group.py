"""Crystallographic space group symmetry constraints for polymer simulations.

This module provides the SpaceGroup class for applying space group symmetry
to reduce computational cost in polymer field theory simulations. By exploiting
crystallographic symmetry, fields can be represented using only irreducible
mesh points rather than the full simulation grid.

Notes
-----
**Beta Feature**: Space group symmetry is an experimental feature. Use with
caution and validate results against full simulations.

**Requires**:
- spglib: Crystallographic space group database and operations
- numpy: Array operations
- scipy: MATLAB file I/O (for testing)
"""

import numpy as np
import json
import spglib
import scipy.io

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
    >>> # Gyroid phase with Ia-3d symmetry
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
        spg_type_info = spglib.get_spacegroup_type(hall_number)

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
        # print(f"International space group symbol: {self.spacegroup_symbol}")
        print(f"Crystal system: {self.crystal_system}")

        # Get the symmetry operations for the correct dimension
        self.symmetry_operations = self.get_symmetry_ops(hall_number)

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

    def hall_numbers_from_ita_symbol(self, international_short):
        """Return every Hall number that belongs to the given ITA symbol."""
        return [
            h for h in range(1, 531)
            if spglib.get_spacegroup_type(h).international_short.strip() == international_short.strip()
        ]

    def hall_numbers_from_ita_number(self, ita):
        """Return every Hall number that belongs to the given ITA number."""
        return [
            h for h in range(1, 531)
            if spglib.get_spacegroup_type(h).number == ita
        ]
    
    def to_reduced_basis(self, fields):
        fields_flat = np.reshape(fields, (fields.shape[0], -1))
        return fields_flat[:, self.reduced_basis_flat_indices_].copy()

    def from_reduced_basis(self, reduced_fields):
        return reduced_fields[:, self.full_to_reduced_map_flat_].copy()

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
        """        Finds the irreducible set of points for a grid of any dimension (1D, 2D, or 3D)
        under the symmetry of the specified space group.
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

        # Infer dimension from grid_size and convert to a NumPy array for vectorized math
        grid_size = np.flip(grid_size)  # Ensure the order is correct for multi-dimensional arrays
        dim = len(grid_size)
        grid_size_arr = np.array(grid_size)
        
        # 1. Create a boolean grid to keep track of visited points.
        #    This works for any dimension.
        indices = np.zeros(grid_size, dtype=np.int32)-1
        
        # 2. Create an empty list to store the representative points.
        irreducible_points = []

        # 3. Iterate through every point in the full grid using np.ndindex.
        #    This is a general way to iterate over a multi-dimensional array.

        count = 0
        for point_coord_int in np.ndindex(*grid_size):
            
            # 5. If we find a point that has not been visited yet...
            if indices[point_coord_int] == -1:
                # ...it's a new representative point. Add it to our irreducible set.
                irreducible_points.append(point_coord_int)
                
                # 6. Find its entire "orbit" (all symmetrically equivalent points)
                #    and mark them as visited.
                
                # Convert integer grid point to fractional coordinates (vectorized).
                # e.g., (i, j, k) -> (i/N1, j/N2, k/N3)
                point_coord_frac = np.array(point_coord_int) / grid_size_arr
                
                for rot_matrix, trans_vector in symmetry_operations:

                    # Apply the symmetry operation: p' = R*p + t
                    new_coord_frac = np.dot(rot_matrix, point_coord_frac) + trans_vector

                    # Convert back to integer grid indices with periodic boundary conditions.
                    # This vectorized approach is robust and dimension-agnostic.
                    # It rounds to nearest integer, casts to int, and applies modulo.
                    coords = np.round(new_coord_frac * grid_size_arr).astype(int)
                    assert np.all(np.isclose(np.abs(coords - new_coord_frac * grid_size_arr), 0.0)) == True, \
                        f"Coordinates should be close to integers after rounding {new_coord_frac*grid_size_arr}."
                    
                    new_coord_int_arr = np.mod(coords, grid_size_arr)
                    
                    # Mark this equivalent point as visited.
                    # We convert the numpy array to a tuple for indexing.
                    indices[tuple(new_coord_int_arr)] = count
                count += 1

        # Reverse the order for correct indexing in multi-dimensional arrays
        indices = np.transpose(indices, (2, 1, 0))
        # Reverse the order of each point
        for i in range(len(irreducible_points)):
            irreducible_points[i] = irreducible_points[i][::-1] 

        return irreducible_points, indices
    
# ==============================================================================
# Main execution block
# ==============================================================================
if __name__ == '__main__':

    # phase = "DG"
    # # Load the input data from a JSON file
    # with open(f'../phases/{phase}.json', 'r') as file:
    #     input_data = json.load(file)

    # # Load the input data from a .mat file
    # input_data = scipy.io.loadmat(f'../phases/{phase}.mat', squeeze_me=True)
    
    input_data = scipy.io.loadmat('fields.mat', squeeze_me=True)
    
    nx = np.array(input_data["nx"])
    lx = np.array(input_data["lx"])
    density = np.reshape(np.array(input_data["phi_A"]), nx)

    print(f"nx: {nx}")
    print(f"lx: {lx}")

    sg = SpaceGroup(nx, "Ia-3d", 530)

    max_diff = 0.0
    max_std = 0.0
    print("\nIrreducible mesh points:")
    for i, point in enumerate(sg.irreducible_mesh):
        diff = np.abs(density[point]-np.mean(density[sg.indices == i]))
        std  = np.std(density[sg.indices == i])
        max_diff = max(max_diff, diff)
        max_std = max(max_std, std)

        # print(f"Index {i} -> {np.sum(indices == i)} points, {point}, {diff}, {std}")
        # print(density[indices == i])

    print(f"Max difference: {max_diff}")
    print(f"Max std: {max_std}")
    
    w = np.zeros((2, np.prod(nx)))
    w[0] = np.array(input_data["w_A"])
    w[1] = np.array(input_data["w_B"])
    
    w_reduced_basis = sg.to_reduced_basis(w)
    print("Reduced basis w:", w_reduced_basis.shape)
    w_converted = sg.from_reduced_basis(w_reduced_basis)
    print("Converted w:", w_converted.shape)
    
    print(w_reduced_basis[0,0:10])
    print(w[0,0:10])
    print(w_converted[0,0:10])
    print(np.mean(w-w_converted, axis=1), np.std(w-w_converted, axis=1))