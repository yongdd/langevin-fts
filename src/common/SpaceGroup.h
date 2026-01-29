/**
 * @file SpaceGroup.h
 * @brief Space group symmetry operations for reduced basis field representation.
 *
 * This class uses spglib to obtain symmetry operations from Hall number and
 * computes the irreducible mesh for reduced basis field representation.
 */

#ifndef SPACE_GROUP_H_
#define SPACE_GROUP_H_

#include <vector>
#include <array>
#include <string>

/**
 * @class SpaceGroup
 * @brief Manages space group symmetry for reduced basis field operations.
 *
 * Uses spglib to obtain symmetry operations and computes irreducible mesh points.
 */
class SpaceGroup
{
private:
    int hall_number_;                          ///< Hall number (1-530)
    int spacegroup_number_;                    ///< ITA space group number (1-230)
    std::string spacegroup_symbol_;            ///< ITA short symbol
    std::string crystal_system_;               ///< Crystal system name

    std::vector<int> nx_;                      ///< Grid dimensions [Nx, Ny, Nz]
    int total_grid_;                           ///< Total grid points (Nx * Ny * Nz)
    int n_irreducible_;                        ///< Number of reduced-basis points
    int n_symmetry_ops_;                       ///< Number of symmetry operations

    // Symmetry operations from spglib
    std::vector<std::array<std::array<int, 3>, 3>> rotations_;    ///< Rotation matrices (n_ops x 3 x 3)
    std::vector<std::array<double, 3>> translations_;              ///< Translation vectors (n_ops x 3)

    std::vector<int> reduced_basis_indices_;   ///< Flat indices of reduced-basis points (size: n_irreducible)
    std::vector<int> full_to_reduced_map_;     ///< Map from full grid to irreducible index (size: total_grid)
    std::vector<int> orbit_counts_;            ///< Number of points in each orbit (size: n_irreducible)

    // Irreducible-basis snapshot (full space group) for enforcing symmetry
    int n_irreducible_full_{0};
    std::vector<int> reduced_basis_indices_full_;
    std::vector<int> full_to_reduced_map_full_;
    std::vector<int> orbit_counts_full_;
    std::vector<int> reduced_to_irreducible_map_; ///< Physical-basis index -> irreducible orbit index

    // Physical basis flags
    bool use_pmmm_physical_basis_{false};
    bool use_m3_physical_basis_{false};

    /**
     * @brief Initialize space group from Hall number.
     */
    void initialize(std::vector<int> nx, int hall_number);

    /**
     * @brief Get symmetry operations from spglib database.
     */
    void get_symmetry_operations();

    /**
     * @brief Find irreducible mesh points using orbit algorithm.
     */
    void find_irreducible_mesh();

    /**
     * @brief Determine crystal system from space group number.
     */
    void determine_crystal_system();

    /**
     * @brief Validate grid dimensions for the crystal system.
     */
    void validate_grid();

    /**
     * @brief Validate grid compatibility with space group symmetry.
     */
    void validate_grid_compatibility();

    /**
     * @brief Get required grid divisor for this space group.
     */
    int get_required_grid_divisor() const;

public:
    /**
     * @brief Construct SpaceGroup from Hall number and grid dimensions.
     *
     * @param nx Grid dimensions [Nx, Ny, Nz]
     * @param hall_number Hall number (1-530) uniquely identifying the space group setting
     */
    SpaceGroup(std::vector<int> nx, int hall_number);

    /**
     * @brief Construct SpaceGroup from ITA symbol and grid dimensions.
     *
     * @param nx Grid dimensions [Nx, Ny, Nz]
     * @param symbol ITA short symbol (e.g., "Im-3m", "Ia-3d")
     * @param hall_number Optional Hall number if symbol has multiple settings
     */
    SpaceGroup(std::vector<int> nx, const std::string& symbol, int hall_number = -1);

    /**
     * @brief Find Hall numbers for an ITA symbol.
     *
     * @param symbol ITA short symbol
     * @return Vector of Hall numbers matching the symbol
     */
    static std::vector<int> hall_numbers_from_symbol(const std::string& symbol);

    ~SpaceGroup() = default;

    /**
     * @brief Convert full field to reduced basis.
     *
     * Extracts field values at reduced basis points.
     *
     * @param full_field Input field on full grid (size: n_fields * total_grid)
     * @param reduced_field Output field on reduced basis (size: n_fields * n_reduced)
     * @param n_fields Number of field components
     */
    void to_reduced_basis(const double* full_field, double* reduced_field, int n_fields) const;
    void reduced_to_irreducible(const double* reduced_field, double* irreducible_field, int n_fields) const;
    void irreducible_to_reduced(const double* irreducible_field, double* reduced_field, int n_fields) const;

    /**
     * @brief Convert reduced basis field to full grid.
     *
     * Broadcasts reduced-basis values to all equivalent grid points.
     *
     * @param reduced_field Input field on reduced basis (size: n_fields * n_reduced)
     * @param full_field Output field on full grid (size: n_fields * total_grid)
     * @param n_fields Number of field components
     */
    void from_reduced_basis(const double* reduced_field, double* full_field, int n_fields) const;

    /**
     * @brief Symmetrize field by averaging over orbits.
     *
     * For each orbit, computes the average value and assigns it to all points.
     *
     * @param field Input/output field on full grid (size: n_fields * total_grid)
     * @param n_fields Number of field components
     */
    void symmetrize(double* field, int n_fields) const;

    /**
     * @brief Symmetrize field (out-of-place version).
     *
     * @param field_in Input field on full grid
     * @param field_out Output symmetrized field on full grid
     * @param n_fields Number of field components
     */
    void symmetrize(const double* field_in, double* field_out, int n_fields) const;

    /**
     * @brief Symmetrize reduced-basis fields using full space-group orbits.
     *
     * When a physical basis is enabled, reduced basis points may include multiple
     * representatives from the same irreducible orbit. This routine averages
     * those representatives to enforce full space-group symmetry.
     *
     * @param reduced_field Input/output field on reduced basis
     * @param n_fields Number of field components
     */
    void symmetrize_reduced_basis(double* reduced_field, int n_fields) const;

    // Getters
    int get_hall_number() const { return hall_number_; }
    int get_spacegroup_number() const { return spacegroup_number_; }
    const std::string& get_spacegroup_symbol() const { return spacegroup_symbol_; }
    const std::string& get_crystal_system() const { return crystal_system_; }
    const std::vector<int>& get_nx() const { return nx_; }
    int get_total_grid() const { return total_grid_; }
    int get_n_reduced_basis() const { return n_irreducible_; }
    int get_n_reduced_basis_full() const { return n_irreducible_full_ > 0 ? n_irreducible_full_ : n_irreducible_; }
    int get_n_symmetry_ops() const { return n_symmetry_ops_; }
    const std::vector<int>& get_reduced_basis_indices() const { return reduced_basis_indices_; }
    const std::vector<int>& get_full_to_reduced_map() const { return full_to_reduced_map_; }
    const std::vector<int>& get_orbit_counts() const { return orbit_counts_; }

    /**
     * @brief Check for mirror planes perpendicular to x, y, and z axes.
     *
     * Returns true if symmetry operations include reflections x->-x,
     * y->-y, and z->-z with zero translation (modulo lattice).
     */
    bool has_mirror_planes_xyz(double tol = 1e-10) const;

    /**
     * @brief Get translational parts for 3m symmetry planes (x,y,z).
     *
     * Searches symmetry operations for reflections:
     *  - z-plane: (x,y,z) -> (x,y,-z) + t_z
     *  - y-plane: (x,y,z) -> (x,-y,z) + t_y
     *  - x-plane: (x,y,z) -> (-x,y,z) + t_x
     *
     * The translational parts are returned as a 9-element array:
     *   [t_zx, t_zy, t_zz,  t_yx, t_yy, t_yz,  t_xx, t_xy, t_xz]
     *
     * Values are normalized to [0, 1).
     *
     * @param g Output array of translational parts.
     * @param tol Tolerance for matching rotations.
     * @return True if all three symmetry planes are found.
     */
    bool get_m3_translations(std::array<double, 9>& g, double tol = 1e-10) const;

    /**
     * @brief Enable Pmmm physical basis (1/8 grid) mapping.
     *
     * Replaces the irreducible basis with the Pmmm physical basis
     * (first octant, size Nx/2 * Ny/2 * Nz/2). Requires 3D even grid
     * and mirror planes along x, y, z.
     */
    void enable_pmmm_physical_basis();
    bool using_pmmm_physical_basis() const { return use_pmmm_physical_basis_; }

    /**
     * @brief Enable 3m physical basis (1/8 even-index grid) mapping.
     *
     * Replaces the irreducible basis with the 3m physical basis
     * (even indices, size Nx/2 * Ny/2 * Nz/2). Requires 3D even grid
     * and valid 3m translations.
     */
    void enable_m3_physical_basis();
    bool using_m3_physical_basis() const { return use_m3_physical_basis_; }

};

#endif // SPACE_GROUP_H_
