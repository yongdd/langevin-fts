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
    int n_irreducible_;                        ///< Number of irreducible mesh points
    int n_symmetry_ops_;                       ///< Number of symmetry operations

    // Symmetry operations from spglib
    std::vector<std::array<std::array<int, 3>, 3>> rotations_;    ///< Rotation matrices (n_ops x 3 x 3)
    std::vector<std::array<double, 3>> translations_;              ///< Translation vectors (n_ops x 3)

    std::vector<int> reduced_basis_indices_;   ///< Flat indices of irreducible points (size: n_irreducible)
    std::vector<int> full_to_reduced_map_;     ///< Map from full grid to irreducible index (size: total_grid)
    std::vector<int> orbit_counts_;            ///< Number of points in each orbit (size: n_irreducible)

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
     * Extracts field values at irreducible mesh points.
     *
     * @param full_field Input field on full grid (size: n_fields * total_grid)
     * @param reduced_field Output field on reduced basis (size: n_fields * n_irreducible)
     * @param n_fields Number of field components
     */
    void to_reduced_basis(const double* full_field, double* reduced_field, int n_fields) const;

    /**
     * @brief Convert reduced basis field to full grid.
     *
     * Broadcasts irreducible values to all equivalent grid points.
     *
     * @param reduced_field Input field on reduced basis (size: n_fields * n_irreducible)
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

    // Getters
    int get_hall_number() const { return hall_number_; }
    int get_spacegroup_number() const { return spacegroup_number_; }
    const std::string& get_spacegroup_symbol() const { return spacegroup_symbol_; }
    const std::string& get_crystal_system() const { return crystal_system_; }
    const std::vector<int>& get_nx() const { return nx_; }
    int get_total_grid() const { return total_grid_; }
    int get_n_irreducible() const { return n_irreducible_; }
    int get_n_symmetry_ops() const { return n_symmetry_ops_; }
    const std::vector<int>& get_reduced_basis_indices() const { return reduced_basis_indices_; }
    const std::vector<int>& get_full_to_reduced_map() const { return full_to_reduced_map_; }
    const std::vector<int>& get_orbit_counts() const { return orbit_counts_; }
};

#endif // SPACE_GROUP_H_
