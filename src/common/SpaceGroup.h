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
 * @brief Physical (CrysFFT) basis layered on top of the irreducible basis.
 *
 * At most one physical basis is active at a time; it selects which
 * crystallographic FFT fast path the solvers may use.
 */
enum class PhysicalBasis
{
    None,     ///< Irreducible (full-group) basis only
    Pmmm,     ///< Pmmm mirror-plane basis (DCT CrysFFT)
    M3,       ///< Recursive 3m basis
    ZMirror   ///< z-mirror basis for hexagonal-axes settings (ObliqueZ CrysFFT)
};

/**
 * @class SpaceGroup
 * @brief Manages space group symmetry for reduced basis field operations.
 *
 * Uses spglib to obtain symmetry operations and computes irreducible mesh points.
 */
class SpaceGroup
{
    template <typename T> friend class ComputationBox;
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
    PhysicalBasis physical_basis_{PhysicalBasis::None};  ///< Active physical basis (at most one)
    mutable bool in_use_{false};  ///< set once a computation box or solver holds this group
    int z_mirror_shift_{0};
    double z_mirror_translation_{0.0};

    /// Initialize space group from Hall number.
    void initialize(std::vector<int> nx, int hall_number);

    /// Get symmetry operations from spglib database.
    void get_symmetry_operations();

    /// Find irreducible mesh points using orbit algorithm.
    void find_irreducible_mesh();

    /// Determine crystal system from space group number.
    void determine_crystal_system();

    /// Validate grid dimensions for the crystal system.
    void validate_grid();

    /// Validate grid compatibility with space group symmetry.
    void validate_grid_compatibility();

    /// Per axis: the grid size must be a multiple of the lcm of the denominators of
    /// the translation components of all symmetry operations along that axis.
    std::array<int, 3> get_required_grid_divisors() const;

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
    /// True for hexagonal/trigonal groups in a hexagonal-axes setting (every rotation
    /// is z-block-diagonal: the cell needs a = b, gamma = 120); false for the
    /// rhombohedral-axes settings of R groups (3-fold about [111]: a = b = c,
    /// alpha = beta = gamma) and for every other crystal system.
    bool uses_hexagonal_axes() const;

    /**
     * @brief Throw unless the cell belongs to the crystal system of the group.
     *
     * Otherwise the symmetry operations are not isometries of the box and a run
     * converges to a constrained, unphysical state. Cubic: a = b = c and right
     * angles; tetragonal: a = b and right angles; orthorhombic: right angles;
     * hexagonal and hexagonal-axes trigonal: a = b, alpha = beta = 90,
     * gamma = 120; rhombohedral-axes trigonal: a = b = c and
     * alpha = beta = gamma. Lengths and angles agree to a relative 1e-6.
     * Monoclinic and triclinic: every rotation preserves the cell metric
     * (R^T G R = G to a relative 1e-6), which puts the monoclinic unique
     * axis perpendicular to the other two.
     *
     * @param lx         Box lengths [a, b, c]
     * @param angles_deg Lattice angles [alpha, beta, gamma] in degrees
     */
    void check_cell(const std::vector<double>& lx, const std::vector<double>& angles_deg) const;
    const std::vector<int>& get_nx() const { return nx_; }
    int get_total_grid() const { return total_grid_; }
    int get_n_reduced_basis() const { return n_irreducible_; }
    int get_n_reduced_basis_full() const { return n_irreducible_full_ > 0 ? n_irreducible_full_ : n_irreducible_; }
    int get_n_symmetry_ops() const { return n_symmetry_ops_; }
    const std::vector<int>& get_reduced_basis_indices() const { return reduced_basis_indices_; }
    const std::vector<int>& get_full_to_reduced_map() const { return full_to_reduced_map_; }
    const std::vector<int>& get_orbit_counts() const { return orbit_counts_; }

    /**
     * @brief True when a full-grid field is constant (within tolerance) on every
     *        orbit of the full space group, also when a physical basis is enabled.
     */
    bool is_invariant(const double* full_field, double tolerance) const;

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
     * @brief Get the translations of all pure-translation operations.
     *
     * Collects the translational parts of every symmetry operation whose
     * rotation matrix is the identity: the trivial (0,0,0) plus the lattice
     * centering translations (e.g. (1/2,1/2,1/2) for I, the three
     * half-half-zero vectors for F). Used by the CrysFFT selector to search
     * alternative mirror coset representatives when the minimal-norm ones
     * give a singular fold-parity permutation.
     *
     * @param translations Output vector of translation vectors in [0, 1).
     */
    void get_pure_translations(std::vector<std::array<double, 3>>& translations) const;

    /**
     * @brief Get translation along z for mirror plane (x,y,-z) symmetry.
     *
     * Searches symmetry operations for reflection:
     *   (x,y,z) -> (x,y,-z) + t
     * Returns the z-translation t_z in [0,1).
     *
     * Only pure z-mirrors with integer x/y translations are accepted.
     *
     * @param t_z Output translation along z in [0,1).
     * @param tol Tolerance for translation matching.
     * @return True if a valid mirror is found.
     */
    bool get_z_mirror_translation(double& t_z, double tol = 1e-10) const;

    /// Enable one physical basis (None restores the irreducible basis).
    void enable_physical_basis(PhysicalBasis basis);
    PhysicalBasis physical_basis() const { return physical_basis_; }
    /// Human-readable name of the active basis ("irreducible", "pmmm", "m3", "z-mirror").
    std::string physical_basis_name() const;
    /// Called by the computation box and the solvers that take this group. Their
    /// reduced-basis sizes and maps are snapshots, so the physical basis is frozen
    /// afterwards (enable_*_physical_basis() throws).
    void mark_in_use() const { in_use_ = true; }
    bool in_use() const { return in_use_; }

    /**
     * @brief Enable Pmmm physical basis (1/8 grid) mapping.
     *
     * Replaces the irreducible basis with the Pmmm physical basis
     * (first octant, size Nx/2 * Ny/2 * Nz/2). Requires 3D even grid
     * and mirror planes along x, y, z.
     */
    void enable_pmmm_physical_basis();
    bool using_pmmm_physical_basis() const { return physical_basis_ == PhysicalBasis::Pmmm; }

    /**
     * @brief Enable 3m physical basis (1/8 even-index grid) mapping.
     *
     * Replaces the irreducible basis with the 3m physical basis
     * (even indices, size Nx/2 * Ny/2 * Nz/2). Requires 3D even grid
     * and valid 3m translations.
     */
    void enable_m3_physical_basis();
    /**
     * @brief Whether the recursive 3m CrysFFT is feasible on this grid.
     *
     * The m3 physical basis only pays off with that transform (even grid,
     * nz >= 16, compatible translation parity); SCFT checks this before
     * enabling the basis and otherwise uses the Pmmm basis.
     */
    bool m3_crysfft_feasible() const;
    bool using_m3_physical_basis() const { return physical_basis_ == PhysicalBasis::M3; }

    /**
     * @brief Enable z-mirror physical basis (half grid along z).
     *
     * Uses a mirror plane perpendicular to z (optionally with a 1/2 translation),
     * enabling a half-grid physical basis along z. Requires 3D even Nz.
     *
     * For t_z = 1/2, Nz must be divisible by 4 to align the half-cell shift.
     */
    void enable_z_mirror_physical_basis();
    bool using_z_mirror_physical_basis() const { return physical_basis_ == PhysicalBasis::ZMirror; }
    int get_z_mirror_shift() const { return z_mirror_shift_; }
    double get_z_mirror_translation() const { return z_mirror_translation_; }

};

#endif // SPACE_GROUP_H_
