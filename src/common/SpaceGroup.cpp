/**
 * @file SpaceGroup.cpp
 * @brief Implementation of SpaceGroup class using spglib for symmetry operations.
 */

#include "SpaceGroup.h"
#include "GridLimits.h"
#include "Exception.h"
#include "CrysFFTSelector.h"
#include <cstring>
#include <cmath>
#include <cctype>
#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <limits>

extern "C" {
#include <spglib.h>
}

SpaceGroup::SpaceGroup(std::vector<int> nx, int hall_number)
{
    initialize(std::move(nx), hall_number);
}

SpaceGroup::SpaceGroup(std::vector<int> nx, const std::string& symbol, int hall_number)
{
    // Find Hall numbers for the symbol
    std::vector<int> hall_numbers = hall_numbers_from_symbol(symbol);

    if (hall_numbers.empty())
    {
        throw_with_line_number("No Hall numbers found for ITA symbol '" + symbol + "'");
    }

    if (hall_number == -1)
    {
        // No Hall number specified
        if (hall_numbers.size() > 1)
        {
            std::string msg = "Multiple Hall numbers found for symbol '" + symbol + "': [";
            for (size_t i = 0; i < hall_numbers.size(); ++i)
            {
                if (i > 0) msg += ", ";
                msg += std::to_string(hall_numbers[i]);
            }
            msg += "]. Please specify one.";
            throw_with_line_number(msg);
        }
        hall_number = hall_numbers[0];
    }
    else
    {
        // Verify provided Hall number is valid for this symbol
        if (std::find(hall_numbers.begin(), hall_numbers.end(), hall_number) == hall_numbers.end())
        {
            std::string msg = "Hall number " + std::to_string(hall_number) +
                " is not valid for symbol '" + symbol + "'. Valid Hall numbers: [";
            for (size_t i = 0; i < hall_numbers.size(); ++i)
            {
                if (i > 0) msg += ", ";
                msg += std::to_string(hall_numbers[i]);
            }
            msg += "]";
            throw_with_line_number(msg);
        }
    }

    initialize(std::move(nx), hall_number);
}

std::vector<int> SpaceGroup::hall_numbers_from_symbol(const std::string& symbol)
{
    std::vector<int> result;
    std::string symbol_trimmed = symbol;

    // Trim whitespace
    symbol_trimmed.erase(0, symbol_trimmed.find_first_not_of(" \t"));
    symbol_trimmed.erase(symbol_trimmed.find_last_not_of(" \t") + 1);

    for (int h = 1; h <= 530; ++h)
    {
        SpglibSpacegroupType spg_type = spg_get_spacegroup_type(h);
        std::string spg_symbol = spg_type.international_short;

        // Trim whitespace from spglib symbol
        spg_symbol.erase(0, spg_symbol.find_first_not_of(" \t"));
        spg_symbol.erase(spg_symbol.find_last_not_of(" \t") + 1);

        if (spg_symbol == symbol_trimmed)
        {
            result.push_back(h);
        }
    }

    return result;
}

void SpaceGroup::initialize(std::vector<int> nx, int hall_number)
{
    nx_ = std::move(nx);
    hall_number_ = hall_number;

    if (nx_.size() != 3)
    {
        throw_with_line_number("SpaceGroup requires 3D grid dimensions");
    }

    if (hall_number_ < 1 || hall_number_ > 530)
    {
        throw_with_line_number("Hall number must be between 1 and 530, got " +
            std::to_string(hall_number_));
    }

    // Compute total grid size, in 64 bits: the space group is created before the
    // computation box and must reject the same grids (see GridLimits.h)
    const long long n_points = static_cast<long long>(nx_[0]) * nx_[1] * nx_[2];
    if (n_points > MAX_GRID_POINTS)
        throw_with_line_number("The grid has " + std::to_string(n_points) + " points; at most " +
            std::to_string(MAX_GRID_POINTS) + " are supported, because grid arrays are indexed "
            "with 32-bit integers and the largest of them hold two fields.");
    total_grid_ = static_cast<int>(n_points);

    // Get space group info from spglib
    SpglibSpacegroupType spg_type = spg_get_spacegroup_type(hall_number_);
    spacegroup_number_ = spg_type.number;
    spacegroup_symbol_ = spg_type.international_short;

    // Determine crystal system
    determine_crystal_system();

    // Validate grid for crystal system (e.g., cubic requires nx=ny=nz)
    validate_grid();

    // Get symmetry operations (needed for grid compatibility check)
    get_symmetry_operations();

    // Validate grid compatibility with symmetry operations
    validate_grid_compatibility();

    // Find irreducible mesh
    find_irreducible_mesh();

    // Preserve full-space-group irreducible mapping
    n_irreducible_full_ = n_irreducible_;
    reduced_basis_indices_full_ = reduced_basis_indices_;
    full_to_reduced_map_full_ = full_to_reduced_map_;
    orbit_counts_full_ = orbit_counts_;
    reduced_to_irreducible_map_.clear();

    // Print info
    std::cout << "---------- Space Group ----------" << std::endl;
    std::cout << "Hall number: " << hall_number_ << std::endl;
    std::cout << "International space group number: " << spacegroup_number_ << std::endl;
    std::cout << "Symbol: " << spacegroup_symbol_ << std::endl;
    std::cout << "Crystal system: " << crystal_system_ << std::endl;
    std::cout << "Number of symmetry operations: " << n_symmetry_ops_ << std::endl;
    std::cout << "Original mesh size: " << total_grid_ << std::endl;
    std::cout << "Reduced basis size (irreducible): " << n_irreducible_ << std::endl;
    if (nx_.size() == 3)
    {
        const bool even_grid = (nx_[0] % 2 == 0 && nx_[1] % 2 == 0 && nx_[2] % 2 == 0);
        std::array<double, 9> m3_part = {0,0,0, 0,0,0, 0,0,0};
        const bool has_m3 = even_grid && get_m3_translations(m3_part);
        if (even_grid && has_mirror_planes_xyz())
        {
            const int pmmm_size = (nx_[0] / 2) * (nx_[1] / 2) * (nx_[2] / 2);
            std::cout << "Pmmm physical basis size (if enabled): " << pmmm_size << std::endl;
        }
        else
        {
            std::cout << "Pmmm physical basis size (if enabled): n/a" << std::endl;
        }
        if (has_m3)
        {
            const int m3_size = (nx_[0] / 2) * (nx_[1] / 2) * (nx_[2] / 2);
            std::cout << "M3 physical basis size (if enabled): " << m3_size << std::endl;
        }
        else
        {
            std::cout << "M3 physical basis size (if enabled): n/a" << std::endl;
        }
    }
}

void SpaceGroup::validate_grid()
{
    if (crystal_system_ == "Cubic")
    {
        if (!(nx_[0] == nx_[1] && nx_[1] == nx_[2]))
        {
            throw_with_line_number("Cubic crystal system requires nx = ny = nz. Got [" +
                std::to_string(nx_[0]) + ", " + std::to_string(nx_[1]) + ", " +
                std::to_string(nx_[2]) + "]");
        }
    }
    else if (crystal_system_ == "Tetragonal" || crystal_system_ == "Trigonal")
    {
        if (nx_[0] != nx_[1])
        {
            throw_with_line_number(crystal_system_ + " crystal system requires nx = ny. Got [" +
                std::to_string(nx_[0]) + ", " + std::to_string(nx_[1]) + ", " +
                std::to_string(nx_[2]) + "]");
        }
    }
    else if (crystal_system_ == "Hexagonal")
    {
        if (nx_[0] != nx_[1])
        {
            throw_with_line_number("Hexagonal crystal system requires nx = ny. Got [" +
                std::to_string(nx_[0]) + ", " + std::to_string(nx_[1]) + ", " +
                std::to_string(nx_[2]) + "]");
        }
    }
}

std::array<int, 3> SpaceGroup::get_required_grid_divisors() const
{
    // Per axis: for every translation component t = p/q (in lowest terms) of a
    // symmetry operation along that axis, the grid size of the axis must be divisible
    // by q (a 1/2 translation along z alone, as in P6_3/mmc, constrains only nz).

    auto get_denominator = [](double t) -> int {
        // Normalize to [0, 1)
        t = t - std::floor(t);
        if (t < 0) t += 1.0;

        // Check common fractions with tolerance
        const double tol = 1e-6;

        // Check for 0
        if (std::abs(t) < tol || std::abs(t - 1.0) < tol) return 1;

        // Check for 1/2
        if (std::abs(t - 0.5) < tol) return 2;

        // Check for 1/3, 2/3
        if (std::abs(t - 1.0/3.0) < tol || std::abs(t - 2.0/3.0) < tol) return 3;

        // Check for 1/4, 3/4
        if (std::abs(t - 0.25) < tol || std::abs(t - 0.75) < tol) return 4;

        // Check for 1/6, 5/6
        if (std::abs(t - 1.0/6.0) < tol || std::abs(t - 5.0/6.0) < tol) return 6;

        // Check for 1/8, 3/8, 5/8, 7/8
        for (int n = 1; n <= 7; n += 2) {
            if (std::abs(t - n/8.0) < tol) return 8;
        }

        // Check for 1/12, 5/12, 7/12, 11/12
        for (int n : {1, 5, 7, 11}) {
            if (std::abs(t - n/12.0) < tol) return 12;
        }

        return 1;  // Unknown fraction, assume no special requirement
    };

    std::array<int, 3> result = {1, 1, 1};

    for (int op = 0; op < n_symmetry_ops_; ++op)
    {
        for (int dim = 0; dim < 3; ++dim)
        {
            int denom = get_denominator(translations_[op][dim]);
            result[dim] = std::lcm(result[dim], denom);
        }
    }

    return result;
}

void SpaceGroup::validate_grid_compatibility()
{
    const std::array<int, 3> divisors = get_required_grid_divisors();

    std::vector<int> incompatible;
    for (int dim = 0; dim < 3; ++dim)
    {
        if (divisors[dim] > 1 && nx_[dim] % divisors[dim] != 0)
            incompatible.push_back(dim);
    }
    if (incompatible.empty())
        return;

    const char* axis_names[] = {"x", "y", "z"};
    std::string msg = "Grid [" + std::to_string(nx_[0]) + ", " +
        std::to_string(nx_[1]) + ", " + std::to_string(nx_[2]) +
        "] is incompatible with space group " + spacegroup_symbol_ +
        " (No. " + std::to_string(spacegroup_number_) + ").\n";
    msg += "The translation parts of the symmetry operations require:";
    for (int dim : incompatible)
    {
        msg += std::string(" ") + axis_names[dim] + "=" + std::to_string(nx_[dim]) +
            " (a multiple of " + std::to_string(divisors[dim]) + ")";
    }

    // Suggest valid grid sizes
    msg += "\nSuggested grid: [";
    for (int dim = 0; dim < 3; ++dim)
    {
        if (dim > 0) msg += ", ";
        const int d = divisors[dim];
        if (d <= 1 || nx_[dim] % d == 0)
        {
            msg += std::to_string(nx_[dim]);
        }
        else
        {
            int lower = (nx_[dim] / d) * d;
            int upper = lower + d;
            int suggested = (upper - nx_[dim] < nx_[dim] - lower || lower == 0) ? upper : lower;
            msg += std::to_string(suggested);
        }
    }
    msg += "]";

    throw_with_line_number(msg);
}

void SpaceGroup::determine_crystal_system()
{
    if (spacegroup_number_ >= 1 && spacegroup_number_ <= 2)
        crystal_system_ = "Triclinic";
    else if (spacegroup_number_ >= 3 && spacegroup_number_ <= 15)
        crystal_system_ = "Monoclinic";
    else if (spacegroup_number_ >= 16 && spacegroup_number_ <= 74)
        crystal_system_ = "Orthorhombic";
    else if (spacegroup_number_ >= 75 && spacegroup_number_ <= 142)
        crystal_system_ = "Tetragonal";
    else if (spacegroup_number_ >= 143 && spacegroup_number_ <= 167)
        crystal_system_ = "Trigonal";
    else if (spacegroup_number_ >= 168 && spacegroup_number_ <= 194)
        crystal_system_ = "Hexagonal";
    else if (spacegroup_number_ >= 195 && spacegroup_number_ <= 230)
        crystal_system_ = "Cubic";
    else
        throw_with_line_number("Invalid space group number: " + std::to_string(spacegroup_number_));
}

void SpaceGroup::get_symmetry_operations()
{
    // Get symmetry operations from spglib database using Hall number
    // Maximum possible symmetry operations is 192 (for Fm-3m)
    constexpr int max_size = 192;
    int rotations[max_size][3][3];
    double translations[max_size][3];

    int n_ops = spg_get_symmetry_from_database(rotations, translations, hall_number_);

    if (n_ops == 0)
    {
        throw_with_line_number("Failed to get symmetry operations for Hall number " +
            std::to_string(hall_number_));
    }

    n_symmetry_ops_ = n_ops;

    // Store symmetry operations
    rotations_.resize(n_ops);
    translations_.resize(n_ops);

    for (int i = 0; i < n_ops; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            for (int k = 0; k < 3; ++k)
            {
                rotations_[i][j][k] = rotations[i][j][k];
            }
            translations_[i][j] = translations[i][j];
        }
    }
}

void SpaceGroup::find_irreducible_mesh()
{
    // Orbit coordinate conventions:
    //
    // (a) Cell-centered sampling x = (i+0.5)/N (default). Exact whenever every
    //     rotation-matrix row has an odd sum AND any axes mixed by a row have
    //     equal grid extents: the 0.5 offsets then recombine to an odd multiple
    //     of 0.5 and grid points map onto grid points. This holds for
    //     cubic/tetragonal/orthorhombic/monoclinic/triclinic settings, and for
    //     rhombohedral-axes trigonal settings (signed permutation rows) only
    //     when Nx = Ny = Nz. Inexact mappings are now detected and rejected
    //     (see the exactness check in the fallback loop below) instead of
    //     being silently rounded onto wrong orbits.
    //
    // (b) Hybrid sampling for hexagonal-axes settings of trigonal/hexagonal
    //     groups: NODE-centered in-plane x = i/Nx, y = j/Ny with CELL-centered
    //     z = (k+0.5)/Nz. In these settings, in-plane rotation rows can have
    //     even sums (e.g. x' = x - y), which annihilate the 0.5 offset and map
    //     cell centers onto cell boundaries — no rounding strategy is
    //     consistent there. Every op in these settings decomposes as a 2x2
    //     in-plane block (+) (+/-1 in z), so node-centering only x,y makes the
    //     in-plane action an exact integer map (Nx = Ny is enforced), while the
    //     z part (+/-z + t_z, odd "row sum") stays exact on cell centers. This
    //     keeps the z-mirror physical basis and the ObliqueZ CrysFFT (DCT-II
    //     fold along z) valid unchanged. Symmetry translations are integral in
    //     grid units by validate_grid_compatibility(), so the whole hybrid map
    //     is exact integer arithmetic — no rounding at all.

    // Initialize full_to_reduced_map with -1 (unvisited)
    full_to_reduced_map_.resize(total_grid_, -1);
    reduced_basis_indices_.clear();

    int Nx = nx_[0];
    int Ny = nx_[1];
    int Nz = nx_[2];

    // Decide whether the hybrid (node-centered x,y; cell-centered z) convention
    // applies: hexagonal/trigonal crystal system, square in-plane grid, all ops
    // z-block-diagonal, and all translations integral in grid units.
    const bool hex_like = (crystal_system_ == "Hexagonal" || crystal_system_ == "Trigonal");
    bool use_hybrid = hex_like && (Nx == Ny);
    std::vector<std::array<int, 3>> trans_grid(n_symmetry_ops_, {0, 0, 0});
    if (use_hybrid)
    {
        for (int op = 0; op < n_symmetry_ops_ && use_hybrid; ++op)
        {
            const auto& R = rotations_[op];
            // Require (2x2 in-plane block) (+) (+/-1 in z)
            if (R[0][2] != 0 || R[1][2] != 0 || R[2][0] != 0 || R[2][1] != 0 ||
                std::abs(R[2][2]) != 1)
            {
                use_hybrid = false;
                break;
            }
            for (int dim = 0; dim < 3; ++dim)
            {
                const double t_grid = translations_[op][dim] * static_cast<double>(nx_[dim]);
                const long t_int = std::lround(t_grid);
                if (std::abs(t_grid - static_cast<double>(t_int)) > 1e-6)
                {
                    use_hybrid = false;
                    break;
                }
                trans_grid[op][dim] = static_cast<int>(((t_int % nx_[dim]) + nx_[dim]) % nx_[dim]);
            }
        }
    }

    auto mod_pos = [](int v, int N) {
        int r = v % N;
        return (r < 0) ? r + N : r;
    };

    int count = 0;

    // Iterate through all grid points
    for (int flat_idx = 0; flat_idx < total_grid_; ++flat_idx)
    {
        // Convert flat index to 3D coordinates
        int ix = flat_idx / (Ny * Nz);
        int iy = (flat_idx / Nz) % Ny;
        int iz = flat_idx % Nz;

        if (full_to_reduced_map_[flat_idx] == -1)
        {
            // This is a new irreducible point
            reduced_basis_indices_.push_back(flat_idx);

            if (use_hybrid)
            {
                // Exact integer orbit: node-centered x,y; cell-centered z.
                // x' = R00 x + R01 y + tx  (units of 1/Nx, valid since Nx == Ny)
                // z' = +z + tz  ->  oz = iz + tz
                // z' = -z + tz  ->  (oz+0.5) = -(iz+0.5) + tz  ->  oz = tz - iz - 1
                for (int op = 0; op < n_symmetry_ops_; ++op)
                {
                    const auto& R = rotations_[op];
                    const auto& tg = trans_grid[op];
                    const int ox = mod_pos(R[0][0] * ix + R[0][1] * iy + tg[0], Nx);
                    const int oy = mod_pos(R[1][0] * ix + R[1][1] * iy + tg[1], Ny);
                    const int oz = (R[2][2] == 1) ? mod_pos(iz + tg[2], Nz)
                                                  : mod_pos(tg[2] - iz - 1, Nz);

                    // Convert to flat index and mark this orbit point
                    const int orbit_flat = (ox * Ny + oy) * Nz + oz;
                    full_to_reduced_map_[orbit_flat] = count;
                }
            }
            else
            {
                // Apply all symmetry operations to find orbit (cell-centered grid)
                const double x = (static_cast<double>(ix) + 0.5) / static_cast<double>(Nx);
                const double y = (static_cast<double>(iy) + 0.5) / static_cast<double>(Ny);
                const double z = (static_cast<double>(iz) + 0.5) / static_cast<double>(Nz);
                const double eps = 1e-8;

                for (int op = 0; op < n_symmetry_ops_; ++op)
                {
                    // Apply rotation + translation in fractional coordinates
                    double rx = rotations_[op][0][0] * x + rotations_[op][0][1] * y + rotations_[op][0][2] * z + translations_[op][0];
                    double ry = rotations_[op][1][0] * x + rotations_[op][1][1] * y + rotations_[op][1][2] * z + translations_[op][1];
                    double rz = rotations_[op][2][0] * x + rotations_[op][2][1] * y + rotations_[op][2][2] * z + translations_[op][2];

                    // Wrap to [0,1)
                    rx -= std::floor(rx);
                    ry -= std::floor(ry);
                    rz -= std::floor(rz);

                    // Map back to cell-centered indices
                    int ox = static_cast<int>(std::floor(rx * Nx + eps));
                    int oy = static_cast<int>(std::floor(ry * Ny + eps));
                    int oz = static_cast<int>(std::floor(rz * Nz + eps));

                    if (ox >= Nx) ox -= Nx;
                    if (oy >= Ny) oy -= Ny;
                    if (oz >= Nz) oz -= Nz;

                    // Exactness check: the mapped coordinate must land on a
                    // cell center (fractional part 0.5 in index units). If it
                    // does not, this op/grid/sampling-convention combination
                    // cannot represent the symmetry consistently (e.g.
                    // rhombohedral-axes settings with unequal Nx, Ny, Nz);
                    // fail loudly rather than silently corrupting the orbits.
                    const double fx = rx * Nx - std::floor(rx * Nx + eps);
                    const double fy = ry * Ny - std::floor(ry * Ny + eps);
                    const double fz = rz * Nz - std::floor(rz * Nz + eps);
                    if (std::abs(fx - 0.5) > 1e-6 || std::abs(fy - 0.5) > 1e-6 ||
                        std::abs(fz - 0.5) > 1e-6)
                    {
                        throw_with_line_number(
                            "Symmetry operation " + std::to_string(op) +
                            " does not map the cell-centered grid onto itself "
                            "(off-center by more than 1e-6 in index units). "
                            "This grid is incompatible with the space group "
                            "under the current sampling convention; for "
                            "rhombohedral-axes settings use Nx = Ny = Nz.");
                    }

                    // Convert to flat index
                    const int orbit_flat = ox * Ny * Nz + oy * Nz + oz;

                    // Mark this orbit point
                    full_to_reduced_map_[orbit_flat] = count;
                }
            }

            ++count;
        }
    }

    n_irreducible_ = count;

    // Compute orbit counts
    orbit_counts_.resize(n_irreducible_, 0);
    for (int i = 0; i < total_grid_; ++i)
    {
        orbit_counts_[full_to_reduced_map_[i]]++;
    }
}

void SpaceGroup::to_reduced_basis(const double* full_field, double* reduced_field, int n_fields) const
{
    const auto& reduced_indices = reduced_basis_indices_;
    const int n_reduced = n_irreducible_;

    for (int f = 0; f < n_fields; ++f)
    {
        const double* src = full_field + f * total_grid_;
        double* dst = reduced_field + f * n_reduced;

        for (int i = 0; i < n_reduced; ++i)
        {
            dst[i] = src[reduced_indices[i]];
        }
    }
}

void SpaceGroup::reduced_to_irreducible(const double* reduced_field, double* irreducible_field, int n_fields) const
{
    if (n_irreducible_full_ <= 0 || reduced_to_irreducible_map_.empty() || orbit_counts_full_.empty())
    {
        const int n_reduced = n_irreducible_;
        for (int f = 0; f < n_fields; ++f)
        {
            std::memcpy(irreducible_field + f * n_reduced,
                        reduced_field + f * n_reduced,
                        sizeof(double) * n_reduced);
        }
        return;
    }

    const int n_reduced = n_irreducible_;
    const int n_irreducible = n_irreducible_full_;

    std::vector<double> orbit_sums(n_irreducible, 0.0);
    std::vector<int> orbit_weights(n_irreducible, 0);

    for (int f = 0; f < n_fields; ++f)
    {
        const double* src = reduced_field + f * n_reduced;
        double* dst = irreducible_field + f * n_irreducible;

        std::fill(orbit_sums.begin(), orbit_sums.end(), 0.0);
        std::fill(orbit_weights.begin(), orbit_weights.end(), 0);

        for (int i = 0; i < n_reduced; ++i)
        {
            const int ridx = reduced_to_irreducible_map_[i];
            const int weight = orbit_counts_[i];
            orbit_sums[ridx] += static_cast<double>(weight) * src[i];
            orbit_weights[ridx] += weight;
        }

        for (int i = 0; i < n_irreducible; ++i)
        {
            if (orbit_weights[i] > 0)
                dst[i] = orbit_sums[i] / static_cast<double>(orbit_weights[i]);
            else
                dst[i] = 0.0;
        }
    }
}

void SpaceGroup::irreducible_to_reduced(const double* irreducible_field, double* reduced_field, int n_fields) const
{
    if (n_irreducible_full_ <= 0 || reduced_to_irreducible_map_.empty())
    {
        const int n_reduced = n_irreducible_;
        for (int f = 0; f < n_fields; ++f)
        {
            std::memcpy(reduced_field + f * n_reduced,
                        irreducible_field + f * n_reduced,
                        sizeof(double) * n_reduced);
        }
        return;
    }

    const int n_reduced = n_irreducible_;
    const int n_irreducible = n_irreducible_full_;

    for (int f = 0; f < n_fields; ++f)
    {
        const double* src = irreducible_field + f * n_irreducible;
        double* dst = reduced_field + f * n_reduced;

        for (int i = 0; i < n_reduced; ++i)
        {
            dst[i] = src[reduced_to_irreducible_map_[i]];
        }
    }
}

void SpaceGroup::from_reduced_basis(const double* reduced_field, double* full_field, int n_fields) const
{
    const auto& full_to_reduced = full_to_reduced_map_;
    const int n_reduced = n_irreducible_;

    for (int f = 0; f < n_fields; ++f)
    {
        const double* src = reduced_field + f * n_reduced;
        double* dst = full_field + f * total_grid_;

        for (int i = 0; i < total_grid_; ++i)
        {
            dst[i] = src[full_to_reduced[i]];
        }
    }
}

void SpaceGroup::symmetrize(double* field, int n_fields) const
{
    const auto& full_to_reduced = full_to_reduced_map_;
    const auto& orbit_counts = orbit_counts_;
    const int n_reduced = n_irreducible_;

    // Temporary buffer for orbit sums
    std::vector<double> orbit_sums(n_reduced);

    for (int f = 0; f < n_fields; ++f)
    {
        double* data = field + f * total_grid_;

        // Reset orbit sums
        std::fill(orbit_sums.begin(), orbit_sums.end(), 0.0);

        // Accumulate values by orbit
        for (int i = 0; i < total_grid_; ++i)
        {
            orbit_sums[full_to_reduced[i]] += data[i];
        }

        // Compute averages and broadcast back
        for (int i = 0; i < total_grid_; ++i)
        {
            int orbit_idx = full_to_reduced[i];
            data[i] = orbit_sums[orbit_idx] / orbit_counts[orbit_idx];
        }
    }
}

void SpaceGroup::symmetrize(const double* field_in, double* field_out, int n_fields) const
{
    // Copy input to output first
    std::memcpy(field_out, field_in, sizeof(double) * n_fields * total_grid_);

    // Then symmetrize in-place
    symmetrize(field_out, n_fields);
}

void SpaceGroup::symmetrize_reduced_basis(double* reduced_field, int n_fields) const
{
    if (physical_basis_ == PhysicalBasis::None)
        return;
    if (reduced_to_irreducible_map_.empty() || orbit_counts_full_.empty())
        return;

    const int n_reduced = n_irreducible_;
    const int n_irreducible = n_irreducible_full_;

    std::vector<double> orbit_sums(n_irreducible, 0.0);
    std::vector<int> orbit_weights(n_irreducible, 0);

    for (int f = 0; f < n_fields; ++f)
    {
        double* data = reduced_field + f * n_reduced;

        std::fill(orbit_sums.begin(), orbit_sums.end(), 0.0);
        std::fill(orbit_weights.begin(), orbit_weights.end(), 0);

        for (int i = 0; i < n_reduced; ++i)
        {
            const int ridx = reduced_to_irreducible_map_[i];
            const int weight = orbit_counts_[i];
            orbit_sums[ridx] += static_cast<double>(weight) * data[i];
            orbit_weights[ridx] += weight;
        }

        for (int i = 0; i < n_reduced; ++i)
        {
            const int ridx = reduced_to_irreducible_map_[i];
            const int weight = orbit_weights[ridx];
            if (weight > 0)
                data[i] = orbit_sums[ridx] / static_cast<double>(weight);
        }
    }
}

bool SpaceGroup::has_mirror_planes_xyz(double tol) const
{
    auto is_zero_mod1 = [tol](double v) {
        double r = std::fabs(v - std::round(v));
        return r < tol;
    };

    bool has_x = false;
    bool has_y = false;
    bool has_z = false;

    for (size_t i = 0; i < rotations_.size(); ++i)
    {
        const auto& R = rotations_[i];
        const auto& t = translations_[i];

        bool t_zero = is_zero_mod1(t[0]) && is_zero_mod1(t[1]) && is_zero_mod1(t[2]);
        if (!t_zero)
            continue;

        if (R[0][0] == -1 && R[1][1] == 1 && R[2][2] == 1 &&
            R[0][1] == 0 && R[0][2] == 0 && R[1][0] == 0 && R[1][2] == 0 && R[2][0] == 0 && R[2][1] == 0)
            has_x = true;
        if (R[0][0] == 1 && R[1][1] == -1 && R[2][2] == 1 &&
            R[0][1] == 0 && R[0][2] == 0 && R[1][0] == 0 && R[1][2] == 0 && R[2][0] == 0 && R[2][1] == 0)
            has_y = true;
        if (R[0][0] == 1 && R[1][1] == 1 && R[2][2] == -1 &&
            R[0][1] == 0 && R[0][2] == 0 && R[1][0] == 0 && R[1][2] == 0 && R[2][0] == 0 && R[2][1] == 0)
            has_z = true;
    }

    return has_x && has_y && has_z;
}

bool SpaceGroup::get_m3_translations(std::array<double, 9>& g, double tol) const
{
    auto wrap01 = [](double v) {
        v -= std::floor(v);
        if (v < 0.0)
            v += 1.0;
        return v;
    };
    auto wrap_half = [](double v) {
        // Map to [-0.5, 0.5) for norm comparison
        v -= std::floor(v + 0.5);
        return v;
    };
    auto norm2 = [&](const std::array<double, 3>& t) {
        double x = wrap_half(t[0]);
        double y = wrap_half(t[1]);
        double z = wrap_half(t[2]);
        return x * x + y * y + z * z;
    };

    bool found_x = false;
    bool found_y = false;
    bool found_z = false;
    double best_x = std::numeric_limits<double>::infinity();
    double best_y = std::numeric_limits<double>::infinity();
    double best_z = std::numeric_limits<double>::infinity();

    for (size_t i = 0; i < rotations_.size(); ++i)
    {
        const auto& R = rotations_[i];
        const auto& t = translations_[i];

        const bool off_diag_zero =
            R[0][1] == 0 && R[0][2] == 0 &&
            R[1][0] == 0 && R[1][2] == 0 &&
            R[2][0] == 0 && R[2][1] == 0;

        if (!off_diag_zero)
            continue;

        const std::array<double, 3> t_val = {t[0], t[1], t[2]};
        const double t_norm = norm2(t_val);

        if (R[0][0] == 1 && R[1][1] == 1 && R[2][2] == -1)
        {
            if (!found_z || t_norm + tol < best_z)
            {
                g[0] = wrap01(t[0]);
                g[1] = wrap01(t[1]);
                g[2] = wrap01(t[2]);
                best_z = t_norm;
                found_z = true;
            }
        }
        if (R[0][0] == 1 && R[1][1] == -1 && R[2][2] == 1)
        {
            if (!found_y || t_norm + tol < best_y)
            {
                g[3] = wrap01(t[0]);
                g[4] = wrap01(t[1]);
                g[5] = wrap01(t[2]);
                best_y = t_norm;
                found_y = true;
            }
        }
        if (R[0][0] == -1 && R[1][1] == 1 && R[2][2] == 1)
        {
            if (!found_x || t_norm + tol < best_x)
            {
                g[6] = wrap01(t[0]);
                g[7] = wrap01(t[1]);
                g[8] = wrap01(t[2]);
                best_x = t_norm;
                found_x = true;
            }
        }
    }

    return found_x && found_y && found_z;
}

void SpaceGroup::get_pure_translations(std::vector<std::array<double, 3>>& translations) const
{
    auto wrap01 = [](double v) {
        v -= std::floor(v);
        if (v < 0.0)
            v += 1.0;
        return v;
    };

    translations.clear();
    for (size_t i = 0; i < rotations_.size(); ++i)
    {
        const auto& R = rotations_[i];
        const bool is_identity =
            R[0][0] == 1 && R[1][1] == 1 && R[2][2] == 1 &&
            R[0][1] == 0 && R[0][2] == 0 &&
            R[1][0] == 0 && R[1][2] == 0 &&
            R[2][0] == 0 && R[2][1] == 0;
        if (!is_identity)
            continue;
        translations.push_back({wrap01(translations_[i][0]),
                                wrap01(translations_[i][1]),
                                wrap01(translations_[i][2])});
    }
}

bool SpaceGroup::get_z_mirror_translation(double& t_z, double tol) const
{
    auto is_zero_mod1 = [tol](double v) {
        double r = std::fabs(v - std::round(v));
        return r < tol;
    };
    auto wrap01 = [](double v) {
        v -= std::floor(v);
        if (v < 0.0)
            v += 1.0;
        return v;
    };
    auto wrap_half = [](double v) {
        v -= std::floor(v + 0.5);
        return v;
    };

    bool found = false;
    double best_norm = std::numeric_limits<double>::infinity();
    double best_tz = 0.0;

    for (size_t i = 0; i < rotations_.size(); ++i)
    {
        const auto& R = rotations_[i];
        const auto& t = translations_[i];

        const bool off_diag_zero =
            R[0][1] == 0 && R[0][2] == 0 &&
            R[1][0] == 0 && R[1][2] == 0 &&
            R[2][0] == 0 && R[2][1] == 0;

        if (!off_diag_zero)
            continue;
        if (!(R[0][0] == 1 && R[1][1] == 1 && R[2][2] == -1))
            continue;

        if (!is_zero_mod1(t[0]) || !is_zero_mod1(t[1]))
            continue;

        const double tz = wrap01(t[2]);
        const double nz = wrap_half(tz);
        const double norm = nz * nz;
        if (!found || norm + tol < best_norm)
        {
            best_norm = norm;
            best_tz = tz;
            found = true;
        }
    }

    if (found)
        t_z = best_tz;
    return found;
}

void SpaceGroup::enable_pmmm_physical_basis()
{
    if (physical_basis_ == PhysicalBasis::Pmmm)
        return;
    if (in_use_)
        throw_with_line_number("The physical basis cannot be changed after a computation box or solver was created with this SpaceGroup (their reduced-basis maps are snapshots); create a new SpaceGroup.");

    if (nx_.size() != 3)
        throw_with_line_number("Pmmm physical basis requires 3D grid.");

    const int Nx = nx_[0];
    const int Ny = nx_[1];
    const int Nz = nx_[2];

    if ((Nx % 2) != 0 || (Ny % 2) != 0 || (Nz % 2) != 0)
        throw_with_line_number("Pmmm physical basis requires even grid dimensions.");

    if (!has_mirror_planes_xyz())
        throw_with_line_number("Space group does not contain mirror planes along x, y, z (Pmmm).");

    const int Nx2 = Nx / 2;
    const int Ny2 = Ny / 2;
    const int Nz2 = Nz / 2;

    n_irreducible_ = Nx2 * Ny2 * Nz2;
    reduced_basis_indices_.resize(n_irreducible_);
    full_to_reduced_map_.resize(total_grid_);
    orbit_counts_.assign(n_irreducible_, 0);

    int idx = 0;
    for (int ix = 0; ix < Nx2; ++ix)
    {
        for (int iy = 0; iy < Ny2; ++iy)
        {
            for (int iz = 0; iz < Nz2; ++iz)
            {
                const int full_idx = (ix * Ny + iy) * Nz + iz;
                reduced_basis_indices_[idx++] = full_idx;
            }
        }
    }

    for (int ix = 0; ix < Nx; ++ix)
    {
        const int ix2 = (ix < Nx2) ? ix : (Nx - 1 - ix);
        for (int iy = 0; iy < Ny; ++iy)
        {
            const int iy2 = (iy < Ny2) ? iy : (Ny - 1 - iy);
            for (int iz = 0; iz < Nz; ++iz)
            {
                const int iz2 = (iz < Nz2) ? iz : (Nz - 1 - iz);
                const int full_idx = (ix * Ny + iy) * Nz + iz;
                const int reduced_idx = (ix2 * Ny2 + iy2) * Nz2 + iz2;
                full_to_reduced_map_[full_idx] = reduced_idx;
                orbit_counts_[reduced_idx] += 1;
            }
        }
    }

    if (!full_to_reduced_map_full_.empty())
    {
        reduced_to_irreducible_map_.resize(n_irreducible_);
        for (int i = 0; i < n_irreducible_; ++i)
        {
            const int full_idx = reduced_basis_indices_[i];
            reduced_to_irreducible_map_[i] = full_to_reduced_map_full_[full_idx];
        }
    }
    else
    {
        reduced_to_irreducible_map_.clear();
    }

    physical_basis_ = PhysicalBasis::Pmmm;
    std::cout << "Pmmm physical basis enabled. Reduced basis size: " << n_irreducible_ << std::endl;
}

bool SpaceGroup::m3_crysfft_feasible() const
{
    if (nx_.size() != 3)
        return false;
    std::array<double, 9> trans_part = {0,0,0, 0,0,0, 0,0,0};
    const std::array<int, 3> nx3 = {nx_[0], nx_[1], nx_[2]};
    return crysfft_recursive_3m_feasible(this, nx3, trans_part);
}

void SpaceGroup::enable_m3_physical_basis()
{
    if (physical_basis_ == PhysicalBasis::M3)
        return;
    if (in_use_)
        throw_with_line_number("The physical basis cannot be changed after a computation box or solver was created with this SpaceGroup (their reduced-basis maps are snapshots); create a new SpaceGroup.");

    if (nx_.size() != 3)
        throw_with_line_number("M3 physical basis requires 3D grid.");

    const int Nx = nx_[0];
    const int Ny = nx_[1];
    const int Nz = nx_[2];

    if ((Nx % 2) != 0 || (Ny % 2) != 0 || (Nz % 2) != 0)
        throw_with_line_number("M3 physical basis requires even grid dimensions.");

    std::array<double, 9> trans_part = {0,0,0, 0,0,0, 0,0,0};
    if (!get_m3_translations(trans_part))
        throw_with_line_number("Space group does not contain 3m translations for M3 physical basis.");

    const int Nx2 = Nx / 2;
    const int Ny2 = Ny / 2;
    const int Nz2 = Nz / 2;

    // The basis is built in locals and committed at the end, so that a failed
    // mapping below leaves the object in its previous state.
    const int n_reduced = Nx2 * Ny2 * Nz2;
    std::vector<int> reduced_basis_indices(n_reduced);
    std::vector<int> full_to_reduced_map(total_grid_, -1);
    std::vector<int> orbit_counts(n_reduced, 0);

    int idx = 0;
    for (int ix = 0; ix < Nx2; ++ix)
    {
        const int fx = 2 * ix;
        for (int iy = 0; iy < Ny2; ++iy)
        {
            const int fy = 2 * iy;
            for (int iz = 0; iz < Nz2; ++iz)
            {
                const int fz = 2 * iz;
                const int full_idx = (fx * Ny + fy) * Nz + fz;
                reduced_basis_indices[idx++] = full_idx;
            }
        }
    }

    struct Op {
        std::array<int, 3> diag;
        std::array<double, 3> t;
    };

    std::vector<Op> ops;
    ops.reserve(rotations_.size());
    for (size_t i = 0; i < rotations_.size(); ++i)
    {
        const auto& R = rotations_[i];
        const auto& t = translations_[i];
        const bool off_diag_zero =
            R[0][1] == 0 && R[0][2] == 0 &&
            R[1][0] == 0 && R[1][2] == 0 &&
            R[2][0] == 0 && R[2][1] == 0;
        if (!off_diag_zero)
            continue;
        if (std::abs(R[0][0]) != 1 || std::abs(R[1][1]) != 1 || std::abs(R[2][2]) != 1)
            continue;
        ops.push_back({{R[0][0], R[1][1], R[2][2]}, {t[0], t[1], t[2]}});
    }

    if (ops.empty())
        throw_with_line_number("Failed to find diagonal symmetry operations for M3 physical basis.");

    auto wrap01 = [](double v) {
        v -= std::floor(v);
        if (v < 0.0)
            v += 1.0;
        return v;
    };
    auto to_index = [&](double v, int N) {
        double w = wrap01(v);
        double scaled = w * static_cast<double>(N);
        int idx = static_cast<int>(std::floor(scaled + 1e-12));
        if (idx >= N)
            idx = 0;
        return idx;
    };
    auto reduced_index = [&](int ix, int iy, int iz) {
        return ((ix / 2) * Ny2 + (iy / 2)) * Nz2 + (iz / 2);
    };

    for (int ix = 0; ix < Nx; ++ix)
    {
        for (int iy = 0; iy < Ny; ++iy)
        {
            for (int iz = 0; iz < Nz; ++iz)
            {
                const int full_idx = (ix * Ny + iy) * Nz + iz;

                int reduced_idx = -1;
                if ((ix % 2 == 0) && (iy % 2 == 0) && (iz % 2 == 0))
                {
                    reduced_idx = reduced_index(ix, iy, iz);
                }
                else
                {
                    const double x = (static_cast<double>(ix) + 0.5) / static_cast<double>(Nx);
                    const double y = (static_cast<double>(iy) + 0.5) / static_cast<double>(Ny);
                    const double z = (static_cast<double>(iz) + 0.5) / static_cast<double>(Nz);

                    for (const auto& op : ops)
                    {
                        const double x2 = op.diag[0] * x + op.t[0];
                        const double y2 = op.diag[1] * y + op.t[1];
                        const double z2 = op.diag[2] * z + op.t[2];

                        const int jx = to_index(x2, Nx);
                        const int jy = to_index(y2, Ny);
                        const int jz = to_index(z2, Nz);

                        if ((jx % 2 == 0) && (jy % 2 == 0) && (jz % 2 == 0))
                        {
                            reduced_idx = reduced_index(jx, jy, jz);
                            break;
                        }
                    }
                }

                if (reduced_idx < 0)
                    throw_with_line_number("Failed to map grid point to M3 physical basis.");

                full_to_reduced_map[full_idx] = reduced_idx;
                orbit_counts[reduced_idx] += 1;
            }
        }
    }

    n_irreducible_ = n_reduced;
    reduced_basis_indices_ = std::move(reduced_basis_indices);
    full_to_reduced_map_ = std::move(full_to_reduced_map);
    orbit_counts_ = std::move(orbit_counts);

    if (!full_to_reduced_map_full_.empty())
    {
        reduced_to_irreducible_map_.resize(n_irreducible_);
        for (int i = 0; i < n_irreducible_; ++i)
        {
            const int full_idx = reduced_basis_indices_[i];
            reduced_to_irreducible_map_[i] = full_to_reduced_map_full_[full_idx];
        }
    }
    else
    {
        reduced_to_irreducible_map_.clear();
    }

    physical_basis_ = PhysicalBasis::M3;
    std::cout << "M3 physical basis enabled. Reduced basis size: " << n_irreducible_ << std::endl;
}

void SpaceGroup::enable_z_mirror_physical_basis()
{
    if (physical_basis_ == PhysicalBasis::ZMirror)
        return;
    if (in_use_)
        throw_with_line_number("The physical basis cannot be changed after a computation box or solver was created with this SpaceGroup (their reduced-basis maps are snapshots); create a new SpaceGroup.");

    if (nx_.size() != 3)
        throw_with_line_number("Z-mirror physical basis requires 3D grid.");

    const int Nx = nx_[0];
    const int Ny = nx_[1];
    const int Nz = nx_[2];

    if ((Nz % 2) != 0)
        throw_with_line_number("Z-mirror physical basis requires even Nz.");

    double t_z = 0.0;
    if (!get_z_mirror_translation(t_z))
        throw_with_line_number("Space group does not contain (x,y,-z)+t mirror for z-physical basis.");

    // Accept only t_z = 0 or 1/2 (within tolerance)
    const double tol = 1e-8;
    double t = t_z;
    if (std::fabs(t - 1.0) < tol)
        t = 0.0;

    int shift = 0;
    if (std::fabs(t) < tol)
    {
        shift = 0;
    }
    else if (std::fabs(t - 0.5) < tol)
    {
        if ((Nz % 4) != 0)
            throw_with_line_number("Z-mirror physical basis with t_z=1/2 requires Nz divisible by 4.");
        shift = Nz / 4;
    }
    else
    {
        throw_with_line_number("Unsupported z-mirror translation (only 0 or 1/2 are supported).");
    }

    const int Nz2 = Nz / 2;
    n_irreducible_ = Nx * Ny * Nz2;
    reduced_basis_indices_.resize(n_irreducible_);
    full_to_reduced_map_.resize(total_grid_);
    orbit_counts_.assign(n_irreducible_, 0);

    int idx = 0;
    for (int ix = 0; ix < Nx; ++ix)
    {
        for (int iy = 0; iy < Ny; ++iy)
        {
            for (int iz = 0; iz < Nz2; ++iz)
            {
                int iz_full = iz + shift;
                if (iz_full >= Nz)
                    iz_full -= Nz;
                const int full_idx = (ix * Ny + iy) * Nz + iz_full;
                reduced_basis_indices_[idx++] = full_idx;
            }
        }
    }

    for (int ix = 0; ix < Nx; ++ix)
    {
        for (int iy = 0; iy < Ny; ++iy)
        {
            for (int iz = 0; iz < Nz; ++iz)
            {
                int iz_shift = iz - shift;
                while (iz_shift < 0)
                    iz_shift += Nz;
                while (iz_shift >= Nz)
                    iz_shift -= Nz;

                int iz2 = (iz_shift < Nz2) ? iz_shift : (Nz - 1 - iz_shift);
                const int full_idx = (ix * Ny + iy) * Nz + iz;
                const int reduced_idx = (ix * Ny + iy) * Nz2 + iz2;
                full_to_reduced_map_[full_idx] = reduced_idx;
                orbit_counts_[reduced_idx] += 1;
            }
        }
    }

    if (!full_to_reduced_map_full_.empty())
    {
        reduced_to_irreducible_map_.resize(n_irreducible_);
        for (int i = 0; i < n_irreducible_; ++i)
        {
            const int full_idx = reduced_basis_indices_[i];
            reduced_to_irreducible_map_[i] = full_to_reduced_map_full_[full_idx];
        }
    }
    else
    {
        reduced_to_irreducible_map_.clear();
    }

    z_mirror_shift_ = shift;
    z_mirror_translation_ = t_z;
    physical_basis_ = PhysicalBasis::ZMirror;

    std::cout << "Z-mirror physical basis enabled (shift=" << z_mirror_shift_
              << "). Reduced basis size: " << n_irreducible_ << std::endl;
}

void SpaceGroup::enable_physical_basis(PhysicalBasis basis)
{
    switch (basis)
    {
        case PhysicalBasis::None:
            if (physical_basis_ != PhysicalBasis::None)
                throw_with_line_number("Restoring the irreducible basis after a physical basis was enabled is not supported; create a new SpaceGroup.");
            break;
        case PhysicalBasis::Pmmm:    enable_pmmm_physical_basis(); break;
        case PhysicalBasis::M3:      enable_m3_physical_basis(); break;
        case PhysicalBasis::ZMirror: enable_z_mirror_physical_basis(); break;
    }
}

bool SpaceGroup::uses_hexagonal_axes() const
{
    if (crystal_system_ != "Hexagonal" && crystal_system_ != "Trigonal")
        return false;
    for (const auto& R : rotations_)
    {
        if (R[0][2] != 0 || R[1][2] != 0 || R[2][0] != 0 || R[2][1] != 0 || std::abs(R[2][2]) != 1)
            return false;   // 3-fold about [111]: rhombohedral axes
    }
    return true;
}

std::string SpaceGroup::physical_basis_name() const
{
    switch (physical_basis_)
    {
        case PhysicalBasis::Pmmm:    return "pmmm";
        case PhysicalBasis::M3:      return "m3";
        case PhysicalBasis::ZMirror: return "z-mirror";
        default:                     return "irreducible";
    }
}
