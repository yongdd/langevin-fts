/**
 * @file SpaceGroup.cpp
 * @brief Implementation of SpaceGroup class using spglib for symmetry operations.
 */

#include "SpaceGroup.h"
#include "Exception.h"
#include <cstring>
#include <cmath>
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

    // Compute total grid size
    total_grid_ = nx_[0] * nx_[1] * nx_[2];

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

    // Print info
    std::cout << "---------- Space Group ----------" << std::endl;
    std::cout << "Hall number: " << hall_number_ << std::endl;
    std::cout << "International space group number: " << spacegroup_number_ << std::endl;
    std::cout << "Symbol: " << spacegroup_symbol_ << std::endl;
    std::cout << "Crystal system: " << crystal_system_ << std::endl;
    std::cout << "Number of symmetry operations: " << n_symmetry_ops_ << std::endl;
    std::cout << "Original mesh size: " << total_grid_ << std::endl;
    std::cout << "Irreducible mesh size: " << n_irreducible_ << std::endl;
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

int SpaceGroup::get_required_grid_divisor() const
{
    // Compute required divisor from actual translation components in symmetry operations
    // For each translation t, if t = p/q (in lowest terms), grid must be divisible by q

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

    int result = 1;

    for (int op = 0; op < n_symmetry_ops_; ++op)
    {
        for (int dim = 0; dim < 3; ++dim)
        {
            int denom = get_denominator(translations_[op][dim]);
            result = std::lcm(result, denom);
        }
    }

    return result;
}

void SpaceGroup::validate_grid_compatibility()
{
    int required_divisor = get_required_grid_divisor();

    if (required_divisor <= 1)
    {
        return;  // No special requirement
    }

    std::vector<int> incompatible;
    for (int dim = 0; dim < 3; ++dim)
    {
        if (nx_[dim] % required_divisor != 0)
        {
            incompatible.push_back(dim);
        }
    }

    if (!incompatible.empty())
    {
        std::string msg = "Grid [" + std::to_string(nx_[0]) + ", " +
            std::to_string(nx_[1]) + ", " + std::to_string(nx_[2]) +
            "] is incompatible with space group " + spacegroup_symbol_ +
            " (No. " + std::to_string(spacegroup_number_) + ").\n";
        msg += "Grid dimensions must be divisible by " + std::to_string(required_divisor) + ".\n";
        msg += "Incompatible dimensions:";

        const char* axis_names[] = {"x", "y", "z"};
        for (int dim : incompatible)
        {
            msg += std::string(" ") + axis_names[dim] + "=" + std::to_string(nx_[dim]);
        }

        // Suggest valid grid sizes
        msg += "\nSuggested grid: [";
        for (int dim = 0; dim < 3; ++dim)
        {
            if (dim > 0) msg += ", ";
            if (nx_[dim] % required_divisor == 0)
            {
                msg += std::to_string(nx_[dim]);
            }
            else
            {
                int lower = (nx_[dim] / required_divisor) * required_divisor;
                int upper = lower + required_divisor;
                int suggested = (upper - nx_[dim] < nx_[dim] - lower || lower == 0) ? upper : lower;
                msg += std::to_string(suggested);
            }
        }
        msg += "]";

        throw_with_line_number(msg);
    }
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
    int max_size = 192;
    int rotations[192][3][3];
    double translations[192][3];

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
    // Initialize full_to_reduced_map with -1 (unvisited)
    full_to_reduced_map_.resize(total_grid_, -1);
    reduced_basis_indices_.clear();

    int Nx = nx_[0];
    int Ny = nx_[1];
    int Nz = nx_[2];

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

            // Apply all symmetry operations to find orbit
            for (int op = 0; op < n_symmetry_ops_; ++op)
            {
                // Apply rotation: R * point
                int rx = rotations_[op][0][0] * ix + rotations_[op][0][1] * iy + rotations_[op][0][2] * iz;
                int ry = rotations_[op][1][0] * ix + rotations_[op][1][1] * iy + rotations_[op][1][2] * iz;
                int rz = rotations_[op][2][0] * ix + rotations_[op][2][1] * iy + rotations_[op][2][2] * iz;

                // Add translation (scaled by grid size and rounded)
                int tx = static_cast<int>(std::round(translations_[op][0] * Nx));
                int ty = static_cast<int>(std::round(translations_[op][1] * Ny));
                int tz = static_cast<int>(std::round(translations_[op][2] * Nz));

                // Apply periodic boundary conditions
                int ox = ((rx + tx) % Nx + Nx) % Nx;
                int oy = ((ry + ty) % Ny + Ny) % Ny;
                int oz = ((rz + tz) % Nz + Nz) % Nz;

                // Convert to flat index
                int orbit_flat = ox * Ny * Nz + oy * Nz + oz;

                // Mark this orbit point
                full_to_reduced_map_[orbit_flat] = count;
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
    for (int f = 0; f < n_fields; ++f)
    {
        const double* src = full_field + f * total_grid_;
        double* dst = reduced_field + f * n_irreducible_;

        for (int i = 0; i < n_irreducible_; ++i)
        {
            dst[i] = src[reduced_basis_indices_[i]];
        }
    }
}

void SpaceGroup::from_reduced_basis(const double* reduced_field, double* full_field, int n_fields) const
{
    for (int f = 0; f < n_fields; ++f)
    {
        const double* src = reduced_field + f * n_irreducible_;
        double* dst = full_field + f * total_grid_;

        for (int i = 0; i < total_grid_; ++i)
        {
            dst[i] = src[full_to_reduced_map_[i]];
        }
    }
}

void SpaceGroup::symmetrize(double* field, int n_fields) const
{
    // Temporary buffer for orbit sums
    std::vector<double> orbit_sums(n_irreducible_);

    for (int f = 0; f < n_fields; ++f)
    {
        double* data = field + f * total_grid_;

        // Reset orbit sums
        std::fill(orbit_sums.begin(), orbit_sums.end(), 0.0);

        // Accumulate values by orbit
        for (int i = 0; i < total_grid_; ++i)
        {
            orbit_sums[full_to_reduced_map_[i]] += data[i];
        }

        // Compute averages and broadcast back
        for (int i = 0; i < total_grid_; ++i)
        {
            int orbit_idx = full_to_reduced_map_[i];
            data[i] = orbit_sums[orbit_idx] / orbit_counts_[orbit_idx];
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
