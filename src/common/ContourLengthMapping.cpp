/**
 * @file ContourLengthMapping.cpp
 * @brief Implementation of ContourLengthMapping class.
 *
 * Provides one-to-one mappings from floating-point block lengths and local Δs
 * values to integer indices. This enables reliable comparison of propagator
 * keys when block lengths are arbitrary floating-point numbers.
 *
 * **Algorithm:**
 *
 * 1. Collect all block contour lengths from all polymers
 * 2. Sort and deduplicate within machine precision
 * 3. For each unique length L:
 *    - n_segment = round(L / global_ds)
 *    - local_ds = L / n_segment
 * 4. Sort and deduplicate local_ds values
 * 5. Build index lookups (1-based indices)
 *
 * @see Molecules::add_polymer for integration point
 */

#include <iostream>
#include <algorithm>
#include <cmath>
#include <cassert>

#include "ContourLengthMapping.h"
#include "Exception.h"

/**
 * @brief Construct mapping with global Δs and tolerance.
 *
 * @param global_ds  Target contour step size
 * @param tolerance  Floating-point comparison tolerance
 */
ContourLengthMapping::ContourLengthMapping(double global_ds, double tolerance)
    : global_ds(global_ds), tolerance(tolerance), is_finalized(false)
{
    if (global_ds <= 0.0)
    {
        throw_with_line_number("global_ds must be positive.");
    }
    if (tolerance <= 0.0)
    {
        throw_with_line_number("tolerance must be positive.");
    }
}

/**
 * @brief Default constructor.
 */
ContourLengthMapping::ContourLengthMapping()
    : global_ds(0.0), tolerance(1e-10), is_finalized(false)
{
}

/**
 * @brief Check if two values are equal within tolerance.
 */
bool ContourLengthMapping::is_equal(double a, double b) const
{
    return std::abs(a - b) < tolerance;
}

/**
 * @brief Find index of value in sorted vector.
 *
 * Uses binary search for efficiency, then checks tolerance.
 *
 * @return 0-based index if found, -1 otherwise
 */
int ContourLengthMapping::find_index(const std::vector<double>& vec, double value) const
{
    if (vec.empty())
        return -1;

    // Binary search for approximate position
    auto it = std::lower_bound(vec.begin(), vec.end(), value - tolerance);

    // Check nearby elements
    for (auto check = it; check != vec.end() && *check <= value + tolerance; ++check)
    {
        if (is_equal(*check, value))
        {
            return static_cast<int>(check - vec.begin());
        }
    }

    // Check element before (in case of floating-point edge cases)
    if (it != vec.begin())
    {
        --it;
        if (is_equal(*it, value))
        {
            return static_cast<int>(it - vec.begin());
        }
    }

    return -1;
}

/**
 * @brief Register a block's contour length.
 */
void ContourLengthMapping::add_block(double contour_length)
{
    if (is_finalized)
    {
        throw_with_line_number("Cannot add blocks after finalize() has been called.");
    }

    if (contour_length <= 0.0)
    {
        throw_with_line_number("contour_length must be positive.");
    }

    pending_lengths.push_back(contour_length);
}

/**
 * @brief Build mappings from collected blocks.
 *
 * After this call:
 * - unique_lengths contains sorted unique contour lengths
 * - unique_ds_values contains sorted unique local Δs values
 * - length_to_ds_index maps length index to ds index
 * - length_to_n_segment maps length index to segment count
 */
void ContourLengthMapping::finalize()
{
    if (is_finalized)
    {
        throw_with_line_number("finalize() has already been called.");
    }

    if (pending_lengths.empty())
    {
        is_finalized = true;
        return;
    }

    // Sort the pending lengths
    std::sort(pending_lengths.begin(), pending_lengths.end());

    // Deduplicate within tolerance
    unique_lengths.clear();
    for (double len : pending_lengths)
    {
        if (unique_lengths.empty() || !is_equal(len, unique_lengths.back()))
        {
            unique_lengths.push_back(len);
        }
    }

    // Compute n_segment and local_ds for each unique length
    std::vector<double> local_ds_values;
    length_to_n_segment.resize(unique_lengths.size());

    for (size_t i = 0; i < unique_lengths.size(); ++i)
    {
        double len = unique_lengths[i];
        int n_seg = static_cast<int>(std::lround(len / global_ds));

        // Ensure at least 1 segment
        if (n_seg < 1)
        {
            n_seg = 1;
        }

        length_to_n_segment[i] = n_seg;
        double local_ds = len / static_cast<double>(n_seg);
        local_ds_values.push_back(local_ds);
    }

    // Sort and deduplicate local_ds values
    std::vector<std::pair<double, size_t>> ds_with_index;
    for (size_t i = 0; i < local_ds_values.size(); ++i)
    {
        ds_with_index.push_back({local_ds_values[i], i});
    }
    std::sort(ds_with_index.begin(), ds_with_index.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    unique_ds_values.clear();
    std::vector<int> ds_value_to_unique_index(local_ds_values.size());

    for (const auto& [ds_val, orig_idx] : ds_with_index)
    {
        if (unique_ds_values.empty() || !is_equal(ds_val, unique_ds_values.back()))
        {
            unique_ds_values.push_back(ds_val);
        }
        ds_value_to_unique_index[orig_idx] = static_cast<int>(unique_ds_values.size() - 1);
    }

    // Build length_to_ds_index mapping
    length_to_ds_index.resize(unique_lengths.size());
    for (size_t i = 0; i < unique_lengths.size(); ++i)
    {
        length_to_ds_index[i] = ds_value_to_unique_index[i];
    }

    // Clear pending data
    pending_lengths.clear();
    is_finalized = true;
}

/**
 * @brief Get integer index for a contour length.
 *
 * @return 1-based index
 */
int ContourLengthMapping::get_length_index(double contour_length) const
{
    if (!is_finalized)
    {
        throw_with_line_number("Must call finalize() before get_length_index().");
    }

    int idx = find_index(unique_lengths, contour_length);
    if (idx < 0)
    {
        throw_with_line_number("Contour length " + std::to_string(contour_length) +
                               " not found in mapping.");
    }

    return idx + 1;  // 1-based index
}

/**
 * @brief Get integer index for local Δs corresponding to a contour length.
 *
 * @return 1-based index
 */
int ContourLengthMapping::get_ds_index(double contour_length) const
{
    if (!is_finalized)
    {
        throw_with_line_number("Must call finalize() before get_ds_index().");
    }

    int len_idx = find_index(unique_lengths, contour_length);
    if (len_idx < 0)
    {
        throw_with_line_number("Contour length " + std::to_string(contour_length) +
                               " not found in mapping.");
    }

    return length_to_ds_index[len_idx] + 1;  // 1-based index
}

/**
 * @brief Get n_segment for a contour length.
 */
int ContourLengthMapping::get_n_segment(double contour_length) const
{
    if (!is_finalized)
    {
        throw_with_line_number("Must call finalize() before get_n_segment().");
    }

    int idx = find_index(unique_lengths, contour_length);
    if (idx < 0)
    {
        throw_with_line_number("Contour length " + std::to_string(contour_length) +
                               " not found in mapping.");
    }

    return length_to_n_segment[idx];
}

/**
 * @brief Get local Δs value for a contour length.
 */
double ContourLengthMapping::get_local_ds(double contour_length) const
{
    if (!is_finalized)
    {
        throw_with_line_number("Must call finalize() before get_local_ds().");
    }

    int idx = find_index(unique_lengths, contour_length);
    if (idx < 0)
    {
        throw_with_line_number("Contour length " + std::to_string(contour_length) +
                               " not found in mapping.");
    }

    return unique_lengths[idx] / static_cast<double>(length_to_n_segment[idx]);
}

/**
 * @brief Get contour length from its index.
 *
 * @param index 1-based index
 */
double ContourLengthMapping::get_length_from_index(int index) const
{
    if (!is_finalized)
    {
        throw_with_line_number("Must call finalize() before get_length_from_index().");
    }

    if (index < 1 || index > static_cast<int>(unique_lengths.size()))
    {
        throw_with_line_number("Length index " + std::to_string(index) +
                               " out of range [1, " + std::to_string(unique_lengths.size()) + "].");
    }

    return unique_lengths[index - 1];
}

/**
 * @brief Get local Δs from its index.
 *
 * @param index 1-based index
 */
double ContourLengthMapping::get_ds_from_index(int index) const
{
    if (!is_finalized)
    {
        throw_with_line_number("Must call finalize() before get_ds_from_index().");
    }

    if (index < 1 || index > static_cast<int>(unique_ds_values.size()))
    {
        throw_with_line_number("Ds index " + std::to_string(index) +
                               " out of range [1, " + std::to_string(unique_ds_values.size()) + "].");
    }

    return unique_ds_values[index - 1];
}

/**
 * @brief Get number of unique contour lengths.
 */
int ContourLengthMapping::get_n_unique_lengths() const
{
    return static_cast<int>(unique_lengths.size());
}

/**
 * @brief Get number of unique local Δs values.
 */
int ContourLengthMapping::get_n_unique_ds() const
{
    return static_cast<int>(unique_ds_values.size());
}

/**
 * @brief Get global Δs value.
 */
double ContourLengthMapping::get_global_ds() const
{
    return global_ds;
}

/**
 * @brief Check if mapping has been finalized.
 */
bool ContourLengthMapping::finalized() const
{
    return is_finalized;
}

/**
 * @brief Print mapping information for debugging.
 */
void ContourLengthMapping::print_mapping() const
{
    if (!is_finalized)
    {
        std::cout << "Mapping not finalized yet." << std::endl;
        return;
    }

    std::cout << "=== ContourLengthMapping ===" << std::endl;
    std::cout << "Global ds: " << global_ds << std::endl;
    std::cout << "Tolerance: " << tolerance << std::endl;
    std::cout << std::endl;

    std::cout << "Unique contour lengths (" << unique_lengths.size() << "):" << std::endl;
    for (size_t i = 0; i < unique_lengths.size(); ++i)
    {
        double local_ds = unique_lengths[i] / static_cast<double>(length_to_n_segment[i]);
        std::cout << "  Index " << (i + 1) << ": length = " << unique_lengths[i]
                  << ", n_segment = " << length_to_n_segment[i]
                  << ", local_ds = " << local_ds
                  << ", ds_index = " << (length_to_ds_index[i] + 1) << std::endl;
    }
    std::cout << std::endl;

    std::cout << "Unique local ds values (" << unique_ds_values.size() << "):" << std::endl;
    for (size_t i = 0; i < unique_ds_values.size(); ++i)
    {
        std::cout << "  Index " << (i + 1) << ": ds = " << unique_ds_values[i] << std::endl;
    }
    std::cout << "===========================" << std::endl;
}
