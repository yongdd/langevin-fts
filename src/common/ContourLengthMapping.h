/**
 * @file ContourLengthMapping.h
 * @brief Provides mappings from floating-point contour lengths and Δs values to integers.
 *
 * This header provides the ContourLengthMapping class which creates one-to-one
 * mappings between floating-point block lengths (contour_length) and integer indices,
 * as well as between local Δs values and integer indices.
 *
 * **Purpose:**
 *
 * For discrete chain models with arbitrary block lengths, the propagator key format
 * needs to encode both the block length and the local contour step size. Since
 * floating-point comparisons are unreliable for exact matching, this class maps
 * unique floating-point values to positive integers within machine precision.
 *
 * **Key Concepts:**
 *
 * - **Contour Length**: The relative block length (N_block / N_Ref)
 * - **Global ds**: The target contour step size specified by user
 * - **Local ds**: Block-specific step size = contour_length / n_segment
 * - **n_segment**: Number of segments = round(contour_length / global_ds)
 *
 * For example, if global_ds = 0.01 and contour_length = 0.505:
 * - n_segment = round(0.505 / 0.01) = 51
 * - local_ds = 0.505 / 51 ≈ 0.00990196
 *
 * The mapping assigns integer indices to unique values:
 * - contour_length: {1 → 0.3, 2 → 0.505, 3 → 0.7, ...}
 * - local_ds: {1 → 0.01, 2 → 0.00990196, ...}
 *
 * **Reference:**
 *
 * This approach is described in J. Chem. Theory Comput. 2025, 21, 3676 (SI):
 * "an alternative method for handling floating-point block lengths is to define
 * a mapping between positive integers and floating-point numbers."
 *
 * @see Molecules for where this mapping is constructed
 * @see PropagatorCode for key generation using these mappings
 */

#ifndef CONTOUR_LENGTH_MAPPING_H_
#define CONTOUR_LENGTH_MAPPING_H_

#include <vector>
#include <map>
#include <string>

/**
 * @class ContourLengthMapping
 * @brief Maps floating-point contour lengths and local Δs values to integer indices.
 *
 * This class collects all unique floating-point values (within machine precision)
 * from polymer blocks and assigns integer indices starting from 1.
 *
 * **Usage:**
 *
 * 1. Construct with global_ds and tolerance
 * 2. Call add_block() for each block to register its contour_length
 * 3. Call finalize() to build the mappings
 * 4. Use get_length_index() and get_ds_index() to look up integers
 *
 * @example
 * @code
 * ContourLengthMapping mapping(0.01);  // global_ds = 0.01
 *
 * // Register all blocks from all polymers
 * for (auto& polymer : polymers) {
 *     for (auto& block : polymer.get_blocks()) {
 *         mapping.add_block(block.contour_length);
 *     }
 * }
 *
 * mapping.finalize();  // Build the mappings
 *
 * // Look up indices
 * int len_idx = mapping.get_length_index(0.505);  // e.g., returns 2
 * int ds_idx = mapping.get_ds_index(0.505);       // e.g., returns 3
 * @endcode
 */
class ContourLengthMapping
{
private:
    /**
     * @brief Global contour step size (target Δs from user input).
     */
    double global_ds;

    /**
     * @brief Tolerance for floating-point comparison (machine precision).
     *
     * Two values are considered equal if |a - b| < tolerance.
     * Default: 1e-10
     */
    double tolerance;

    /**
     * @brief Vector of unique contour lengths (sorted).
     *
     * After finalize(), contains all unique block lengths found.
     */
    std::vector<double> unique_lengths;

    /**
     * @brief Vector of unique local Δs values (sorted).
     *
     * After finalize(), contains all unique local_ds values computed from blocks.
     */
    std::vector<double> unique_ds_values;

    /**
     * @brief Map from contour_length index to local_ds index.
     *
     * length_to_ds_index[i] gives the ds index for unique_lengths[i].
     */
    std::vector<int> length_to_ds_index;

    /**
     * @brief Map from contour_length index to n_segment.
     *
     * length_to_n_segment[i] gives the segment count for unique_lengths[i].
     */
    std::vector<int> length_to_n_segment;

    /**
     * @brief Whether finalize() has been called.
     */
    bool is_finalized;

    /**
     * @brief Temporary storage for contour lengths before finalization.
     */
    std::vector<double> pending_lengths;

    /**
     * @brief Check if two floating-point values are equal within tolerance.
     *
     * @param a First value
     * @param b Second value
     * @return true if |a - b| < tolerance
     */
    bool is_equal(double a, double b) const;

    /**
     * @brief Find index of value in sorted vector, or -1 if not found.
     *
     * @param vec Sorted vector to search
     * @param value Value to find
     * @return Index (0-based) if found, -1 otherwise
     */
    int find_index(const std::vector<double>& vec, double value) const;

public:
    /**
     * @brief Construct a ContourLengthMapping.
     *
     * @param global_ds  Global contour step size (target Δs)
     * @param tolerance  Tolerance for floating-point comparison (default: 1e-10)
     */
    ContourLengthMapping(double global_ds, double tolerance = 1e-10);

    /**
     * @brief Default constructor (creates uninitialized mapping).
     */
    ContourLengthMapping();

    /**
     * @brief Destructor.
     */
    ~ContourLengthMapping() {}

    /**
     * @brief Register a block's contour length.
     *
     * Call this for each block before calling finalize().
     *
     * @param contour_length Block's contour length (floating-point)
     */
    void add_block(double contour_length);

    /**
     * @brief Build the mappings from collected blocks.
     *
     * Must be called after all add_block() calls and before any lookups.
     * This method:
     * 1. Sorts and deduplicates contour lengths
     * 2. Computes n_segment and local_ds for each
     * 3. Sorts and deduplicates local_ds values
     * 4. Builds index mappings
     */
    void finalize();

    /**
     * @brief Get integer index for a contour length.
     *
     * @param contour_length Block's contour length
     * @return Integer index (1-based), or throws if not found
     *
     * @throws Exception if contour_length not in mapping
     */
    int get_length_index(double contour_length) const;

    /**
     * @brief Get integer index for local Δs corresponding to a contour length.
     *
     * @param contour_length Block's contour length
     * @return Integer index (1-based) for the local Δs value
     *
     * @throws Exception if contour_length not in mapping
     */
    int get_ds_index(double contour_length) const;

    /**
     * @brief Get n_segment for a contour length.
     *
     * @param contour_length Block's contour length
     * @return Number of segments for this block
     *
     * @throws Exception if contour_length not in mapping
     */
    int get_n_segment(double contour_length) const;

    /**
     * @brief Get local Δs value for a contour length.
     *
     * @param contour_length Block's contour length
     * @return Local Δs = contour_length / n_segment
     *
     * @throws Exception if contour_length not in mapping
     */
    double get_local_ds(double contour_length) const;

    /**
     * @brief Get contour length from its index.
     *
     * @param index Integer index (1-based)
     * @return Contour length value
     *
     * @throws Exception if index out of range
     */
    double get_length_from_index(int index) const;

    /**
     * @brief Get local Δs from its index.
     *
     * @param index Integer index (1-based)
     * @return Local Δs value
     *
     * @throws Exception if index out of range
     */
    double get_ds_from_index(int index) const;

    /**
     * @brief Get number of unique contour lengths.
     * @return Count of unique lengths
     */
    int get_n_unique_lengths() const;

    /**
     * @brief Get number of unique local Δs values.
     * @return Count of unique Δs values
     */
    int get_n_unique_ds() const;

    /**
     * @brief Get global Δs value.
     * @return Global contour step size
     */
    double get_global_ds() const;

    /**
     * @brief Check if mapping has been finalized.
     * @return true if finalize() has been called
     */
    bool finalized() const;

    /**
     * @brief Print mapping information for debugging.
     */
    void print_mapping() const;
};

#endif
