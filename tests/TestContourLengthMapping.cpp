/*******************************************************************************
 * WARNING TO AI ASSISTANTS (Claude, ChatGPT, Copilot, etc.):
 * DO NOT MODIFY TEST PARAMETERS WITHOUT EXPLICIT PERMISSION FROM THE USER.
 * - NEVER increase tolerance values (e.g., 1e-7 -> 1e-6)
 * - NEVER decrease field strength or standard deviation values
 * - NEVER change grid sizes, box dimensions, or polymer parameters
 * - NEVER weaken any test conditions to make tests pass
 * These parameters are carefully calibrated. If a test fails, report the
 * failure to the user rather than modifying the test to pass.
 ******************************************************************************/

/**
 * @file TestContourLengthMapping.cpp
 * @brief Unit test for ContourLengthMapping class.
 *
 * Tests the mapping of floating-point contour lengths and local Δs values
 * to integer indices.
 */

#include <iostream>
#include <cmath>
#include <vector>
#include <array>
#include <limits>

#include "Molecules.h"
#include "ContourLengthMapping.h"

int main()
{
    try
    {
        std::cout << "=== Test 1: Basic ContourLengthMapping ===" << std::endl;

        // Test with global_ds = 0.01
        ContourLengthMapping mapping(0.01);

        // Add some block lengths
        mapping.add_block(0.3);   // 30 segments, local_ds = 0.01
        mapping.add_block(0.5);   // 50 segments, local_ds = 0.01
        mapping.add_block(0.505); // 51 segments, local_ds = 0.505/51 ≈ 0.00990196
        mapping.add_block(0.7);   // 70 segments, local_ds = 0.01
        mapping.add_block(0.3);   // duplicate - should be deduplicated

        mapping.finalize();
        mapping.print_mapping();

        // Verify unique lengths
        if (mapping.get_n_unique_lengths() != 4)
        {
            std::cout << "FAILED: Expected 4 unique lengths, got "
                      << mapping.get_n_unique_lengths() << std::endl;
            return -1;
        }

        // Verify index lookups
        int idx_03 = mapping.get_length_index(0.3);
        int idx_05 = mapping.get_length_index(0.5);
        int idx_0505 = mapping.get_length_index(0.505);
        int idx_07 = mapping.get_length_index(0.7);

        std::cout << "Index for 0.3: " << idx_03 << std::endl;
        std::cout << "Index for 0.5: " << idx_05 << std::endl;
        std::cout << "Index for 0.505: " << idx_0505 << std::endl;
        std::cout << "Index for 0.7: " << idx_07 << std::endl;

        // Indices should be 1, 2, 3, 4 (sorted order)
        if (idx_03 != 1 || idx_05 != 2 || idx_0505 != 3 || idx_07 != 4)
        {
            std::cout << "FAILED: Incorrect length indices" << std::endl;
            return -1;
        }

        // Verify n_segment values
        if (mapping.get_n_segment(0.3) != 30)
        {
            std::cout << "FAILED: n_segment(0.3) should be 30, got "
                      << mapping.get_n_segment(0.3) << std::endl;
            return -1;
        }
        if (mapping.get_n_segment(0.505) != 51)
        {
            std::cout << "FAILED: n_segment(0.505) should be 51, got "
                      << mapping.get_n_segment(0.505) << std::endl;
            return -1;
        }

        // Verify local_ds values
        double local_ds_03 = mapping.get_local_ds(0.3);
        double local_ds_0505 = mapping.get_local_ds(0.505);

        std::cout << "Local ds for 0.3: " << local_ds_03 << std::endl;
        std::cout << "Local ds for 0.505: " << local_ds_0505 << std::endl;

        if (!(std::abs(local_ds_03 - 0.01) <= 1e-10))
        {
            std::cout << "FAILED: local_ds(0.3) should be 0.01" << std::endl;
            return -1;
        }
        if (!(std::abs(local_ds_0505 - 0.505 / 51.0) <= 1e-10))
        {
            std::cout << "FAILED: local_ds(0.505) should be 0.505/51" << std::endl;
            return -1;
        }

        // Verify ds index (0.3, 0.5, 0.7 all have local_ds = 0.01)
        int ds_idx_03 = mapping.get_ds_index(0.3);
        int ds_idx_05 = mapping.get_ds_index(0.5);
        int ds_idx_0505 = mapping.get_ds_index(0.505);
        int ds_idx_07 = mapping.get_ds_index(0.7);

        std::cout << "DS index for 0.3: " << ds_idx_03 << std::endl;
        std::cout << "DS index for 0.5: " << ds_idx_05 << std::endl;
        std::cout << "DS index for 0.505: " << ds_idx_0505 << std::endl;
        std::cout << "DS index for 0.7: " << ds_idx_07 << std::endl;

        // 0.3, 0.5, 0.7 should have the same ds_index (0.01)
        // 0.505 should have a different ds_index
        if (ds_idx_03 != ds_idx_05 || ds_idx_03 != ds_idx_07)
        {
            std::cout << "FAILED: 0.3, 0.5, 0.7 should have same ds_index" << std::endl;
            return -1;
        }
        if (ds_idx_0505 == ds_idx_03)
        {
            std::cout << "FAILED: 0.505 should have different ds_index from 0.3" << std::endl;
            return -1;
        }

        std::cout << "Test 1 PASSED" << std::endl << std::endl;

        // =====================================================
        std::cout << "=== Test 2: Integration with Molecules ===" << std::endl;

        // Use continuous model since we test non-integer block lengths (0.505)
        std::map<std::string, double> bond_lengths = {{"A", 1.0}, {"B", 1.0}};
        Molecules molecules("continuous", 0.01, bond_lengths);

        // Add a polymer with various block lengths
        std::vector<BlockInput> blocks1 = {
            {"A", 0.25, 0, 1},  // 25 segments
            {"B", 0.505, 1, 2}  // 51 segments (non-integer ratio with ds, allowed for continuous)
        };
        molecules.add_polymer(0.5, blocks1);

        // Add another polymer
        std::vector<BlockInput> blocks2 = {
            {"A", 0.25, 0, 1},  // same as above - should deduplicate
            {"B", 0.75, 1, 2}   // 75 segments
        };
        molecules.add_polymer(0.5, blocks2);

        // Finalize the mapping
        molecules.finalize_contour_length_mapping();

        const auto& mol_mapping = molecules.get_contour_length_mapping();
        mol_mapping.print_mapping();

        // Should have 3 unique lengths: 0.25, 0.505, 0.75
        if (mol_mapping.get_n_unique_lengths() != 3)
        {
            std::cout << "FAILED: Expected 3 unique lengths, got "
                      << mol_mapping.get_n_unique_lengths() << std::endl;
            return -1;
        }

        std::cout << "Test 2 PASSED" << std::endl << std::endl;

        // =====================================================
        std::cout << "=== Test 3: Machine precision comparison ===" << std::endl;

        ContourLengthMapping mapping3(0.01);

        // Add values that differ only by machine precision
        mapping3.add_block(0.5);
        mapping3.add_block(0.5 + 1e-14);  // should be treated as same
        mapping3.add_block(0.5 - 1e-14);  // should be treated as same
        mapping3.add_block(0.5 + 1e-8);   // should be different

        mapping3.finalize();
        mapping3.print_mapping();

        // Should have 2 unique lengths (0.5 and 0.5+1e-8)
        if (mapping3.get_n_unique_lengths() != 2)
        {
            std::cout << "FAILED: Expected 2 unique lengths (machine precision test), got "
                      << mapping3.get_n_unique_lengths() << std::endl;
            return -1;
        }

        std::cout << "Test 3 PASSED" << std::endl << std::endl;

        // =====================================================
        std::cout << "=== Test 4: Reverse mapping (index to value) ===" << std::endl;

        ContourLengthMapping mapping4(0.01);
        mapping4.add_block(0.3);
        mapping4.add_block(0.505);
        mapping4.add_block(0.7);
        mapping4.finalize();

        // Test get_length_from_index (reverse mapping)
        double len1 = mapping4.get_length_from_index(1);  // should be 0.3
        double len2 = mapping4.get_length_from_index(2);  // should be 0.505
        double len3 = mapping4.get_length_from_index(3);  // should be 0.7

        std::cout << "Index 1 -> length: " << len1 << std::endl;
        std::cout << "Index 2 -> length: " << len2 << std::endl;
        std::cout << "Index 3 -> length: " << len3 << std::endl;

        if (!(std::abs(len1 - 0.3) <= 1e-10))
        {
            std::cout << "FAILED: get_length_from_index(1) should return 0.3" << std::endl;
            return -1;
        }
        if (!(std::abs(len2 - 0.505) <= 1e-10))
        {
            std::cout << "FAILED: get_length_from_index(2) should return 0.505" << std::endl;
            return -1;
        }
        if (!(std::abs(len3 - 0.7) <= 1e-10))
        {
            std::cout << "FAILED: get_length_from_index(3) should return 0.7" << std::endl;
            return -1;
        }

        // Test get_ds_from_index (reverse mapping for ds)
        int n_unique_ds = mapping4.get_n_unique_ds();
        std::cout << "Number of unique ds values: " << n_unique_ds << std::endl;

        for (int i = 0; i < n_unique_ds; ++i)
        {
            double ds_val = mapping4.get_ds_from_index(i);
            std::cout << "DS Index " << i << " -> ds: " << ds_val << std::endl;
        }

        // Verify round-trip: index -> length -> index
        for (int i = 1; i <= mapping4.get_n_unique_lengths(); ++i)
        {
            double length = mapping4.get_length_from_index(i);
            int idx = mapping4.get_length_index(length);
            if (idx != i)
            {
                std::cout << "FAILED: Round-trip failed for index " << i << std::endl;
                return -1;
            }
        }

        std::cout << "Round-trip verification passed!" << std::endl;
        std::cout << "Test 4 PASSED" << std::endl << std::endl;

        // =====================================================
        std::cout << "=== Test 5: Half-integer boundary robustness ===" << std::endl;

        // Test case: contour_length/ds close to 3.5 (mentioned in issue)
        // Values like 3.4999999999 and 3.5000000001 should both round to 4
        double ds5 = 1.0;
        ContourLengthMapping mapping5(ds5);

        // Add values that are very close to 3.5*ds but on different sides
        double len_below = 3.5 * ds5 - 1e-12;  // Just below 3.5
        double len_above = 3.5 * ds5 + 1e-12;  // Just above 3.5
        double len_exact = 3.5 * ds5;          // Exactly 3.5

        mapping5.add_block(len_below);
        mapping5.add_block(len_above);
        mapping5.add_block(len_exact);

        mapping5.finalize();
        mapping5.print_mapping();

        // All three should have the same n_segment (4, since 3.5 rounds away from zero)
        int n_seg_below = mapping5.get_n_segment(len_below);
        int n_seg_above = mapping5.get_n_segment(len_above);
        int n_seg_exact = mapping5.get_n_segment(len_exact);

        std::cout << "n_segment for len_below (3.5-1e-12): " << n_seg_below << std::endl;
        std::cout << "n_segment for len_above (3.5+1e-12): " << n_seg_above << std::endl;
        std::cout << "n_segment for len_exact (3.5): " << n_seg_exact << std::endl;

        // Verify all are 4 (std::lround(3.5) = 4)
        if (n_seg_below != 4 || n_seg_above != 4 || n_seg_exact != 4)
        {
            std::cout << "FAILED: Values near 3.5 should all round to 4" << std::endl;
            return -1;
        }

        // Test another boundary: values near 2.5*ds
        ContourLengthMapping mapping5b(ds5);
        double len_2_5_below = 2.5 * ds5 - 1e-12;
        double len_2_5_above = 2.5 * ds5 + 1e-12;
        mapping5b.add_block(len_2_5_below);
        mapping5b.add_block(len_2_5_above);
        mapping5b.finalize();

        int n_seg_2_5_below = mapping5b.get_n_segment(len_2_5_below);
        int n_seg_2_5_above = mapping5b.get_n_segment(len_2_5_above);

        std::cout << "n_segment for 2.5-1e-12: " << n_seg_2_5_below << std::endl;
        std::cout << "n_segment for 2.5+1e-12: " << n_seg_2_5_above << std::endl;

        if (n_seg_2_5_below != 3 || n_seg_2_5_above != 3)
        {
            std::cout << "FAILED: Values near 2.5 should all round to 3" << std::endl;
            return -1;
        }

        std::cout << "Test 5 PASSED" << std::endl << std::endl;

        // Distinct segment counts must never share a length, even with an
        // explicitly loose comparison tolerance near a rounding boundary.
        ContourLengthMapping boundary(0.01, 1e-10);
        const double lower = 0.495 - 2e-11;
        const double upper = 0.495 + 2e-11;
        boundary.add_block(lower);
        boundary.add_block(upper);
        boundary.finalize();
        if (boundary.get_n_unique_lengths() != 2 || boundary.get_n_segment(lower) != 49 ||
            boundary.get_n_segment(upper) != 50 ||
            boundary.get_length_index(lower) == boundary.get_length_index(upper))
        {
            std::cout << "FAILED: lengths with different segment counts were merged" << std::endl;
            return -1;
        }

        // A fixed absolute tolerance for local ds loses significant contour
        // length after hundreds of steps, and breaks inverse aggregation lookups.
        ContourLengthMapping precise(0.001);
        for (double length : {0.1, 0.4, 0.50000004})
            precise.add_block(length);
        precise.finalize();
        if (precise.get_n_unique_ds() != 2 || precise.get_ds_index(0.4) == precise.get_ds_index(0.50000004))
        {
            std::cout << "FAILED: distinct local contour steps were merged" << std::endl;
            return -1;
        }
        for (double length : {0.1, 0.4, 0.50000004})
            if (precise.get_length_index_from_n_segment(precise.get_n_segment(length), precise.get_ds_index(length))
                != precise.get_length_index(length))
            {
                std::cout << "FAILED: segment-count / ds-index inverse lookup" << std::endl;
                return -1;
            }

        // Positive short blocks are not the zero-length aggregation sentinel.
        ContourLengthMapping short_block(1e-11);
        short_block.add_block(5e-11);
        short_block.finalize();
        if (short_block.get_length_index(5e-11) == 0 || short_block.get_n_segment(5e-11) != 5)
        {
            std::cout << "FAILED: a positive short contour was treated as zero" << std::endl;
            return -1;
        }

        Molecules discrete("discrete", 0.01, bond_lengths);
        discrete.add_polymer(1.0, {{"A", 0.5, 0, 1}, {"B", 0.5 + 4e-10, 1, 2}});
        discrete.finalize_contour_length_mapping();
        if (discrete.get_contour_length_mapping().get_n_unique_ds() != 1 ||
            discrete.get_polymer(0).get_alpha() != 1.0)
        {
            std::cout << "FAILED: accepted discrete roundoff changed the global contour step" << std::endl;
            return -1;
        }

        // The shared scalar canonicalizer and molecule construction must choose
        // exactly the same lengths, including tiny steps and half-integer snaps.
        double accepted_upper = 0.5 + 1e-9;
        while (accepted_upper - 0.5 > 1e-9)
            accepted_upper = std::nextafter(accepted_upper, 0.5);
        double accepted_lower = 0.5 - 1e-9;
        while (0.5 - accepted_lower > 1e-9)
            accepted_lower = std::nextafter(accepted_lower, 0.5);
        const std::vector<std::array<double, 3>> accepted_discrete = {
            {0.5 + 4e-10, 0.01, 0.5},
            {0.5 - 4e-10, 0.01, 0.5},
            {accepted_upper, 0.01, 0.5},
            {accepted_lower, 0.01, 0.5},
            {1.0009e-6, 1e-6, 1e-6},
            {0.25e-9, 1e-9, 1e-9},
            {1.5e-9 - 5e-19, 1e-9, 2e-9},
            {1.5e-9 - 2e-18, 1e-9, 1e-9},
        };
        for (const auto& [length, step, expected] : accepted_discrete)
        {
            const double canonical = ContourLengthMapping::discrete_contour_length(length, step);
            Molecules molecule("discrete", step, bond_lengths);
            molecule.add_polymer(1.0, {{"A", length, 0, 1}});
            if (canonical != expected || molecule.get_polymer(0).get_alpha() != canonical ||
                molecule.get_polymer(0).get_blocks()[0].contour_length != canonical)
            {
                std::cout << "FAILED: discrete canonicalizer differs from molecule length" << std::endl;
                return -1;
            }
        }

        const double nan = std::numeric_limits<double>::quiet_NaN();
        const double inf = std::numeric_limits<double>::infinity();
        const std::vector<std::array<double, 2>> rejected_discrete = {
            {std::nextafter(accepted_upper, inf), 0.01},
            {std::nextafter(accepted_lower, 0.0), 0.01},
            {0.495, 0.01}, {0.0, 0.01}, {-0.1, 0.01}, {nan, 0.01}, {inf, 0.01},
            {0.5, 0.0}, {0.5, -0.01}, {0.5, nan}, {0.5, inf}, {1.0, 1e-12},
        };
        for (const auto& [length, step] : rejected_discrete)
        {
            bool scalar_rejected = false, molecule_rejected = false;
            try { ContourLengthMapping::discrete_contour_length(length, step); }
            catch (const std::exception&) { scalar_rejected = true; }
            try
            {
                Molecules molecule("discrete", step, bond_lengths);
                molecule.add_polymer(1.0, {{"A", length, 0, 1}});
            }
            catch (const std::exception&) { molecule_rejected = true; }
            if (!scalar_rejected || !molecule_rejected)
            {
                std::cout << "FAILED: invalid discrete length accepted by one construction path" << std::endl;
                return -1;
            }
        }

        // =====================================================
        std::cout << "=== All tests PASSED ===" << std::endl;
        return 0;
    }
    catch (std::exception& exc)
    {
        std::cout << exc.what() << std::endl;
        return -1;
    }
}
