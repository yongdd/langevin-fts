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
 * @file TestDisplayArchitecture.cpp
 * @brief Test the display_architecture() method for various polymer topologies.
 */

#include <iostream>
#include <map>
#include <vector>

#include "Molecules.h"
#include "Polymer.h"

int main()
{
    try
    {
        std::map<std::string, double> bond_lengths = {{"A", 1.0}, {"B", 1.0}, {"C", 1.0}};
        double ds = 0.1;

        // Test 1: Linear AB diblock
        std::cout << "--- Test 1: Linear AB Diblock ---" << std::endl;
        {
            std::vector<BlockInput> blocks = {
                {"A", 0.3, 0, 1},
                {"B", 0.7, 1, 2}
            };
            Molecules molecules("Continuous", ds, bond_lengths);
            molecules.add_polymer(1.0, blocks, {});
            molecules.get_polymer(0).display_architecture();
        }

        // Test 2: Linear ABC triblock
        std::cout << "\n--- Test 2: Linear ABC Triblock ---" << std::endl;
        {
            std::vector<BlockInput> blocks = {
                {"A", 0.3, 0, 1},
                {"B", 0.4, 1, 2},
                {"C", 0.3, 2, 3}
            };
            Molecules molecules("Continuous", ds, bond_lengths);
            molecules.add_polymer(1.0, blocks, {});
            molecules.get_polymer(0).display_architecture();
        }

        // Test 3: 3-arm star polymer
        std::cout << "\n--- Test 3: 3-Arm Star Polymer ---" << std::endl;
        {
            std::vector<BlockInput> blocks = {
                {"A", 0.33, 0, 1},
                {"B", 0.33, 0, 2},
                {"C", 0.34, 0, 3}
            };
            Molecules molecules("Continuous", ds, bond_lengths);
            molecules.add_polymer(1.0, blocks, {});
            molecules.get_polymer(0).display_architecture();
        }

        // Test 4: Star with secondary branches (dendritic)
        std::cout << "\n--- Test 4: Dendritic (Star with Secondary Branches) ---" << std::endl;
        {
            std::vector<BlockInput> blocks = {
                {"A", 0.2, 0, 1},   // Core to arm 1
                {"B", 0.2, 0, 2},   // Core to arm 2
                {"C", 0.2, 0, 3},   // Core to arm 3
                {"A", 0.15, 1, 4},  // Arm 1 branch 1
                {"B", 0.15, 1, 5}   // Arm 1 branch 2
            };
            Molecules molecules("Continuous", ds, bond_lengths);
            molecules.add_polymer(1.0, blocks, {});
            molecules.get_polymer(0).display_architecture();
        }

        // Test 5: AB diblock with side chains (comb-like)
        std::cout << "\n--- Test 5: AB Diblock with Side Chain ---" << std::endl;
        {
            std::vector<BlockInput> blocks = {
                {"A", 0.3, 0, 1},   // Main backbone start
                {"B", 0.3, 1, 2},   // Main backbone middle
                {"A", 0.3, 2, 3},   // Main backbone end
                {"C", 0.2, 1, 4}    // Side chain from middle
            };
            Molecules molecules("Continuous", ds, bond_lengths);
            molecules.add_polymer(1.0, blocks, {});
            molecules.get_polymer(0).display_architecture();
        }

        // Test 6: More complex dendritic
        std::cout << "\n--- Test 6: Complex Dendritic ---" << std::endl;
        {
            std::vector<BlockInput> blocks = {
                {"A", 0.2, 0, 1},   // Level 0 to level 1
                {"B", 0.15, 1, 2},  // Branch 1 from 1
                {"B", 0.15, 1, 3},  // Branch 2 from 1
                {"C", 0.1, 2, 4},   // Sub-branch from 2
                {"C", 0.1, 2, 5},   // Sub-branch from 2
                {"C", 0.1, 3, 6},   // Sub-branch from 3
                {"C", 0.1, 3, 7}    // Sub-branch from 3
            };
            Molecules molecules("Continuous", ds, bond_lengths);
            molecules.add_polymer(1.0, blocks, {});
            molecules.get_polymer(0).display_architecture();
        }

        // Test 7: Multiple polymers using display_architectures()
        std::cout << "\n--- Test 7: Multiple Polymers (display_architectures) ---" << std::endl;
        {
            Molecules molecules("Continuous", ds, bond_lengths);

            // Polymer 0: AB diblock
            molecules.add_polymer(0.6, {
                {"A", 0.3, 0, 1},
                {"B", 0.7, 1, 2}
            }, {});

            // Polymer 1: 3-arm star
            molecules.add_polymer(0.4, {
                {"A", 0.33, 0, 1},
                {"B", 0.33, 0, 2},
                {"C", 0.34, 0, 3}
            }, {});

            // Display all architectures at once
            molecules.display_architectures();
        }

        std::cout << "\n--- All Tests Complete ---" << std::endl;
        return 0;
    }
    catch(std::exception& exc)
    {
        std::cout << exc.what() << std::endl;
        return -1;
    }
}
