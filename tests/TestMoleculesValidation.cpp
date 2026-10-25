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
 * @file TestMoleculesValidation.cpp
 * @brief Input checks of Molecules, Polymer and the propagator keys.
 *
 * - A monomer name may end with an underscore: the length index that follows
 *   the name in a propagator code is split off after it.
 * - Out-of-range solvent and block indices throw instead of reading past the end.
 * - A rejected polymer registers no contour lengths in the mapping.
 * - NaN lengths are rejected like non-positive ones.
 */

#include <iostream>
#include <functional>
#include <limits>
#include <string>
#include <vector>

#include "Molecules.h"
#include "Polymer.h"
#include "PropagatorCode.h"
#include "PropagatorComputationOptimizer.h"
#include "ContourLengthMapping.h"
#include "ValidationUtils.h"

static bool throws(const std::function<void()>& call)
{
    try
    {
        call();
    }
    catch (std::exception&)
    {
        return true;
    }
    return false;
}

static bool check(bool condition, const std::string& what)
{
    std::cout << (condition ? "  ok: " : "  FAILED: ") << what << std::endl;
    return condition;
}

int main()
{
    bool ok = true;
    try
    {
        const double nan = std::numeric_limits<double>::quiet_NaN();

        std::cout << "=== Monomer names ending with an underscore ===" << std::endl;
        ok &= check(PropagatorCode::get_key_from_code("A_3") == "A_", "key of code A_3");
        ok &= check(PropagatorCode::get_key_from_code("(B12)A_13") == "(B12)A_", "key of code (B12)A_13");
        {
            // A linear chain and a star with two identical arms (aggregated keys)
            Molecules molecules("continuous", 0.1, {{"A_", 1.0}, {"B", 1.0}, {"_", 1.5}});
            molecules.add_polymer(0.5, {{"A_", 0.5, 0, 1}, {"B", 0.5, 1, 2}}, {});
            molecules.add_polymer(0.5, {{"A_", 0.4, 0, 1}, {"A_", 0.4, 0, 2}, {"_", 0.3, 0, 3}}, {});
            PropagatorComputationOptimizer optimizer(&molecules, true);

            bool types_ok = !optimizer.get_computation_propagators().empty();
            for (const auto& [key, edge] : optimizer.get_computation_propagators())
            {
                const std::string monomer_type = PropagatorCode::get_monomer_type_from_key(key);
                types_ok = types_ok && monomer_type == edge.monomer_type
                    && validation::contains(molecules.get_bond_lengths(), monomer_type);
            }
            ok &= check(types_ok, "every propagator key ends with a known monomer type");

            const ContourLengthMapping& mapping = molecules.get_contour_length_mapping();
            const std::string code = "A_" + std::to_string(mapping.get_length_index(0.4));
            const std::string key = "A_+" + std::to_string(mapping.get_ds_index(0.4));
            ok &= check(PropagatorCode::get_key_from_code(code, mapping) == key, "key of code " + code + " with the mapping");
        }

        std::cout << "=== Index checks ===" << std::endl;
        {
            Molecules molecules("continuous", 0.1, {{"A", 1.0}, {"B", 1.0}});
            molecules.add_polymer(0.8, {{"A", 1.0, 0, 1}}, {});
            molecules.add_solvent(0.2, "B");
            ok &= check(std::get<1>(molecules.get_solvent(0)) == "B", "solvent 0");
            ok &= check(throws([&]{ molecules.get_solvent(1); }), "solvent 1 of 1 throws");
            ok &= check(throws([&]{ molecules.get_solvent(-1); }), "solvent -1 throws");

            Polymer& polymer = molecules.get_polymer(0);
            ok &= check(polymer.get_n_segment(0) == 10, "segments of block 0");
            ok &= check(throws([&]{ polymer.get_n_segment(1); }), "block 1 of 1 throws");
            ok &= check(throws([&]{ polymer.get_n_segment(-1); }), "block -1 throws");
        }

        std::cout << "=== Rejected polymers leave no contour lengths ===" << std::endl;
        {
            Molecules molecules("continuous", 0.05, {{"A", 1.0}, {"B", 1.0}});
            ok &= check(throws([&]{ molecules.add_polymer(1.0, {{"A", 0.37, 0, 1}, {"C", 0.5, 1, 2}}, {}); }),
                "a polymer with an unknown monomer type is rejected");
            ok &= check(throws([&]{ molecules.add_polymer(1.0, {{"A", 0.37, 0, 1}, {"B", nan, 1, 2}}, {}); }),
                "a polymer with a NaN contour length is rejected");
            molecules.add_polymer(1.0, {{"A", 0.5, 0, 1}, {"B", 0.5, 1, 2}}, {});
            molecules.finalize_contour_length_mapping();
            const ContourLengthMapping& mapping = molecules.get_contour_length_mapping();
            ok &= check(molecules.get_n_polymer_types() == 1, "one polymer type");
            ok &= check(mapping.get_n_unique_lengths() == 1 && mapping.get_n_unique_ds() == 1,
                "only the accepted polymer's contour length and ds are mapped");
        }

        std::cout << "=== NaN values ===" << std::endl;
        {
            Molecules molecules("continuous", 0.1, {{"A", nan}});
            ok &= check(throws([&]{ molecules.add_polymer(1.0, {{"A", 1.0, 0, 1}}, {}); }), "a NaN bond length is rejected");
        }
        ok &= check(throws([&]{ validation::require_positive(nan, "x"); }), "require_positive(NaN) throws");
        ok &= check(throws([&]{ validation::require_non_negative(nan, "x"); }), "require_non_negative(NaN) throws");
        ok &= check(throws([&]{ validation::require_in_range(nan, 0.0, 1.0, "x"); }), "require_in_range(NaN) throws");
        ok &= check(throws([&]{ validation::require_all_positive(std::vector<double>{1.0, nan}, "x"); }),
            "require_all_positive({1, NaN}) throws");
        ok &= check(!throws([&]{
            validation::require_positive(0.5, "x");
            validation::require_positive(1, "x");
            validation::require_non_negative(0, "x");
            validation::require_in_range(1.0, 0.0, 1.0, "x");
            validation::require_all_positive(std::vector<double>{1.0, 2.0}, "x");
        }), "valid values pass");
        ok &= check(throws([&]{ validation::require_positive(0, "x"); })
            && throws([&]{ validation::require_non_negative(-1, "x"); })
            && throws([&]{ validation::require_in_range(1.5, 0.0, 1.0, "x"); }),
            "out-of-range values throw");
    }
    catch (std::exception& exc)
    {
        std::cout << "Unexpected exception: " << exc.what() << std::endl;
        return -1;
    }

    if (!ok)
    {
        std::cout << "TestMoleculesValidation FAILED" << std::endl;
        return -1;
    }
    std::cout << "TestMoleculesValidation passed" << std::endl;
    return 0;
}
