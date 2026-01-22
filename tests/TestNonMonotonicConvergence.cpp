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
 * @file TestNonMonotonicConvergence.cpp
 * @brief Test that the non-monotonic convergence fix works correctly.
 *
 * This test verifies that when two different Ns values produce the same
 * total segment count due to rounding, they produce identical partition functions.
 *
 * For f = 0.375, the same-segment pairs are:
 * - Ns = 12, 13: both have n_A=5, n_B=8, total=13
 * - Ns = 20, 21: both have n_A=8, n_B=13, total=21
 *
 * The test runs propagator computations for these Ns values and verifies that
 * pairs with the same segment counts produce identical partition functions.
 *
 * Currently tests RQM4 and RK2 methods which correctly use per-block local_ds.
 * Tests all platforms (CPU-MKL, CUDA).
 */

#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <map>
#include <vector>
#include <string>

#include "Exception.h"
#include "ComputationBox.h"
#include "Polymer.h"
#include "Molecules.h"
#include "PropagatorComputationOptimizer.h"
#include "PropagatorComputation.h"
#include "AbstractFactory.h"
#include "PlatformSelector.h"

// Test parameters
const double F_A = 0.375;           // A-block fraction
const double TOLERANCE = 1e-10;     // Tolerance for partition function comparison

// Grid parameters
const int NX = 16;
const int NY = 16;
const int NZ = 16;
const double LX = 4.0;
const double LY = 4.0;
const double LZ = 4.0;

/**
 * @brief Result structure for propagator computation.
 */
struct PropagatorResult {
    double Q;                    // Partition function
    std::vector<double> stress;  // Stress tensor (empty if not computed)
    bool stress_computed;        // Whether stress was computed
};

/**
 * @brief Run propagator computation and return the partition function and stress.
 *
 * @param compute_stress If true, compute stress tensor (skip for CN-ADI methods)
 */
PropagatorResult run_propagator_computation(
    AbstractFactory<double>* factory,
    int Ns,
    double f,
    const std::string& numerical_method,
    const std::string& chain_model,
    double* w_a,
    double* w_b,
    bool compute_stress = true)
{
    double ds = 1.0 / Ns;

    // Create computation box
    std::vector<int> nx = {NX, NY, NZ};
    std::vector<double> lx = {LX, LY, LZ};
    ComputationBox<double>* cb = factory->create_computation_box(nx, lx, {});

    // Create molecules
    Molecules* molecules = factory->create_molecules_information(
        chain_model, ds, {{"A", 1.0}, {"B", 1.0}});

    std::vector<BlockInput> blocks = {
        {"A", f, 0, 1},
        {"B", 1.0 - f, 1, 2},
    };
    molecules->add_polymer(1.0, blocks, {});

    // Create propagator computation optimizer
    PropagatorComputationOptimizer* pco = new PropagatorComputationOptimizer(molecules, false);

    // Create solver
    PropagatorComputation<double>* solver = factory->create_propagator_computation(
        cb, molecules, pco, numerical_method);

    // Compute propagators
    solver->compute_propagators({{"A", w_a}, {"B", w_b}}, {});

    // Get partition function
    PropagatorResult result;
    result.Q = solver->get_total_partition(0);
    result.stress_computed = false;

    // Compute stress if requested (skip for CN-ADI methods)
    if (compute_stress)
    {
        solver->compute_stress();
        result.stress = solver->get_stress();
        result.stress_computed = true;
    }

    // Cleanup
    delete solver;
    delete pco;
    delete molecules;
    delete cb;

    return result;
}

/**
 * @brief Check if a numerical method is a CN-ADI variant.
 */
bool is_cn_adi_method(const std::string& method)
{
    return method.find("cn-adi") != std::string::npos;
}

/**
 * @brief Test same-segment pairs for a given platform and method.
 */
bool test_same_segment_pairs(
    AbstractFactory<double>* factory,
    const std::string& platform_name,
    const std::string& numerical_method,
    const std::string& chain_model)
{
    bool compute_stress = !is_cn_adi_method(numerical_method);

    std::cout << "  Testing " << platform_name << " / " << numerical_method
              << " / " << chain_model;
    if (!compute_stress)
        std::cout << " (stress skipped)";
    std::cout << "..." << std::endl;

    const int M = NX * NY * NZ;

    // Initialize fields with lamellar pattern
    double* w_a = new double[M];
    double* w_b = new double[M];

    for (int i = 0; i < NX; i++)
    {
        for (int j = 0; j < NY; j++)
        {
            for (int k = 0; k < NZ; k++)
            {
                int idx = i * NY * NZ + j * NZ + k;
                double phase = 2.0 * M_PI * k / NZ;
                w_a[idx] =  5.0 * std::cos(phase);
                w_b[idx] = -5.0 * std::cos(phase);
            }
        }
    }

    // Same-segment pairs: (Ns1, Ns2) where both produce same total segments
    std::vector<std::pair<int, int>> pairs = {
        {12, 13},  // both have total=13
        {20, 21},  // both have total=21
    };

    bool all_passed = true;

    for (const auto& pair : pairs)
    {
        int Ns1 = pair.first;
        int Ns2 = pair.second;

        PropagatorResult r1 = run_propagator_computation(factory, Ns1, F_A, numerical_method, chain_model, w_a, w_b, compute_stress);
        PropagatorResult r2 = run_propagator_computation(factory, Ns2, F_A, numerical_method, chain_model, w_a, w_b, compute_stress);

        // Check partition function
        double Q_diff = std::abs(r1.Q - r2.Q);
        double Q_rel_diff = Q_diff / std::abs(r1.Q);

        std::cout << "    Ns=" << Ns1 << " vs Ns=" << Ns2
                  << ": Q1=" << std::scientific << std::setprecision(10) << r1.Q
                  << ", Q2=" << r2.Q
                  << ", rel_diff=" << Q_rel_diff << std::endl;

        if (Q_rel_diff > TOLERANCE)
        {
            std::cout << "    FAILED: Q relative difference " << Q_rel_diff
                      << " > " << TOLERANCE << std::endl;
            all_passed = false;
        }

        // Check stress (if computed)
        if (r1.stress_computed && r2.stress_computed)
        {
            // Stress tensor has 6 components: xx, yy, zz, xy, xz, yz
            for (size_t i = 0; i < r1.stress.size() && i < r2.stress.size(); i++)
            {
                double s_diff = std::abs(r1.stress[i] - r2.stress[i]);
                double s_max = std::max(std::abs(r1.stress[i]), std::abs(r2.stress[i]));
                double s_rel_diff = s_diff / (s_max + 1e-15);

                if (s_rel_diff > TOLERANCE && s_diff > 1e-12)
                {
                    std::cout << "    FAILED: stress[" << i << "] relative difference " << s_rel_diff
                              << " > " << TOLERANCE
                              << " (s1=" << r1.stress[i] << ", s2=" << r2.stress[i] << ")" << std::endl;
                    all_passed = false;
                }
            }
            std::cout << "      stress: s1[0]=" << r1.stress[0] << ", s2[0]=" << r2.stress[0] << std::endl;
        }
    }

    delete[] w_a;
    delete[] w_b;

    return all_passed;
}

int main()
{
    try
    {
        bool all_tests_passed = true;

        std::cout << "========================================" << std::endl;
        std::cout << "Test: Non-Monotonic Convergence Fix" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Parameters: f=" << F_A << std::endl;
        std::cout << "Grid: " << NX << "x" << NY << "x" << NZ << std::endl;
        std::cout << "Box: " << LX << "x" << LY << "x" << LZ << std::endl;
        std::cout << std::endl;

        // Test configurations: (numerical_method, chain_model)
        // All methods now use per-block local_ds for the Boltzmann factor
        // Note: Discrete chains are not tested here because f=0.375 is not an
        // integer multiple of ds=1/Ns for the Ns values tested (12, 13, 20, 21).
        // Discrete chains require block lengths to be integer multiples of ds.
        std::vector<std::pair<std::string, std::string>> configs = {
            {"rqm4", "Continuous"},
            {"rk2", "Continuous"},
            {"etdrk4", "Continuous"},
        };

        // Get available platforms
        std::vector<std::string> platforms = PlatformSelector::avail_platforms();

        // Test both reduce_memory=false and reduce_memory=true
        std::vector<bool> reduce_memory_options = {false, true};

        for (bool reduce_memory : reduce_memory_options)
        {
            std::cout << "*** reduce_memory=" << (reduce_memory ? "true" : "false") << " ***" << std::endl;

            for (const std::string& platform : platforms)
            {
                std::cout << "=== Platform: " << platform << " ===" << std::endl;

                AbstractFactory<double>* factory = PlatformSelector::create_factory_real(platform, reduce_memory);

                for (const auto& config : configs)
                {
                    const std::string& method = config.first;
                    const std::string& chain = config.second;

                    bool passed = test_same_segment_pairs(factory, platform, method, chain);

                    if (!passed)
                        all_tests_passed = false;
                }

                delete factory;
                std::cout << std::endl;
            }
        }

        std::cout << "========================================" << std::endl;
        if (all_tests_passed)
        {
            std::cout << "All tests PASSED" << std::endl;
            return 0;
        }
        else
        {
            std::cout << "Some tests FAILED" << std::endl;
            return -1;
        }
    }
    catch (std::exception& exc)
    {
        std::cout << exc.what() << std::endl;
        return -1;
    }
}
