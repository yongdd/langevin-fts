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
 * @file TestDCT2Propagator.cpp
 * @brief Test DCT-2 based pseudo-spectral propagator with reflecting boundaries.
 *
 * This test verifies that the propagator solver produces results matching
 * the analytical DCT-2 diffusion calculation from devel/DFT/diffusion_pseudo_dct.ipynb.
 *
 * The test uses:
 * - 1D domain with L=4.0, N=12 grid points
 * - Reflecting boundary conditions (Neumann BC)
 * - Zero potential field (w=0)
 * - Single homopolymer block
 * - Gaussian initial condition centered at L/2
 *
 * Tests are run on all available platforms (CPU-MKL, CUDA) with both
 * standard and memory-saving modes.
 *
 * The DCT-2 pseudo-spectral method for the diffusion equation:
 *     dq/ds = (b^2/6) nabla^2 q - w*q
 *
 * With w=0 and reflecting BC, the solution is:
 *     q_hat(k, s+ds) = q_hat(k, s) * exp(-b^2/6 * k^2 * ds)
 *
 * where k = n*pi/L for the DCT-2 transform.
 *
 * Expected values from notebook (devel/DFT/diffusion_pseudo_dct.ipynb):
 *
 * Initial condition (Gaussian, sigma=0.5, centered at L/2):
 * q_init = [0.0012038599948282, 0.01110899653824231, 0.06572852861653045, ...]
 *
 * After one step (ds=0.1, b=1.0):
 * q_out = [0.00265976695175955, 0.01771721116200349, 0.0850483724745189, ...]
 */

#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <map>
#include <string>

#include "Exception.h"
#include "ComputationBox.h"
#include "Molecules.h"
#include "Polymer.h"
#include "PropagatorComputationOptimizer.h"
#include "PropagatorComputation.h"
#include "AbstractFactory.h"
#include "PlatformSelector.h"

// Test parameters
const int N = 12;
const double L = 4.0;
const double dx = L / N;
const double ds = 0.1;
const double sigma = 0.5;
const double b = 1.0;

// Expected values from notebook (full precision)
const std::vector<double> expected_q_init = {
    0.0012038599948282,  0.01110899653824231, 0.06572852861653045,
    0.2493522087772961,  0.6065306597126334,  0.9459594689067654,
    0.9459594689067655,  0.6065306597126334,  0.24935220877729644,
    0.06572852861653053, 0.01110899653824231, 0.00120385999482821
};

const std::vector<double> expected_q_out = {
    0.00265976695175955, 0.01771721116200349, 0.0850483724745189,
    0.27580183068075836, 0.6042556834496476,  0.894400857827608,
    0.894400857827608,   0.6042556834496476,  0.2758018306807586,
    0.085048372474519,   0.01771721116200349, 0.00265976695175955
};

/**
 * Run DCT-2 propagator tests for a specific platform and memory mode.
 *
 * @param platform Platform name ("cuda" or "cpu-mkl")
 * @param reduce_memory Enable memory-saving mode
 * @return 0 on success, -1 on failure
 */
int run_tests(const std::string& platform, bool reduce_memory)
{
    std::cout << std::fixed << std::setprecision(17);

    std::cout << "\n============================================================" << std::endl;
    std::cout << "Platform: " << platform << std::endl;
    std::cout << "Memory saving mode: " << (reduce_memory ? "ON" : "OFF") << std::endl;
    std::cout << "============================================================" << std::endl;

    // Create factory
    AbstractFactory<double>* factory = PlatformSelector::create_factory_real(platform, reduce_memory);
    factory->display_info();

    // Staggered grid (half-sample symmetric)
    std::vector<double> x(N);
    for (int i = 0; i < N; ++i)
    {
        x[i] = (i + 0.5) * dx;
    }

    // Gaussian initial condition centered at L/2
    std::vector<double> q_init(N);
    for (int i = 0; i < N; ++i)
    {
        q_init[i] = std::exp(-std::pow(x[i] - L/2, 2) / (2 * sigma * sigma));
    }

    std::cout << "\nParameters:" << std::endl;
    std::cout << "  N = " << N << std::endl;
    std::cout << "  L = " << L << std::endl;
    std::cout << "  dx = " << dx << std::endl;
    std::cout << "  ds = " << ds << std::endl;
    std::cout << "  sigma = " << sigma << std::endl;
    std::cout << "  b = " << b << std::endl;

    std::cout << "\nInitial condition (Gaussian):" << std::endl;
    std::cout << "  q_init = [";
    for (int i = 0; i < N; ++i)
    {
        std::cout << q_init[i];
        if (i < N - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // Verify initial condition matches notebook
    double init_error = 0.0;
    for (int i = 0; i < N; ++i)
    {
        init_error = std::max(init_error, std::abs(q_init[i] - expected_q_init[i]));
    }
    std::cout << "  Init error vs notebook: " << std::scientific << init_error << std::fixed << std::endl;

    // Zero potential field
    std::vector<double> w_zero(N, 0.0);

    //=======================================================================
    // Test 1: Homopolymer with DCT-2 Reflecting BC and Custom Initial Condition
    //=======================================================================
    std::cout << "\n------------------------------------------------------------" << std::endl;
    std::cout << "Test 1: Homopolymer Propagator with Reflecting BC" << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;

    // Create molecules for homopolymer
    std::map<std::string, double> bond_lengths = {{"A", b}};
    Molecules* molecules = factory->create_molecules_information("Continuous", ds, bond_lengths);

    // Single A block with 1 step (ds length)
    // Use custom initial condition "G" at vertex 0
    std::vector<BlockInput> blocks = {
        {"A", ds, 0, 1}   // A block: 1 step
    };
    std::map<int, std::string> chain_end_to_q_init = {{0, "G"}};  // vertex 0 -> "G"
    molecules->add_polymer(1.0, blocks, chain_end_to_q_init);

    std::cout << "Polymer: Homopolymer (A block)" << std::endl;
    std::cout << "  A block length: " << ds << " (1 step)" << std::endl;
    std::cout << "  Initial condition: Gaussian at vertex 0" << std::endl;

    // Create computation box with reflecting BC
    ComputationBox<double>* cb = factory->create_computation_box(
        {N}, {L}, {"reflecting", "reflecting"});

    // Create propagator optimizer and solver
    PropagatorComputationOptimizer* optimizer = new PropagatorComputationOptimizer(molecules, false);
    PropagatorComputation<double>* solver = factory->create_propagator_computation(cb, molecules, optimizer, "rqm4");

    // Compute propagators with zero field and custom initial condition
    solver->compute_propagators({{"A", w_zero.data()}}, {{"G", q_init.data()}});

    // Get propagator after A block (step 1)
    std::vector<double> q_out(N);
    solver->get_chain_propagator(q_out.data(), 0, 0, 1, 1);

    std::cout << "\nPropagator after 1 step (A block):" << std::endl;
    std::cout << "  q_out = [";
    for (int i = 0; i < N; ++i)
    {
        std::cout << q_out[i];
        if (i < N - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    double sum_out = 0.0;
    for (int i = 0; i < N; ++i)
        sum_out += q_out[i];
    std::cout << "  sum(q_out) = " << sum_out << std::endl;

    // Compare with expected values from notebook
    double max_error = 0.0;
    for (int i = 0; i < N; ++i)
    {
        max_error = std::max(max_error, std::abs(q_out[i] - expected_q_out[i]));
    }
    std::cout << "\nMax error vs notebook: " << std::scientific << max_error << std::fixed << std::endl;

    // Verify symmetry
    double sym_error = 0.0;
    for (int i = 0; i < N/2; ++i)
    {
        sym_error = std::max(sym_error, std::abs(q_out[i] - q_out[N-1-i]));
    }
    std::cout << "Symmetry error: " << std::scientific << sym_error << std::fixed << std::endl;

    // Verify mass conservation
    double initial_mass = 0.0;
    for (int i = 0; i < N; ++i)
        initial_mass += q_init[i] * dx;

    double final_mass = 0.0;
    for (int i = 0; i < N; ++i)
        final_mass += q_out[i] * dx;

    double mass_error = std::abs(final_mass - initial_mass) / initial_mass;
    std::cout << "\nMass conservation:" << std::endl;
    std::cout << "  Initial mass: " << initial_mass << std::endl;
    std::cout << "  Final mass:   " << final_mass << std::endl;
    std::cout << "  Relative error: " << std::scientific << mass_error << std::fixed << std::endl;

    bool test1_passed = (mass_error < 1e-6) && (max_error < 1e-6);
    std::cout << "\nTest 1: " << (test1_passed ? "PASSED" : "FAILED") << std::endl;

    // Cleanup first test
    delete solver;
    delete optimizer;
    delete molecules;
    delete cb;

    delete factory;

    //=======================================================================
    // Summary
    //=======================================================================
    std::cout << "\n------------------------------------------------------------" << std::endl;
    std::cout << "SUMMARY for " << platform;
    if (reduce_memory) std::cout << " (memory-saving)";
    std::cout << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;
    std::cout << "  Test (Homopolymer DCT-2): " << (test1_passed ? "PASSED" : "FAILED") << std::endl;

    return test1_passed ? 0 : -1;
}

int main()
{
    try
    {
        std::cout << "============================================================" << std::endl;
        std::cout << "Test: DCT-2 Propagator with Reflecting Boundaries" << std::endl;
        std::cout << "============================================================" << std::endl;
        std::cout << "Testing on all available platforms with standard and" << std::endl;
        std::cout << "memory-saving modes." << std::endl;

        int result = 0;
        int test_count = 0;
        int pass_count = 0;

        // Get available platforms
        std::vector<std::string> platforms = PlatformSelector::avail_platforms();

        std::cout << "\nAvailable platforms: ";
        for (const auto& p : platforms)
            std::cout << p << " ";
        std::cout << std::endl;

        // Run tests on each platform with both memory modes
        for (const std::string& platform : platforms)
        {
            // Test with standard memory mode
            test_count++;
            if (run_tests(platform, false) == 0)
                pass_count++;
            else
                result = -1;

            // Test with memory-saving mode
            test_count++;
            if (run_tests(platform, true) == 0)
                pass_count++;
            else
                result = -1;
        }

        //=======================================================================
        // Final Summary
        //=======================================================================
        std::cout << "\n============================================================" << std::endl;
        std::cout << "FINAL TEST SUMMARY" << std::endl;
        std::cout << "============================================================" << std::endl;
        std::cout << "Tests passed: " << pass_count << "/" << test_count << std::endl;

        if (result == 0)
        {
            std::cout << "All tests PASSED!" << std::endl;
        }
        else
        {
            std::cout << "Some tests FAILED!" << std::endl;
        }
        std::cout << "============================================================" << std::endl;

        return result;
    }
    catch (std::exception& exc)
    {
        std::cout << exc.what() << std::endl;
        return -1;
    }
}
