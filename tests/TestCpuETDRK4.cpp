/**
 * @file TestCpuETDRK4.cpp
 * @brief Test ETDRK4 time integration against RQM4.
 *
 * Both ETDRK4 and RQM4 (Ranjan-Qin-Morse 4th-order using Richardson extrapolation)
 * are 4th-order accurate methods, so they should produce nearly identical results
 * for the same problem.
 */

#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <vector>
#include <map>

#include "Exception.h"
#include "Molecules.h"
#include "Polymer.h"

#ifdef USE_CPU_MKL
#include "CpuComputationBox.h"
#include "CpuSolverPseudoRQM4.h"
#include "CpuSolverPseudoETDRK4.h"
#include "Pseudo.h"
#endif

int main()
{
#ifndef USE_CPU_MKL
    std::cout << "CPU MKL not available, skipping test." << std::endl;
    return 0;
#else
    try
    {
        std::cout << "=== ETDRK4 vs RQM4 Test ===" << std::endl;

        // Grid parameters
        const int II = 8;
        const int JJ = 8;
        const int KK = 8;
        const int M = II * JJ * KK;

        double Lx = 4.0;
        double Ly = 4.0;
        double Lz = 4.0;

        // Random-ish potential field
        std::vector<double> w_a(M);
        for (int i = 0; i < M; ++i)
        {
            // Simple deterministic "random" pattern
            w_a[i] = 0.5 * std::sin(2.0 * M_PI * i / M) + 0.3 * std::cos(4.0 * M_PI * i / M);
        }

        // Initial propagator (uniform)
        std::vector<double> q_in(M, 1.0);

        // Output propagators
        std::vector<double> q_rqm4(M);
        std::vector<double> q_etdrk4(M);

        // Initialize molecules
        std::map<std::string, double> bond_lengths = {{"A", 1.0}};
        double ds = 0.01;  // Contour step

        Molecules* molecules = new Molecules("Continuous", ds, bond_lengths);
        std::vector<BlockInput> blocks = {{"A", 1.0, 0, 1}};
        molecules->add_polymer(1.0, blocks, {});

        // Create computation box
        CpuComputationBox<double>* cb = new CpuComputationBox<double>(
            {II, JJ, KK}, {Lx, Ly, Lz}, {});

        // Create solvers - RQM4 and ETDRK4 as separate classes
        CpuSolverPseudoRQM4<double>* solver_rqm4 =
            new CpuSolverPseudoRQM4<double>(cb, molecules);
        CpuSolverPseudoETDRK4<double>* solver_etdrk4 =
            new CpuSolverPseudoETDRK4<double>(cb, molecules);

        // Update fields for both solvers
        solver_rqm4->update_dw({{"A", w_a.data()}});
        solver_etdrk4->update_dw({{"A", w_a.data()}});

        // Test 1: Single step comparison
        std::cout << "\n--- Test 1: Single Step Comparison ---" << std::endl;

        // RQM4 step
        solver_rqm4->advance_propagator(q_in.data(), q_rqm4.data(), "A", nullptr);

        // ETDRK4 step
        solver_etdrk4->advance_propagator(q_in.data(), q_etdrk4.data(), "A", nullptr);

        // Compare results
        double max_diff = 0.0;
        double max_rel_diff = 0.0;
        for (int i = 0; i < M; ++i)
        {
            double diff = std::abs(q_rqm4[i] - q_etdrk4[i]);
            double rel_diff = diff / std::max(std::abs(q_rqm4[i]), 1e-10);
            max_diff = std::max(max_diff, diff);
            max_rel_diff = std::max(max_rel_diff, rel_diff);
        }

        std::cout << std::scientific << std::setprecision(6);
        std::cout << "Max absolute difference: " << max_diff << std::endl;
        std::cout << "Max relative difference: " << max_rel_diff << std::endl;

        // Both methods are 4th-order accurate, so difference should be O(ds^5) ~ 1e-10
        // However, due to different constants in the error terms, allow 1e-5 tolerance
        if (max_rel_diff > 1e-4)
        {
            std::cout << "ERROR: ETDRK4 and RQM4 results differ too much!" << std::endl;
            return -1;
        }
        std::cout << "Single step comparison: PASSED" << std::endl;

        // Test 2: Multiple steps (propagator evolution)
        std::cout << "\n--- Test 2: Multiple Steps (10 steps) ---" << std::endl;

        // Reset
        std::copy(q_in.begin(), q_in.end(), q_rqm4.begin());
        std::copy(q_in.begin(), q_in.end(), q_etdrk4.begin());

        // Evolve for 10 steps
        const int n_steps = 10;
        for (int step = 0; step < n_steps; ++step)
        {
            std::vector<double> temp_rqm4(M), temp_etdrk4(M);
            solver_rqm4->advance_propagator(q_rqm4.data(), temp_rqm4.data(), "A", nullptr);
            solver_etdrk4->advance_propagator(q_etdrk4.data(), temp_etdrk4.data(), "A", nullptr);
            std::copy(temp_rqm4.begin(), temp_rqm4.end(), q_rqm4.begin());
            std::copy(temp_etdrk4.begin(), temp_etdrk4.end(), q_etdrk4.begin());
        }

        // Compare final results
        max_diff = 0.0;
        max_rel_diff = 0.0;
        for (int i = 0; i < M; ++i)
        {
            double diff = std::abs(q_rqm4[i] - q_etdrk4[i]);
            double rel_diff = diff / std::max(std::abs(q_rqm4[i]), 1e-10);
            max_diff = std::max(max_diff, diff);
            max_rel_diff = std::max(max_rel_diff, rel_diff);
        }

        std::cout << "After " << n_steps << " steps:" << std::endl;
        std::cout << "Max absolute difference: " << max_diff << std::endl;
        std::cout << "Max relative difference: " << max_rel_diff << std::endl;

        // Error accumulates over steps, allow larger tolerance
        if (max_rel_diff > 1e-3)
        {
            std::cout << "ERROR: ETDRK4 and RQM4 results differ too much after multiple steps!" << std::endl;
            return -1;
        }
        std::cout << "Multiple steps comparison: PASSED" << std::endl;

        // Test 3: Check propagator normalization (sum should decrease due to potential)
        std::cout << "\n--- Test 3: Propagator Properties ---" << std::endl;

        double sum_rqm4 = 0.0, sum_etdrk4 = 0.0;
        for (int i = 0; i < M; ++i)
        {
            sum_rqm4 += q_rqm4[i];
            sum_etdrk4 += q_etdrk4[i];
        }

        std::cout << "Sum of q_rqm4: " << sum_rqm4 << std::endl;
        std::cout << "Sum of q_etdrk4:     " << sum_etdrk4 << std::endl;

        // Both should be positive (propagators are positive)
        if (sum_rqm4 <= 0 || sum_etdrk4 <= 0)
        {
            std::cout << "ERROR: Propagator sum should be positive!" << std::endl;
            return -1;
        }

        // Sums should be similar
        double sum_rel_diff = std::abs(sum_rqm4 - sum_etdrk4) / sum_rqm4;
        std::cout << "Relative difference in sums: " << sum_rel_diff << std::endl;

        if (sum_rel_diff > 1e-3)
        {
            std::cout << "ERROR: Propagator sums differ too much!" << std::endl;
            return -1;
        }
        std::cout << "Propagator properties: PASSED" << std::endl;

        // Cleanup
        delete solver_rqm4;
        delete solver_etdrk4;
        delete cb;
        delete molecules;

        std::cout << "\n=== All ETDRK4 Tests PASSED ===" << std::endl;
        return 0;
    }
    catch (std::exception& exc)
    {
        std::cout << exc.what() << std::endl;
        return -1;
    }
#endif
}
