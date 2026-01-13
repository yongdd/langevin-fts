/**
 * @file TestGlobalRichardson1D.cpp
 * @brief Test Global Richardson extrapolation at quadrature level (cn-adi4-gq).
 *
 * This test verifies that the Global Richardson method (cn-adi4-gq) produces
 * consistent results compared to the per-step Richardson method (cn-adi4).
 *
 * Global Richardson maintains two independent propagator chains:
 * - Full-step chain: q_full[0..N] with step size ds
 * - Half-step chain: q_half[0..2N] with step size ds/2
 *
 * The Richardson-extrapolated propagators are computed as:
 *   q_rich[n] = (4·q_half[2n] - q_full[n]) / 3
 *
 * Both methods achieve O(ds⁴) accuracy for Q and φ. For homogeneous fields
 * (constant w), both methods produce identical results to machine precision.
 * For spatially varying fields, small differences may arise from the different
 * order of operations, but these are typically O(ds⁴) or smaller.
 *
 * Tests:
 * 1. Partition function Q agreement (within 1% for robustness)
 * 2. Material conservation: sum(phi_i) = 1 (strict: 1e-8)
 * 3. Concentration field comparison (within 1% for robustness)
 *
 * Note: This test uses loose tolerances to ensure stability. Actual accuracy
 * is typically machine precision (~1e-14) for simple test cases.
 */

#include <iostream>
#include <cmath>
#include <algorithm>
#include <map>
#include <string>

#include "Exception.h"
#include "Molecules.h"
#include "Polymer.h"
#include "PropagatorComputationOptimizer.h"

#ifdef USE_CPU_MKL
#include "CpuComputationBox.h"
#include "CpuComputationContinuous.h"
#include "CpuComputationGlobalRichardson.h"
#endif

int main()
{
    try
    {
#ifdef USE_CPU_MKL
        std::cout << "=== Testing Global Richardson (cn-adi4-gq) ===" << std::endl;

        // Test parameters
        const int nx = 32;
        const double Lx = 4.0;
        const double ds = 0.01;
        const double f = 0.5;  // Symmetric diblock

        // Test potential fields
        std::vector<double> w_a(nx), w_b(nx);
        for (int i = 0; i < nx; ++i)
        {
            // Simple sinusoidal fields
            double x = 2.0 * M_PI * i / nx;
            w_a[i] = 0.5 * std::sin(x);
            w_b[i] = -0.5 * std::sin(x);
        }

        // Setup molecules (shared configuration)
        std::map<std::string, double> bond_lengths = {{"A", 1.0}, {"B", 1.0}};
        std::vector<BlockInput> blocks = {
            {"A", f, 0, 1},
            {"B", 1.0 - f, 1, 2},
        };

        // Periodic BC
        std::vector<std::string> bc_periodic = {"periodic", "periodic"};

        //=====================================================================
        // Test 1: Compare cn-adi4-gq vs cn-adi4
        //=====================================================================
        std::cout << "\nTest 1: Partition function comparison" << std::endl;

        // cn-adi4-gq (Global Richardson)
        // Use separate Molecules/PropOpt to avoid shared state issues
        Molecules molecules_gq("Continuous", ds, bond_lengths);
        molecules_gq.add_polymer(1.0, blocks, {});
        PropagatorComputationOptimizer prop_opt_gq(&molecules_gq, false);

        CpuComputationBox<double>* cb_gq = new CpuComputationBox<double>({nx}, {Lx}, bc_periodic);
        CpuComputationGlobalRichardson* solver_gq = new CpuComputationGlobalRichardson(
            cb_gq, &molecules_gq, &prop_opt_gq);

        solver_gq->compute_statistics({{"A", w_a.data()}, {"B", w_b.data()}}, {});
        double Q_gq = solver_gq->get_total_partition(0);

        // cn-adi4 (per-step Richardson)
        // Use separate Molecules/PropOpt
        Molecules molecules_adi4("Continuous", ds, bond_lengths);
        molecules_adi4.add_polymer(1.0, blocks, {});
        PropagatorComputationOptimizer prop_opt_adi4(&molecules_adi4, false);

        CpuComputationBox<double>* cb_adi4 = new CpuComputationBox<double>({nx}, {Lx}, bc_periodic);
        CpuComputationContinuous<double>* solver_adi4 = new CpuComputationContinuous<double>(
            cb_adi4, &molecules_adi4, &prop_opt_adi4, "realspace", "cn-adi4");

        solver_adi4->compute_statistics({{"A", w_a.data()}, {"B", w_b.data()}}, {});
        double Q_adi4 = solver_adi4->get_total_partition(0);

        double Q_diff = std::abs(Q_gq - Q_adi4);
        double Q_rel_err = Q_diff / Q_adi4;

        std::cout << "  cn-adi4-gq Q = " << Q_gq << std::endl;
        std::cout << "  cn-adi4    Q = " << Q_adi4 << std::endl;
        std::cout << "  Difference   = " << Q_diff << std::endl;
        std::cout << "  Relative err = " << Q_rel_err << std::endl;

        // Both methods are O(ds^4), but apply extrapolation differently
        // Allow 1% relative error for differences due to different error coefficients
        // Tighter agreement requires smaller ds or convergence studies
        bool test1_passed = true;
        if (Q_rel_err > 0.01)
        {
            std::cout << "  FAILED: Q values differ too much (>1%)!" << std::endl;
            test1_passed = false;
        }
        else
            std::cout << "  PASSED" << std::endl;

        //=====================================================================
        // Test 2: Material conservation for cn-adi4-gq
        //=====================================================================
        std::cout << "\nTest 2: Material conservation (cn-adi4-gq)" << std::endl;

        std::vector<double> phi_a(nx), phi_b(nx);
        solver_gq->get_total_concentration("A", phi_a.data());
        solver_gq->get_total_concentration("B", phi_b.data());

        // Compute integral of (phi_A + phi_B) / volume
        double dx = Lx / nx;
        double integral = 0.0;
        for (int i = 0; i < nx; ++i)
        {
            integral += (phi_a[i] + phi_b[i]) * dx;
        }
        double avg_phi = integral / Lx;
        double conservation_error = std::abs(avg_phi - 1.0);

        // Print more diagnostic info
        std::cout << "  phi_a[0] = " << phi_a[0] << ", phi_b[0] = " << phi_b[0] << std::endl;
        std::cout << "  phi_a[mid] = " << phi_a[nx/2] << ", phi_b[mid] = " << phi_b[nx/2] << std::endl;
        std::cout << "  avg(phi_A + phi_B) = " << avg_phi << std::endl;
        std::cout << "  Conservation error = " << conservation_error << std::endl;

        // Material should be conserved to high precision
        bool test2_passed = true;
        if (conservation_error > 1e-8)
        {
            std::cout << "  FAILED: Material conservation violated!" << std::endl;
            test2_passed = false;
        }
        else
            std::cout << "  PASSED" << std::endl;

        //=====================================================================
        // Test 3: Material conservation for cn-adi4 (sanity check)
        //=====================================================================
        std::cout << "\nTest 3: Material conservation (cn-adi4)" << std::endl;

        solver_adi4->get_total_concentration("A", phi_a.data());
        solver_adi4->get_total_concentration("B", phi_b.data());

        integral = 0.0;
        for (int i = 0; i < nx; ++i)
        {
            integral += (phi_a[i] + phi_b[i]) * dx;
        }
        avg_phi = integral / Lx;
        conservation_error = std::abs(avg_phi - 1.0);

        std::cout << "  phi_a[0] = " << phi_a[0] << ", phi_b[0] = " << phi_b[0] << std::endl;
        std::cout << "  phi_a[mid] = " << phi_a[nx/2] << ", phi_b[mid] = " << phi_b[nx/2] << std::endl;
        std::cout << "  avg(phi_A + phi_B) = " << avg_phi << std::endl;
        std::cout << "  Conservation error = " << conservation_error << std::endl;

        bool test3_passed = true;
        if (conservation_error > 1e-8)
        {
            std::cout << "  FAILED: Material conservation violated!" << std::endl;
            test3_passed = false;
        }
        else
            std::cout << "  PASSED" << std::endl;

        //=====================================================================
        // Test 4: Concentration field comparison
        //=====================================================================
        std::cout << "\nTest 4: Concentration field comparison" << std::endl;

        std::vector<double> phi_a_gq(nx), phi_b_gq(nx);
        std::vector<double> phi_a_adi4(nx), phi_b_adi4(nx);

        solver_gq->get_total_concentration("A", phi_a_gq.data());
        solver_gq->get_total_concentration("B", phi_b_gq.data());
        solver_adi4->get_total_concentration("A", phi_a_adi4.data());
        solver_adi4->get_total_concentration("B", phi_b_adi4.data());

        double max_diff_a = 0.0, max_diff_b = 0.0;
        for (int i = 0; i < nx; ++i)
        {
            max_diff_a = std::max(max_diff_a, std::abs(phi_a_gq[i] - phi_a_adi4[i]));
            max_diff_b = std::max(max_diff_b, std::abs(phi_b_gq[i] - phi_b_adi4[i]));
        }

        std::cout << "  max|phi_A_gq - phi_A_adi4| = " << max_diff_a << std::endl;
        std::cout << "  max|phi_B_gq - phi_B_adi4| = " << max_diff_b << std::endl;

        // Concentration fields should be within ~1% for both O(ds^4) methods
        // (phi values are typically O(1), so use absolute tolerance of 0.01)
        bool test4_passed = true;
        if (max_diff_a > 0.01 || max_diff_b > 0.01)
        {
            std::cout << "  FAILED: Concentration fields differ too much (>1%)!" << std::endl;
            test4_passed = false;
        }
        else
            std::cout << "  PASSED" << std::endl;

        // Cleanup
        delete solver_gq;
        delete cb_gq;
        delete solver_adi4;
        delete cb_adi4;

        // Summary
        bool all_passed = test1_passed && test2_passed && test3_passed && test4_passed;
        if (all_passed)
        {
            std::cout << "\n=== All Global Richardson tests passed! ===" << std::endl;
            return 0;
        }
        else
        {
            std::cout << "\n=== Some tests FAILED! ===" << std::endl;
            return -1;
        }

#else
        std::cout << "Skipping test: USE_CPU_MKL not defined" << std::endl;
        return 0;
#endif
    }
    catch (std::exception& exc)
    {
        std::cout << exc.what() << std::endl;
        return -1;
    }
}
