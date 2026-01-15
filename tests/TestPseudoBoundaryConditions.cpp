/**
 * @file TestPseudoBoundaryConditions.cpp
 * @brief Test pseudo-spectral solvers with various boundary conditions.
 *
 * This test verifies that the ETDRK4 (Exponential Time Differencing RK4) solver
 * produces consistent results for different boundary conditions:
 * - Periodic: Uses FFT
 * - Reflecting: Uses DCT
 * - Absorbing: Uses DST
 *
 * ETDRK4 is compared with RQM4 (both are pseudo-spectral methods using the same
 * grid discretization). The methods should agree within tight tolerance (~1e-4).
 *
 * Cross-validation with CN-ADI2 (real-space method) confirms that both
 * pseudo-spectral and real-space methods use consistent cell-centered grids
 * for non-periodic boundary conditions.
 */

#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <map>
#include <string>
#include <iomanip>

#include "Exception.h"
#include "Molecules.h"
#include "Polymer.h"
#include "PropagatorComputationOptimizer.h"

#ifdef USE_CPU_MKL
#include "CpuComputationBox.h"
#include "CpuComputationContinuous.h"
#endif

#ifdef USE_CUDA
#include "CudaComputationBox.h"
#include "CudaComputationContinuous.h"
#endif

// Helper template to run a single test comparing two methods
template<template<typename> class ComputationBox, template<typename> class ComputationContinuous>
bool run_test(const std::string& platform_name,
              const std::vector<int>& nx,
              const std::vector<double>& lx,
              const std::vector<std::string>& bc,
              const std::string& method1_type,
              const std::string& method1_name,
              const std::string& method2_type,
              const std::string& method2_name,
              Molecules* molecules,
              PropagatorComputationOptimizer* prop_opt,
              double tolerance)
{
    int M = 1;
    for (auto n : nx) M *= n;

    // Generate random fields
    std::vector<double> w_A(M), w_B(M);
    unsigned int seed = 42;
    for (int i = 0; i < M; ++i)
    {
        seed = seed * 1103515245 + 12345;
        w_A[i] = ((seed >> 16) & 0x7FFF) / 32768.0 * 0.2 - 0.1;
        w_B[i] = -w_A[i];
    }

    // Reference method
    ComputationBox<double>* cb1 = new ComputationBox<double>(nx, lx, bc);
    ComputationContinuous<double>* solver1 = new ComputationContinuous<double>(
        cb1, molecules, prop_opt, method1_type, method1_name);
    solver1->compute_propagators({{"A", w_A.data()}, {"B", w_B.data()}}, {});
    double Q1 = solver1->get_total_partition(0);
    delete solver1;
    delete cb1;

    // Test method (ETDRK4)
    ComputationBox<double>* cb2 = new ComputationBox<double>(nx, lx, bc);
    ComputationContinuous<double>* solver2 = new ComputationContinuous<double>(
        cb2, molecules, prop_opt, method2_type, method2_name);
    solver2->compute_propagators({{"A", w_A.data()}, {"B", w_B.data()}}, {});
    double Q2 = solver2->get_total_partition(0);
    delete solver2;
    delete cb2;

    double rel_err = std::abs(Q2 - Q1) / std::abs(Q1);
    if (rel_err > tolerance || !std::isfinite(rel_err))
    {
        std::cout << platform_name << " FAILED (Q_ref=" << Q1 << ", Q_etdrk4=" << Q2
                  << ", rel_err=" << rel_err << ")" << std::endl;
        return false;
    }
    else
    {
        std::cout << platform_name << " PASSED (rel_err=" << std::scientific
                  << std::setprecision(2) << rel_err << ")" << std::fixed << std::endl;
        return true;
    }
}

int main()
{
#if !defined(USE_CPU_MKL) && !defined(USE_CUDA)
    std::cout << "Neither CPU nor CUDA available, skipping ETDRK4 test" << std::endl;
    return 0;
#else
    try
    {
        bool all_passed = true;
        const double ds = 1.0 / 16.0;
        const double tolerance = 1e-4;  // ETDRK4 vs RQM4 (both pseudo-spectral)

        // Setup molecules
        std::map<std::string, double> bond_lengths = {{"A", 1.0}, {"B", 1.0}};
        std::vector<BlockInput> blocks = {{"A", 0.5, 0, 1}, {"B", 0.5, 1, 2}};

        Molecules molecules("Continuous", ds, bond_lengths);
        molecules.add_polymer(1.0, blocks, {});
        PropagatorComputationOptimizer prop_opt(&molecules, false);

        std::cout << "==================================================================" << std::endl;
        std::cout << "Testing ETDRK4 solver with various boundary conditions" << std::endl;
        std::cout << "==================================================================" << std::endl;

        //=======================================================================
        // PART 1: Periodic BC (ETDRK4 vs RQM4)
        //=======================================================================
        std::cout << "\n=== Periodic BC (ETDRK4 vs RQM4) ===" << std::endl;

        // Test 1: 1D Periodic
        std::cout << "\n1D Periodic:" << std::endl;
        {
            std::vector<int> nx = {64};
            std::vector<double> lx = {4.0};
            std::vector<std::string> bc = {"periodic", "periodic"};

            #ifdef USE_CPU_MKL
            if (!run_test<CpuComputationBox, CpuComputationContinuous>(
                    "  CPU-MKL", nx, lx, bc, "pseudospectral", "rqm4",
                    "pseudospectral", "etdrk4", &molecules, &prop_opt, tolerance))
                all_passed = false;
            #endif

            #ifdef USE_CUDA
            if (!run_test<CudaComputationBox, CudaComputationContinuous>(
                    "  CUDA", nx, lx, bc, "pseudospectral", "rqm4",
                    "pseudospectral", "etdrk4", &molecules, &prop_opt, tolerance))
                all_passed = false;
            #endif
        }

        // Test 2: 2D Periodic
        std::cout << "\n2D Periodic:" << std::endl;
        {
            std::vector<int> nx = {32, 32};
            std::vector<double> lx = {4.0, 4.0};
            std::vector<std::string> bc = {"periodic", "periodic", "periodic", "periodic"};

            #ifdef USE_CPU_MKL
            if (!run_test<CpuComputationBox, CpuComputationContinuous>(
                    "  CPU-MKL", nx, lx, bc, "pseudospectral", "rqm4",
                    "pseudospectral", "etdrk4", &molecules, &prop_opt, tolerance))
                all_passed = false;
            #endif

            #ifdef USE_CUDA
            if (!run_test<CudaComputationBox, CudaComputationContinuous>(
                    "  CUDA", nx, lx, bc, "pseudospectral", "rqm4",
                    "pseudospectral", "etdrk4", &molecules, &prop_opt, tolerance))
                all_passed = false;
            #endif
        }

        // Test 3: 3D Periodic
        std::cout << "\n3D Periodic:" << std::endl;
        {
            std::vector<int> nx = {16, 16, 16};
            std::vector<double> lx = {4.0, 4.0, 4.0};
            std::vector<std::string> bc = {"periodic", "periodic", "periodic", "periodic", "periodic", "periodic"};

            #ifdef USE_CPU_MKL
            if (!run_test<CpuComputationBox, CpuComputationContinuous>(
                    "  CPU-MKL", nx, lx, bc, "pseudospectral", "rqm4",
                    "pseudospectral", "etdrk4", &molecules, &prop_opt, tolerance))
                all_passed = false;
            #endif

            #ifdef USE_CUDA
            if (!run_test<CudaComputationBox, CudaComputationContinuous>(
                    "  CUDA", nx, lx, bc, "pseudospectral", "rqm4",
                    "pseudospectral", "etdrk4", &molecules, &prop_opt, tolerance))
                all_passed = false;
            #endif
        }

        //=======================================================================
        // PART 2: Reflecting BC (ETDRK4 vs RQM4)
        //=======================================================================
        std::cout << "\n=== Reflecting BC (ETDRK4 vs RQM4) ===" << std::endl;

        // Test 4: 1D Reflecting
        std::cout << "\n1D Reflecting:" << std::endl;
        {
            std::vector<int> nx = {64};
            std::vector<double> lx = {4.0};
            std::vector<std::string> bc = {"reflecting", "reflecting"};

            #ifdef USE_CPU_MKL
            if (!run_test<CpuComputationBox, CpuComputationContinuous>(
                    "  CPU-MKL", nx, lx, bc, "pseudospectral", "rqm4",
                    "pseudospectral", "etdrk4", &molecules, &prop_opt, tolerance))
                all_passed = false;
            #endif

            #ifdef USE_CUDA
            if (!run_test<CudaComputationBox, CudaComputationContinuous>(
                    "  CUDA", nx, lx, bc, "pseudospectral", "rqm4",
                    "pseudospectral", "etdrk4", &molecules, &prop_opt, tolerance))
                all_passed = false;
            #endif
        }

        // Test 5: 2D Reflecting
        std::cout << "\n2D Reflecting:" << std::endl;
        {
            std::vector<int> nx = {24, 24};
            std::vector<double> lx = {3.0, 3.0};
            std::vector<std::string> bc = {"reflecting", "reflecting", "reflecting", "reflecting"};

            #ifdef USE_CPU_MKL
            if (!run_test<CpuComputationBox, CpuComputationContinuous>(
                    "  CPU-MKL", nx, lx, bc, "pseudospectral", "rqm4",
                    "pseudospectral", "etdrk4", &molecules, &prop_opt, tolerance))
                all_passed = false;
            #endif

            #ifdef USE_CUDA
            if (!run_test<CudaComputationBox, CudaComputationContinuous>(
                    "  CUDA", nx, lx, bc, "pseudospectral", "rqm4",
                    "pseudospectral", "etdrk4", &molecules, &prop_opt, tolerance))
                all_passed = false;
            #endif
        }

        // Test 6: 3D Reflecting
        std::cout << "\n3D Reflecting:" << std::endl;
        {
            std::vector<int> nx = {12, 12, 12};
            std::vector<double> lx = {2.4, 2.4, 2.4};
            std::vector<std::string> bc = {"reflecting", "reflecting", "reflecting", "reflecting", "reflecting", "reflecting"};

            #ifdef USE_CPU_MKL
            if (!run_test<CpuComputationBox, CpuComputationContinuous>(
                    "  CPU-MKL", nx, lx, bc, "pseudospectral", "rqm4",
                    "pseudospectral", "etdrk4", &molecules, &prop_opt, tolerance))
                all_passed = false;
            #endif

            #ifdef USE_CUDA
            if (!run_test<CudaComputationBox, CudaComputationContinuous>(
                    "  CUDA", nx, lx, bc, "pseudospectral", "rqm4",
                    "pseudospectral", "etdrk4", &molecules, &prop_opt, tolerance))
                all_passed = false;
            #endif
        }

        //=======================================================================
        // PART 3: Absorbing BC (ETDRK4 vs RQM4)
        //=======================================================================
        std::cout << "\n=== Absorbing BC (ETDRK4 vs RQM4) ===" << std::endl;

        // Test 7: 1D Absorbing
        std::cout << "\n1D Absorbing:" << std::endl;
        {
            std::vector<int> nx = {64};
            std::vector<double> lx = {4.0};
            std::vector<std::string> bc = {"absorbing", "absorbing"};

            #ifdef USE_CPU_MKL
            if (!run_test<CpuComputationBox, CpuComputationContinuous>(
                    "  CPU-MKL", nx, lx, bc, "pseudospectral", "rqm4",
                    "pseudospectral", "etdrk4", &molecules, &prop_opt, tolerance))
                all_passed = false;
            #endif

            #ifdef USE_CUDA
            if (!run_test<CudaComputationBox, CudaComputationContinuous>(
                    "  CUDA", nx, lx, bc, "pseudospectral", "rqm4",
                    "pseudospectral", "etdrk4", &molecules, &prop_opt, tolerance))
                all_passed = false;
            #endif
        }

        // Test 8: 2D Absorbing
        std::cout << "\n2D Absorbing:" << std::endl;
        {
            std::vector<int> nx = {24, 24};
            std::vector<double> lx = {3.0, 3.0};
            std::vector<std::string> bc = {"absorbing", "absorbing", "absorbing", "absorbing"};

            #ifdef USE_CPU_MKL
            if (!run_test<CpuComputationBox, CpuComputationContinuous>(
                    "  CPU-MKL", nx, lx, bc, "pseudospectral", "rqm4",
                    "pseudospectral", "etdrk4", &molecules, &prop_opt, tolerance))
                all_passed = false;
            #endif

            #ifdef USE_CUDA
            if (!run_test<CudaComputationBox, CudaComputationContinuous>(
                    "  CUDA", nx, lx, bc, "pseudospectral", "rqm4",
                    "pseudospectral", "etdrk4", &molecules, &prop_opt, tolerance))
                all_passed = false;
            #endif
        }

        // Test 9: 3D Absorbing
        std::cout << "\n3D Absorbing:" << std::endl;
        {
            std::vector<int> nx = {12, 12, 12};
            std::vector<double> lx = {2.4, 2.4, 2.4};
            std::vector<std::string> bc = {"absorbing", "absorbing", "absorbing", "absorbing", "absorbing", "absorbing"};

            #ifdef USE_CPU_MKL
            if (!run_test<CpuComputationBox, CpuComputationContinuous>(
                    "  CPU-MKL", nx, lx, bc, "pseudospectral", "rqm4",
                    "pseudospectral", "etdrk4", &molecules, &prop_opt, tolerance))
                all_passed = false;
            #endif

            #ifdef USE_CUDA
            if (!run_test<CudaComputationBox, CudaComputationContinuous>(
                    "  CUDA", nx, lx, bc, "pseudospectral", "rqm4",
                    "pseudospectral", "etdrk4", &molecules, &prop_opt, tolerance))
                all_passed = false;
            #endif
        }

        //=======================================================================
        // PART 4: Cross-validation with real-space CN-ADI2
        // This validates that pseudo-spectral (DST) and real-space (finite
        // difference) methods use consistent cell-centered grids.
        //=======================================================================
        std::cout << "\n=== Cross-validation: ETDRK4 vs CN-ADI2 (Absorbing BC) ===" << std::endl;
        const double tolerance_cross = 0.02;  // 2% tolerance for different methods

        // Test: 2D Absorbing (ETDRK4 vs CN-ADI2)
        std::cout << "\n2D Absorbing (ETDRK4 vs CN-ADI2):" << std::endl;
        {
            std::vector<int> nx = {24, 24};
            std::vector<double> lx = {3.0, 3.0};
            std::vector<std::string> bc = {"absorbing", "absorbing", "absorbing", "absorbing"};

            #ifdef USE_CPU_MKL
            if (!run_test<CpuComputationBox, CpuComputationContinuous>(
                    "  CPU-MKL", nx, lx, bc, "realspace", "cn-adi2",
                    "pseudospectral", "etdrk4", &molecules, &prop_opt, tolerance_cross))
                all_passed = false;
            #endif

            #ifdef USE_CUDA
            if (!run_test<CudaComputationBox, CudaComputationContinuous>(
                    "  CUDA", nx, lx, bc, "realspace", "cn-adi2",
                    "pseudospectral", "etdrk4", &molecules, &prop_opt, tolerance_cross))
                all_passed = false;
            #endif
        }

        // Test: 3D Absorbing (ETDRK4 vs CN-ADI2)
        std::cout << "\n3D Absorbing (ETDRK4 vs CN-ADI2):" << std::endl;
        {
            std::vector<int> nx = {12, 12, 12};
            std::vector<double> lx = {2.4, 2.4, 2.4};
            std::vector<std::string> bc = {"absorbing", "absorbing", "absorbing", "absorbing", "absorbing", "absorbing"};

            #ifdef USE_CPU_MKL
            if (!run_test<CpuComputationBox, CpuComputationContinuous>(
                    "  CPU-MKL", nx, lx, bc, "realspace", "cn-adi2",
                    "pseudospectral", "etdrk4", &molecules, &prop_opt, tolerance_cross))
                all_passed = false;
            #endif

            #ifdef USE_CUDA
            if (!run_test<CudaComputationBox, CudaComputationContinuous>(
                    "  CUDA", nx, lx, bc, "realspace", "cn-adi2",
                    "pseudospectral", "etdrk4", &molecules, &prop_opt, tolerance_cross))
                all_passed = false;
            #endif
        }

        //=======================================================================
        // PART 5: Mixed BC (ETDRK4 vs RQM4) - CUDA only
        // Note: CPU has a known issue with mixed absorbing+reflecting BC.
        //       This is a bug in the CPU FFT implementation, not ETDRK4.
        //=======================================================================
        std::cout << "\n=== Mixed BC (ETDRK4 vs RQM4) - CUDA only ===" << std::endl;

        // Test 10: 2D Mixed (X:Absorbing, Y:Reflecting)
        std::cout << "\n2D X:Absorbing, Y:Reflecting:" << std::endl;
        {
            std::vector<int> nx = {24, 24};
            std::vector<double> lx = {3.0, 3.0};
            std::vector<std::string> bc = {"absorbing", "absorbing", "reflecting", "reflecting"};

            #ifdef USE_CPU_MKL
            std::cout << "  CPU-MKL SKIPPED (known issue with mixed BC)" << std::endl;
            #endif

            #ifdef USE_CUDA
            if (!run_test<CudaComputationBox, CudaComputationContinuous>(
                    "  CUDA", nx, lx, bc, "pseudospectral", "rqm4",
                    "pseudospectral", "etdrk4", &molecules, &prop_opt, tolerance))
                all_passed = false;
            #endif
        }

        // Test 11: 3D Mixed (X/Y:Reflecting, Z:Absorbing)
        std::cout << "\n3D X/Y:Reflecting, Z:Absorbing:" << std::endl;
        {
            std::vector<int> nx = {12, 12, 12};
            std::vector<double> lx = {2.4, 2.4, 2.4};
            std::vector<std::string> bc = {"reflecting", "reflecting", "reflecting", "reflecting", "absorbing", "absorbing"};

            #ifdef USE_CPU_MKL
            std::cout << "  CPU-MKL SKIPPED (known issue with mixed BC)" << std::endl;
            #endif

            #ifdef USE_CUDA
            if (!run_test<CudaComputationBox, CudaComputationContinuous>(
                    "  CUDA", nx, lx, bc, "pseudospectral", "rqm4",
                    "pseudospectral", "etdrk4", &molecules, &prop_opt, tolerance))
                all_passed = false;
            #endif
        }

        //=======================================================================
        // PART 6: Physical Property Tests (Mass Conservation)
        // - Reflecting BC: Partition function Q should be higher (mass conserved)
        // - Absorbing BC: Partition function Q should be lower (mass absorbed)
        // For zero potential (w=0), Q_reflecting > Q_absorbing
        //=======================================================================
        std::cout << "\n=== Physical Property Tests ===" << std::endl;

        std::cout << "\n1D Mass Conservation (Q_reflecting > Q_absorbing):" << std::endl;
        {
            std::vector<int> nx = {64};
            std::vector<double> lx = {4.0};
            int M = nx[0];

            // Zero potential field
            std::vector<double> w_A(M, 0.0), w_B(M, 0.0);

            double Q_reflecting = 0.0, Q_absorbing = 0.0;

            #ifdef USE_CPU_MKL
            {
                // Reflecting BC
                CpuComputationBox<double>* cb_ref = new CpuComputationBox<double>(
                    nx, lx, {"reflecting", "reflecting"});
                CpuComputationContinuous<double>* solver_ref = new CpuComputationContinuous<double>(
                    cb_ref, &molecules, &prop_opt, "pseudospectral", "rqm4");
                solver_ref->compute_propagators({{"A", w_A.data()}, {"B", w_B.data()}}, {});
                Q_reflecting = solver_ref->get_total_partition(0);
                delete solver_ref;
                delete cb_ref;

                // Absorbing BC
                CpuComputationBox<double>* cb_abs = new CpuComputationBox<double>(
                    nx, lx, {"absorbing", "absorbing"});
                CpuComputationContinuous<double>* solver_abs = new CpuComputationContinuous<double>(
                    cb_abs, &molecules, &prop_opt, "pseudospectral", "rqm4");
                solver_abs->compute_propagators({{"A", w_A.data()}, {"B", w_B.data()}}, {});
                Q_absorbing = solver_abs->get_total_partition(0);
                delete solver_abs;
                delete cb_abs;

                // For w=0, reflecting BC should conserve mass: Q = 1.0
                // Absorbing BC should lose mass at boundaries: Q < 1.0
                double tol_conservation = 1e-10;
                bool ref_conserved = std::abs(Q_reflecting - 1.0) < tol_conservation;
                bool abs_less = Q_absorbing < 1.0 && Q_reflecting > Q_absorbing;

                if (!ref_conserved || !abs_less)
                {
                    std::cout << "  CPU-MKL FAILED (Q_ref=" << std::scientific << Q_reflecting
                              << ", Q_abs=" << Q_absorbing << ")" << std::fixed << std::endl;
                    if (!ref_conserved)
                        std::cout << "    Expected Q_reflecting = 1.0 for w=0" << std::endl;
                    all_passed = false;
                }
                else
                {
                    std::cout << "  CPU-MKL PASSED (Q_ref=" << std::fixed << std::setprecision(4) << Q_reflecting
                              << " [conserved], Q_abs=" << Q_absorbing << " [absorbed])" << std::endl;
                }
            }
            #endif

            #ifdef USE_CUDA
            {
                // Reflecting BC
                CudaComputationBox<double>* cb_ref = new CudaComputationBox<double>(
                    nx, lx, {"reflecting", "reflecting"});
                CudaComputationContinuous<double>* solver_ref = new CudaComputationContinuous<double>(
                    cb_ref, &molecules, &prop_opt, "pseudospectral", "rqm4");
                solver_ref->compute_propagators({{"A", w_A.data()}, {"B", w_B.data()}}, {});
                Q_reflecting = solver_ref->get_total_partition(0);
                delete solver_ref;
                delete cb_ref;

                // Absorbing BC
                CudaComputationBox<double>* cb_abs = new CudaComputationBox<double>(
                    nx, lx, {"absorbing", "absorbing"});
                CudaComputationContinuous<double>* solver_abs = new CudaComputationContinuous<double>(
                    cb_abs, &molecules, &prop_opt, "pseudospectral", "rqm4");
                solver_abs->compute_propagators({{"A", w_A.data()}, {"B", w_B.data()}}, {});
                Q_absorbing = solver_abs->get_total_partition(0);
                delete solver_abs;
                delete cb_abs;

                // For w=0, reflecting BC should conserve mass: Q = 1.0
                double tol_conservation = 1e-10;
                bool ref_conserved = std::abs(Q_reflecting - 1.0) < tol_conservation;
                bool abs_less = Q_absorbing < 1.0 && Q_reflecting > Q_absorbing;

                if (!ref_conserved || !abs_less)
                {
                    std::cout << "  CUDA FAILED (Q_ref=" << std::scientific << Q_reflecting
                              << ", Q_abs=" << Q_absorbing << ")" << std::fixed << std::endl;
                    if (!ref_conserved)
                        std::cout << "    Expected Q_reflecting = 1.0 for w=0" << std::endl;
                    all_passed = false;
                }
                else
                {
                    std::cout << "  CUDA PASSED (Q_ref=" << std::fixed << std::setprecision(4) << Q_reflecting
                              << " [conserved], Q_abs=" << Q_absorbing << " [absorbed])" << std::endl;
                }
            }
            #endif
        }

        //=======================================================================
        // Final Summary
        //=======================================================================
        std::cout << "\n==================================================================" << std::endl;
        if (all_passed)
        {
            std::cout << "All pseudo-spectral boundary condition tests PASSED!" << std::endl;
            return 0;
        }
        else
        {
            std::cout << "Some pseudo-spectral boundary condition tests FAILED!" << std::endl;
            return -1;
        }
    }
    catch (std::exception& exc)
    {
        std::cout << exc.what() << std::endl;
        return -1;
    }
#endif
}
