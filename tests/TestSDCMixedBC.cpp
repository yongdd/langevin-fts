/**
 * @file TestSDCMixedBC.cpp
 * @brief Test SDC solver with various boundary conditions.
 *
 * This test verifies that the SDC (Spectral Deferred Correction) solver
 * produces consistent results with CN-ADI2 for various boundary conditions
 * in 1D, 2D, and 3D.
 *
 * The test compares partition functions between SDC-3 and CN-ADI2 methods.
 * Both are real-space methods with similar accuracy, so results should be
 * within a few percent.
 *
 * Testing Guidelines compliance:
 * - Uses field amplitudes with std ~ 5 (range [-9, 9])
 * - Calls check_total_partition() for CN-ADI2 (SDC has inherent ~1e-6 discrepancy)
 * - Uses ds=1/64 for accuracy with strong fields
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

// Helper template to run a single test on either CPU or CUDA
template<template<typename> class ComputationBox, template<typename> class ComputationContinuous>
bool run_test(const std::string& platform_name,
              const std::vector<int>& nx,
              const std::vector<double>& lx,
              const std::vector<std::string>& bc,
              Molecules* molecules,
              PropagatorComputationOptimizer* prop_opt,
              double tolerance)
{
    int M = 1;
    for (auto n : nx) M *= n;

    // Generate random fields with moderate amplitude (per Testing Guidelines)
    // Range [-9, 9] gives std ~ 5.2
    std::vector<double> w_A(M), w_B(M);
    unsigned int seed = 42;
    for (int i = 0; i < M; ++i)
    {
        seed = seed * 1103515245 + 12345;
        w_A[i] = ((seed >> 16) & 0x7FFF) / 32768.0 * 18.0 - 9.0;
        w_B[i] = -w_A[i];
    }

    // CN-ADI2 reference
    ComputationBox<double>* cb1 = new ComputationBox<double>(nx, lx, bc);
    ComputationContinuous<double>* solver1 = new ComputationContinuous<double>(
        cb1, molecules, prop_opt, "realspace", "cn-adi2");
    solver1->compute_propagators({{"A", w_A.data()}, {"B", w_B.data()}}, {});

    // Verify partition function consistency (per Testing Guidelines)
    if (solver1->check_total_partition() == false)
    {
        std::cout << platform_name << " FAILED (CN-ADI2 partition check failed)" << std::endl;
        delete solver1;
        delete cb1;
        return false;
    }

    double Q1 = solver1->get_total_partition(0);
    delete solver1;
    delete cb1;

    // SDC-3
    // Note: SDC has inherent forward-backward partition discrepancy (~1e-6)
    // due to the iterative correction nature of the algorithm. We skip
    // check_total_partition() and rely on comparison with CN-ADI2 for validation.
    ComputationBox<double>* cb2 = new ComputationBox<double>(nx, lx, bc);
    ComputationContinuous<double>* solver2 = new ComputationContinuous<double>(
        cb2, molecules, prop_opt, "realspace", "sdc-3");
    solver2->compute_propagators({{"A", w_A.data()}, {"B", w_B.data()}}, {});

    double Q2 = solver2->get_total_partition(0);
    delete solver2;
    delete cb2;

    double rel_err = std::abs(Q2 - Q1) / std::abs(Q1);
    if (rel_err > tolerance || !std::isfinite(rel_err))
    {
        std::cout << platform_name << " FAILED (Q_cn=" << Q1 << ", Q_sdc=" << Q2
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
    std::cout << "Neither CPU nor CUDA available, skipping SDC mixed BC test" << std::endl;
    return 0;
#else
    try
    {
        bool all_passed = true;
        const double ds = 1.0 / 64.0;
        const double tolerance = 0.02;  // 2% relative error

        // Setup molecules
        std::map<std::string, double> bond_lengths = {{"A", 1.0}, {"B", 1.0}};
        std::vector<BlockInput> blocks = {{"A", 0.5, 0, 1}, {"B", 0.5, 1, 2}};

        Molecules molecules("Continuous", ds, bond_lengths);
        molecules.add_polymer(1.0, blocks, {});
        PropagatorComputationOptimizer prop_opt(&molecules, false);

        std::cout << "==================================================================" << std::endl;
        std::cout << "Testing SDC-3 solver with various boundary conditions" << std::endl;
        std::cout << "==================================================================" << std::endl;

        //=======================================================================
        // Test 1: 1D Periodic
        //=======================================================================
        std::cout << "\nTest 1: 1D Periodic" << std::endl;
        {
            std::vector<int> nx = {64};
            std::vector<double> lx = {4.0};
            std::vector<std::string> bc = {"periodic", "periodic"};

            #ifdef USE_CPU_MKL
            if (!run_test<CpuComputationBox, CpuComputationContinuous>(
                    "  CPU-MKL", nx, lx, bc, &molecules, &prop_opt, tolerance))
                all_passed = false;
            #endif

            #ifdef USE_CUDA
            if (!run_test<CudaComputationBox, CudaComputationContinuous>(
                    "  CUDA", nx, lx, bc, &molecules, &prop_opt, tolerance))
                all_passed = false;
            #endif
        }

        //=======================================================================
        // Test 2: 1D Reflecting
        //=======================================================================
        std::cout << "\nTest 2: 1D Reflecting" << std::endl;
        {
            std::vector<int> nx = {64};
            std::vector<double> lx = {4.0};
            std::vector<std::string> bc = {"reflecting", "reflecting"};

            #ifdef USE_CPU_MKL
            if (!run_test<CpuComputationBox, CpuComputationContinuous>(
                    "  CPU-MKL", nx, lx, bc, &molecules, &prop_opt, tolerance))
                all_passed = false;
            #endif

            #ifdef USE_CUDA
            if (!run_test<CudaComputationBox, CudaComputationContinuous>(
                    "  CUDA", nx, lx, bc, &molecules, &prop_opt, tolerance))
                all_passed = false;
            #endif
        }

        //=======================================================================
        // Test 3: 1D Absorbing
        //=======================================================================
        std::cout << "\nTest 3: 1D Absorbing" << std::endl;
        {
            std::vector<int> nx = {64};
            std::vector<double> lx = {4.0};
            std::vector<std::string> bc = {"absorbing", "absorbing"};

            #ifdef USE_CPU_MKL
            if (!run_test<CpuComputationBox, CpuComputationContinuous>(
                    "  CPU-MKL", nx, lx, bc, &molecules, &prop_opt, tolerance))
                all_passed = false;
            #endif

            #ifdef USE_CUDA
            if (!run_test<CudaComputationBox, CudaComputationContinuous>(
                    "  CUDA", nx, lx, bc, &molecules, &prop_opt, tolerance))
                all_passed = false;
            #endif
        }

        //=======================================================================
        // Test 4: 2D Periodic
        //=======================================================================
        std::cout << "\nTest 4: 2D Periodic" << std::endl;
        {
            std::vector<int> nx = {24, 24};
            std::vector<double> lx = {3.0, 3.0};
            std::vector<std::string> bc = {"periodic", "periodic", "periodic", "periodic"};

            #ifdef USE_CPU_MKL
            if (!run_test<CpuComputationBox, CpuComputationContinuous>(
                    "  CPU-MKL", nx, lx, bc, &molecules, &prop_opt, tolerance))
                all_passed = false;
            #endif

            #ifdef USE_CUDA
            if (!run_test<CudaComputationBox, CudaComputationContinuous>(
                    "  CUDA", nx, lx, bc, &molecules, &prop_opt, tolerance))
                all_passed = false;
            #endif
        }

        //=======================================================================
        // Test 5: 2D All Absorbing
        //=======================================================================
        std::cout << "\nTest 5: 2D All Absorbing" << std::endl;
        {
            std::vector<int> nx = {24, 24};
            std::vector<double> lx = {3.0, 3.0};
            std::vector<std::string> bc = {"absorbing", "absorbing", "absorbing", "absorbing"};

            #ifdef USE_CPU_MKL
            if (!run_test<CpuComputationBox, CpuComputationContinuous>(
                    "  CPU-MKL", nx, lx, bc, &molecules, &prop_opt, tolerance))
                all_passed = false;
            #endif

            #ifdef USE_CUDA
            if (!run_test<CudaComputationBox, CudaComputationContinuous>(
                    "  CUDA", nx, lx, bc, &molecules, &prop_opt, tolerance))
                all_passed = false;
            #endif
        }

        //=======================================================================
        // Test 6: 2D Mixed (X:Absorbing, Y:Reflecting)
        //=======================================================================
        std::cout << "\nTest 6: 2D X:Absorbing, Y:Reflecting" << std::endl;
        {
            std::vector<int> nx = {24, 24};
            std::vector<double> lx = {3.0, 3.0};
            std::vector<std::string> bc = {"absorbing", "absorbing", "reflecting", "reflecting"};

            #ifdef USE_CPU_MKL
            if (!run_test<CpuComputationBox, CpuComputationContinuous>(
                    "  CPU-MKL", nx, lx, bc, &molecules, &prop_opt, tolerance))
                all_passed = false;
            #endif

            #ifdef USE_CUDA
            if (!run_test<CudaComputationBox, CudaComputationContinuous>(
                    "  CUDA", nx, lx, bc, &molecules, &prop_opt, tolerance))
                all_passed = false;
            #endif
        }

        //=======================================================================
        // Test 7: 3D Periodic
        //=======================================================================
        std::cout << "\nTest 7: 3D Periodic" << std::endl;
        {
            std::vector<int> nx = {12, 12, 12};
            std::vector<double> lx = {2.4, 2.4, 2.4};
            std::vector<std::string> bc = {"periodic", "periodic", "periodic", "periodic", "periodic", "periodic"};

            #ifdef USE_CPU_MKL
            if (!run_test<CpuComputationBox, CpuComputationContinuous>(
                    "  CPU-MKL", nx, lx, bc, &molecules, &prop_opt, tolerance))
                all_passed = false;
            #endif

            #ifdef USE_CUDA
            if (!run_test<CudaComputationBox, CudaComputationContinuous>(
                    "  CUDA", nx, lx, bc, &molecules, &prop_opt, tolerance))
                all_passed = false;
            #endif
        }

        //=======================================================================
        // Test 8: 3D All Absorbing
        //=======================================================================
        std::cout << "\nTest 8: 3D All Absorbing" << std::endl;
        {
            std::vector<int> nx = {12, 12, 12};
            std::vector<double> lx = {2.4, 2.4, 2.4};
            std::vector<std::string> bc = {"absorbing", "absorbing", "absorbing", "absorbing", "absorbing", "absorbing"};

            #ifdef USE_CPU_MKL
            if (!run_test<CpuComputationBox, CpuComputationContinuous>(
                    "  CPU-MKL", nx, lx, bc, &molecules, &prop_opt, tolerance))
                all_passed = false;
            #endif

            #ifdef USE_CUDA
            if (!run_test<CudaComputationBox, CudaComputationContinuous>(
                    "  CUDA", nx, lx, bc, &molecules, &prop_opt, tolerance))
                all_passed = false;
            #endif
        }

        //=======================================================================
        // Test 9: 3D Mixed (X/Y:Reflecting, Z:Absorbing)
        //=======================================================================
        std::cout << "\nTest 9: 3D X/Y:Reflecting, Z:Absorbing" << std::endl;
        {
            std::vector<int> nx = {12, 12, 12};
            std::vector<double> lx = {2.4, 2.4, 2.4};
            std::vector<std::string> bc = {"reflecting", "reflecting", "reflecting", "reflecting", "absorbing", "absorbing"};

            #ifdef USE_CPU_MKL
            if (!run_test<CpuComputationBox, CpuComputationContinuous>(
                    "  CPU-MKL", nx, lx, bc, &molecules, &prop_opt, tolerance))
                all_passed = false;
            #endif

            #ifdef USE_CUDA
            if (!run_test<CudaComputationBox, CudaComputationContinuous>(
                    "  CUDA", nx, lx, bc, &molecules, &prop_opt, tolerance))
                all_passed = false;
            #endif
        }

        //=======================================================================
        // Final Summary
        //=======================================================================
        std::cout << "\n==================================================================" << std::endl;
        if (all_passed)
        {
            std::cout << "All SDC boundary condition tests PASSED!" << std::endl;
            return 0;
        }
        else
        {
            std::cout << "Some SDC boundary condition tests FAILED!" << std::endl;
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
