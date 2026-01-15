/**
 * @file TestCNADI4MixedBC.cpp
 * @brief Test CN-ADI4 solvers with absorbing and mixed boundary conditions.
 *
 * This test verifies that the CN-ADI4 (4th-order Crank-Nicolson ADI) solvers
 * produce consistent results with CN-ADI2 for absorbing boundary conditions
 * in 2D and 3D.
 *
 * The test compares partition functions between CN-ADI4 variants and CN-ADI2:
 * - CN-ADI4-LR: Local Richardson extrapolation (applied at each contour step)
 * - CN-ADI4-GR: Global Richardson extrapolation (applied across full chain)
 *
 * Results should be within a few percent for these test cases.
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
              const std::string& method_name,
              const std::vector<int>& nx,
              const std::vector<double>& lx,
              const std::vector<std::string>& bc,
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

    // CN-ADI2 reference
    ComputationBox<double>* cb1 = new ComputationBox<double>(nx, lx, bc);
    ComputationContinuous<double>* solver1 = new ComputationContinuous<double>(
        cb1, molecules, prop_opt, "realspace", "cn-adi2");
    solver1->compute_propagators({{"A", w_A.data()}, {"B", w_B.data()}}, {});
    double Q1 = solver1->get_total_partition(0);
    delete solver1;
    delete cb1;

    // CN-ADI4 (LR or GR)
    ComputationBox<double>* cb2 = new ComputationBox<double>(nx, lx, bc);
    ComputationContinuous<double>* solver2 = new ComputationContinuous<double>(
        cb2, molecules, prop_opt, "realspace", method_name);
    solver2->compute_propagators({{"A", w_A.data()}, {"B", w_B.data()}}, {});
    double Q2 = solver2->get_total_partition(0);
    delete solver2;
    delete cb2;

    double rel_err = std::abs(Q2 - Q1) / std::abs(Q1);
    if (rel_err > tolerance || !std::isfinite(rel_err))
    {
        std::cout << platform_name << " FAILED (Q_cn2=" << Q1 << ", Q_cn4=" << Q2
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

// Run tests for a specific method (cn-adi4-lr or cn-adi4-gr)
template<template<typename> class ComputationBox, template<typename> class ComputationContinuous>
bool run_method_tests(const std::string& platform_name,
                      const std::string& method_name,
                      Molecules* molecules,
                      PropagatorComputationOptimizer* prop_opt,
                      double tolerance)
{
    bool all_passed = true;

    // Test 1: 2D All Absorbing
    {
        std::vector<int> nx = {24, 24};
        std::vector<double> lx = {3.0, 3.0};
        std::vector<std::string> bc = {"absorbing", "absorbing", "absorbing", "absorbing"};

        std::cout << "    2D Absorbing:      ";
        if (!run_test<ComputationBox, ComputationContinuous>(
                "", method_name, nx, lx, bc, molecules, prop_opt, tolerance))
            all_passed = false;
    }

    // Test 2: 2D Mixed (X:Absorbing, Y:Reflecting)
    {
        std::vector<int> nx = {24, 24};
        std::vector<double> lx = {3.0, 3.0};
        std::vector<std::string> bc = {"absorbing", "absorbing", "reflecting", "reflecting"};

        std::cout << "    2D Mixed:          ";
        if (!run_test<ComputationBox, ComputationContinuous>(
                "", method_name, nx, lx, bc, molecules, prop_opt, tolerance))
            all_passed = false;
    }

    // Test 3: 3D All Absorbing
    {
        std::vector<int> nx = {12, 12, 12};
        std::vector<double> lx = {2.4, 2.4, 2.4};
        std::vector<std::string> bc = {"absorbing", "absorbing", "absorbing", "absorbing", "absorbing", "absorbing"};

        std::cout << "    3D Absorbing:      ";
        if (!run_test<ComputationBox, ComputationContinuous>(
                "", method_name, nx, lx, bc, molecules, prop_opt, tolerance))
            all_passed = false;
    }

    // Test 4: 3D Mixed (X/Y:Reflecting, Z:Absorbing)
    {
        std::vector<int> nx = {12, 12, 12};
        std::vector<double> lx = {2.4, 2.4, 2.4};
        std::vector<std::string> bc = {"reflecting", "reflecting", "reflecting", "reflecting", "absorbing", "absorbing"};

        std::cout << "    3D Mixed:          ";
        if (!run_test<ComputationBox, ComputationContinuous>(
                "", method_name, nx, lx, bc, molecules, prop_opt, tolerance))
            all_passed = false;
    }

    return all_passed;
}

int main()
{
#if !defined(USE_CPU_MKL) && !defined(USE_CUDA)
    std::cout << "Neither CPU nor CUDA available, skipping CN-ADI4 mixed BC test" << std::endl;
    return 0;
#else
    try
    {
        bool all_passed = true;
        const double ds = 1.0 / 16.0;
        const double tolerance = 0.02;  // 2% relative error

        // Setup molecules
        std::map<std::string, double> bond_lengths = {{"A", 1.0}, {"B", 1.0}};
        std::vector<BlockInput> blocks = {{"A", 0.5, 0, 1}, {"B", 0.5, 1, 2}};

        Molecules molecules("Continuous", ds, bond_lengths);
        molecules.add_polymer(1.0, blocks, {});
        PropagatorComputationOptimizer prop_opt(&molecules, false);

        std::cout << "==================================================================" << std::endl;
        std::cout << "Testing CN-ADI4 solvers with absorbing boundary conditions" << std::endl;
        std::cout << "==================================================================" << std::endl;

        //=======================================================================
        // CN-ADI4-LR (Local Richardson)
        //=======================================================================
        std::cout << "\n--- CN-ADI4-LR (Local Richardson) ---" << std::endl;

        #ifdef USE_CPU_MKL
        std::cout << "  CPU-MKL:" << std::endl;
        if (!run_method_tests<CpuComputationBox, CpuComputationContinuous>(
                "CPU-MKL", "cn-adi4-lr", &molecules, &prop_opt, tolerance))
            all_passed = false;
        #endif

        #ifdef USE_CUDA
        std::cout << "  CUDA:" << std::endl;
        if (!run_method_tests<CudaComputationBox, CudaComputationContinuous>(
                "CUDA", "cn-adi4-lr", &molecules, &prop_opt, tolerance))
            all_passed = false;
        #endif

        //=======================================================================
        // CN-ADI4-GR (Global Richardson)
        //=======================================================================
        std::cout << "\n--- CN-ADI4-GR (Global Richardson) ---" << std::endl;

        #ifdef USE_CPU_MKL
        std::cout << "  CPU-MKL:" << std::endl;
        if (!run_method_tests<CpuComputationBox, CpuComputationContinuous>(
                "CPU-MKL", "cn-adi4-gr", &molecules, &prop_opt, tolerance))
            all_passed = false;
        #endif

        #ifdef USE_CUDA
        std::cout << "  CUDA:" << std::endl;
        if (!run_method_tests<CudaComputationBox, CudaComputationContinuous>(
                "CUDA", "cn-adi4-gr", &molecules, &prop_opt, tolerance))
            all_passed = false;
        #endif

        //=======================================================================
        // Final Summary
        //=======================================================================
        std::cout << "\n==================================================================" << std::endl;
        if (all_passed)
        {
            std::cout << "All CN-ADI4 absorbing BC tests PASSED!" << std::endl;
            return 0;
        }
        else
        {
            std::cout << "Some CN-ADI4 absorbing BC tests FAILED!" << std::endl;
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
