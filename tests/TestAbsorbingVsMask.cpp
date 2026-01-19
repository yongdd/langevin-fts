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
 * @file TestAbsorbingVsMask.cpp
 * @brief Compare absorbing boundary conditions with periodic BC + mask approach.
 *
 * This test compares two approaches for modeling absorbing surfaces:
 *
 * 1. **Absorbing BC (DST-based)**: Uses Discrete Sine Transform which naturally
 *    enforces q=0 at cell faces (the domain boundaries at x=0 and x=L).
 *
 * 2. **Periodic BC + Mask**: Uses periodic BC with FFT, but applies a mask that
 *    sets the propagator to zero at boundary grid cells.
 *
 * **Important Distinction**:
 * - DST enforces zeros at cell faces (boundaries between cells)
 * - Mask enforces zeros at grid cell centers
 *
 * These are fundamentally different boundary treatments:
 * - For the same grid, they won't produce identical results
 * - The mask approach requires adding extra "boundary cells" that act as absorbers
 * - As the grid becomes finer, both approaches should converge to the same continuum limit
 *
 * **Physical Interpretation**:
 * - Absorbing BC: Models a surface at the domain boundary that absorbs chains
 * - Mask approach: Models impenetrable/absorbing particles at specific grid points
 *
 * This test validates that both approaches produce physically reasonable results
 * with partition functions that are reasonably close (within ~10-30% depending on grid).
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

/**
 * @brief Create a boundary mask for simulating absorbing BC with periodic FFT.
 *
 * Sets mask=0 for grid cells at the domain boundary (first and last cells in each direction).
 * This creates a "shell" of absorbing cells around the interior.
 *
 * @param nx Grid dimensions
 * @return Mask array (1=accessible, 0=blocked)
 */
std::vector<double> create_boundary_mask(const std::vector<int>& nx)
{
    int dim = nx.size();
    int M = 1;
    for (auto n : nx) M *= n;

    std::vector<double> mask(M, 1.0);

    if (dim == 1)
    {
        // 1D: mask first and last cells
        mask[0] = 0.0;
        mask[nx[0] - 1] = 0.0;
    }
    else if (dim == 2)
    {
        for (int ix = 0; ix < nx[0]; ++ix)
        {
            for (int iy = 0; iy < nx[1]; ++iy)
            {
                int idx = iy + nx[1] * ix;
                // Mask boundary cells (first/last in each direction)
                if (ix == 0 || ix == nx[0] - 1 || iy == 0 || iy == nx[1] - 1)
                {
                    mask[idx] = 0.0;
                }
            }
        }
    }
    else if (dim == 3)
    {
        for (int ix = 0; ix < nx[0]; ++ix)
        {
            for (int iy = 0; iy < nx[1]; ++iy)
            {
                for (int iz = 0; iz < nx[2]; ++iz)
                {
                    int idx = iz + nx[2] * (iy + nx[1] * ix);
                    // Mask boundary cells
                    if (ix == 0 || ix == nx[0] - 1 ||
                        iy == 0 || iy == nx[1] - 1 ||
                        iz == 0 || iz == nx[2] - 1)
                    {
                        mask[idx] = 0.0;
                    }
                }
            }
        }
    }

    return mask;
}

/**
 * @brief Run comparison test for a given platform.
 */
template<template<typename> class ComputationBox, template<typename> class ComputationContinuous>
bool run_comparison(const std::string& platform_name,
                    const std::vector<int>& nx,
                    const std::vector<double>& lx,
                    Molecules* molecules,
                    PropagatorComputationOptimizer* prop_opt,
                    double tolerance)
{
    int dim = nx.size();
    int M = 1;
    for (auto n : nx) M *= n;

    // Generate random fields with std ~ 5 (as per testing guidelines)
    std::vector<double> w_A(M), w_B(M);
    unsigned int seed = 12345;
    for (int i = 0; i < M; ++i)
    {
        // Linear congruential generator
        seed = seed * 1103515245 + 12345;
        double u1 = ((seed >> 16) & 0x7FFF) / 32768.0;
        seed = seed * 1103515245 + 12345;
        double u2 = ((seed >> 16) & 0x7FFF) / 32768.0;
        // Box-Muller transform for normal distribution
        double z = std::sqrt(-2.0 * std::log(u1 + 1e-10)) * std::cos(2.0 * M_PI * u2);
        w_A[i] = z * 5.0;  // std = 5
        w_B[i] = -w_A[i];
    }

    // Build BC strings
    std::vector<std::string> bc_absorbing, bc_periodic;
    for (int d = 0; d < dim; ++d)
    {
        bc_absorbing.push_back("absorbing");
        bc_absorbing.push_back("absorbing");
        bc_periodic.push_back("periodic");
        bc_periodic.push_back("periodic");
    }

    // Create boundary mask for periodic BC
    std::vector<double> mask = create_boundary_mask(nx);

    // Method 1: Absorbing BC (DST-based)
    ComputationBox<double>* cb_absorbing = new ComputationBox<double>(nx, lx, bc_absorbing);
    ComputationContinuous<double>* solver_absorbing = new ComputationContinuous<double>(
        cb_absorbing, molecules, prop_opt, "pseudospectral", "rqm4");
    solver_absorbing->compute_propagators({{"A", w_A.data()}, {"B", w_B.data()}}, {});
    double Q_absorbing = solver_absorbing->get_total_partition(0);
    delete solver_absorbing;
    delete cb_absorbing;

    // Method 2: Periodic BC with boundary mask
    ComputationBox<double>* cb_masked = new ComputationBox<double>(nx, lx, bc_periodic, mask.data());
    ComputationContinuous<double>* solver_masked = new ComputationContinuous<double>(
        cb_masked, molecules, prop_opt, "pseudospectral", "rqm4");
    solver_masked->compute_propagators({{"A", w_A.data()}, {"B", w_B.data()}}, {});
    double Q_masked = solver_masked->get_total_partition(0);
    delete solver_masked;
    delete cb_masked;

    // Calculate relative difference
    // Note: These won't be identical - DST zeros at cell faces, mask zeros at cell centers
    double rel_diff = std::abs(Q_absorbing - Q_masked) / std::abs(Q_absorbing);

    std::cout << "  " << platform_name << ": Q_absorbing=" << std::scientific << std::setprecision(6) << Q_absorbing
              << ", Q_masked=" << Q_masked
              << ", rel_diff=" << std::setprecision(2) << rel_diff << std::fixed;

    if (rel_diff > tolerance || !std::isfinite(rel_diff) || !std::isfinite(Q_absorbing) || !std::isfinite(Q_masked))
    {
        std::cout << " [FAILED - difference too large]" << std::endl;
        return false;
    }
    else
    {
        std::cout << " [PASSED]" << std::endl;
        return true;
    }
}

/**
 * @brief Test that both approaches show similar physical behavior.
 *
 * For zero field (w=0), both absorbing BC and mask should:
 * - Have Q < 1.0 (mass is absorbed at boundaries)
 * - Absorbing BC typically has slightly higher Q (zeros at faces, not cell centers)
 */
template<template<typename> class ComputationBox, template<typename> class ComputationContinuous>
bool run_physical_test(const std::string& platform_name,
                       const std::vector<int>& nx,
                       const std::vector<double>& lx,
                       Molecules* molecules,
                       PropagatorComputationOptimizer* prop_opt)
{
    int dim = nx.size();
    int M = 1;
    for (auto n : nx) M *= n;

    // Zero field
    std::vector<double> w_A(M, 0.0), w_B(M, 0.0);

    // Build BC strings
    std::vector<std::string> bc_absorbing, bc_periodic, bc_reflecting;
    for (int d = 0; d < dim; ++d)
    {
        bc_absorbing.push_back("absorbing");
        bc_absorbing.push_back("absorbing");
        bc_periodic.push_back("periodic");
        bc_periodic.push_back("periodic");
        bc_reflecting.push_back("reflecting");
        bc_reflecting.push_back("reflecting");
    }

    std::vector<double> mask = create_boundary_mask(nx);

    // Reflecting BC (reference: should have Q = 1.0 for w=0)
    ComputationBox<double>* cb_ref = new ComputationBox<double>(nx, lx, bc_reflecting);
    ComputationContinuous<double>* solver_ref = new ComputationContinuous<double>(
        cb_ref, molecules, prop_opt, "pseudospectral", "rqm4");
    solver_ref->compute_propagators({{"A", w_A.data()}, {"B", w_B.data()}}, {});
    double Q_reflecting = solver_ref->get_total_partition(0);
    delete solver_ref;
    delete cb_ref;

    // Absorbing BC
    ComputationBox<double>* cb_absorbing = new ComputationBox<double>(nx, lx, bc_absorbing);
    ComputationContinuous<double>* solver_absorbing = new ComputationContinuous<double>(
        cb_absorbing, molecules, prop_opt, "pseudospectral", "rqm4");
    solver_absorbing->compute_propagators({{"A", w_A.data()}, {"B", w_B.data()}}, {});
    double Q_absorbing = solver_absorbing->get_total_partition(0);
    delete solver_absorbing;
    delete cb_absorbing;

    // Periodic BC with mask
    ComputationBox<double>* cb_masked = new ComputationBox<double>(nx, lx, bc_periodic, mask.data());
    ComputationContinuous<double>* solver_masked = new ComputationContinuous<double>(
        cb_masked, molecules, prop_opt, "pseudospectral", "rqm4");
    solver_masked->compute_propagators({{"A", w_A.data()}, {"B", w_B.data()}}, {});
    double Q_masked = solver_masked->get_total_partition(0);
    delete solver_masked;
    delete cb_masked;

    std::cout << "  " << platform_name << ": Q_reflecting=" << std::fixed << std::setprecision(6) << Q_reflecting
              << ", Q_absorbing=" << Q_absorbing
              << ", Q_masked=" << Q_masked;

    // Physical checks:
    // 1. Reflecting should conserve mass: Q_reflecting ≈ 1.0
    // 2. Absorbing should lose mass: Q_absorbing < 1.0
    // 3. Masked should also lose mass: Q_masked < 1.0
    // 4. Ordering should be: Q_reflecting > Q_absorbing and Q_reflecting > Q_masked
    bool reflecting_conserved = std::abs(Q_reflecting - 1.0) < 1e-8;
    bool absorbing_loses = Q_absorbing < 1.0;
    bool masked_loses = Q_masked < 1.0;
    bool reflecting_higher = Q_reflecting > Q_absorbing && Q_reflecting > Q_masked;

    if (reflecting_conserved && absorbing_loses && masked_loses && reflecting_higher)
    {
        std::cout << " [PASSED]" << std::endl;
        return true;
    }
    else
    {
        std::cout << " [FAILED";
        if (!reflecting_conserved) std::cout << " - reflecting not conserved";
        if (!absorbing_loses) std::cout << " - absorbing not losing mass";
        if (!masked_loses) std::cout << " - masked not losing mass";
        if (!reflecting_higher) std::cout << " - ordering wrong";
        std::cout << "]" << std::endl;
        return false;
    }
}

int main()
{
#if !defined(USE_CPU_MKL) && !defined(USE_CUDA)
    std::cout << "Neither CPU nor CUDA available, skipping test" << std::endl;
    return 0;
#else
    try
    {
        bool all_passed = true;
        const double ds = 1.0 / 100.0;

        // Setup molecules
        std::map<std::string, double> bond_lengths = {{"A", 1.0}, {"B", 1.0}};
        std::vector<BlockInput> blocks = {{"A", 0.5, 0, 1}, {"B", 0.5, 1, 2}};

        Molecules molecules("Continuous", ds, bond_lengths);
        molecules.add_polymer(1.0, blocks, {});
        PropagatorComputationOptimizer prop_opt(&molecules, false);

        std::cout << "==================================================================" << std::endl;
        std::cout << "Comparing Absorbing BC (DST) vs Periodic BC + Mask" << std::endl;
        std::cout << "==================================================================" << std::endl;
        std::cout << "\nNote: These approaches are NOT equivalent:" << std::endl;
        std::cout << "  - DST enforces q=0 at cell FACES (domain boundaries)" << std::endl;
        std::cout << "  - Mask enforces q=0 at cell CENTERS (grid points)" << std::endl;
        std::cout << "  Results will differ, but both model absorbing surfaces.\n" << std::endl;

        //=======================================================================
        // Part 1: Physical behavior test (zero field)
        //=======================================================================
        std::cout << "=== Physical Behavior Test (w=0) ===" << std::endl;
        std::cout << "Expected: Q_reflecting ≈ 1.0 > Q_absorbing, Q_masked < 1.0\n" << std::endl;

        // 1D test
        std::cout << "1D (nx=64, lx=4.0):" << std::endl;
        {
            std::vector<int> nx = {64};
            std::vector<double> lx = {4.0};

            #ifdef USE_CPU_MKL
            if (!run_physical_test<CpuComputationBox, CpuComputationContinuous>(
                    "CPU-MKL", nx, lx, &molecules, &prop_opt))
                all_passed = false;
            #endif

            #ifdef USE_CUDA
            if (!run_physical_test<CudaComputationBox, CudaComputationContinuous>(
                    "CUDA", nx, lx, &molecules, &prop_opt))
                all_passed = false;
            #endif
        }

        // 2D test
        std::cout << "\n2D (nx=32x32, lx=4.0x4.0):" << std::endl;
        {
            std::vector<int> nx = {32, 32};
            std::vector<double> lx = {4.0, 4.0};

            #ifdef USE_CPU_MKL
            if (!run_physical_test<CpuComputationBox, CpuComputationContinuous>(
                    "CPU-MKL", nx, lx, &molecules, &prop_opt))
                all_passed = false;
            #endif

            #ifdef USE_CUDA
            if (!run_physical_test<CudaComputationBox, CudaComputationContinuous>(
                    "CUDA", nx, lx, &molecules, &prop_opt))
                all_passed = false;
            #endif
        }

        // 3D test
        std::cout << "\n3D (nx=16x16x16, lx=4.0x4.0x4.0):" << std::endl;
        {
            std::vector<int> nx = {16, 16, 16};
            std::vector<double> lx = {4.0, 4.0, 4.0};

            #ifdef USE_CPU_MKL
            if (!run_physical_test<CpuComputationBox, CpuComputationContinuous>(
                    "CPU-MKL", nx, lx, &molecules, &prop_opt))
                all_passed = false;
            #endif

            #ifdef USE_CUDA
            if (!run_physical_test<CudaComputationBox, CudaComputationContinuous>(
                    "CUDA", nx, lx, &molecules, &prop_opt))
                all_passed = false;
            #endif
        }

        //=======================================================================
        // Part 2: Quantitative comparison with random fields
        //=======================================================================
        std::cout << "\n=== Quantitative Comparison (random fields, std=5) ===" << std::endl;
        std::cout << "Note: Large tolerance (50%) expected due to different BC implementations\n" << std::endl;

        // The tolerance is large because the methods are fundamentally different
        // DST: zeros at x=0, x=L (cell faces)
        // Mask: zeros at x=dx/2, x=L-dx/2 (first and last cell centers)
        const double tolerance = 0.5;  // 50% tolerance

        // 1D comparison
        std::cout << "1D (nx=64, lx=4.0):" << std::endl;
        {
            std::vector<int> nx = {64};
            std::vector<double> lx = {4.0};

            #ifdef USE_CPU_MKL
            if (!run_comparison<CpuComputationBox, CpuComputationContinuous>(
                    "CPU-MKL", nx, lx, &molecules, &prop_opt, tolerance))
                all_passed = false;
            #endif

            #ifdef USE_CUDA
            if (!run_comparison<CudaComputationBox, CudaComputationContinuous>(
                    "CUDA", nx, lx, &molecules, &prop_opt, tolerance))
                all_passed = false;
            #endif
        }

        // 2D comparison
        std::cout << "\n2D (nx=32x32, lx=4.0x4.0):" << std::endl;
        {
            std::vector<int> nx = {32, 32};
            std::vector<double> lx = {4.0, 4.0};

            #ifdef USE_CPU_MKL
            if (!run_comparison<CpuComputationBox, CpuComputationContinuous>(
                    "CPU-MKL", nx, lx, &molecules, &prop_opt, tolerance))
                all_passed = false;
            #endif

            #ifdef USE_CUDA
            if (!run_comparison<CudaComputationBox, CudaComputationContinuous>(
                    "CUDA", nx, lx, &molecules, &prop_opt, tolerance))
                all_passed = false;
            #endif
        }

        // 3D comparison
        std::cout << "\n3D (nx=16x16x16, lx=4.0x4.0x4.0):" << std::endl;
        {
            std::vector<int> nx = {16, 16, 16};
            std::vector<double> lx = {4.0, 4.0, 4.0};

            #ifdef USE_CPU_MKL
            if (!run_comparison<CpuComputationBox, CpuComputationContinuous>(
                    "CPU-MKL", nx, lx, &molecules, &prop_opt, tolerance))
                all_passed = false;
            #endif

            #ifdef USE_CUDA
            if (!run_comparison<CudaComputationBox, CudaComputationContinuous>(
                    "CUDA", nx, lx, &molecules, &prop_opt, tolerance))
                all_passed = false;
            #endif
        }

        //=======================================================================
        // Summary
        //=======================================================================
        std::cout << "\n==================================================================" << std::endl;
        if (all_passed)
        {
            std::cout << "All absorbing BC vs mask comparison tests PASSED!" << std::endl;
            std::cout << "\nConclusion:" << std::endl;
            std::cout << "  Both approaches model absorbing surfaces but differ in implementation:" << std::endl;
            std::cout << "  - Absorbing BC (DST): More accurate for smooth boundaries" << std::endl;
            std::cout << "  - Periodic + Mask: More flexible for complex geometries" << std::endl;
            return 0;
        }
        else
        {
            std::cout << "Some absorbing BC vs mask comparison tests FAILED!" << std::endl;
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
