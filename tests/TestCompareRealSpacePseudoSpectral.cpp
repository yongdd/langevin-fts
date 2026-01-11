/**
 * @file TestCompareRealSpacePseudoSpectral.cpp
 * @brief Compare real-space and pseudo-spectral methods for non-periodic BCs.
 *
 * This test verifies that both numerical methods produce consistent results
 * for propagator computation with non-periodic boundary conditions:
 *
 * - Pseudo-spectral: DCT (reflecting) / DST (absorbing) transforms
 * - Real-space: Crank-Nicolson finite differences with ADI splitting
 *
 * Both methods should converge to the same solution as grid resolution
 * increases. The pseudo-spectral method has higher accuracy (spectral
 * convergence) while the real-space method is more flexible.
 *
 * Tests cover:
 * - 1D reflecting, absorbing, and mixed BCs
 * - 2D reflecting, absorbing, and mixed BCs
 * - 3D reflecting, absorbing, and mixed BCs
 *
 * Platform support:
 * - CPU (MKL): Both pseudo-spectral (DCT/DST) and real-space
 * - CUDA: Real-space only for non-periodic BCs (pseudo-spectral uses FFT only)
 *
 * The test compares:
 * - CPU pseudo-spectral vs CPU real-space
 * - CUDA real-space vs CPU pseudo-spectral (to verify CUDA implementation)
 */

#include <iostream>
#include <cmath>
#include <numbers>
#include <algorithm>
#include <vector>
#include <map>
#include <string>

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
 * @brief Check if boundary conditions are mixed (different on each side of a direction).
 *
 * Pseudo-spectral methods require matching BCs on both sides of each direction
 * because FFT/DCT/DST transforms apply to the entire direction.
 */
bool has_mixed_bc(const std::vector<std::string>& bc)
{
    int dim = bc.size() / 2;
    for (int d = 0; d < dim; ++d)
    {
        if (bc[2*d] != bc[2*d + 1])
            return true;
    }
    return false;
}

/**
 * @brief Run a single comparison test between pseudo-spectral and real-space methods.
 *
 * @param test_name   Name of the test for output
 * @param nx          Grid dimensions
 * @param lx          Box dimensions
 * @param bc          Boundary conditions
 * @param n_steps     Number of propagator steps
 * @param tolerance   Maximum allowed relative error
 * @return true if test passed, false otherwise
 *
 * Note: For mixed BCs (different on each side of a direction), only real-space
 * is tested since pseudo-spectral doesn't support mixed BCs.
 */
template <typename T>
bool run_comparison_test(
    const std::string& test_name,
    const std::vector<int>& nx,
    const std::vector<double>& lx,
    const std::vector<std::string>& bc,
    int n_steps,
    double tolerance)
{
    const double ds = 0.01;
    int dim = nx.size();
    int M = 1;
    for (int i = 0; i < dim; ++i)
        M *= nx[i];

    std::map<std::string, double> bond_lengths = {{"A", 1.0}};
    std::vector<BlockInput> blocks = {{"A", 1.0, 0, 1}};

    Molecules molecules("Continuous", ds, bond_lengths);
    molecules.add_polymer(1.0, blocks, {});
    PropagatorComputationOptimizer prop_opt(&molecules, false);

    // Create Gaussian initial condition
    std::vector<double> q_init(M);
    if (dim == 1)
    {
        double dx = lx[0] / nx[0];
        for (int i = 0; i < nx[0]; ++i)
        {
            double x = (i + 0.5) * dx;
            q_init[i] = std::exp(-std::pow(x - lx[0]/2, 2) / (2 * 0.4 * 0.4));
        }
    }
    else if (dim == 2)
    {
        double dx = lx[0] / nx[0];
        double dy = lx[1] / nx[1];
        for (int i = 0; i < nx[0]; ++i)
        {
            double x = (i + 0.5) * dx;
            for (int j = 0; j < nx[1]; ++j)
            {
                double y = (j + 0.5) * dy;
                int idx = i * nx[1] + j;
                q_init[idx] = std::exp(-((x - lx[0]/2)*(x - lx[0]/2) + (y - lx[1]/2)*(y - lx[1]/2)) / (2 * 0.4 * 0.4));
            }
        }
    }
    else if (dim == 3)
    {
        double dx = lx[0] / nx[0];
        double dy = lx[1] / nx[1];
        double dz = lx[2] / nx[2];
        for (int i = 0; i < nx[0]; ++i)
        {
            double x = (i + 0.5) * dx;
            for (int j = 0; j < nx[1]; ++j)
            {
                double y = (j + 0.5) * dy;
                for (int k = 0; k < nx[2]; ++k)
                {
                    double z = (k + 0.5) * dz;
                    int idx = (i * nx[1] + j) * nx[2] + k;
                    q_init[idx] = std::exp(-((x - lx[0]/2)*(x - lx[0]/2) + (y - lx[1]/2)*(y - lx[1]/2) + (z - lx[2]/2)*(z - lx[2]/2)) / (2 * 0.5 * 0.5));
                }
            }
        }
    }

    std::vector<double> w(M, 0.0);
    std::vector<double> q_ps(M), q_rs(M);
    std::vector<double> q_out(M);
    double max_diff, rel_err;
    bool all_passed = true;

    // Check if BCs are mixed (different on each side of a direction)
    // Pseudo-spectral doesn't support mixed BCs
    bool mixed_bc = has_mixed_bc(bc);

    // ====================================================================
    // CPU Tests: Compare pseudo-spectral vs real-space
    // Both methods support non-periodic BCs on CPU (DCT/DST for pseudo-spectral)
    // For mixed BCs, only test real-space (pseudo-spectral not supported)
    // ====================================================================
#ifdef USE_CPU_MKL
    std::vector<double> q_cpu_ref(M);  // Reference for CUDA comparison

    if (mixed_bc)
    {
        // Mixed BCs: only test real-space
        std::cout << "  [CPU-MKL RS only] " << std::flush;
        CpuComputationBox<T>* cb_rs = new CpuComputationBox<T>(nx, lx, bc);
        CpuComputationContinuous<T>* solver_rs = new CpuComputationContinuous<T>(cb_rs, &molecules, &prop_opt, "realspace");

        // Run real-space solver
        solver_rs->compute_propagators({{"A", w.data()}}, {});
        q_rs = q_init;
        for (int step = 0; step < n_steps; ++step)
        {
            solver_rs->advance_propagator_single_segment(q_rs.data(), q_out.data(), "A");
            std::swap(q_rs, q_out);
        }

        // Just verify it ran without errors and produced reasonable output
        double max_val = *std::max_element(q_rs.begin(), q_rs.end());
        if (max_val > 0 && std::isfinite(max_val))
        {
            std::cout << "PASSED (max_q=" << max_val << ")" << std::endl;
        }
        else
        {
            std::cout << "FAILED (invalid output)" << std::endl;
            all_passed = false;
        }

        q_cpu_ref = q_rs;  // Use CPU real-space as reference for CUDA

        delete solver_rs;
        delete cb_rs;
    }
    else
    {
        // Matching BCs: compare pseudo-spectral vs real-space
        std::cout << "  [CPU-MKL PS vs RS] " << std::flush;
        CpuComputationBox<T>* cb_ps = new CpuComputationBox<T>(nx, lx, bc);
        CpuComputationBox<T>* cb_rs = new CpuComputationBox<T>(nx, lx, bc);

        CpuComputationContinuous<T>* solver_ps = new CpuComputationContinuous<T>(cb_ps, &molecules, &prop_opt, "pseudospectral");
        CpuComputationContinuous<T>* solver_rs = new CpuComputationContinuous<T>(cb_rs, &molecules, &prop_opt, "realspace");

        // Run pseudo-spectral solver
        solver_ps->compute_propagators({{"A", w.data()}}, {});
        q_ps = q_init;
        for (int step = 0; step < n_steps; ++step)
        {
            solver_ps->advance_propagator_single_segment(q_ps.data(), q_out.data(), "A");
            std::swap(q_ps, q_out);
        }

        // Run real-space solver
        solver_rs->compute_propagators({{"A", w.data()}}, {});
        q_rs = q_init;
        for (int step = 0; step < n_steps; ++step)
        {
            solver_rs->advance_propagator_single_segment(q_rs.data(), q_out.data(), "A");
            std::swap(q_rs, q_out);
        }

        max_diff = 0.0;
        for (int i = 0; i < M; ++i)
            max_diff = std::max(max_diff, std::abs(q_ps[i] - q_rs[i]));
        rel_err = max_diff / (*std::max_element(q_ps.begin(), q_ps.end()));

        if (rel_err > tolerance)
        {
            std::cout << "FAILED (rel_err=" << rel_err << ")" << std::endl;
            all_passed = false;
        }
        else
        {
            std::cout << "PASSED (rel_err=" << rel_err << ")" << std::endl;
        }

        q_cpu_ref = q_ps;  // Use CPU pseudo-spectral as reference for CUDA

        delete solver_ps;
        delete solver_rs;
        delete cb_ps;
        delete cb_rs;
    }

    // ====================================================================
    // CUDA Tests: Compare CUDA real-space vs CPU reference
    // For matching BCs: compare with CPU pseudo-spectral
    // For mixed BCs: compare with CPU real-space
    // ====================================================================
#ifdef USE_CUDA
    std::cout << "  [CUDA RS vs CPU " << (mixed_bc ? "RS" : "PS") << "] " << std::flush;
    {
        CudaComputationBox<T>* cb_cuda_rs = new CudaComputationBox<T>(nx, lx, bc);
        CudaComputationContinuous<T>* solver_cuda_rs = new CudaComputationContinuous<T>(cb_cuda_rs, &molecules, &prop_opt, "realspace");

        // Run CUDA real-space solver
        solver_cuda_rs->compute_propagators({{"A", w.data()}}, {});
        q_rs = q_init;
        for (int step = 0; step < n_steps; ++step)
        {
            solver_cuda_rs->advance_propagator_single_segment(q_rs.data(), q_out.data(), "A");
            std::swap(q_rs, q_out);
        }

        // Compare CUDA real-space result with CPU reference
        max_diff = 0.0;
        for (int i = 0; i < M; ++i)
            max_diff = std::max(max_diff, std::abs(q_cpu_ref[i] - q_rs[i]));
        rel_err = max_diff / (*std::max_element(q_cpu_ref.begin(), q_cpu_ref.end()));

        // For mixed BCs comparing RS vs RS, use tighter tolerance
        double cuda_tol = mixed_bc ? 1e-10 : tolerance;
        if (rel_err > cuda_tol)
        {
            std::cout << "FAILED (rel_err=" << rel_err << ")" << std::endl;
            all_passed = false;
        }
        else
        {
            std::cout << "PASSED (rel_err=" << rel_err << ")" << std::endl;
        }

        delete solver_cuda_rs;
        delete cb_cuda_rs;
    }
#endif
#endif

    return all_passed;
}

int main()
{
    try
    {
        bool all_passed = true;

        // Tolerance for comparison (real-space is 2nd order, pseudo-spectral is spectral)
        const double tol_1d = 0.01;   // 1% relative error for 1D
        const double tol_2d = 0.02;   // 2% relative error for 2D
        const double tol_3d = 0.05;   // 5% relative error for 3D

        const int n_steps_1d = 30;
        const int n_steps_2d = 30;
        const int n_steps_3d = 20;

        //=======================================================================
        // Test 1: 1D Reflecting BC
        //=======================================================================
        std::cout << "\nTest 1: 1D Reflecting BC" << std::endl;
        if (!run_comparison_test<double>(
            "1D Reflecting BC",
            {64}, {4.0},
            {"reflecting", "reflecting"},
            n_steps_1d, tol_1d))
            all_passed = false;

        //=======================================================================
        // Test 2: 1D Absorbing BC
        //=======================================================================
        std::cout << "\nTest 2: 1D Absorbing BC" << std::endl;
        if (!run_comparison_test<double>(
            "1D Absorbing BC",
            {64}, {4.0},
            {"absorbing", "absorbing"},
            n_steps_1d, tol_1d))
            all_passed = false;

        //=======================================================================
        // Test 3: 2D Reflecting BC
        //=======================================================================
        std::cout << "\nTest 3: 2D Reflecting BC" << std::endl;
        if (!run_comparison_test<double>(
            "2D Reflecting BC",
            {32, 24}, {4.0, 3.0},
            {"reflecting", "reflecting", "reflecting", "reflecting"},
            n_steps_2d, tol_2d))
            all_passed = false;

        //=======================================================================
        // Test 4: 2D Absorbing BC
        //=======================================================================
        std::cout << "\nTest 4: 2D Absorbing BC" << std::endl;
        if (!run_comparison_test<double>(
            "2D Absorbing BC",
            {32, 24}, {4.0, 3.0},
            {"absorbing", "absorbing", "absorbing", "absorbing"},
            n_steps_2d, tol_2d))
            all_passed = false;

        //=======================================================================
        // Test 5: 1D Mixed BC (Reflecting left, Absorbing right)
        //=======================================================================
        std::cout << "\nTest 5: 1D Mixed BC (Reflect-Absorb)" << std::endl;
        if (!run_comparison_test<double>(
            "1D Mixed BC",
            {64}, {4.0},
            {"reflecting", "absorbing"},
            n_steps_1d, tol_1d))
            all_passed = false;

        //=======================================================================
        // Test 6: 2D Mixed BC (Reflecting in X, Absorbing in Y)
        //=======================================================================
        std::cout << "\nTest 6: 2D Mixed BC (X:Reflect, Y:Absorb)" << std::endl;
        if (!run_comparison_test<double>(
            "2D Mixed BC X:Reflect Y:Absorb",
            {32, 24}, {4.0, 3.0},
            {"reflecting", "reflecting", "absorbing", "absorbing"},
            n_steps_2d, tol_2d))
            all_passed = false;

        //=======================================================================
        // Test 7: 2D Mixed BC (Absorbing in X, Reflecting in Y)
        //=======================================================================
        std::cout << "\nTest 7: 2D Mixed BC (X:Absorb, Y:Reflect)" << std::endl;
        if (!run_comparison_test<double>(
            "2D Mixed BC X:Absorb Y:Reflect",
            {32, 24}, {4.0, 3.0},
            {"absorbing", "absorbing", "reflecting", "reflecting"},
            n_steps_2d, tol_2d))
            all_passed = false;

        //=======================================================================
        // Test 8: 3D Reflecting BC
        //=======================================================================
        std::cout << "\nTest 8: 3D Reflecting BC" << std::endl;
        if (!run_comparison_test<double>(
            "3D Reflecting BC",
            {16, 16, 16}, {4.0, 4.0, 4.0},
            {"reflecting", "reflecting", "reflecting", "reflecting", "reflecting", "reflecting"},
            n_steps_3d, tol_3d))
            all_passed = false;

        //=======================================================================
        // Test 9: 3D Absorbing BC
        //=======================================================================
        std::cout << "\nTest 9: 3D Absorbing BC" << std::endl;
        if (!run_comparison_test<double>(
            "3D Absorbing BC",
            {16, 16, 16}, {4.0, 4.0, 4.0},
            {"absorbing", "absorbing", "absorbing", "absorbing", "absorbing", "absorbing"},
            n_steps_3d, tol_3d))
            all_passed = false;

        //=======================================================================
        // Test 10: 3D Mixed BC (Reflecting in X/Y, Absorbing in Z)
        //=======================================================================
        std::cout << "\nTest 10: 3D Mixed BC (X:Reflect, Y:Reflect, Z:Absorb)" << std::endl;
        if (!run_comparison_test<double>(
            "3D Mixed BC",
            {16, 16, 16}, {4.0, 4.0, 4.0},
            {"reflecting", "reflecting", "reflecting", "reflecting", "absorbing", "absorbing"},
            n_steps_3d, tol_3d))
            all_passed = false;

        //=======================================================================
        // Final Summary
        //=======================================================================
        if (all_passed)
        {
            std::cout << "\nAll tests passed!" << std::endl;
            return 0;
        }
        else
        {
            std::cout << "\nSome tests FAILED!" << std::endl;
            return -1;
        }
    }
    catch (std::exception& exc)
    {
        std::cout << exc.what() << std::endl;
        return -1;
    }
}
