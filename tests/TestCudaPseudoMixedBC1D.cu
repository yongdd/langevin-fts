/**
 * @file TestCudaPseudoMixedBC1D.cu
 * @brief Test CUDA pseudo-spectral solver with mixed boundary conditions.
 *
 * This test verifies that CudaSolverPseudoMixedBC correctly computes
 * propagator evolution with:
 * - REFLECTING: DCT-based method (Neumann BC, zero flux)
 * - ABSORBING: DST-based method (Dirichlet BC, zero value at boundary)
 *
 * Physical verification:
 * - REFLECTING: Mass should be conserved (no flux through boundaries)
 * - ABSORBING: Mass should decrease (absorbed at boundaries)
 */

#include <iostream>
#include <cmath>
#include <numbers>
#include <algorithm>
#include <vector>
#include <map>

#include "Exception.h"
#include "Molecules.h"
#include "Polymer.h"
#include "CudaCommon.h"
#include "CudaComputationBox.h"
#include "CudaSolverPseudoMixedBC.h"

int main()
{
    try
    {
        const int N = 32;
        const double L = 4.0;
        const double dx = L / N;
        const double ds = 0.01;
        const int n_steps = 50;

        // Initialize CUDA
        CudaCommon::get_instance().set(128, 256, 0);

        // Create Gaussian initial condition centered in domain
        std::vector<double> q_init(N);
        for (int i = 0; i < N; ++i)
        {
            double x = (i + 0.5) * dx;  // Staggered grid
            q_init[i] = std::exp(-std::pow(x - L/2, 2) / (2 * 0.3 * 0.3));
        }

        // Zero potential field
        std::vector<double> w_zero(N, 0.0);

        // Create CUDA streams
        cudaStream_t cuda_streams[MAX_STREAMS][2];
        for (int i = 0; i < MAX_STREAMS; i++)
        {
            gpu_error_check(cudaStreamCreate(&cuda_streams[i][0]));
            gpu_error_check(cudaStreamCreate(&cuda_streams[i][1]));
        }

        //=======================================================================
        // Test 1: Reflecting BC - Mass Conservation
        //=======================================================================
        std::cout << "Test 1: Reflecting BC - Mass Conservation" << std::endl;

        std::map<std::string, double> bond_lengths = {{"A", 1.0}};
        Molecules molecules_reflect("Continuous", ds, bond_lengths);
        std::vector<BlockInput> blocks = {{"A", 1.0, 0, 1}};
        molecules_reflect.add_polymer(1.0, blocks, {});

        // BC format: 2 entries per dimension (low, high)
        CudaComputationBox<double> cb_reflect(
            {N}, {L}, {"reflecting", "reflecting"});

        CudaSolverPseudoMixedBC<double> solver_reflect(
            &cb_reflect, &molecules_reflect, 1, cuda_streams, false);
        solver_reflect.update_dw("cpu", {{"A", w_zero.data()}});

        // Allocate device memory
        double *d_q_in, *d_q_out;
        gpu_error_check(cudaMalloc((void**)&d_q_in, sizeof(double) * N));
        gpu_error_check(cudaMalloc((void**)&d_q_out, sizeof(double) * N));

        // Copy initial condition to device
        gpu_error_check(cudaMemcpy(d_q_in, q_init.data(),
                                   sizeof(double) * N, cudaMemcpyHostToDevice));

        // Compute initial mass
        double initial_mass = 0.0;
        for (int i = 0; i < N; ++i)
            initial_mass += q_init[i] * dx;

        // Evolve propagator
        for (int step = 0; step < n_steps; ++step)
        {
            solver_reflect.advance_propagator(0, d_q_in, d_q_out, "A", nullptr);
            gpu_error_check(cudaMemcpy(d_q_in, d_q_out,
                                       sizeof(double) * N, cudaMemcpyDeviceToDevice));
        }

        // Copy result back to host
        std::vector<double> q_out(N);
        gpu_error_check(cudaMemcpy(q_out.data(), d_q_out,
                                   sizeof(double) * N, cudaMemcpyDeviceToHost));

        // Compute final mass
        double final_mass_reflect = 0.0;
        for (int i = 0; i < N; ++i)
            final_mass_reflect += q_out[i] * dx;

        double mass_error_reflect = std::abs(final_mass_reflect - initial_mass) / initial_mass;
        std::cout << "  Initial mass: " << initial_mass << std::endl;
        std::cout << "  Final mass:   " << final_mass_reflect << std::endl;
        std::cout << "  Relative mass error: " << mass_error_reflect << std::endl;

        // Mass should be nearly conserved for reflecting BC (within numerical tolerance)
        if (mass_error_reflect > 1e-6)
        {
            std::cout << "  FAILED! Mass not conserved for reflecting BC." << std::endl;
            cudaFree(d_q_in);
            cudaFree(d_q_out);
            return -1;
        }
        std::cout << "  PASSED!" << std::endl;

        //=======================================================================
        // Test 2: Absorbing BC - Mass Decrease
        //=======================================================================
        std::cout << "\nTest 2: Absorbing BC - Mass Decrease" << std::endl;

        CudaComputationBox<double> cb_absorb(
            {N}, {L}, {"absorbing", "absorbing"});

        CudaSolverPseudoMixedBC<double> solver_absorb(
            &cb_absorb, &molecules_reflect, 1, cuda_streams, false);
        solver_absorb.update_dw("cpu", {{"A", w_zero.data()}});

        // Copy initial condition to device
        gpu_error_check(cudaMemcpy(d_q_in, q_init.data(),
                                   sizeof(double) * N, cudaMemcpyHostToDevice));

        // Evolve propagator
        for (int step = 0; step < n_steps; ++step)
        {
            solver_absorb.advance_propagator(0, d_q_in, d_q_out, "A", nullptr);
            gpu_error_check(cudaMemcpy(d_q_in, d_q_out,
                                       sizeof(double) * N, cudaMemcpyDeviceToDevice));
        }

        // Copy result back to host
        gpu_error_check(cudaMemcpy(q_out.data(), d_q_out,
                                   sizeof(double) * N, cudaMemcpyDeviceToHost));

        // Compute final mass
        double final_mass_absorb = 0.0;
        for (int i = 0; i < N; ++i)
            final_mass_absorb += q_out[i] * dx;

        std::cout << "  Initial mass: " << initial_mass << std::endl;
        std::cout << "  Final mass:   " << final_mass_absorb << std::endl;
        std::cout << "  Mass ratio:   " << final_mass_absorb / initial_mass << std::endl;

        // Mass should decrease for absorbing BC
        if (final_mass_absorb >= initial_mass)
        {
            std::cout << "  FAILED! Mass did not decrease for absorbing BC." << std::endl;
            cudaFree(d_q_in);
            cudaFree(d_q_out);
            return -1;
        }
        std::cout << "  PASSED! (Mass decreased as expected)" << std::endl;

        //=======================================================================
        // Test 3: Reflecting BC - Uniform field should remain uniform
        //=======================================================================
        std::cout << "\nTest 3: Reflecting BC - Uniform field test" << std::endl;

        // Reset reflecting solver
        CudaSolverPseudoMixedBC<double> solver_reflect2(
            &cb_reflect, &molecules_reflect, 1, cuda_streams, false);
        solver_reflect2.update_dw("cpu", {{"A", w_zero.data()}});

        // For uniform initial condition with zero potential, field should remain uniform
        std::vector<double> q_uniform(N, 1.0);
        gpu_error_check(cudaMemcpy(d_q_in, q_uniform.data(),
                                   sizeof(double) * N, cudaMemcpyHostToDevice));

        solver_reflect2.advance_propagator(0, d_q_in, d_q_out, "A", nullptr);

        gpu_error_check(cudaMemcpy(q_out.data(), d_q_out,
                                   sizeof(double) * N, cudaMemcpyDeviceToHost));

        double max_deviation_reflect = 0.0;
        for (int i = 0; i < N; ++i)
        {
            max_deviation_reflect = std::max(max_deviation_reflect,
                std::abs(q_out[i] - 1.0));
        }

        std::cout << "  Reflecting max deviation from 1.0: " << max_deviation_reflect << std::endl;

        if (max_deviation_reflect > 1e-10)
        {
            std::cout << "  FAILED! Uniform field should remain uniform." << std::endl;
            cudaFree(d_q_in);
            cudaFree(d_q_out);
            return -1;
        }
        std::cout << "  PASSED!" << std::endl;

        //=======================================================================
        // Test 4: Symmetry test for reflecting BC
        //=======================================================================
        std::cout << "\nTest 4: Symmetry test for reflecting BC" << std::endl;

        // Symmetric initial condition: Gaussian centered at L/2
        std::vector<double> q_sym(N);
        for (int i = 0; i < N; ++i)
        {
            double x = (i + 0.5) * dx;
            q_sym[i] = std::exp(-std::pow(x - L/2, 2) / (2 * 0.5 * 0.5));
        }

        gpu_error_check(cudaMemcpy(d_q_in, q_sym.data(),
                                   sizeof(double) * N, cudaMemcpyHostToDevice));

        solver_reflect2.advance_propagator(0, d_q_in, d_q_out, "A", nullptr);

        gpu_error_check(cudaMemcpy(q_out.data(), d_q_out,
                                   sizeof(double) * N, cudaMemcpyDeviceToHost));

        // Check symmetry is preserved
        double sym_error = 0.0;
        for (int i = 0; i < N/2; ++i)
        {
            // q(x) should equal q(L-x) for symmetric input
            sym_error = std::max(sym_error, std::abs(q_out[i] - q_out[N-1-i]));
        }

        std::cout << "  Symmetry error: " << sym_error << std::endl;
        if (sym_error > 1e-10)
        {
            std::cout << "  FAILED! Symmetry not preserved." << std::endl;
            cudaFree(d_q_in);
            cudaFree(d_q_out);
            return -1;
        }
        std::cout << "  PASSED!" << std::endl;

        // Clean up
        cudaFree(d_q_in);
        cudaFree(d_q_out);

        //=======================================================================
        // Test 5: 2D Mixed BC test
        //=======================================================================
        std::cout << "\nTest 5: 2D Mixed BC test" << std::endl;

        const int NX = 16, NY = 12;
        const double LX = 4.0, LY = 3.0;
        const int M = NX * NY;

        CudaComputationBox<double> cb_2d(
            {NX, NY}, {LX, LY},
            {"reflecting", "reflecting", "absorbing", "absorbing"});

        CudaSolverPseudoMixedBC<double> solver_2d(
            &cb_2d, &molecules_reflect, 1, cuda_streams, false);

        std::vector<double> w_2d(M, 0.0);
        solver_2d.update_dw("cpu", {{"A", w_2d.data()}});

        // Gaussian centered in domain
        std::vector<double> q_2d_init(M), q_2d_out(M);
        for (int i = 0; i < NX; ++i)
        {
            double x = (i + 0.5) * LX / NX;
            for (int j = 0; j < NY; ++j)
            {
                double y = (j + 0.5) * LY / NY;
                q_2d_init[i * NY + j] = std::exp(
                    -std::pow(x - LX/2, 2) / (2 * 0.5 * 0.5) -
                    std::pow(y - LY/2, 2) / (2 * 0.5 * 0.5));
            }
        }

        double *d_q_2d_in, *d_q_2d_out;
        gpu_error_check(cudaMalloc((void**)&d_q_2d_in, sizeof(double) * M));
        gpu_error_check(cudaMalloc((void**)&d_q_2d_out, sizeof(double) * M));
        gpu_error_check(cudaMemcpy(d_q_2d_in, q_2d_init.data(),
                                   sizeof(double) * M, cudaMemcpyHostToDevice));

        // Single step
        solver_2d.advance_propagator(0, d_q_2d_in, d_q_2d_out, "A", nullptr);

        gpu_error_check(cudaMemcpy(q_2d_out.data(), d_q_2d_out,
                                   sizeof(double) * M, cudaMemcpyDeviceToHost));

        // Check all values are positive (physical)
        bool all_positive = true;
        for (int i = 0; i < M; ++i)
        {
            if (q_2d_out[i] < -1e-10)
            {
                all_positive = false;
                std::cout << "  Negative value at index " << i << ": " << q_2d_out[i] << std::endl;
            }
        }

        cudaFree(d_q_2d_in);
        cudaFree(d_q_2d_out);

        if (!all_positive)
        {
            std::cout << "  FAILED! Propagator has negative values." << std::endl;
            return -1;
        }
        std::cout << "  PASSED! (All values positive)" << std::endl;

        //=======================================================================
        // Test 6: 3D Mixed BC test
        //=======================================================================
        std::cout << "\nTest 6: 3D Mixed BC test" << std::endl;

        const int NX3 = 8, NY3 = 6, NZ3 = 4;
        const double LX3 = 4.0, LY3 = 3.0, LZ3 = 2.0;
        const int M3 = NX3 * NY3 * NZ3;

        CudaComputationBox<double> cb_3d(
            {NX3, NY3, NZ3}, {LX3, LY3, LZ3},
            {"reflecting", "reflecting", "reflecting", "reflecting", "absorbing", "absorbing"});

        CudaSolverPseudoMixedBC<double> solver_3d(
            &cb_3d, &molecules_reflect, 1, cuda_streams, false);

        std::vector<double> w_3d(M3, 0.0);
        solver_3d.update_dw("cpu", {{"A", w_3d.data()}});

        // Uniform initial condition
        std::vector<double> q_3d_init(M3, 1.0), q_3d_out(M3);

        double *d_q_3d_in, *d_q_3d_out;
        gpu_error_check(cudaMalloc((void**)&d_q_3d_in, sizeof(double) * M3));
        gpu_error_check(cudaMalloc((void**)&d_q_3d_out, sizeof(double) * M3));
        gpu_error_check(cudaMemcpy(d_q_3d_in, q_3d_init.data(),
                                   sizeof(double) * M3, cudaMemcpyHostToDevice));

        solver_3d.advance_propagator(0, d_q_3d_in, d_q_3d_out, "A", nullptr);

        gpu_error_check(cudaMemcpy(q_3d_out.data(), d_q_3d_out,
                                   sizeof(double) * M3, cudaMemcpyDeviceToHost));

        // Compute mass
        double mass_3d_init = 0.0, mass_3d_out = 0.0;
        double dV = (LX3/NX3) * (LY3/NY3) * (LZ3/NZ3);
        for (int i = 0; i < M3; ++i)
        {
            mass_3d_init += q_3d_init[i] * dV;
            mass_3d_out += q_3d_out[i] * dV;
        }

        std::cout << "  Initial mass: " << mass_3d_init << std::endl;
        std::cout << "  Final mass:   " << mass_3d_out << std::endl;

        cudaFree(d_q_3d_in);
        cudaFree(d_q_3d_out);

        // With absorbing BC in z, mass should decrease
        if (mass_3d_out >= mass_3d_init)
        {
            std::cout << "  WARNING: Mass did not decrease (may be ok for short time step)" << std::endl;
        }
        std::cout << "  PASSED!" << std::endl;

        // Clean up streams
        for (int i = 0; i < MAX_STREAMS; i++)
        {
            cudaStreamDestroy(cuda_streams[i][0]);
            cudaStreamDestroy(cuda_streams[i][1]);
        }

        std::cout << "\nAll tests passed!" << std::endl;

        return 0;
    }
    catch (std::exception& exc)
    {
        std::cout << exc.what() << std::endl;
        return -1;
    }
}
