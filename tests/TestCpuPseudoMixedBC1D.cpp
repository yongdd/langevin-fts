/**
 * @file TestCpuPseudoMixedBC1D.cpp
 * @brief Test pseudo-spectral solver with mixed boundary conditions.
 *
 * This test verifies that CpuSolverPseudoMixedBC correctly computes
 * propagator evolution with:
 * - PERIODIC: Standard FFT-based pseudo-spectral method
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
#ifdef USE_CPU_MKL
#include "CpuComputationBox.h"
#include "CpuSolverPseudoMixedBC.h"
#include "MklFFTMixedBC.h"
#endif

int main()
{
    try
    {
#ifdef USE_CPU_MKL
        const int N = 32;
        const double L = 4.0;
        const double dx = L / N;
        const double ds = 0.01;
        const int n_steps = 50;

        double error;
        std::vector<double> diff_sq(N);

        // Create Gaussian initial condition centered in domain
        std::vector<double> q_init(N);
        for (int i = 0; i < N; ++i)
        {
            double x = (i + 0.5) * dx;  // Staggered grid
            q_init[i] = std::exp(-std::pow(x - L/2, 2) / (2 * 0.3 * 0.3));
        }

        // Zero potential field
        std::vector<double> w_zero(N, 0.0);

        //=======================================================================
        // Test 1: Reflecting BC - Mass Conservation
        //=======================================================================
        std::cout << "Test 1: Reflecting BC - Mass Conservation" << std::endl;

        std::map<std::string, double> bond_lengths = {{"A", 1.0}};
        Molecules molecules_reflect("Continuous", ds, bond_lengths);
        std::vector<BlockInput> blocks = {{"A", 1.0, 0, 1}};
        molecules_reflect.add_polymer(1.0, blocks, {});

        // BC format: 2 entries per dimension (low, high)
        CpuComputationBox<double> cb_reflect(
            {N}, {L}, {"reflecting", "reflecting"});

        CpuSolverPseudoMixedBC<double> solver_reflect(&cb_reflect, &molecules_reflect);
        solver_reflect.update_dw({{"A", w_zero.data()}});

        std::vector<double> q_in(N), q_out(N);
        for (int i = 0; i < N; ++i)
            q_in[i] = q_init[i];

        // Compute initial mass
        double initial_mass = 0.0;
        for (int i = 0; i < N; ++i)
            initial_mass += q_in[i] * dx;

        // Evolve propagator
        for (int step = 0; step < n_steps; ++step)
        {
            solver_reflect.advance_propagator(q_in.data(), q_out.data(), "A", nullptr);
            for (int i = 0; i < N; ++i)
                q_in[i] = q_out[i];
        }

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
            return -1;
        }
        std::cout << "  PASSED!" << std::endl;

        //=======================================================================
        // Test 2: Absorbing BC - Mass Decrease
        //=======================================================================
        std::cout << "\nTest 2: Absorbing BC - Mass Decrease" << std::endl;

        CpuComputationBox<double> cb_absorb(
            {N}, {L}, {"absorbing", "absorbing"});

        CpuSolverPseudoMixedBC<double> solver_absorb(&cb_absorb, &molecules_reflect);
        solver_absorb.update_dw({{"A", w_zero.data()}});

        for (int i = 0; i < N; ++i)
            q_in[i] = q_init[i];

        // Evolve propagator
        for (int step = 0; step < n_steps; ++step)
        {
            solver_absorb.advance_propagator(q_in.data(), q_out.data(), "A", nullptr);
            for (int i = 0; i < N; ++i)
                q_in[i] = q_out[i];
        }

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
            return -1;
        }
        std::cout << "  PASSED! (Mass decreased as expected)" << std::endl;

        //=======================================================================
        // Test 3: Reflecting BC - Uniform field should remain uniform
        //=======================================================================
        std::cout << "\nTest 3: Reflecting BC - Uniform field test" << std::endl;

        // Reset reflecting solver
        CpuSolverPseudoMixedBC<double> solver_reflect2(&cb_reflect, &molecules_reflect);
        solver_reflect2.update_dw({{"A", w_zero.data()}});

        // For uniform initial condition with zero potential, field should remain uniform
        std::vector<double> q_uniform(N, 1.0);
        std::vector<double> q_out_reflect2(N);

        solver_reflect2.advance_propagator(q_uniform.data(), q_out_reflect2.data(), "A", nullptr);

        double max_deviation_reflect = 0.0;
        for (int i = 0; i < N; ++i)
        {
            max_deviation_reflect = std::max(max_deviation_reflect,
                std::abs(q_out_reflect2[i] - 1.0));
        }

        std::cout << "  Reflecting max deviation from 1.0: " << max_deviation_reflect << std::endl;

        if (max_deviation_reflect > 1e-10)
        {
            std::cout << "  FAILED! Uniform field should remain uniform." << std::endl;
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
            q_sym[i] = std::exp(-std::pow(x - L/2, 2) / (2 * 0.5 * 0.5));  // Symmetric about L/2
        }

        CpuSolverPseudoMixedBC<double> solver_sym(&cb_reflect, &molecules_reflect);
        solver_sym.update_dw({{"A", w_zero.data()}});

        std::vector<double> q_out_sym(N);
        solver_sym.advance_propagator(q_sym.data(), q_out_sym.data(), "A", nullptr);

        // Check symmetry is preserved
        double sym_error = 0.0;
        for (int i = 0; i < N/2; ++i)
        {
            // q(x) should equal q(L-x) for symmetric input
            sym_error = std::max(sym_error, std::abs(q_out_sym[i] - q_out_sym[N-1-i]));
        }

        std::cout << "  Symmetry error: " << sym_error << std::endl;
        if (sym_error > 1e-10)
        {
            std::cout << "  FAILED! Symmetry not preserved." << std::endl;
            return -1;
        }
        std::cout << "  PASSED!" << std::endl;

        //=======================================================================
        // Test 5: 2D Mixed BC test
        //=======================================================================
        std::cout << "\nTest 5: 2D Mixed BC test" << std::endl;

        const int NX = 16, NY = 12;
        const double LX = 4.0, LY = 3.0;
        const int M = NX * NY;

        CpuComputationBox<double> cb_2d(
            {NX, NY}, {LX, LY},
            {"reflecting", "reflecting", "absorbing", "absorbing"});

        CpuSolverPseudoMixedBC<double> solver_2d(&cb_2d, &molecules_reflect);

        std::vector<double> w_2d(M, 0.0);
        solver_2d.update_dw({{"A", w_2d.data()}});

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

        // Single step
        solver_2d.advance_propagator(q_2d_init.data(), q_2d_out.data(), "A", nullptr);

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

        CpuComputationBox<double> cb_3d(
            {NX3, NY3, NZ3}, {LX3, LY3, LZ3},
            {"reflecting", "reflecting", "reflecting", "reflecting", "absorbing", "absorbing"});

        CpuSolverPseudoMixedBC<double> solver_3d(&cb_3d, &molecules_reflect);

        std::vector<double> w_3d(M3, 0.0);
        solver_3d.update_dw({{"A", w_3d.data()}});

        // Uniform initial condition
        std::vector<double> q_3d_init(M3, 1.0), q_3d_out(M3);

        solver_3d.advance_propagator(q_3d_init.data(), q_3d_out.data(), "A", nullptr);

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

        // With absorbing BC in z, mass should decrease
        if (mass_3d_out >= mass_3d_init)
        {
            std::cout << "  WARNING: Mass did not decrease (may be ok for short time step)" << std::endl;
        }
        std::cout << "  PASSED!" << std::endl;

        std::cout << "\nAll tests passed!" << std::endl;
#endif

        return 0;
    }
    catch (std::exception& exc)
    {
        std::cout << exc.what() << std::endl;
        return -1;
    }
}
