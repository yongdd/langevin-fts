/**
 * @file TestCpuPseudoMixedBC2D.cpp
 * @brief Test 2D pseudo-spectral solver with mixed boundary conditions.
 *
 * This test verifies that CpuSolverPseudoMixedBC correctly computes
 * 2D propagator evolution with various boundary condition combinations:
 * - All Reflecting: Mass should be conserved
 * - All Absorbing: Mass should decrease
 * - Mixed (Reflecting/Absorbing): Partial mass loss
 *
 * Physical meaning:
 * - REFLECTING: Impenetrable wall (zero flux)
 * - ABSORBING: Surface that absorbs polymer ends (zero value)
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
        const int NX = 24;
        const int NY = 20;
        const int M = NX * NY;
        const double LX = 4.0;
        const double LY = 3.0;
        const double dx = LX / NX;
        const double dy = LY / NY;
        const double ds = 0.01;
        const int n_steps = 30;

        double error;

        // Create Gaussian initial condition centered in domain
        std::vector<double> q_init(M);
        for (int i = 0; i < NX; ++i)
        {
            double x = (i + 0.5) * dx;
            for (int j = 0; j < NY; ++j)
            {
                double y = (j + 0.5) * dy;
                q_init[i * NY + j] = std::exp(
                    -std::pow(x - LX/2, 2) / (2 * 0.4 * 0.4) -
                    std::pow(y - LY/2, 2) / (2 * 0.4 * 0.4));
            }
        }

        // Zero potential field
        std::vector<double> w_zero(M, 0.0);

        // Set up molecules
        std::map<std::string, double> bond_lengths = {{"A", 1.0}};
        Molecules molecules("Continuous", ds, bond_lengths);
        std::vector<BlockInput> blocks = {{"A", 1.0, 0, 1}};
        molecules.add_polymer(1.0, blocks, {});

        // Compute volume element
        double dV = dx * dy;

        //=======================================================================
        // Test 1: All Reflecting BC - Mass Conservation
        //=======================================================================
        std::cout << "Test 1: 2D All Reflecting BC - Mass Conservation" << std::endl;

        CpuComputationBox<double> cb_reflect(
            {NX, NY}, {LX, LY}, {"reflecting", "reflecting", "reflecting", "reflecting"});

        CpuSolverPseudoMixedBC<double> solver_reflect(&cb_reflect, &molecules);
        solver_reflect.update_dw({{"A", w_zero.data()}});

        std::vector<double> q_in(M), q_out(M);
        for (int i = 0; i < M; ++i)
            q_in[i] = q_init[i];

        // Compute initial mass
        double initial_mass = 0.0;
        for (int i = 0; i < M; ++i)
            initial_mass += q_in[i] * dV;

        // Evolve propagator
        for (int step = 0; step < n_steps; ++step)
        {
            solver_reflect.advance_propagator(q_in.data(), q_out.data(), "A", nullptr);
            for (int i = 0; i < M; ++i)
                q_in[i] = q_out[i];
        }

        // Compute final mass
        double final_mass_reflect = 0.0;
        for (int i = 0; i < M; ++i)
            final_mass_reflect += q_out[i] * dV;

        double mass_error_reflect = std::abs(final_mass_reflect - initial_mass) / initial_mass;
        std::cout << "  Initial mass: " << initial_mass << std::endl;
        std::cout << "  Final mass:   " << final_mass_reflect << std::endl;
        std::cout << "  Relative mass error: " << mass_error_reflect << std::endl;

        if (mass_error_reflect > 1e-6)
        {
            std::cout << "  FAILED! Mass not conserved for reflecting BC." << std::endl;
            return -1;
        }
        std::cout << "  PASSED!" << std::endl;

        //=======================================================================
        // Test 2: All Absorbing BC - Mass Decrease
        //=======================================================================
        std::cout << "\nTest 2: 2D All Absorbing BC - Mass Decrease" << std::endl;

        CpuComputationBox<double> cb_absorb(
            {NX, NY}, {LX, LY}, {"absorbing", "absorbing", "absorbing", "absorbing"});

        CpuSolverPseudoMixedBC<double> solver_absorb(&cb_absorb, &molecules);
        solver_absorb.update_dw({{"A", w_zero.data()}});

        for (int i = 0; i < M; ++i)
            q_in[i] = q_init[i];

        // Evolve propagator
        for (int step = 0; step < n_steps; ++step)
        {
            solver_absorb.advance_propagator(q_in.data(), q_out.data(), "A", nullptr);
            for (int i = 0; i < M; ++i)
                q_in[i] = q_out[i];
        }

        // Compute final mass
        double final_mass_absorb = 0.0;
        for (int i = 0; i < M; ++i)
            final_mass_absorb += q_out[i] * dV;

        std::cout << "  Initial mass: " << initial_mass << std::endl;
        std::cout << "  Final mass:   " << final_mass_absorb << std::endl;
        std::cout << "  Mass ratio:   " << final_mass_absorb / initial_mass << std::endl;

        if (final_mass_absorb >= initial_mass)
        {
            std::cout << "  FAILED! Mass did not decrease for absorbing BC." << std::endl;
            return -1;
        }
        std::cout << "  PASSED! (Mass decreased as expected)" << std::endl;

        //=======================================================================
        // Test 3: Mixed BC (Reflecting X, Absorbing Y)
        //=======================================================================
        std::cout << "\nTest 3: 2D Mixed BC (Reflecting X, Absorbing Y)" << std::endl;

        CpuComputationBox<double> cb_mixed1(
            {NX, NY}, {LX, LY}, {"reflecting", "reflecting", "absorbing", "absorbing"});

        CpuSolverPseudoMixedBC<double> solver_mixed1(&cb_mixed1, &molecules);
        solver_mixed1.update_dw({{"A", w_zero.data()}});

        for (int i = 0; i < M; ++i)
            q_in[i] = q_init[i];

        // Evolve propagator
        for (int step = 0; step < n_steps; ++step)
        {
            solver_mixed1.advance_propagator(q_in.data(), q_out.data(), "A", nullptr);
            for (int i = 0; i < M; ++i)
                q_in[i] = q_out[i];
        }

        // Compute final mass
        double final_mass_mixed1 = 0.0;
        for (int i = 0; i < M; ++i)
            final_mass_mixed1 += q_out[i] * dV;

        std::cout << "  Initial mass: " << initial_mass << std::endl;
        std::cout << "  Final mass:   " << final_mass_mixed1 << std::endl;
        std::cout << "  Mass ratio:   " << final_mass_mixed1 / initial_mass << std::endl;

        // Mixed BC should lose less mass than all absorbing
        if (final_mass_mixed1 >= initial_mass)
        {
            std::cout << "  FAILED! Mass should decrease with absorbing BC in Y." << std::endl;
            return -1;
        }
        if (final_mass_mixed1 <= final_mass_absorb)
        {
            std::cout << "  FAILED! Mixed BC should lose less mass than all-absorbing." << std::endl;
            return -1;
        }
        std::cout << "  PASSED!" << std::endl;

        //=======================================================================
        // Test 4: Mixed BC (Absorbing X, Reflecting Y)
        //=======================================================================
        std::cout << "\nTest 4: 2D Mixed BC (Absorbing X, Reflecting Y)" << std::endl;

        CpuComputationBox<double> cb_mixed2(
            {NX, NY}, {LX, LY}, {"absorbing", "absorbing", "reflecting", "reflecting"});

        CpuSolverPseudoMixedBC<double> solver_mixed2(&cb_mixed2, &molecules);
        solver_mixed2.update_dw({{"A", w_zero.data()}});

        for (int i = 0; i < M; ++i)
            q_in[i] = q_init[i];

        // Evolve propagator
        for (int step = 0; step < n_steps; ++step)
        {
            solver_mixed2.advance_propagator(q_in.data(), q_out.data(), "A", nullptr);
            for (int i = 0; i < M; ++i)
                q_in[i] = q_out[i];
        }

        // Compute final mass
        double final_mass_mixed2 = 0.0;
        for (int i = 0; i < M; ++i)
            final_mass_mixed2 += q_out[i] * dV;

        std::cout << "  Initial mass: " << initial_mass << std::endl;
        std::cout << "  Final mass:   " << final_mass_mixed2 << std::endl;
        std::cout << "  Mass ratio:   " << final_mass_mixed2 / initial_mass << std::endl;

        if (final_mass_mixed2 >= initial_mass)
        {
            std::cout << "  FAILED! Mass should decrease with absorbing BC in X." << std::endl;
            return -1;
        }
        std::cout << "  PASSED!" << std::endl;

        //=======================================================================
        // Test 5: Uniform field should remain uniform (Reflecting BC)
        //=======================================================================
        std::cout << "\nTest 5: 2D Uniform field test (Reflecting BC)" << std::endl;

        std::vector<double> q_uniform(M, 1.0);
        std::vector<double> q_out_uniform(M);

        CpuSolverPseudoMixedBC<double> solver_uniform(&cb_reflect, &molecules);
        solver_uniform.update_dw({{"A", w_zero.data()}});

        solver_uniform.advance_propagator(q_uniform.data(), q_out_uniform.data(), "A", nullptr);

        double max_deviation = 0.0;
        for (int i = 0; i < M; ++i)
        {
            max_deviation = std::max(max_deviation, std::abs(q_out_uniform[i] - 1.0));
        }

        std::cout << "  Max deviation from 1.0: " << max_deviation << std::endl;

        if (max_deviation > 1e-10)
        {
            std::cout << "  FAILED! Uniform field should remain uniform." << std::endl;
            return -1;
        }
        std::cout << "  PASSED!" << std::endl;

        //=======================================================================
        // Test 6: Symmetry test (symmetric about center in both directions)
        //=======================================================================
        std::cout << "\nTest 6: 2D Symmetry test (Reflecting BC)" << std::endl;

        // Create symmetric initial condition
        std::vector<double> q_sym(M);
        for (int i = 0; i < NX; ++i)
        {
            double x = (i + 0.5) * dx;
            for (int j = 0; j < NY; ++j)
            {
                double y = (j + 0.5) * dy;
                q_sym[i * NY + j] = std::exp(
                    -std::pow(x - LX/2, 2) / (2 * 0.5 * 0.5) -
                    std::pow(y - LY/2, 2) / (2 * 0.5 * 0.5));
            }
        }

        std::vector<double> q_out_sym(M);

        CpuSolverPseudoMixedBC<double> solver_sym(&cb_reflect, &molecules);
        solver_sym.update_dw({{"A", w_zero.data()}});

        solver_sym.advance_propagator(q_sym.data(), q_out_sym.data(), "A", nullptr);

        // Check symmetry: q(i,j) should equal q(NX-1-i, NY-1-j) for centered Gaussian
        double sym_error = 0.0;
        for (int i = 0; i < NX/2; ++i)
        {
            for (int j = 0; j < NY/2; ++j)
            {
                int idx1 = i * NY + j;
                int idx2 = (NX - 1 - i) * NY + (NY - 1 - j);
                sym_error = std::max(sym_error, std::abs(q_out_sym[idx1] - q_out_sym[idx2]));
            }
        }

        std::cout << "  Symmetry error: " << sym_error << std::endl;

        if (sym_error > 1e-10)
        {
            std::cout << "  FAILED! Symmetry not preserved." << std::endl;
            return -1;
        }
        std::cout << "  PASSED!" << std::endl;

        //=======================================================================
        // Test 7: All values should remain positive
        //=======================================================================
        std::cout << "\nTest 7: 2D Positivity test" << std::endl;

        for (int i = 0; i < M; ++i)
            q_in[i] = q_init[i];

        solver_mixed1.advance_propagator(q_in.data(), q_out.data(), "A", nullptr);

        bool all_positive = true;
        double min_val = q_out[0];
        for (int i = 0; i < M; ++i)
        {
            min_val = std::min(min_val, q_out[i]);
            if (q_out[i] < -1e-10)
            {
                all_positive = false;
            }
        }

        std::cout << "  Minimum value: " << min_val << std::endl;

        if (!all_positive)
        {
            std::cout << "  FAILED! Propagator has negative values." << std::endl;
            return -1;
        }
        std::cout << "  PASSED! (All values positive)" << std::endl;

        std::cout << "\nAll 2D pseudo-spectral tests passed!" << std::endl;
#endif

        return 0;
    }
    catch (std::exception& exc)
    {
        std::cout << exc.what() << std::endl;
        return -1;
    }
}
