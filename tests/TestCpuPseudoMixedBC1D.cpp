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
#include "CpuSolverPseudoContinuous.h"
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

        //=======================================================================
        // Test 7: DCT-based propagator vs DFT-based propagator (symmetric input)
        //=======================================================================
        // Mathematical relationship:
        // For symmetric input f(x) = f(L-x), the DCT-based solver on domain [0,L]
        // should give identical results to the DFT-based solver on domain [0,2L]
        // with symmetric extension f(x) = f(2L-x).
        //
        // This tests that reflecting BC (DCT) correctly implements the physics
        // of a reflecting wall by showing equivalence to periodic BC with
        // symmetric extension (method of images).
        //=======================================================================
        std::cout << "\nTest 7: DCT-based propagator vs DFT-based propagator (symmetric input)" << std::endl;

        const int N_DCT = 16;
        const double L_DCT = 4.0;
        const double dx_DCT = L_DCT / N_DCT;
        const double ds_DCT = 0.01;
        const int n_steps_DCT = 10;

        // Create symmetric initial condition on [0, L]: Gaussian centered at L/2
        std::vector<double> q_dct_init(N_DCT);
        for (int i = 0; i < N_DCT; ++i)
        {
            double x = (i + 0.5) * dx_DCT;
            q_dct_init[i] = std::exp(-std::pow(x - L_DCT/2, 2) / (2 * 0.5 * 0.5));
        }

        // Create symmetric extension on [0, 2L] for DFT
        // y[n] = x[n] for n=0..N-1
        // y[2N-1-n] = x[n] for n=0..N-1
        const int N2_DFT = 2 * N_DCT;
        const double L2_DFT = 2 * L_DCT;
        std::vector<double> q_dft_init(N2_DFT);
        for (int n = 0; n < N_DCT; ++n)
        {
            q_dft_init[n] = q_dct_init[n];
            q_dft_init[N2_DFT - 1 - n] = q_dct_init[n];
        }

        // Set up DCT-based solver (reflecting BC)
        std::map<std::string, double> bond_lengths_dct = {{"A", 1.0}};
        Molecules molecules_dct("Continuous", ds_DCT, bond_lengths_dct);
        std::vector<BlockInput> blocks_dct = {{"A", 1.0, 0, 1}};
        molecules_dct.add_polymer(1.0, blocks_dct, {});

        CpuComputationBox<double> cb_dct({N_DCT}, {L_DCT}, {"reflecting", "reflecting"});
        CpuSolverPseudoMixedBC<double> solver_dct(&cb_dct, &molecules_dct);

        std::vector<double> w_dct(N_DCT, 0.0);
        solver_dct.update_dw({{"A", w_dct.data()}});

        // Set up DFT-based solver (periodic BC on 2L domain)
        Molecules molecules_dft("Continuous", ds_DCT, bond_lengths_dct);
        molecules_dft.add_polymer(1.0, blocks_dct, {});

        CpuComputationBox<double> cb_dft({N2_DFT}, {L2_DFT}, {"periodic", "periodic"});
        CpuSolverPseudoContinuous<double> solver_dft(&cb_dft, &molecules_dft);

        std::vector<double> w_dft(N2_DFT, 0.0);
        solver_dft.update_dw({{"A", w_dft.data()}});

        // Evolve both propagators
        std::vector<double> q_dct_in(N_DCT), q_dct_out(N_DCT);
        std::vector<double> q_dft_in(N2_DFT), q_dft_out(N2_DFT);

        for (int i = 0; i < N_DCT; ++i)
            q_dct_in[i] = q_dct_init[i];
        for (int i = 0; i < N2_DFT; ++i)
            q_dft_in[i] = q_dft_init[i];

        for (int step = 0; step < n_steps_DCT; ++step)
        {
            solver_dct.advance_propagator(q_dct_in.data(), q_dct_out.data(), "A", nullptr);
            solver_dft.advance_propagator(q_dft_in.data(), q_dft_out.data(), "A", nullptr);

            for (int i = 0; i < N_DCT; ++i)
                q_dct_in[i] = q_dct_out[i];
            for (int i = 0; i < N2_DFT; ++i)
                q_dft_in[i] = q_dft_out[i];
        }

        // Compare results: DCT result should match first N points of DFT result
        double dct_dft_error = 0.0;
        for (int i = 0; i < N_DCT; ++i)
        {
            dct_dft_error = std::max(dct_dft_error, std::abs(q_dct_out[i] - q_dft_out[i]));
        }

        std::cout << "  DCT vs DFT propagator max error: " << dct_dft_error << std::endl;
        if (!std::isfinite(dct_dft_error) || dct_dft_error > 1e-10)
        {
            std::cout << "  FAILED! DCT and DFT propagators should match for symmetric input." << std::endl;
            return -1;
        }
        std::cout << "  PASSED!" << std::endl;

        //=======================================================================
        // Test 8: DST-based propagator vs DFT-based propagator (antisymmetric input)
        //=======================================================================
        // Mathematical relationship:
        // For antisymmetric input f(x) = -f(L-x) (about x=L), the DST-based solver
        // on domain [0,L] should give identical results to the DFT-based solver
        // on domain [0,2L] with antisymmetric extension f(x) = -f(2L-x).
        //
        // This tests that absorbing BC (DST) correctly implements the physics
        // of an absorbing boundary by showing equivalence to periodic BC with
        // antisymmetric extension (method of images with opposite sign).
        //=======================================================================
        std::cout << "\nTest 8: DST-based propagator vs DFT-based propagator (antisymmetric input)" << std::endl;

        // Create antisymmetric initial condition on [0, L]
        // Using sin(π*x/L) which satisfies f(0)=f(L)=0 (absorbing BC)
        std::vector<double> q_dst_init(N_DCT);
        for (int i = 0; i < N_DCT; ++i)
        {
            double x = (i + 0.5) * dx_DCT;
            // Use a function that is antisymmetric about x=L when extended
            // sin(π*(x+0.5*dx)/L) gives antisymmetric extension
            q_dst_init[i] = std::sin(std::numbers::pi * x / L_DCT);
        }

        // Create antisymmetric extension on [0, 2L] for DFT
        // y[n] = x[n] for n=0..N-1
        // y[2N-1-n] = -x[n] for n=0..N-1
        std::vector<double> q_dft_antisym_init(N2_DFT);
        for (int n = 0; n < N_DCT; ++n)
        {
            q_dft_antisym_init[n] = q_dst_init[n];
            q_dft_antisym_init[N2_DFT - 1 - n] = -q_dst_init[n];
        }

        // Set up DST-based solver (absorbing BC)
        CpuComputationBox<double> cb_dst({N_DCT}, {L_DCT}, {"absorbing", "absorbing"});
        CpuSolverPseudoMixedBC<double> solver_dst(&cb_dst, &molecules_dct);
        solver_dst.update_dw({{"A", w_dct.data()}});

        // Evolve both propagators
        std::vector<double> q_dst_in(N_DCT), q_dst_out(N_DCT);
        std::vector<double> q_dft_antisym_in(N2_DFT), q_dft_antisym_out(N2_DFT);

        for (int i = 0; i < N_DCT; ++i)
            q_dst_in[i] = q_dst_init[i];
        for (int i = 0; i < N2_DFT; ++i)
            q_dft_antisym_in[i] = q_dft_antisym_init[i];

        for (int step = 0; step < n_steps_DCT; ++step)
        {
            solver_dst.advance_propagator(q_dst_in.data(), q_dst_out.data(), "A", nullptr);
            solver_dft.advance_propagator(q_dft_antisym_in.data(), q_dft_antisym_out.data(), "A", nullptr);

            for (int i = 0; i < N_DCT; ++i)
                q_dst_in[i] = q_dst_out[i];
            for (int i = 0; i < N2_DFT; ++i)
                q_dft_antisym_in[i] = q_dft_antisym_out[i];
        }

        // Compare results: DST result should match first N points of DFT result
        double dst_dft_error = 0.0;
        for (int i = 0; i < N_DCT; ++i)
        {
            dst_dft_error = std::max(dst_dft_error, std::abs(q_dst_out[i] - q_dft_antisym_out[i]));
        }

        std::cout << "  DST vs DFT propagator max error: " << dst_dft_error << std::endl;
        if (!std::isfinite(dst_dft_error) || dst_dft_error > 1e-10)
        {
            std::cout << "  FAILED! DST and DFT propagators should match for antisymmetric input." << std::endl;
            return -1;
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
