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
 * @file TestCpuFFTMixedBC.cpp
 * @brief Test DCT-II/III and DST-II/III transforms for mixed boundary conditions.
 *
 * This test verifies that FftwFFT correctly implements:
 * - DCT-II forward / DCT-III backward for reflecting (Neumann) BCs
 * - DST-II forward / DST-III backward for absorbing (Dirichlet) BCs
 *
 * Tests cover 1D, 2D, and 3D cases.
 *
 * The primary verification is round-trip: backward(forward(x)) = x
 *
 * Also verifies the mathematical relationship between DCT/DST and DFT:
 * - DCT-2 of N points equals Re(DFT) of 2N-point symmetric extension
 * - DST-2 of N points equals -Im(DFT) of 2N-point antisymmetric extension
 */

#include <iostream>
#include <cmath>
#include <numbers>
#include <complex>
#include <algorithm>
#include <random>
#include <vector>

#include "Exception.h"
#include "ComputationBox.h"
#ifdef USE_CPU_FFTW
#include "FftwFFT.h"
#endif

int main()
{
    try
    {
#ifdef USE_CPU_FFTW
        const int N = 12;
        double error;
        std::vector<double> diff_sq(N);

        // Test input data
        double data_init[N];
        for (int i = 0; i < N; ++i)
            data_init[i] = i + 1;  // [1, 2, 3, ..., 12]

        double data_r[N];
        double data_k[N];

        //=======================================================================
        // Test 1: DCT-II / DCT-III (Reflecting BC) Round-trip
        //=======================================================================
        std::cout << "Test 1: DCT-II/III (Reflecting BC) Round-trip" << std::endl;

        std::array<int, 1> nx_1d = {N};
        std::array<BoundaryCondition, 1> bc_reflect = {BoundaryCondition::REFLECTING};
        FftwFFT<double, 1> fft_dct(nx_1d, bc_reflect);

        // Forward DCT-II
        for (int i = 0; i < N; ++i)
            data_r[i] = data_init[i];

        fft_dct.forward(data_r, data_k);

        // Backward DCT-III (inverse) - should recover original
        fft_dct.backward(data_k, data_r);

        for (int i = 0; i < N; ++i)
            diff_sq[i] = std::pow(data_r[i] - data_init[i], 2);

        error = std::sqrt(*std::max_element(diff_sq.begin(), diff_sq.end()));
        std::cout << "  Round-trip Error: " << error << std::endl;
        if (!std::isfinite(error) || error > 1e-7)
        {
            std::cout << "  FAILED!" << std::endl;
            return -1;
        }
        std::cout << "  PASSED!" << std::endl;

        //=======================================================================
        // Test 2: DST-II / DST-III (Absorbing BC)
        //=======================================================================
        std::cout << "\nTest 2: DST-II/III (Absorbing BC)" << std::endl;

        std::array<BoundaryCondition, 1> bc_absorb = {BoundaryCondition::ABSORBING};
        FftwFFT<double, 1> fft_dst(nx_1d, bc_absorb);

        // Forward DST-II
        for (int i = 0; i < N; ++i)
            data_r[i] = data_init[i];

        fft_dst.forward(data_r, data_k);

        // Backward DST-III (inverse) - should recover original data
        fft_dst.backward(data_k, data_r);

        std::cout << "DST-II/III Round-trip:" << std::endl;
        for (int i = 0; i < N; ++i)
            diff_sq[i] = std::pow(data_r[i] - data_init[i], 2);

        error = std::sqrt(*std::max_element(diff_sq.begin(), diff_sq.end()));
        std::cout << "  Round-trip Error: " << error << std::endl;
        if (!std::isfinite(error) || error > 1e-7)
        {
            std::cout << "  FAILED!" << std::endl;
            return -1;
        }
        std::cout << "  PASSED!" << std::endl;

        //=======================================================================
        // Test 3: Random data round-trip test
        //=======================================================================
        std::cout << "\nTest 3: Random data round-trip" << std::endl;

        std::mt19937 gen(42);
        std::uniform_real_distribution<> dist(0.0, 1.0);

        double random_data[N];
        for (int i = 0; i < N; ++i)
            random_data[i] = dist(gen);

        // DCT round-trip
        for (int i = 0; i < N; ++i)
            data_r[i] = random_data[i];

        fft_dct.forward(data_r, data_k);
        fft_dct.backward(data_k, data_r);

        for (int i = 0; i < N; ++i)
            diff_sq[i] = std::pow(data_r[i] - random_data[i], 2);

        error = std::sqrt(*std::max_element(diff_sq.begin(), diff_sq.end()));
        std::cout << "  DCT Random Round-trip Error: " << error << std::endl;
        if (!std::isfinite(error) || error > 1e-7)
        {
            std::cout << "  FAILED!" << std::endl;
            return -1;
        }

        // DST round-trip
        for (int i = 0; i < N; ++i)
            data_r[i] = random_data[i];

        fft_dst.forward(data_r, data_k);
        fft_dst.backward(data_k, data_r);

        for (int i = 0; i < N; ++i)
            diff_sq[i] = std::pow(data_r[i] - random_data[i], 2);

        error = std::sqrt(*std::max_element(diff_sq.begin(), diff_sq.end()));
        std::cout << "  DST Random Round-trip Error: " << error << std::endl;
        if (!std::isfinite(error) || error > 1e-7)
        {
            std::cout << "  FAILED!" << std::endl;
            return -1;
        }
        std::cout << "  PASSED!" << std::endl;

        //=======================================================================
        // Test 4: 2D Mixed BC test
        //=======================================================================
        std::cout << "\nTest 4: 2D Mixed BC Round-trip" << std::endl;

        const int NX = 8, NY = 6;
        const int M = NX * NY;
        std::vector<double> data2d_init(M), data2d_r(M), data2d_k(M);

        for (int i = 0; i < M; ++i)
            data2d_init[i] = dist(gen);

        std::array<int, 2> nx_2d = {NX, NY};
        std::array<BoundaryCondition, 2> bc_2d = {BoundaryCondition::REFLECTING, BoundaryCondition::ABSORBING};
        FftwFFT<double, 2> fft_2d(nx_2d, bc_2d);

        for (int i = 0; i < M; ++i)
            data2d_r[i] = data2d_init[i];

        fft_2d.forward(data2d_r.data(), data2d_k.data());
        fft_2d.backward(data2d_k.data(), data2d_r.data());

        std::vector<double> diff_sq_2d(M);
        for (int i = 0; i < M; ++i)
            diff_sq_2d[i] = std::pow(data2d_r[i] - data2d_init[i], 2);

        error = std::sqrt(*std::max_element(diff_sq_2d.begin(), diff_sq_2d.end()));
        std::cout << "  2D Mixed BC Round-trip Error: " << error << std::endl;
        if (!std::isfinite(error) || error > 1e-7)
        {
            std::cout << "  FAILED!" << std::endl;
            return -1;
        }
        std::cout << "  PASSED!" << std::endl;

        //=======================================================================
        // Test 5: 3D Mixed BC test
        //=======================================================================
        std::cout << "\nTest 5: 3D Mixed BC Round-trip" << std::endl;

        const int NX3 = 5, NY3 = 4, NZ3 = 3;
        const int M3 = NX3 * NY3 * NZ3;
        std::vector<double> data3d_init(M3), data3d_r(M3), data3d_k(M3);

        for (int i = 0; i < M3; ++i)
            data3d_init[i] = dist(gen);

        std::array<int, 3> nx_3d = {NX3, NY3, NZ3};
        std::array<BoundaryCondition, 3> bc_3d = {
            BoundaryCondition::REFLECTING,
            BoundaryCondition::ABSORBING,
            BoundaryCondition::REFLECTING
        };
        FftwFFT<double, 3> fft_3d(nx_3d, bc_3d);

        for (int i = 0; i < M3; ++i)
            data3d_r[i] = data3d_init[i];

        fft_3d.forward(data3d_r.data(), data3d_k.data());
        fft_3d.backward(data3d_k.data(), data3d_r.data());

        std::vector<double> diff_sq_3d(M3);
        for (int i = 0; i < M3; ++i)
            diff_sq_3d[i] = std::pow(data3d_r[i] - data3d_init[i], 2);

        error = std::sqrt(*std::max_element(diff_sq_3d.begin(), diff_sq_3d.end()));
        std::cout << "  3D Mixed BC Round-trip Error: " << error << std::endl;
        if (!std::isfinite(error) || error > 1e-7)
        {
            std::cout << "  FAILED!" << std::endl;
            return -1;
        }
        std::cout << "  PASSED!" << std::endl;

        //=======================================================================
        // Test 6: DCT-2 vs DFT with symmetric extension
        //=======================================================================
        // Mathematical relationship:
        // If we create a symmetric extension of N-point data to 2N points:
        //   y[n] = x[n] for n=0..N-1
        //   y[2N-1-n] = x[n] for n=0..N-1
        // Then: DCT-II(x)[k] = Re(DFT(y)[k]) (with appropriate scaling)
        //
        // More precisely, for our DCT-II definition:
        //   X[k] = sum_{n=0}^{N-1} x[n] * cos(π*k*(n+0.5)/N)
        // This equals (1/2) * Re(DFT of 2N-point whole-sample symmetric extension)
        //=======================================================================
        std::cout << "\nTest 6: DCT-2 vs DFT with symmetric extension" << std::endl;

        const int N_TEST = 8;

        // Create test data
        std::vector<double> x_data(N_TEST);
        for (int i = 0; i < N_TEST; ++i)
            x_data[i] = dist(gen);

        // Compute DCT-2 using our implementation
        std::vector<double> dct_result(N_TEST);
        std::array<int, 1> nx_test = {N_TEST};
        std::array<BoundaryCondition, 1> bc_test_reflect = {BoundaryCondition::REFLECTING};
        FftwFFT<double, 1> fft_dct_test(nx_test, bc_test_reflect);
        fft_dct_test.forward(x_data.data(), dct_result.data());

        // Create symmetric extension for DFT (whole-sample symmetric)
        // For DCT-II with cosine basis cos(π*k*(n+0.5)/N), the extension is:
        // y[n] = x[n] for n=0..N-1
        // y[2N-1-n] = x[n] for n=0..N-1
        const int N2 = 2 * N_TEST;
        std::vector<double> y_sym(N2);
        for (int n = 0; n < N_TEST; ++n)
        {
            y_sym[n] = x_data[n];
            y_sym[N2 - 1 - n] = x_data[n];
        }

        // Compute DFT of symmetric extension
        std::array<int, 1> nx_2n = {N2};
        FftwFFT<double, 1> fft_dft(nx_2n);
        std::vector<std::complex<double>> dft_result(N2 / 2 + 1);
        fft_dft.forward(y_sym.data(), dft_result.data());

        // Compare: DCT-II[k] should equal 0.5 * Re(DFT[k]) * exp(i*π*k/(2N)) for k > 0
        // Due to the half-sample shift, the relationship involves a phase factor
        // For k=0: DCT-II[0] = sum x[n] = 0.5 * DFT[0]
        // For k>0: DCT-II[k] = Re(DFT[k] * exp(-i*π*k/(2N)))
        double dct_dft_error = 0.0;
        const double PI = std::numbers::pi;

        for (int k = 0; k < N_TEST; ++k)
        {
            // Phase factor for half-sample shift: exp(-i*π*k/(2N))
            double phase = -PI * k / N2;
            std::complex<double> phase_factor(std::cos(phase), std::sin(phase));

            // DFT coefficient with phase correction
            std::complex<double> dft_shifted = dft_result[k] * phase_factor;
            double dft_real = dft_shifted.real();

            // Compare with DCT-II result (with scaling factor of 0.5)
            double expected = dft_real * 0.5;
            double actual = dct_result[k];

            dct_dft_error = std::max(dct_dft_error, std::abs(actual - expected));
        }

        std::cout << "  DCT-2 vs DFT max error: " << dct_dft_error << std::endl;
        if (!std::isfinite(dct_dft_error) || dct_dft_error > 1e-10)
        {
            std::cout << "  FAILED!" << std::endl;
            return -1;
        }
        std::cout << "  PASSED!" << std::endl;

        //=======================================================================
        // Test 7: DST-2 vs DFT with antisymmetric extension
        //=======================================================================
        // Mathematical relationship:
        // If we create an antisymmetric extension of N-point data to 2N points:
        //   y[n] = x[n] for n=0..N-1
        //   y[2N-1-n] = -x[n] for n=0..N-1
        // Then: DST-II(x)[k] = -Im(DFT(y)[k]) (with appropriate scaling)
        //
        // For our DST-II definition:
        //   X[k] = sum_{n=0}^{N-1} x[n] * sin(π*(k+1)*(n+0.5)/N)
        //=======================================================================
        std::cout << "\nTest 7: DST-2 vs DFT with antisymmetric extension" << std::endl;

        // Compute DST-2 using our implementation
        std::vector<double> dst_result(N_TEST);
        std::array<BoundaryCondition, 1> bc_test_absorb = {BoundaryCondition::ABSORBING};
        FftwFFT<double, 1> fft_dst_test(nx_test, bc_test_absorb);
        fft_dst_test.forward(x_data.data(), dst_result.data());

        // Create antisymmetric extension for DFT (whole-sample antisymmetric)
        // y[n] = x[n] for n=0..N-1
        // y[2N-1-n] = -x[n] for n=0..N-1
        std::vector<double> y_antisym(N2);
        for (int n = 0; n < N_TEST; ++n)
        {
            y_antisym[n] = x_data[n];
            y_antisym[N2 - 1 - n] = -x_data[n];
        }

        // Compute DFT of antisymmetric extension
        fft_dft.forward(y_antisym.data(), dft_result.data());

        // Compare: DST-II[k] should equal -0.5 * Im(DFT[k+1]) * exp(i*π*(k+1)/(2N))
        // Due to the indexing and half-sample shift
        double dst_dft_error = 0.0;

        for (int k = 0; k < N_TEST; ++k)
        {
            // For DST-II, we use frequency index k+1
            int freq_idx = k + 1;

            // Phase factor for half-sample shift: exp(-i*π*(k+1)/(2N))
            double phase = -PI * freq_idx / N2;
            std::complex<double> phase_factor(std::cos(phase), std::sin(phase));

            // DFT coefficient with phase correction
            std::complex<double> dft_shifted = dft_result[freq_idx] * phase_factor;
            double dft_imag = -dft_shifted.imag();

            // Compare with DST-II result (with scaling factor of 0.5)
            double expected = dft_imag * 0.5;
            double actual = dst_result[k];

            dst_dft_error = std::max(dst_dft_error, std::abs(actual - expected));
        }

        std::cout << "  DST-2 vs DFT max error: " << dst_dft_error << std::endl;
        if (!std::isfinite(dst_dft_error) || dst_dft_error > 1e-10)
        {
            std::cout << "  FAILED!" << std::endl;
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
