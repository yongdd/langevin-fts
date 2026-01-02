/**
 * @file TestCpuFFTMixedBC2D.cpp
 * @brief Test 2D DCT/DST transforms for mixed boundary conditions.
 *
 * This test verifies that MklFFTMixedBC correctly implements 2D transforms:
 * - DCT-II forward / DCT-III backward for reflecting (Neumann) BCs
 * - DST-II forward / DST-III backward for absorbing (Dirichlet) BCs
 * - Mixed combinations (e.g., reflecting in x, absorbing in y)
 *
 * The primary verification is round-trip: backward(forward(x)) = x
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
#ifdef USE_CPU_MKL
#include "MklFFTMixedBC.h"
#endif

int main()
{
    try
    {
#ifdef USE_CPU_MKL
        const int NX = 8;
        const int NY = 6;
        const int M = NX * NY;
        double error;
        std::vector<double> diff_sq(M);

        // Random data for testing
        std::mt19937 gen(42);
        std::uniform_real_distribution<> dist(0.0, 1.0);

        std::vector<double> data_init(M);
        for (int i = 0; i < M; ++i)
            data_init[i] = dist(gen);

        std::vector<double> data_r(M);
        std::vector<double> data_k(M);

        std::array<int, 2> nx_2d = {NX, NY};

        //=======================================================================
        // Test 1: All Reflecting BC (DCT-II/III)
        //=======================================================================
        std::cout << "Test 1: 2D All Reflecting BC (DCT-II/III)" << std::endl;

        std::array<BoundaryCondition, 2> bc_reflect = {
            BoundaryCondition::REFLECTING, BoundaryCondition::REFLECTING};
        MklFFTMixedBC<double, 2> fft_reflect(nx_2d, bc_reflect);

        for (int i = 0; i < M; ++i)
            data_r[i] = data_init[i];

        fft_reflect.forward(data_r.data(), data_k.data());
        fft_reflect.backward(data_k.data(), data_r.data());

        for (int i = 0; i < M; ++i)
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
        // Test 2: All Absorbing BC (DST-II/III)
        //=======================================================================
        std::cout << "\nTest 2: 2D All Absorbing BC (DST-II/III)" << std::endl;

        std::array<BoundaryCondition, 2> bc_absorb = {
            BoundaryCondition::ABSORBING, BoundaryCondition::ABSORBING};
        MklFFTMixedBC<double, 2> fft_absorb(nx_2d, bc_absorb);

        for (int i = 0; i < M; ++i)
            data_r[i] = data_init[i];

        fft_absorb.forward(data_r.data(), data_k.data());
        fft_absorb.backward(data_k.data(), data_r.data());

        for (int i = 0; i < M; ++i)
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
        // Test 3: Mixed BC (Reflecting in X, Absorbing in Y)
        //=======================================================================
        std::cout << "\nTest 3: 2D Mixed BC (Reflecting X, Absorbing Y)" << std::endl;

        std::array<BoundaryCondition, 2> bc_mixed1 = {
            BoundaryCondition::REFLECTING, BoundaryCondition::ABSORBING};
        MklFFTMixedBC<double, 2> fft_mixed1(nx_2d, bc_mixed1);

        for (int i = 0; i < M; ++i)
            data_r[i] = data_init[i];

        fft_mixed1.forward(data_r.data(), data_k.data());
        fft_mixed1.backward(data_k.data(), data_r.data());

        for (int i = 0; i < M; ++i)
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
        // Test 4: Mixed BC (Absorbing in X, Reflecting in Y)
        //=======================================================================
        std::cout << "\nTest 4: 2D Mixed BC (Absorbing X, Reflecting Y)" << std::endl;

        std::array<BoundaryCondition, 2> bc_mixed2 = {
            BoundaryCondition::ABSORBING, BoundaryCondition::REFLECTING};
        MklFFTMixedBC<double, 2> fft_mixed2(nx_2d, bc_mixed2);

        for (int i = 0; i < M; ++i)
            data_r[i] = data_init[i];

        fft_mixed2.forward(data_r.data(), data_k.data());
        fft_mixed2.backward(data_k.data(), data_r.data());

        for (int i = 0; i < M; ++i)
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
        // Test 5: Larger grid test
        //=======================================================================
        std::cout << "\nTest 5: 2D Larger Grid (16x12)" << std::endl;

        const int NX2 = 16, NY2 = 12;
        const int M2 = NX2 * NY2;

        std::array<int, 2> nx_2d_large = {NX2, NY2};
        std::vector<double> data_init2(M2), data_r2(M2), data_k2(M2);
        std::vector<double> diff_sq2(M2);

        for (int i = 0; i < M2; ++i)
            data_init2[i] = dist(gen);

        std::array<BoundaryCondition, 2> bc_large = {
            BoundaryCondition::REFLECTING, BoundaryCondition::ABSORBING};
        MklFFTMixedBC<double, 2> fft_large(nx_2d_large, bc_large);

        for (int i = 0; i < M2; ++i)
            data_r2[i] = data_init2[i];

        fft_large.forward(data_r2.data(), data_k2.data());
        fft_large.backward(data_k2.data(), data_r2.data());

        for (int i = 0; i < M2; ++i)
            diff_sq2[i] = std::pow(data_r2[i] - data_init2[i], 2);

        error = std::sqrt(*std::max_element(diff_sq2.begin(), diff_sq2.end()));
        std::cout << "  Round-trip Error: " << error << std::endl;
        if (!std::isfinite(error) || error > 1e-7)
        {
            std::cout << "  FAILED!" << std::endl;
            return -1;
        }
        std::cout << "  PASSED!" << std::endl;

        //=======================================================================
        // Test 6: Structured data (Gaussian) test
        //=======================================================================
        std::cout << "\nTest 6: 2D Gaussian Input Data" << std::endl;

        const double LX = 4.0, LY = 3.0;
        const double dxg = LX / NX, dyg = LY / NY;

        std::vector<double> gaussian_data(M);
        for (int i = 0; i < NX; ++i)
        {
            double x = (i + 0.5) * dxg;
            for (int j = 0; j < NY; ++j)
            {
                double y = (j + 0.5) * dyg;
                gaussian_data[i * NY + j] = std::exp(
                    -std::pow(x - LX/2, 2) / (2 * 0.5 * 0.5) -
                    -std::pow(y - LY/2, 2) / (2 * 0.5 * 0.5));
            }
        }

        for (int i = 0; i < M; ++i)
            data_r[i] = gaussian_data[i];

        fft_mixed1.forward(data_r.data(), data_k.data());
        fft_mixed1.backward(data_k.data(), data_r.data());

        for (int i = 0; i < M; ++i)
            diff_sq[i] = std::pow(data_r[i] - gaussian_data[i], 2);

        error = std::sqrt(*std::max_element(diff_sq.begin(), diff_sq.end()));
        std::cout << "  Round-trip Error: " << error << std::endl;
        if (!std::isfinite(error) || error > 1e-7)
        {
            std::cout << "  FAILED!" << std::endl;
            return -1;
        }
        std::cout << "  PASSED!" << std::endl;

        std::cout << "\nAll 2D FFT tests passed!" << std::endl;
#endif

        return 0;
    }
    catch (std::exception& exc)
    {
        std::cout << exc.what() << std::endl;
        return -1;
    }
}
