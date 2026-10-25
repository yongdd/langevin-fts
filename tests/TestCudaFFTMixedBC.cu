/**
 * @file TestCudaFFT.cu
 * @brief Test CUDA DCT-II/III and DST-II/III transforms for mixed boundary conditions.
 *
 * This test verifies that CudaFFT correctly implements:
 * - DCT-II forward / DCT-III backward for reflecting (Neumann) BCs
 * - DST-II forward / DST-III backward for absorbing (Dirichlet) BCs
 *
 * Tests cover 1D, 2D, and 3D cases.
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
#include <array>

#include "CudaCommon.h"
#include "CudaFFT.h"
#include "TestNanSafe.h"

int main()
{
    try
    {
        const int N = 12;
        double error;
        std::vector<double> diff_sq(N);

        // Test input data
        std::vector<double> data_init(N);
        for (int i = 0; i < N; ++i)
            data_init[i] = i + 1;  // [1, 2, 3, ..., 12]

        std::vector<double> data_r(N);
        std::vector<double> data_k(N);

        // GPU buffers
        double* d_data_r;
        double* d_data_k;
        CUDA_CHECK(cudaMalloc((void**)&d_data_r, sizeof(double) * N));
        CUDA_CHECK(cudaMalloc((void**)&d_data_k, sizeof(double) * N));

        //=======================================================================
        // Test 0: Odd lengths along reflecting axes. CudaFFT's forward is the
        // unnormalized DCT-II / DST-II per axis (FFTW REDFT10 / RODFT10), which
        // a direct sum checks; absorbing axes must stay even.
        //=======================================================================
        {
            std::cout << "Test 0: odd reflecting axes (1D N=11, 3D 6x4x3 R/A/R) against direct sums" << std::endl;
            auto basis = [](BoundaryCondition bc, int k, int n, int len) {
                const double pi = std::numbers::pi;
                return (bc == BoundaryCondition::REFLECTING) ? 2.0 * std::cos(pi * k * (2 * n + 1) / (2.0 * len))
                                                             : 2.0 * std::sin(pi * (k + 1) * (2 * n + 1) / (2.0 * len));
            };
            std::mt19937 gen0(7);
            std::normal_distribution<double> normal(0.0, 5.0);

            // 1D, N = 11
            {
                const int N1 = 11;
                std::vector<double> x(N1), coeff(N1), back(N1), ref(N1, 0.0);
                for (double& v : x)
                    v = normal(gen0);
                for (int k = 0; k < N1; ++k)
                    for (int n = 0; n < N1; ++n)
                        ref[k] += x[n] * basis(BoundaryCondition::REFLECTING, k, n, N1);
                CudaFFT<double, 1> fft_odd({N1}, {BoundaryCondition::REFLECTING});
                CudaDeviceBuffer<double> d_x(N1), d_c(N1);
                CUDA_CHECK(cudaMemcpy(d_x, x.data(), sizeof(double) * N1, cudaMemcpyHostToDevice));
                fft_odd.forward(d_x, d_c);
                CUDA_CHECK(cudaMemcpy(coeff.data(), d_c, sizeof(double) * N1, cudaMemcpyDeviceToHost));
                fft_odd.backward(d_c, d_x);
                CUDA_CHECK(cudaMemcpy(back.data(), d_x, sizeof(double) * N1, cudaMemcpyDeviceToHost));
                double err_fwd = 0.0, err_back = 0.0, scale = 0.0;
                for (int i = 0; i < N1; ++i)
                {
                    err_fwd = max_or_nan(err_fwd, std::abs(coeff[i] - ref[i]));
                    err_back = max_or_nan(err_back, std::abs(back[i] - x[i]));
                    scale = std::max(scale, std::abs(ref[i]));
                }
                std::cout << "  1D N=11: forward vs direct " << err_fwd / scale << ", round trip " << err_back << std::endl;
                if (!(err_fwd / scale < 1e-12) || !(err_back < 1e-12))
                {
                    std::cout << "  FAILED!" << std::endl;
                    return -1;
                }
            }

            // 3D, 6 (reflecting) x 4 (absorbing) x 3 (reflecting)
            {
                const std::array<int, 3> n3 = {6, 4, 3};
                const std::array<BoundaryCondition, 3> bc3 = {
                    BoundaryCondition::REFLECTING, BoundaryCondition::ABSORBING, BoundaryCondition::REFLECTING};
                const int M3 = n3[0] * n3[1] * n3[2];
                std::vector<double> x(M3), coeff(M3), back(M3), ref(M3, 0.0);
                for (double& v : x)
                    v = normal(gen0);
                for (int k0 = 0; k0 < n3[0]; ++k0)
                    for (int k1 = 0; k1 < n3[1]; ++k1)
                        for (int k2 = 0; k2 < n3[2]; ++k2)
                        {
                            double sum = 0.0;
                            for (int i0 = 0; i0 < n3[0]; ++i0)
                                for (int i1 = 0; i1 < n3[1]; ++i1)
                                    for (int i2 = 0; i2 < n3[2]; ++i2)
                                        sum += x[(i0 * n3[1] + i1) * n3[2] + i2] * basis(bc3[0], k0, i0, n3[0]) *
                                               basis(bc3[1], k1, i1, n3[1]) * basis(bc3[2], k2, i2, n3[2]);
                            ref[(k0 * n3[1] + k1) * n3[2] + k2] = sum;
                        }
                CudaFFT<double, 3> fft_odd3(n3, bc3);
                CudaDeviceBuffer<double> d_x(M3), d_c(M3);
                CUDA_CHECK(cudaMemcpy(d_x, x.data(), sizeof(double) * M3, cudaMemcpyHostToDevice));
                fft_odd3.forward(d_x, d_c);
                CUDA_CHECK(cudaMemcpy(coeff.data(), d_c, sizeof(double) * M3, cudaMemcpyDeviceToHost));
                fft_odd3.backward(d_c, d_x);
                CUDA_CHECK(cudaMemcpy(back.data(), d_x, sizeof(double) * M3, cudaMemcpyDeviceToHost));
                double err_fwd = 0.0, err_back = 0.0, scale = 0.0;
                for (int i = 0; i < M3; ++i)
                {
                    err_fwd = max_or_nan(err_fwd, std::abs(coeff[i] - ref[i]));
                    err_back = max_or_nan(err_back, std::abs(back[i] - x[i]));
                    scale = std::max(scale, std::abs(ref[i]));
                }
                std::cout << "  3D 6x4x3 R/A/R: forward vs direct " << err_fwd / scale << ", round trip " << err_back << std::endl;
                if (!(err_fwd / scale < 1e-12) || !(err_back < 1e-12))
                {
                    std::cout << "  FAILED!" << std::endl;
                    return -1;
                }
            }

            // An odd absorbing axis is refused
            bool refused = false;
            try
            {
                CudaFFT<double, 2> fft_bad({6, 5}, {BoundaryCondition::REFLECTING, BoundaryCondition::ABSORBING});
            }
            catch (std::exception&)
            {
                refused = true;
            }
            std::cout << "  odd absorbing axis refused: " << (refused ? "yes" : "NO") << std::endl;
            if (!refused)
            {
                std::cout << "  FAILED!" << std::endl;
                return -1;
            }
            std::cout << "  PASSED!" << std::endl;
        }

        //=======================================================================
        // Test 1: DCT-II / DCT-III (Reflecting BC) Round-trip
        //=======================================================================
        std::cout << "\nTest 1: DCT-II/III (Reflecting BC) Round-trip" << std::endl;

        std::array<int, 1> nx_1d = {N};
        std::array<BoundaryCondition, 1> bc_reflect = {BoundaryCondition::REFLECTING};
        CudaFFT<double, 1> fft_dct(nx_1d, bc_reflect);

        // Copy input to device
        CUDA_CHECK(cudaMemcpy(d_data_r, data_init.data(),
                                   sizeof(double) * N, cudaMemcpyHostToDevice));

        // Forward DCT-II
        fft_dct.forward(d_data_r, d_data_k);

        // Backward DCT-III (inverse) - should recover original
        fft_dct.backward(d_data_k, d_data_r);

        // Copy result back to host
        CUDA_CHECK(cudaMemcpy(data_r.data(), d_data_r,
                                   sizeof(double) * N, cudaMemcpyDeviceToHost));

        for (int i = 0; i < N; ++i)
            diff_sq[i] = std::pow(data_r[i] - data_init[i], 2);

        error = std::sqrt(max_or_nan(diff_sq));
        std::cout << "  Round-trip Error: " << error << std::endl;
        if (!std::isfinite(error) || error > 1e-7)
        {
            std::cout << "  FAILED!" << std::endl;
            cudaFree(d_data_r);
            cudaFree(d_data_k);
            return -1;
        }
        std::cout << "  PASSED!" << std::endl;

        //=======================================================================
        // Test 2: DST-II / DST-III (Absorbing BC) Round-trip
        //=======================================================================
        std::cout << "\nTest 2: DST-II/III (Absorbing BC) Round-trip" << std::endl;

        std::array<BoundaryCondition, 1> bc_absorb = {BoundaryCondition::ABSORBING};
        CudaFFT<double, 1> fft_dst(nx_1d, bc_absorb);

        // Copy input to device
        CUDA_CHECK(cudaMemcpy(d_data_r, data_init.data(),
                                   sizeof(double) * N, cudaMemcpyHostToDevice));

        // Forward DST-II
        fft_dst.forward(d_data_r, d_data_k);

        // Backward DST-III (inverse) - should recover original
        fft_dst.backward(d_data_k, d_data_r);

        // Copy result back to host
        CUDA_CHECK(cudaMemcpy(data_r.data(), d_data_r,
                                   sizeof(double) * N, cudaMemcpyDeviceToHost));

        for (int i = 0; i < N; ++i)
            diff_sq[i] = std::pow(data_r[i] - data_init[i], 2);

        error = std::sqrt(max_or_nan(diff_sq));
        std::cout << "  Round-trip Error: " << error << std::endl;
        if (!std::isfinite(error) || error > 1e-7)
        {
            std::cout << "  FAILED!" << std::endl;
            cudaFree(d_data_r);
            cudaFree(d_data_k);
            return -1;
        }
        std::cout << "  PASSED!" << std::endl;

        //=======================================================================
        // Test 3: Random data round-trip test
        //=======================================================================
        std::cout << "\nTest 3: Random data round-trip" << std::endl;

        std::mt19937 gen(42);
        std::uniform_real_distribution<> dist(0.0, 1.0);

        std::vector<double> random_data(N);
        for (int i = 0; i < N; ++i)
            random_data[i] = dist(gen);

        // DCT round-trip
        CUDA_CHECK(cudaMemcpy(d_data_r, random_data.data(),
                                   sizeof(double) * N, cudaMemcpyHostToDevice));

        fft_dct.forward(d_data_r, d_data_k);
        fft_dct.backward(d_data_k, d_data_r);

        CUDA_CHECK(cudaMemcpy(data_r.data(), d_data_r,
                                   sizeof(double) * N, cudaMemcpyDeviceToHost));

        for (int i = 0; i < N; ++i)
            diff_sq[i] = std::pow(data_r[i] - random_data[i], 2);

        error = std::sqrt(max_or_nan(diff_sq));
        std::cout << "  DCT Random Round-trip Error: " << error << std::endl;
        if (!std::isfinite(error) || error > 1e-7)
        {
            std::cout << "  FAILED!" << std::endl;
            cudaFree(d_data_r);
            cudaFree(d_data_k);
            return -1;
        }

        // DST round-trip
        CUDA_CHECK(cudaMemcpy(d_data_r, random_data.data(),
                                   sizeof(double) * N, cudaMemcpyHostToDevice));

        fft_dst.forward(d_data_r, d_data_k);
        fft_dst.backward(d_data_k, d_data_r);

        CUDA_CHECK(cudaMemcpy(data_r.data(), d_data_r,
                                   sizeof(double) * N, cudaMemcpyDeviceToHost));

        for (int i = 0; i < N; ++i)
            diff_sq[i] = std::pow(data_r[i] - random_data[i], 2);

        error = std::sqrt(max_or_nan(diff_sq));
        std::cout << "  DST Random Round-trip Error: " << error << std::endl;
        if (!std::isfinite(error) || error > 1e-7)
        {
            std::cout << "  FAILED!" << std::endl;
            cudaFree(d_data_r);
            cudaFree(d_data_k);
            return -1;
        }
        std::cout << "  PASSED!" << std::endl;

        // Clean up
        cudaFree(d_data_r);
        cudaFree(d_data_k);

        //=======================================================================
        // Test 4: 2D Mixed BC test
        //=======================================================================
        std::cout << "\nTest 4: 2D Mixed BC Round-trip" << std::endl;

        const int NX = 8, NY = 6;
        const int M = NX * NY;
        std::vector<double> data2d_init(M), data2d_r(M), data2d_k(M);

        for (int i = 0; i < M; ++i)
            data2d_init[i] = dist(gen);

        double* d_data2d_r;
        double* d_data2d_k;
        CUDA_CHECK(cudaMalloc((void**)&d_data2d_r, sizeof(double) * M));
        CUDA_CHECK(cudaMalloc((void**)&d_data2d_k, sizeof(double) * M));

        std::array<int, 2> nx_2d = {NX, NY};
        std::array<BoundaryCondition, 2> bc_2d = {BoundaryCondition::REFLECTING, BoundaryCondition::ABSORBING};
        CudaFFT<double, 2> fft_2d(nx_2d, bc_2d);

        CUDA_CHECK(cudaMemcpy(d_data2d_r, data2d_init.data(),
                                   sizeof(double) * M, cudaMemcpyHostToDevice));

        fft_2d.forward(d_data2d_r, d_data2d_k);
        fft_2d.backward(d_data2d_k, d_data2d_r);

        CUDA_CHECK(cudaMemcpy(data2d_r.data(), d_data2d_r,
                                   sizeof(double) * M, cudaMemcpyDeviceToHost));

        std::vector<double> diff_sq_2d(M);
        for (int i = 0; i < M; ++i)
            diff_sq_2d[i] = std::pow(data2d_r[i] - data2d_init[i], 2);

        error = std::sqrt(max_or_nan(diff_sq_2d));
        std::cout << "  2D Mixed BC Round-trip Error: " << error << std::endl;
        if (!std::isfinite(error) || error > 1e-7)
        {
            std::cout << "  FAILED!" << std::endl;
            cudaFree(d_data2d_r);
            cudaFree(d_data2d_k);
            return -1;
        }
        std::cout << "  PASSED!" << std::endl;

        cudaFree(d_data2d_r);
        cudaFree(d_data2d_k);

        //=======================================================================
        // Test 5: 3D Mixed BC test (the reflecting z axis has an odd length)
        //=======================================================================
        std::cout << "\nTest 5: 3D Mixed BC Round-trip" << std::endl;
        {
        const int NX3 = 6, NY3 = 4, NZ3 = 3;  // Only dim 2 is odd
        const int M3 = NX3 * NY3 * NZ3;
        std::vector<double> data3d_init(M3), data3d_r(M3), data3d_k(M3);

        for (int i = 0; i < M3; ++i)
            data3d_init[i] = dist(gen);

        double* d_data3d_r;
        double* d_data3d_k;
        CUDA_CHECK(cudaMalloc((void**)&d_data3d_r, sizeof(double) * M3));
        CUDA_CHECK(cudaMalloc((void**)&d_data3d_k, sizeof(double) * M3));

        std::array<int, 3> nx_3d = {NX3, NY3, NZ3};
        std::array<BoundaryCondition, 3> bc_3d = {
            BoundaryCondition::REFLECTING,
            BoundaryCondition::ABSORBING,
            BoundaryCondition::REFLECTING
        };
        CudaFFT<double, 3> fft_3d(nx_3d, bc_3d);

        CUDA_CHECK(cudaMemcpy(d_data3d_r, data3d_init.data(),
                                   sizeof(double) * M3, cudaMemcpyHostToDevice));

        fft_3d.forward(d_data3d_r, d_data3d_k);
        fft_3d.backward(d_data3d_k, d_data3d_r);

        CUDA_CHECK(cudaMemcpy(data3d_r.data(), d_data3d_r,
                                   sizeof(double) * M3, cudaMemcpyDeviceToHost));

        std::vector<double> diff_sq_3d(M3);
        for (int i = 0; i < M3; ++i)
            diff_sq_3d[i] = std::pow(data3d_r[i] - data3d_init[i], 2);

        error = std::sqrt(max_or_nan(diff_sq_3d));
        std::cout << "  3D Mixed BC Round-trip Error: " << error << std::endl;
        if (!std::isfinite(error) || error > 1e-7)
        {
            std::cout << "  FAILED!" << std::endl;
            cudaFree(d_data3d_r);
            cudaFree(d_data3d_k);
            return -1;
        }
        std::cout << "  PASSED!" << std::endl;

        cudaFree(d_data3d_r);
        cudaFree(d_data3d_k);
        }

        //=======================================================================
        // Periodic complex interface: backward() preserves its input spectrum
        // (cuFFT Z2D may overwrite its input even out of place)
        //=======================================================================
        {
            std::cout << "\nPeriodic backward(complex) preserves its input" << std::endl;
            const std::array<int, 3> nx_p = {8, 6, 10};
            const std::array<BoundaryCondition, 3> bc_p = {
                BoundaryCondition::PERIODIC, BoundaryCondition::PERIODIC, BoundaryCondition::PERIODIC};
            const int MP = nx_p[0] * nx_p[1] * nx_p[2];
            const int MP_COMPLEX = nx_p[0] * nx_p[1] * (nx_p[2] / 2 + 1);
            CudaFFT<double, 3> fft_p(nx_p, bc_p);

            std::vector<double> data_p(MP), data_p_back(MP);
            for (int i = 0; i < MP; ++i)
                data_p[i] = std::sin(0.37 * i) + 0.1 * i / MP;
            CudaDeviceBuffer<double> d_p(MP), d_p_back(MP);
            CudaDeviceBuffer<cuDoubleComplex> d_pk(MP_COMPLEX);
            std::complex<double>* pk = reinterpret_cast<std::complex<double>*>(d_pk.get());
            CUDA_CHECK(cudaMemcpy(d_p, data_p.data(), sizeof(double) * MP, cudaMemcpyHostToDevice));

            fft_p.forward(d_p, pk);
            std::vector<std::complex<double>> spectrum(MP_COMPLEX), spectrum_after(MP_COMPLEX);
            CUDA_CHECK(cudaMemcpy(spectrum.data(), pk, sizeof(cuDoubleComplex) * MP_COMPLEX, cudaMemcpyDeviceToHost));
            fft_p.backward(pk, d_p_back);
            CUDA_CHECK(cudaMemcpy(spectrum_after.data(), pk, sizeof(cuDoubleComplex) * MP_COMPLEX, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(data_p_back.data(), d_p_back, sizeof(double) * MP, cudaMemcpyDeviceToHost));

            if (spectrum != spectrum_after)
            {
                std::cout << "  FAILED! backward() changed its input spectrum" << std::endl;
                return -1;
            }
            std::vector<double> diff_p(MP);
            for (int i = 0; i < MP; ++i)
                diff_p[i] = std::abs(data_p_back[i] - data_p[i]);
            error = max_or_nan(diff_p);
            std::cout << "  Round-trip Error: " << error << std::endl;
            if (!std::isfinite(error) || error > 1e-12)
            {
                std::cout << "  FAILED!" << std::endl;
                return -1;
            }
            std::cout << "  PASSED!" << std::endl;
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
