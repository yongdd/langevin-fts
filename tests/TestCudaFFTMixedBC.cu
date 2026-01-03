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
        gpu_error_check(cudaMalloc((void**)&d_data_r, sizeof(double) * N));
        gpu_error_check(cudaMalloc((void**)&d_data_k, sizeof(double) * N));

        //=======================================================================
        // Test 1: DCT-II / DCT-III (Reflecting BC) Round-trip
        //=======================================================================
        std::cout << "Test 1: DCT-II/III (Reflecting BC) Round-trip" << std::endl;

        std::array<int, 1> nx_1d = {N};
        std::array<BoundaryCondition, 1> bc_reflect = {BoundaryCondition::REFLECTING};
        CudaFFT<double, 1> fft_dct(nx_1d, bc_reflect);

        // Copy input to device
        gpu_error_check(cudaMemcpy(d_data_r, data_init.data(),
                                   sizeof(double) * N, cudaMemcpyHostToDevice));

        // Forward DCT-II
        fft_dct.forward(d_data_r, d_data_k);

        // Backward DCT-III (inverse) - should recover original
        fft_dct.backward(d_data_k, d_data_r);

        // Copy result back to host
        gpu_error_check(cudaMemcpy(data_r.data(), d_data_r,
                                   sizeof(double) * N, cudaMemcpyDeviceToHost));

        for (int i = 0; i < N; ++i)
            diff_sq[i] = std::pow(data_r[i] - data_init[i], 2);

        error = std::sqrt(*std::max_element(diff_sq.begin(), diff_sq.end()));
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
        gpu_error_check(cudaMemcpy(d_data_r, data_init.data(),
                                   sizeof(double) * N, cudaMemcpyHostToDevice));

        // Forward DST-II
        fft_dst.forward(d_data_r, d_data_k);

        // Backward DST-III (inverse) - should recover original
        fft_dst.backward(d_data_k, d_data_r);

        // Copy result back to host
        gpu_error_check(cudaMemcpy(data_r.data(), d_data_r,
                                   sizeof(double) * N, cudaMemcpyDeviceToHost));

        for (int i = 0; i < N; ++i)
            diff_sq[i] = std::pow(data_r[i] - data_init[i], 2);

        error = std::sqrt(*std::max_element(diff_sq.begin(), diff_sq.end()));
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
        gpu_error_check(cudaMemcpy(d_data_r, random_data.data(),
                                   sizeof(double) * N, cudaMemcpyHostToDevice));

        fft_dct.forward(d_data_r, d_data_k);
        fft_dct.backward(d_data_k, d_data_r);

        gpu_error_check(cudaMemcpy(data_r.data(), d_data_r,
                                   sizeof(double) * N, cudaMemcpyDeviceToHost));

        for (int i = 0; i < N; ++i)
            diff_sq[i] = std::pow(data_r[i] - random_data[i], 2);

        error = std::sqrt(*std::max_element(diff_sq.begin(), diff_sq.end()));
        std::cout << "  DCT Random Round-trip Error: " << error << std::endl;
        if (!std::isfinite(error) || error > 1e-7)
        {
            std::cout << "  FAILED!" << std::endl;
            cudaFree(d_data_r);
            cudaFree(d_data_k);
            return -1;
        }

        // DST round-trip
        gpu_error_check(cudaMemcpy(d_data_r, random_data.data(),
                                   sizeof(double) * N, cudaMemcpyHostToDevice));

        fft_dst.forward(d_data_r, d_data_k);
        fft_dst.backward(d_data_k, d_data_r);

        gpu_error_check(cudaMemcpy(data_r.data(), d_data_r,
                                   sizeof(double) * N, cudaMemcpyDeviceToHost));

        for (int i = 0; i < N; ++i)
            diff_sq[i] = std::pow(data_r[i] - random_data[i], 2);

        error = std::sqrt(*std::max_element(diff_sq.begin(), diff_sq.end()));
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
        gpu_error_check(cudaMalloc((void**)&d_data2d_r, sizeof(double) * M));
        gpu_error_check(cudaMalloc((void**)&d_data2d_k, sizeof(double) * M));

        std::array<int, 2> nx_2d = {NX, NY};
        std::array<BoundaryCondition, 2> bc_2d = {BoundaryCondition::REFLECTING, BoundaryCondition::ABSORBING};
        CudaFFT<double, 2> fft_2d(nx_2d, bc_2d);

        gpu_error_check(cudaMemcpy(d_data2d_r, data2d_init.data(),
                                   sizeof(double) * M, cudaMemcpyHostToDevice));

        fft_2d.forward(d_data2d_r, d_data2d_k);
        fft_2d.backward(d_data2d_k, d_data2d_r);

        gpu_error_check(cudaMemcpy(data2d_r.data(), d_data2d_r,
                                   sizeof(double) * M, cudaMemcpyDeviceToHost));

        std::vector<double> diff_sq_2d(M);
        for (int i = 0; i < M; ++i)
            diff_sq_2d[i] = std::pow(data2d_r[i] - data2d_init[i], 2);

        error = std::sqrt(*std::max_element(diff_sq_2d.begin(), diff_sq_2d.end()));
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
        // Test 5: 3D Mixed BC test
        //=======================================================================
        std::cout << "\nTest 5: 3D Mixed BC Round-trip" << std::endl;

        const int NX3 = 5, NY3 = 4, NZ3 = 3;
        const int M3 = NX3 * NY3 * NZ3;
        std::vector<double> data3d_init(M3), data3d_r(M3), data3d_k(M3);

        for (int i = 0; i < M3; ++i)
            data3d_init[i] = dist(gen);

        double* d_data3d_r;
        double* d_data3d_k;
        gpu_error_check(cudaMalloc((void**)&d_data3d_r, sizeof(double) * M3));
        gpu_error_check(cudaMalloc((void**)&d_data3d_k, sizeof(double) * M3));

        std::array<int, 3> nx_3d = {NX3, NY3, NZ3};
        std::array<BoundaryCondition, 3> bc_3d = {
            BoundaryCondition::REFLECTING,
            BoundaryCondition::ABSORBING,
            BoundaryCondition::REFLECTING
        };
        CudaFFT<double, 3> fft_3d(nx_3d, bc_3d);

        gpu_error_check(cudaMemcpy(d_data3d_r, data3d_init.data(),
                                   sizeof(double) * M3, cudaMemcpyHostToDevice));

        fft_3d.forward(d_data3d_r, d_data3d_k);
        fft_3d.backward(d_data3d_k, d_data3d_r);

        gpu_error_check(cudaMemcpy(data3d_r.data(), d_data3d_r,
                                   sizeof(double) * M3, cudaMemcpyDeviceToHost));

        std::vector<double> diff_sq_3d(M3);
        for (int i = 0; i < M3; ++i)
            diff_sq_3d[i] = std::pow(data3d_r[i] - data3d_init[i], 2);

        error = std::sqrt(*std::max_element(diff_sq_3d.begin(), diff_sq_3d.end()));
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

        std::cout << "\nAll tests passed!" << std::endl;

        return 0;
    }
    catch (std::exception& exc)
    {
        std::cout << exc.what() << std::endl;
        return -1;
    }
}
