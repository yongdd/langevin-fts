/**
 * @file TestCudaDCT.cu
 * @brief Test CUDA DCT Types 1-4 against FFTW reference.
 *
 * Compares CudaDCT output with FFTW REDFT00/10/01/11 for all DCT types.
 */

#include <iostream>
#include <cmath>
#include <vector>
#include <cstdlib>
#include <algorithm>
#include <cuda_runtime.h>
#include <fftw3.h>

#include "../src/platforms/cuda/CudaDCT.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

//==============================================================================
// Test 1D DCT-1 (FFTW_REDFT00)
//==============================================================================
bool test_dct1_1d(int N)
{
    printf("Testing 1D DCT-1: N=%d (N+1=%d points)\n", N, N + 1);

    int size = N + 1;  // DCT-1 has N+1 points

    // Allocate host memory
    std::vector<double> h_input(size);
    std::vector<double> h_fftw(size);
    std::vector<double> h_cuda(size);

    // Initialize with test data
    for (int i = 0; i < size; ++i) {
        h_input[i] = sin(M_PI * i / N) + 0.5 * cos(2 * M_PI * i / N);
    }

    // === FFTW DCT-1 (REDFT00) ===
    std::copy(h_input.begin(), h_input.end(), h_fftw.begin());
    fftw_plan plan = fftw_plan_r2r_1d(size, h_fftw.data(), h_fftw.data(), FFTW_REDFT00, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);

    // === CUDA DCT-1 ===
    double* d_data;
    cudaMalloc(&d_data, sizeof(double) * size);
    cudaMemcpy(d_data, h_input.data(), sizeof(double) * size, cudaMemcpyHostToDevice);

    CudaDCT dct(size, CUDA_DCT_1);
    dct.execute(d_data);

    cudaMemcpy(h_cuda.data(), d_data, sizeof(double) * size, cudaMemcpyDeviceToHost);
    cudaFree(d_data);

    // Compare results
    double max_err = 0;
    for (int i = 0; i < size; ++i) {
        double err = std::abs(h_fftw[i] - h_cuda[i]);
        max_err = std::max(max_err, err);
    }

    // Relative error (scale by max value)
    double max_val = *std::max_element(h_fftw.begin(), h_fftw.end(),
        [](double a, double b) { return std::abs(a) < std::abs(b); });
    double rel_err = max_err / std::abs(max_val);

    printf("  Max abs error: %.6e, Relative error: %.6e\n", max_err, rel_err);

    bool passed = rel_err < 1e-10;
    printf("  %s\n\n", passed ? "PASSED!" : "FAILED!");

    return passed;
}

//==============================================================================
// Test 1D DCT-2 (FFTW_REDFT10)
//==============================================================================
bool test_dct2_1d(int N)
{
    printf("Testing 1D DCT-2: N=%d points\n", N);

    // Allocate host memory
    std::vector<double> h_input(N);
    std::vector<double> h_fftw(N);
    std::vector<double> h_cuda(N);

    // Initialize with test data
    for (int i = 0; i < N; ++i) {
        h_input[i] = sin(M_PI * (i + 0.5) / N) + 0.5 * cos(2 * M_PI * (i + 0.5) / N);
    }

    // === FFTW DCT-2 (REDFT10) ===
    std::copy(h_input.begin(), h_input.end(), h_fftw.begin());
    fftw_plan plan = fftw_plan_r2r_1d(N, h_fftw.data(), h_fftw.data(), FFTW_REDFT10, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);

    // === CUDA DCT-2 ===
    double* d_data;
    cudaMalloc(&d_data, sizeof(double) * N);
    cudaMemcpy(d_data, h_input.data(), sizeof(double) * N, cudaMemcpyHostToDevice);

    CudaDCT dct(N, CUDA_DCT_2);
    dct.execute(d_data);

    cudaMemcpy(h_cuda.data(), d_data, sizeof(double) * N, cudaMemcpyDeviceToHost);
    cudaFree(d_data);

    // Compare results
    double max_err = 0;
    for (int i = 0; i < N; ++i) {
        double err = std::abs(h_fftw[i] - h_cuda[i]);
        max_err = std::max(max_err, err);
    }

    double max_val = *std::max_element(h_fftw.begin(), h_fftw.end(),
        [](double a, double b) { return std::abs(a) < std::abs(b); });
    double rel_err = max_err / std::abs(max_val);

    printf("  Max abs error: %.6e, Relative error: %.6e\n", max_err, rel_err);

    bool passed = rel_err < 1e-10;
    printf("  %s\n\n", passed ? "PASSED!" : "FAILED!");

    return passed;
}

//==============================================================================
// Test 1D DCT-3 (FFTW_REDFT01)
//==============================================================================
bool test_dct3_1d(int N)
{
    printf("Testing 1D DCT-3: N=%d points\n", N);

    // Allocate host memory
    std::vector<double> h_input(N);
    std::vector<double> h_fftw(N);
    std::vector<double> h_cuda(N);

    // Initialize with test data
    for (int i = 0; i < N; ++i) {
        h_input[i] = sin(M_PI * i / N) + 0.5 * cos(2 * M_PI * i / N);
    }

    // === FFTW DCT-3 (REDFT01) ===
    std::copy(h_input.begin(), h_input.end(), h_fftw.begin());
    fftw_plan plan = fftw_plan_r2r_1d(N, h_fftw.data(), h_fftw.data(), FFTW_REDFT01, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);

    // === CUDA DCT-3 ===
    double* d_data;
    cudaMalloc(&d_data, sizeof(double) * N);
    cudaMemcpy(d_data, h_input.data(), sizeof(double) * N, cudaMemcpyHostToDevice);

    CudaDCT dct(N, CUDA_DCT_3);
    dct.execute(d_data);

    cudaMemcpy(h_cuda.data(), d_data, sizeof(double) * N, cudaMemcpyDeviceToHost);
    cudaFree(d_data);

    // Compare results
    double max_err = 0;
    for (int i = 0; i < N; ++i) {
        double err = std::abs(h_fftw[i] - h_cuda[i]);
        max_err = std::max(max_err, err);
    }

    double max_val = *std::max_element(h_fftw.begin(), h_fftw.end(),
        [](double a, double b) { return std::abs(a) < std::abs(b); });
    double rel_err = max_err / std::abs(max_val);

    printf("  Max abs error: %.6e, Relative error: %.6e\n", max_err, rel_err);

    bool passed = rel_err < 1e-10;
    printf("  %s\n\n", passed ? "PASSED!" : "FAILED!");

    return passed;
}

//==============================================================================
// Test 1D DCT-4 (FFTW_REDFT11)
//==============================================================================
bool test_dct4_1d(int N)
{
    printf("Testing 1D DCT-4: N=%d points\n", N);

    // Allocate host memory
    std::vector<double> h_input(N);
    std::vector<double> h_fftw(N);
    std::vector<double> h_cuda(N);

    // Initialize with test data
    for (int i = 0; i < N; ++i) {
        h_input[i] = sin(M_PI * (i + 0.5) / N) + 0.5 * cos(2 * M_PI * (i + 0.5) / N);
    }

    // === FFTW DCT-4 (REDFT11) ===
    std::copy(h_input.begin(), h_input.end(), h_fftw.begin());
    fftw_plan plan = fftw_plan_r2r_1d(N, h_fftw.data(), h_fftw.data(), FFTW_REDFT11, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);

    // === CUDA DCT-4 ===
    double* d_data;
    cudaMalloc(&d_data, sizeof(double) * N);
    cudaMemcpy(d_data, h_input.data(), sizeof(double) * N, cudaMemcpyHostToDevice);

    CudaDCT dct(N, CUDA_DCT_4);
    dct.execute(d_data);

    cudaMemcpy(h_cuda.data(), d_data, sizeof(double) * N, cudaMemcpyDeviceToHost);
    cudaFree(d_data);

    // Compare results
    double max_err = 0;
    for (int i = 0; i < N; ++i) {
        double err = std::abs(h_fftw[i] - h_cuda[i]);
        max_err = std::max(max_err, err);
    }

    double max_val = *std::max_element(h_fftw.begin(), h_fftw.end(),
        [](double a, double b) { return std::abs(a) < std::abs(b); });
    double rel_err = max_err / std::abs(max_val);

    printf("  Max abs error: %.6e, Relative error: %.6e\n", max_err, rel_err);

    bool passed = rel_err < 1e-10;
    printf("  %s\n\n", passed ? "PASSED!" : "FAILED!");

    return passed;
}

//==============================================================================
// Test 2D DCT-1
//==============================================================================
bool test_dct1_2d(int Nx, int Ny)
{
    printf("Testing 2D DCT-1: %d x %d (data size: %d x %d)\n",
           Nx, Ny, Nx + 1, Ny + 1);

    int Nx1 = Nx + 1, Ny1 = Ny + 1;
    int M = Nx1 * Ny1;

    std::vector<double> h_input(M);
    std::vector<double> h_fftw(M);
    std::vector<double> h_cuda(M);

    // Initialize with test data
    for (int ix = 0; ix < Nx1; ++ix) {
        for (int iy = 0; iy < Ny1; ++iy) {
            int idx = ix * Ny1 + iy;
            double r2 = ix * ix + iy * iy;
            h_input[idx] = exp(-r2 / 50.0);
        }
    }

    // === FFTW 2D DCT-1 (REDFT00) ===
    std::copy(h_input.begin(), h_input.end(), h_fftw.begin());
    fftw_plan plan = fftw_plan_r2r_2d(Nx1, Ny1, h_fftw.data(), h_fftw.data(),
                                       FFTW_REDFT00, FFTW_REDFT00, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);

    // === CUDA 2D DCT-1 ===
    double* d_data;
    cudaMalloc(&d_data, sizeof(double) * M);
    cudaMemcpy(d_data, h_input.data(), sizeof(double) * M, cudaMemcpyHostToDevice);

    CudaDCT2D dct(Nx, Ny, CUDA_DCT_1);
    dct.execute(d_data);

    cudaMemcpy(h_cuda.data(), d_data, sizeof(double) * M, cudaMemcpyDeviceToHost);
    cudaFree(d_data);

    // Compare results
    double max_err = 0;
    for (int i = 0; i < M; ++i) {
        double err = std::abs(h_fftw[i] - h_cuda[i]);
        max_err = std::max(max_err, err);
    }

    double max_val = *std::max_element(h_fftw.begin(), h_fftw.end(),
        [](double a, double b) { return std::abs(a) < std::abs(b); });
    double rel_err = max_err / std::abs(max_val);

    printf("  Max abs error: %.6e, Relative error: %.6e\n", max_err, rel_err);

    bool passed = rel_err < 1e-10;
    printf("  %s\n\n", passed ? "PASSED!" : "FAILED!");

    return passed;
}

//==============================================================================
// Test 2D DCT-2
//==============================================================================
bool test_dct2_2d(int Nx, int Ny)
{
    printf("Testing 2D DCT-2: %d x %d\n", Nx, Ny);

    int M = Nx * Ny;

    std::vector<double> h_input(M);
    std::vector<double> h_fftw(M);
    std::vector<double> h_cuda(M);

    // Initialize with test data
    for (int ix = 0; ix < Nx; ++ix) {
        for (int iy = 0; iy < Ny; ++iy) {
            int idx = ix * Ny + iy;
            double r2 = (ix + 0.5) * (ix + 0.5) + (iy + 0.5) * (iy + 0.5);
            h_input[idx] = exp(-r2 / 50.0);
        }
    }

    // === FFTW 2D DCT-2 (REDFT10) ===
    std::copy(h_input.begin(), h_input.end(), h_fftw.begin());
    fftw_plan plan = fftw_plan_r2r_2d(Nx, Ny, h_fftw.data(), h_fftw.data(),
                                       FFTW_REDFT10, FFTW_REDFT10, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);

    // === CUDA 2D DCT-2 ===
    double* d_data;
    cudaMalloc(&d_data, sizeof(double) * M);
    cudaMemcpy(d_data, h_input.data(), sizeof(double) * M, cudaMemcpyHostToDevice);

    CudaDCT2D dct(Nx, Ny, CUDA_DCT_2);
    dct.execute(d_data);

    cudaMemcpy(h_cuda.data(), d_data, sizeof(double) * M, cudaMemcpyDeviceToHost);
    cudaFree(d_data);

    // Compare results
    double max_err = 0;
    for (int i = 0; i < M; ++i) {
        double err = std::abs(h_fftw[i] - h_cuda[i]);
        max_err = std::max(max_err, err);
    }

    double max_val = *std::max_element(h_fftw.begin(), h_fftw.end(),
        [](double a, double b) { return std::abs(a) < std::abs(b); });
    double rel_err = max_err / std::abs(max_val);

    printf("  Max abs error: %.6e, Relative error: %.6e\n", max_err, rel_err);

    bool passed = rel_err < 1e-10;
    printf("  %s\n\n", passed ? "PASSED!" : "FAILED!");

    return passed;
}

//==============================================================================
// Test 2D DCT-3
//==============================================================================
bool test_dct3_2d(int Nx, int Ny)
{
    printf("Testing 2D DCT-3: %d x %d\n", Nx, Ny);

    int M = Nx * Ny;

    std::vector<double> h_input(M);
    std::vector<double> h_fftw(M);
    std::vector<double> h_cuda(M);

    // Initialize with test data
    for (int ix = 0; ix < Nx; ++ix) {
        for (int iy = 0; iy < Ny; ++iy) {
            int idx = ix * Ny + iy;
            double r2 = ix * ix + iy * iy;
            h_input[idx] = exp(-r2 / 50.0);
        }
    }

    // === FFTW 2D DCT-3 (REDFT01) ===
    std::copy(h_input.begin(), h_input.end(), h_fftw.begin());
    fftw_plan plan = fftw_plan_r2r_2d(Nx, Ny, h_fftw.data(), h_fftw.data(),
                                       FFTW_REDFT01, FFTW_REDFT01, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);

    // === CUDA 2D DCT-3 ===
    double* d_data;
    cudaMalloc(&d_data, sizeof(double) * M);
    cudaMemcpy(d_data, h_input.data(), sizeof(double) * M, cudaMemcpyHostToDevice);

    CudaDCT2D dct(Nx, Ny, CUDA_DCT_3);
    dct.execute(d_data);

    cudaMemcpy(h_cuda.data(), d_data, sizeof(double) * M, cudaMemcpyDeviceToHost);
    cudaFree(d_data);

    // Compare results
    double max_err = 0;
    for (int i = 0; i < M; ++i) {
        double err = std::abs(h_fftw[i] - h_cuda[i]);
        max_err = std::max(max_err, err);
    }

    double max_val = *std::max_element(h_fftw.begin(), h_fftw.end(),
        [](double a, double b) { return std::abs(a) < std::abs(b); });
    double rel_err = max_err / std::abs(max_val);

    printf("  Max abs error: %.6e, Relative error: %.6e\n", max_err, rel_err);

    bool passed = rel_err < 1e-10;
    printf("  %s\n\n", passed ? "PASSED!" : "FAILED!");

    return passed;
}

//==============================================================================
// Test 2D DCT-4
//==============================================================================
bool test_dct4_2d(int Nx, int Ny)
{
    printf("Testing 2D DCT-4: %d x %d\n", Nx, Ny);

    int M = Nx * Ny;

    std::vector<double> h_input(M);
    std::vector<double> h_fftw(M);
    std::vector<double> h_cuda(M);

    // Initialize with test data
    for (int ix = 0; ix < Nx; ++ix) {
        for (int iy = 0; iy < Ny; ++iy) {
            int idx = ix * Ny + iy;
            double r2 = (ix + 0.5) * (ix + 0.5) + (iy + 0.5) * (iy + 0.5);
            h_input[idx] = exp(-r2 / 50.0);
        }
    }

    // === FFTW 2D DCT-4 (REDFT11) ===
    std::copy(h_input.begin(), h_input.end(), h_fftw.begin());
    fftw_plan plan = fftw_plan_r2r_2d(Nx, Ny, h_fftw.data(), h_fftw.data(),
                                       FFTW_REDFT11, FFTW_REDFT11, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);

    // === CUDA 2D DCT-4 ===
    double* d_data;
    cudaMalloc(&d_data, sizeof(double) * M);
    cudaMemcpy(d_data, h_input.data(), sizeof(double) * M, cudaMemcpyHostToDevice);

    CudaDCT2D dct(Nx, Ny, CUDA_DCT_4);
    dct.execute(d_data);

    cudaMemcpy(h_cuda.data(), d_data, sizeof(double) * M, cudaMemcpyDeviceToHost);
    cudaFree(d_data);

    // Compare results
    double max_err = 0;
    for (int i = 0; i < M; ++i) {
        double err = std::abs(h_fftw[i] - h_cuda[i]);
        max_err = std::max(max_err, err);
    }

    double max_val = *std::max_element(h_fftw.begin(), h_fftw.end(),
        [](double a, double b) { return std::abs(a) < std::abs(b); });
    double rel_err = max_err / std::abs(max_val);

    printf("  Max abs error: %.6e, Relative error: %.6e\n", max_err, rel_err);

    bool passed = rel_err < 1e-10;
    printf("  %s\n\n", passed ? "PASSED!" : "FAILED!");

    return passed;
}

//==============================================================================
// Test 3D DCT-1
//==============================================================================
bool test_dct1_3d(int Nx, int Ny, int Nz)
{
    printf("Testing 3D DCT-1: %d x %d x %d (data size: %d x %d x %d)\n",
           Nx, Ny, Nz, Nx + 1, Ny + 1, Nz + 1);

    int Nx1 = Nx + 1, Ny1 = Ny + 1, Nz1 = Nz + 1;
    int M = Nx1 * Ny1 * Nz1;

    // Allocate host memory
    std::vector<double> h_input(M);
    std::vector<double> h_fftw(M);
    std::vector<double> h_cuda(M);

    // Initialize with test data (symmetric Gaussian)
    for (int ix = 0; ix < Nx1; ++ix) {
        for (int iy = 0; iy < Ny1; ++iy) {
            for (int iz = 0; iz < Nz1; ++iz) {
                int idx = (ix * Ny1 + iy) * Nz1 + iz;
                double r2 = ix * ix + iy * iy + iz * iz;
                h_input[idx] = exp(-r2 / 50.0);
            }
        }
    }

    // === FFTW 3D DCT-1 (REDFT00) ===
    std::copy(h_input.begin(), h_input.end(), h_fftw.begin());
    fftw_plan plan = fftw_plan_r2r_3d(Nx1, Ny1, Nz1, h_fftw.data(), h_fftw.data(),
                                       FFTW_REDFT00, FFTW_REDFT00, FFTW_REDFT00,
                                       FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);

    // === CUDA 3D DCT-1 ===
    double* d_data;
    cudaMalloc(&d_data, sizeof(double) * M);
    cudaMemcpy(d_data, h_input.data(), sizeof(double) * M, cudaMemcpyHostToDevice);

    CudaDCT3D dct(Nx, Ny, Nz, CUDA_DCT_1);
    dct.execute(d_data);

    cudaMemcpy(h_cuda.data(), d_data, sizeof(double) * M, cudaMemcpyDeviceToHost);
    cudaFree(d_data);

    // Compare results
    double max_err = 0;
    for (int i = 0; i < M; ++i) {
        double err = std::abs(h_fftw[i] - h_cuda[i]);
        max_err = std::max(max_err, err);
    }

    double max_val = *std::max_element(h_fftw.begin(), h_fftw.end(),
        [](double a, double b) { return std::abs(a) < std::abs(b); });
    double rel_err = max_err / std::abs(max_val);

    printf("  Max abs error: %.6e, Relative error: %.6e\n", max_err, rel_err);

    bool passed = rel_err < 1e-10;
    printf("  %s\n\n", passed ? "PASSED!" : "FAILED!");

    return passed;
}

//==============================================================================
// Test 3D DCT-2
//==============================================================================
bool test_dct2_3d(int Nx, int Ny, int Nz)
{
    printf("Testing 3D DCT-2: %d x %d x %d\n", Nx, Ny, Nz);

    int M = Nx * Ny * Nz;

    std::vector<double> h_input(M);
    std::vector<double> h_fftw(M);
    std::vector<double> h_cuda(M);

    // Initialize with test data
    for (int ix = 0; ix < Nx; ++ix) {
        for (int iy = 0; iy < Ny; ++iy) {
            for (int iz = 0; iz < Nz; ++iz) {
                int idx = (ix * Ny + iy) * Nz + iz;
                double r2 = (ix + 0.5) * (ix + 0.5) + (iy + 0.5) * (iy + 0.5) + (iz + 0.5) * (iz + 0.5);
                h_input[idx] = exp(-r2 / 50.0);
            }
        }
    }

    // === FFTW 3D DCT-2 (REDFT10) ===
    std::copy(h_input.begin(), h_input.end(), h_fftw.begin());
    fftw_plan plan = fftw_plan_r2r_3d(Nx, Ny, Nz, h_fftw.data(), h_fftw.data(),
                                       FFTW_REDFT10, FFTW_REDFT10, FFTW_REDFT10,
                                       FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);

    // === CUDA 3D DCT-2 ===
    double* d_data;
    cudaMalloc(&d_data, sizeof(double) * M);
    cudaMemcpy(d_data, h_input.data(), sizeof(double) * M, cudaMemcpyHostToDevice);

    CudaDCT3D dct(Nx, Ny, Nz, CUDA_DCT_2);
    dct.execute(d_data);

    cudaMemcpy(h_cuda.data(), d_data, sizeof(double) * M, cudaMemcpyDeviceToHost);
    cudaFree(d_data);

    // Compare results
    double max_err = 0;
    for (int i = 0; i < M; ++i) {
        double err = std::abs(h_fftw[i] - h_cuda[i]);
        max_err = std::max(max_err, err);
    }

    double max_val = *std::max_element(h_fftw.begin(), h_fftw.end(),
        [](double a, double b) { return std::abs(a) < std::abs(b); });
    double rel_err = max_err / std::abs(max_val);

    printf("  Max abs error: %.6e, Relative error: %.6e\n", max_err, rel_err);

    bool passed = rel_err < 1e-10;
    printf("  %s\n\n", passed ? "PASSED!" : "FAILED!");

    return passed;
}

//==============================================================================
// Test 3D DCT-3
//==============================================================================
bool test_dct3_3d(int Nx, int Ny, int Nz)
{
    printf("Testing 3D DCT-3: %d x %d x %d\n", Nx, Ny, Nz);

    int M = Nx * Ny * Nz;

    std::vector<double> h_input(M);
    std::vector<double> h_fftw(M);
    std::vector<double> h_cuda(M);

    // Initialize with test data
    for (int ix = 0; ix < Nx; ++ix) {
        for (int iy = 0; iy < Ny; ++iy) {
            for (int iz = 0; iz < Nz; ++iz) {
                int idx = (ix * Ny + iy) * Nz + iz;
                double r2 = ix * ix + iy * iy + iz * iz;
                h_input[idx] = exp(-r2 / 50.0);
            }
        }
    }

    // === FFTW 3D DCT-3 (REDFT01) ===
    std::copy(h_input.begin(), h_input.end(), h_fftw.begin());
    fftw_plan plan = fftw_plan_r2r_3d(Nx, Ny, Nz, h_fftw.data(), h_fftw.data(),
                                       FFTW_REDFT01, FFTW_REDFT01, FFTW_REDFT01,
                                       FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);

    // === CUDA 3D DCT-3 ===
    double* d_data;
    cudaMalloc(&d_data, sizeof(double) * M);
    cudaMemcpy(d_data, h_input.data(), sizeof(double) * M, cudaMemcpyHostToDevice);

    CudaDCT3D dct(Nx, Ny, Nz, CUDA_DCT_3);
    dct.execute(d_data);

    cudaMemcpy(h_cuda.data(), d_data, sizeof(double) * M, cudaMemcpyDeviceToHost);
    cudaFree(d_data);

    // Compare results
    double max_err = 0;
    for (int i = 0; i < M; ++i) {
        double err = std::abs(h_fftw[i] - h_cuda[i]);
        max_err = std::max(max_err, err);
    }

    double max_val = *std::max_element(h_fftw.begin(), h_fftw.end(),
        [](double a, double b) { return std::abs(a) < std::abs(b); });
    double rel_err = max_err / std::abs(max_val);

    printf("  Max abs error: %.6e, Relative error: %.6e\n", max_err, rel_err);

    bool passed = rel_err < 1e-10;
    printf("  %s\n\n", passed ? "PASSED!" : "FAILED!");

    return passed;
}

//==============================================================================
// Test 3D DCT-4
//==============================================================================
bool test_dct4_3d(int Nx, int Ny, int Nz)
{
    printf("Testing 3D DCT-4: %d x %d x %d\n", Nx, Ny, Nz);

    int M = Nx * Ny * Nz;

    // Allocate host memory
    std::vector<double> h_input(M);
    std::vector<double> h_fftw(M);
    std::vector<double> h_cuda(M);

    // Initialize with test data
    for (int ix = 0; ix < Nx; ++ix) {
        for (int iy = 0; iy < Ny; ++iy) {
            for (int iz = 0; iz < Nz; ++iz) {
                int idx = (ix * Ny + iy) * Nz + iz;
                double r2 = (ix + 0.5) * (ix + 0.5) + (iy + 0.5) * (iy + 0.5) + (iz + 0.5) * (iz + 0.5);
                h_input[idx] = exp(-r2 / 50.0);
            }
        }
    }

    // === FFTW 3D DCT-4 (REDFT11) ===
    std::copy(h_input.begin(), h_input.end(), h_fftw.begin());
    fftw_plan plan = fftw_plan_r2r_3d(Nx, Ny, Nz, h_fftw.data(), h_fftw.data(),
                                       FFTW_REDFT11, FFTW_REDFT11, FFTW_REDFT11,
                                       FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);

    // === CUDA 3D DCT-4 ===
    double* d_data;
    cudaMalloc(&d_data, sizeof(double) * M);
    cudaMemcpy(d_data, h_input.data(), sizeof(double) * M, cudaMemcpyHostToDevice);

    CudaDCT3D dct(Nx, Ny, Nz, CUDA_DCT_4);
    dct.execute(d_data);

    cudaMemcpy(h_cuda.data(), d_data, sizeof(double) * M, cudaMemcpyDeviceToHost);
    cudaFree(d_data);

    // Compare results
    double max_err = 0;
    for (int i = 0; i < M; ++i) {
        double err = std::abs(h_fftw[i] - h_cuda[i]);
        max_err = std::max(max_err, err);
    }

    double max_val = *std::max_element(h_fftw.begin(), h_fftw.end(),
        [](double a, double b) { return std::abs(a) < std::abs(b); });
    double rel_err = max_err / std::abs(max_val);

    printf("  Max abs error: %.6e, Relative error: %.6e\n", max_err, rel_err);

    bool passed = rel_err < 1e-10;
    printf("  %s\n\n", passed ? "PASSED!" : "FAILED!");

    return passed;
}

//==============================================================================
// Test round-trip (forward + backward should recover input)
//==============================================================================
bool test_dct1_roundtrip(int N)
{
    printf("Testing DCT-1 round-trip: N=%d (size=%d)\n", N, N + 1);

    int size = N + 1;

    std::vector<double> h_input(size);
    std::vector<double> h_result(size);

    // Initialize
    for (int i = 0; i < size; ++i) {
        h_input[i] = sin(M_PI * i / N) + 0.5 * cos(2 * M_PI * i / N);
    }

    // FFTW: forward + backward
    std::copy(h_input.begin(), h_input.end(), h_result.begin());

    fftw_plan plan = fftw_plan_r2r_1d(size, h_result.data(), h_result.data(), FFTW_REDFT00, FFTW_ESTIMATE);
    fftw_execute(plan);  // Forward
    fftw_execute(plan);  // Backward (DCT-1 is self-inverse)
    fftw_destroy_plan(plan);

    // Compute actual scaling factor
    double scale = h_result[1] / h_input[1];  // Use a middle element
    printf("  Actual scaling factor: %.4f (expected 2*N=%.4f)\n", scale, 2.0 * N);

    // FFTW DCT-1: round-trip multiplies by 2*(n-1) = 2*N for size=N+1
    double norm = 1.0 / (2.0 * N);
    for (int i = 0; i < size; ++i) {
        h_result[i] *= norm;
    }

    // Check recovery
    double max_err = 0;
    for (int i = 0; i < size; ++i) {
        double err = std::abs(h_input[i] - h_result[i]);
        max_err = std::max(max_err, err);
    }

    double max_val = *std::max_element(h_input.begin(), h_input.end(),
        [](double a, double b) { return std::abs(a) < std::abs(b); });
    double rel_err = max_err / std::abs(max_val);

    printf("  FFTW round-trip: Max abs error: %.6e, Relative error: %.6e\n", max_err, rel_err);

    bool passed = rel_err < 1e-12;
    printf("  %s\n\n", passed ? "PASSED!" : "FAILED!");

    return passed;
}

//==============================================================================
// Test DCT-2/DCT-3 inverse pair
//==============================================================================
bool test_dct23_inverse(int N)
{
    printf("Testing DCT-2/DCT-3 inverse pair: N=%d\n", N);

    std::vector<double> h_input(N);
    std::vector<double> h_result(N);

    // Initialize
    for (int i = 0; i < N; ++i) {
        h_input[i] = sin(M_PI * (i + 0.5) / N) + 0.5 * cos(2 * M_PI * (i + 0.5) / N);
    }

    // FFTW: DCT-2 then DCT-3 should recover input (up to scale)
    std::copy(h_input.begin(), h_input.end(), h_result.begin());

    fftw_plan plan2 = fftw_plan_r2r_1d(N, h_result.data(), h_result.data(), FFTW_REDFT10, FFTW_ESTIMATE);
    fftw_execute(plan2);
    fftw_destroy_plan(plan2);

    fftw_plan plan3 = fftw_plan_r2r_1d(N, h_result.data(), h_result.data(), FFTW_REDFT01, FFTW_ESTIMATE);
    fftw_execute(plan3);
    fftw_destroy_plan(plan3);

    // Normalization: 2N for round-trip
    double norm = 1.0 / (2.0 * N);
    for (int i = 0; i < N; ++i) {
        h_result[i] *= norm;
    }

    // Check recovery
    double max_err = 0;
    for (int i = 0; i < N; ++i) {
        double err = std::abs(h_input[i] - h_result[i]);
        max_err = std::max(max_err, err);
    }

    double max_val = *std::max_element(h_input.begin(), h_input.end(),
        [](double a, double b) { return std::abs(a) < std::abs(b); });
    double rel_err = max_err / std::abs(max_val);

    printf("  FFTW DCT-2 + DCT-3 round-trip: rel_err = %.6e\n", rel_err);

    bool passed = rel_err < 1e-12;
    printf("  %s\n\n", passed ? "PASSED!" : "FAILED!");

    return passed;
}

//==============================================================================
// Test CUDA DCT-1 round-trip
//==============================================================================
bool test_cuda_dct1_roundtrip(int N)
{
    printf("Testing CUDA DCT-1 round-trip: N=%d (size=%d)\n", N, N + 1);

    int size = N + 1;

    std::vector<double> h_input(size);
    std::vector<double> h_result(size);

    // Initialize
    for (int i = 0; i < size; ++i) {
        h_input[i] = sin(M_PI * i / N) + 0.5 * cos(2 * M_PI * i / N);
    }

    // CUDA: forward + backward
    double* d_data;
    cudaMalloc(&d_data, sizeof(double) * size);
    cudaMemcpy(d_data, h_input.data(), sizeof(double) * size, cudaMemcpyHostToDevice);

    CudaDCT dct(size, CUDA_DCT_1);
    dct.execute(d_data);  // Forward
    dct.execute(d_data);  // Backward (DCT-1 is self-inverse)

    cudaMemcpy(h_result.data(), d_data, sizeof(double) * size, cudaMemcpyDeviceToHost);
    cudaFree(d_data);

    // Compute actual scaling factor
    double scale = h_result[1] / h_input[1];
    printf("  Actual scaling factor: %.4f (expected 2*N=%.4f)\n", scale, 2.0 * N);

    // Apply normalization: 1/(2*N)
    double norm = 1.0 / (2.0 * N);
    for (int i = 0; i < size; ++i) {
        h_result[i] *= norm;
    }

    // Check recovery
    double max_err = 0;
    for (int i = 0; i < size; ++i) {
        double err = std::abs(h_input[i] - h_result[i]);
        max_err = std::max(max_err, err);
    }

    double max_val = *std::max_element(h_input.begin(), h_input.end(),
        [](double a, double b) { return std::abs(a) < std::abs(b); });
    double rel_err = max_err / std::abs(max_val);

    printf("  Max abs error: %.6e, Relative error: %.6e\n", max_err, rel_err);

    bool passed = rel_err < 1e-12;
    printf("  %s\n\n", passed ? "PASSED!" : "FAILED!");

    return passed;
}

//==============================================================================
// Test 3D CUDA DCT-1 round-trip
//==============================================================================
bool test_cuda_dct1_3d_roundtrip(int Nx, int Ny, int Nz)
{
    printf("Testing CUDA 3D DCT-1 round-trip: %d x %d x %d\n", Nx, Ny, Nz);

    int Nx1 = Nx + 1, Ny1 = Ny + 1, Nz1 = Nz + 1;
    int M = Nx1 * Ny1 * Nz1;

    std::vector<double> h_input(M);
    std::vector<double> h_result(M);

    // Initialize - use non-zero values at interior point
    for (int ix = 0; ix < Nx1; ++ix) {
        for (int iy = 0; iy < Ny1; ++iy) {
            for (int iz = 0; iz < Nz1; ++iz) {
                int idx = (ix * Ny1 + iy) * Nz1 + iz;
                double r2 = ix * ix + iy * iy + iz * iz;
                h_input[idx] = exp(-r2 / 50.0);
            }
        }
    }

    // CUDA: forward + backward
    double* d_data;
    cudaMalloc(&d_data, sizeof(double) * M);
    cudaMemcpy(d_data, h_input.data(), sizeof(double) * M, cudaMemcpyHostToDevice);

    CudaDCT3D dct(Nx, Ny, Nz, CUDA_DCT_1);
    dct.execute(d_data);  // Forward
    dct.execute(d_data);  // Backward

    cudaMemcpy(h_result.data(), d_data, sizeof(double) * M, cudaMemcpyDeviceToHost);
    cudaFree(d_data);

    // Compute actual scaling at interior point
    int test_idx = (1 * Ny1 + 1) * Nz1 + 1;  // Point (1,1,1)
    double scale = h_result[test_idx] / h_input[test_idx];
    printf("  Actual scaling factor: %.4f (expected 8*Nx*Ny*Nz=%.4f)\n", scale, 8.0 * Nx * Ny * Nz);

    // Normalization: 1/(8*Nx*Ny*Nz) for round-trip
    double norm = 1.0 / (8.0 * Nx * Ny * Nz);
    for (int i = 0; i < M; ++i) {
        h_result[i] *= norm;
    }

    // Check recovery
    double max_err = 0;
    for (int i = 0; i < M; ++i) {
        double err = std::abs(h_input[i] - h_result[i]);
        max_err = std::max(max_err, err);
    }

    double max_val = *std::max_element(h_input.begin(), h_input.end(),
        [](double a, double b) { return std::abs(a) < std::abs(b); });
    double rel_err = max_err / std::abs(max_val);

    printf("  Max abs error: %.6e, Relative error: %.6e\n", max_err, rel_err);

    bool passed = rel_err < 1e-10;
    printf("  %s\n\n", passed ? "PASSED!" : "FAILED!");

    return passed;
}

//==============================================================================
// Test 2D Mixed DCT (vs FFTW)
//==============================================================================
bool test_mixed_2d(int Nx, int Ny, CudaDCTType type_x, CudaDCTType type_y)
{
    const char* type_names[] = {"DCT-1", "DCT-2", "DCT-3", "DCT-4"};
    printf("Testing 2D Mixed (%s, %s): %d x %d\n",
           type_names[type_x], type_names[type_y], Nx, Ny);

    int M = Nx * Ny;

    std::vector<double> h_input(M);
    std::vector<double> h_fftw(M);
    std::vector<double> h_cuda(M);

    // Initialize
    for (int ix = 0; ix < Nx; ++ix) {
        for (int iy = 0; iy < Ny; ++iy) {
            int idx = ix * Ny + iy;
            h_input[idx] = sin(M_PI * (ix + 0.5) / Nx) * cos(M_PI * (iy + 0.5) / Ny)
                         + 0.5 * cos(2 * M_PI * (ix + 0.5) / Nx) * sin(2 * M_PI * (iy + 0.5) / Ny);
        }
    }

    // FFTW reference: separate 1D transforms per dimension
    std::copy(h_input.begin(), h_input.end(), h_fftw.begin());

    // Y dimension first (contiguous)
    fftw_r2r_kind kind_y = (type_y == CUDA_DCT_2) ? FFTW_REDFT10 :
                           (type_y == CUDA_DCT_3) ? FFTW_REDFT01 : FFTW_REDFT11;
    for (int ix = 0; ix < Nx; ++ix) {
        fftw_plan plan = fftw_plan_r2r_1d(Ny, &h_fftw[ix * Ny], &h_fftw[ix * Ny], kind_y, FFTW_ESTIMATE);
        fftw_execute(plan);
        fftw_destroy_plan(plan);
    }

    // X dimension (need transpose for FFTW or do row-by-row)
    std::vector<double> h_col(Nx);
    fftw_r2r_kind kind_x = (type_x == CUDA_DCT_2) ? FFTW_REDFT10 :
                           (type_x == CUDA_DCT_3) ? FFTW_REDFT01 : FFTW_REDFT11;
    for (int iy = 0; iy < Ny; ++iy) {
        // Extract column
        for (int ix = 0; ix < Nx; ++ix) {
            h_col[ix] = h_fftw[ix * Ny + iy];
        }
        fftw_plan plan = fftw_plan_r2r_1d(Nx, h_col.data(), h_col.data(), kind_x, FFTW_ESTIMATE);
        fftw_execute(plan);
        fftw_destroy_plan(plan);
        // Put back
        for (int ix = 0; ix < Nx; ++ix) {
            h_fftw[ix * Ny + iy] = h_col[ix];
        }
    }

    // CUDA
    double* d_data;
    cudaMalloc(&d_data, sizeof(double) * M);
    cudaMemcpy(d_data, h_input.data(), sizeof(double) * M, cudaMemcpyHostToDevice);

    CudaDCT2D dct(Nx, Ny, type_x, type_y);
    dct.execute(d_data);

    cudaMemcpy(h_cuda.data(), d_data, sizeof(double) * M, cudaMemcpyDeviceToHost);
    cudaFree(d_data);

    // Compare
    double max_err = 0;
    for (int i = 0; i < M; ++i) {
        double err = std::abs(h_fftw[i] - h_cuda[i]);
        max_err = std::max(max_err, err);
    }

    double max_val = *std::max_element(h_fftw.begin(), h_fftw.end(),
        [](double a, double b) { return std::abs(a) < std::abs(b); });
    double rel_err = max_err / std::abs(max_val);

    printf("  Max abs error: %.6e, Relative error: %.6e\n", max_err, rel_err);

    bool passed = rel_err < 1e-10;
    printf("  %s\n\n", passed ? "PASSED!" : "FAILED!");

    return passed;
}

//==============================================================================
// Test 3D Mixed DCT (vs FFTW)
//==============================================================================
bool test_mixed_3d(int Nx, int Ny, int Nz, CudaDCTType type_x, CudaDCTType type_y, CudaDCTType type_z)
{
    const char* type_names[] = {"DCT-1", "DCT-2", "DCT-3", "DCT-4"};
    printf("Testing 3D Mixed (%s, %s, %s): %d x %d x %d\n",
           type_names[type_x], type_names[type_y], type_names[type_z], Nx, Ny, Nz);

    int M = Nx * Ny * Nz;

    std::vector<double> h_input(M);
    std::vector<double> h_fftw(M);
    std::vector<double> h_cuda(M);

    // Initialize
    for (int ix = 0; ix < Nx; ++ix) {
        for (int iy = 0; iy < Ny; ++iy) {
            for (int iz = 0; iz < Nz; ++iz) {
                int idx = (ix * Ny + iy) * Nz + iz;
                h_input[idx] = sin(M_PI * (ix + 0.5) / Nx)
                             * cos(M_PI * (iy + 0.5) / Ny)
                             * sin(M_PI * (iz + 0.5) / Nz);
            }
        }
    }

    // FFTW reference
    std::copy(h_input.begin(), h_input.end(), h_fftw.begin());

    // Z dimension (contiguous)
    fftw_r2r_kind kind_z = (type_z == CUDA_DCT_2) ? FFTW_REDFT10 :
                           (type_z == CUDA_DCT_3) ? FFTW_REDFT01 : FFTW_REDFT11;
    for (int ix = 0; ix < Nx; ++ix) {
        for (int iy = 0; iy < Ny; ++iy) {
            int offset = (ix * Ny + iy) * Nz;
            fftw_plan plan = fftw_plan_r2r_1d(Nz, &h_fftw[offset], &h_fftw[offset], kind_z, FFTW_ESTIMATE);
            fftw_execute(plan);
            fftw_destroy_plan(plan);
        }
    }

    // Y dimension
    fftw_r2r_kind kind_y = (type_y == CUDA_DCT_2) ? FFTW_REDFT10 :
                           (type_y == CUDA_DCT_3) ? FFTW_REDFT01 : FFTW_REDFT11;
    std::vector<double> h_tmp_y(Ny);
    for (int ix = 0; ix < Nx; ++ix) {
        for (int iz = 0; iz < Nz; ++iz) {
            for (int iy = 0; iy < Ny; ++iy) {
                h_tmp_y[iy] = h_fftw[(ix * Ny + iy) * Nz + iz];
            }
            fftw_plan plan = fftw_plan_r2r_1d(Ny, h_tmp_y.data(), h_tmp_y.data(), kind_y, FFTW_ESTIMATE);
            fftw_execute(plan);
            fftw_destroy_plan(plan);
            for (int iy = 0; iy < Ny; ++iy) {
                h_fftw[(ix * Ny + iy) * Nz + iz] = h_tmp_y[iy];
            }
        }
    }

    // X dimension
    fftw_r2r_kind kind_x = (type_x == CUDA_DCT_2) ? FFTW_REDFT10 :
                           (type_x == CUDA_DCT_3) ? FFTW_REDFT01 : FFTW_REDFT11;
    std::vector<double> h_tmp_x(Nx);
    for (int iy = 0; iy < Ny; ++iy) {
        for (int iz = 0; iz < Nz; ++iz) {
            for (int ix = 0; ix < Nx; ++ix) {
                h_tmp_x[ix] = h_fftw[(ix * Ny + iy) * Nz + iz];
            }
            fftw_plan plan = fftw_plan_r2r_1d(Nx, h_tmp_x.data(), h_tmp_x.data(), kind_x, FFTW_ESTIMATE);
            fftw_execute(plan);
            fftw_destroy_plan(plan);
            for (int ix = 0; ix < Nx; ++ix) {
                h_fftw[(ix * Ny + iy) * Nz + iz] = h_tmp_x[ix];
            }
        }
    }

    // CUDA
    double* d_data;
    cudaMalloc(&d_data, sizeof(double) * M);
    cudaMemcpy(d_data, h_input.data(), sizeof(double) * M, cudaMemcpyHostToDevice);

    CudaDCT3D dct(Nx, Ny, Nz, type_x, type_y, type_z);
    dct.execute(d_data);

    cudaMemcpy(h_cuda.data(), d_data, sizeof(double) * M, cudaMemcpyDeviceToHost);
    cudaFree(d_data);

    // Compare
    double max_err = 0;
    for (int i = 0; i < M; ++i) {
        double err = std::abs(h_fftw[i] - h_cuda[i]);
        max_err = std::max(max_err, err);
    }

    double max_val = *std::max_element(h_fftw.begin(), h_fftw.end(),
        [](double a, double b) { return std::abs(a) < std::abs(b); });
    double rel_err = max_err / std::abs(max_val);

    printf("  Max abs error: %.6e, Relative error: %.6e\n", max_err, rel_err);

    bool passed = rel_err < 1e-10;
    printf("  %s\n\n", passed ? "PASSED!" : "FAILED!");

    return passed;
}

//==============================================================================
// Main
//==============================================================================
int main()
{
    std::cout << "========== CUDA DCT vs FFTW Test ==========\n\n";

    bool all_passed = true;

    // Test 1D DCT-1
    std::cout << "--- 1D DCT-1 Tests (vs FFTW REDFT00) ---\n\n";
    all_passed &= test_dct1_1d(16);
    all_passed &= test_dct1_1d(32);
    all_passed &= test_dct1_1d(64);

    // Test 1D DCT-2
    std::cout << "--- 1D DCT-2 Tests (vs FFTW REDFT10) ---\n\n";
    all_passed &= test_dct2_1d(16);
    all_passed &= test_dct2_1d(32);
    all_passed &= test_dct2_1d(64);

    // Test 1D DCT-3
    std::cout << "--- 1D DCT-3 Tests (vs FFTW REDFT01) ---\n\n";
    all_passed &= test_dct3_1d(16);
    all_passed &= test_dct3_1d(32);
    all_passed &= test_dct3_1d(64);

    // Test 1D DCT-4
    std::cout << "--- 1D DCT-4 Tests (vs FFTW REDFT11) ---\n\n";
    all_passed &= test_dct4_1d(16);
    all_passed &= test_dct4_1d(32);
    all_passed &= test_dct4_1d(64);

    // Test 2D DCT-1
    std::cout << "--- 2D DCT-1 Tests (vs FFTW REDFT00) ---\n\n";
    all_passed &= test_dct1_2d(8, 8);
    all_passed &= test_dct1_2d(16, 16);
    all_passed &= test_dct1_2d(32, 32);

    // Test 2D DCT-2
    std::cout << "--- 2D DCT-2 Tests (vs FFTW REDFT10) ---\n\n";
    all_passed &= test_dct2_2d(8, 8);
    all_passed &= test_dct2_2d(16, 16);
    all_passed &= test_dct2_2d(32, 32);

    // Test 2D DCT-3
    std::cout << "--- 2D DCT-3 Tests (vs FFTW REDFT01) ---\n\n";
    all_passed &= test_dct3_2d(8, 8);
    all_passed &= test_dct3_2d(16, 16);
    all_passed &= test_dct3_2d(32, 32);

    // Test 2D DCT-4
    std::cout << "--- 2D DCT-4 Tests (vs FFTW REDFT11) ---\n\n";
    all_passed &= test_dct4_2d(8, 8);
    all_passed &= test_dct4_2d(16, 16);
    all_passed &= test_dct4_2d(32, 32);

    // Test 3D DCT-1
    std::cout << "--- 3D DCT-1 Tests (vs FFTW REDFT00) ---\n\n";
    all_passed &= test_dct1_3d(8, 8, 8);
    all_passed &= test_dct1_3d(16, 16, 16);
    all_passed &= test_dct1_3d(32, 32, 32);

    // Test 3D DCT-2
    std::cout << "--- 3D DCT-2 Tests (vs FFTW REDFT10) ---\n\n";
    all_passed &= test_dct2_3d(8, 8, 8);
    all_passed &= test_dct2_3d(16, 16, 16);
    all_passed &= test_dct2_3d(32, 32, 32);

    // Test 3D DCT-3
    std::cout << "--- 3D DCT-3 Tests (vs FFTW REDFT01) ---\n\n";
    all_passed &= test_dct3_3d(8, 8, 8);
    all_passed &= test_dct3_3d(16, 16, 16);
    all_passed &= test_dct3_3d(32, 32, 32);

    // Test 3D DCT-4
    std::cout << "--- 3D DCT-4 Tests (vs FFTW REDFT11) ---\n\n";
    all_passed &= test_dct4_3d(8, 8, 8);
    all_passed &= test_dct4_3d(16, 16, 16);
    all_passed &= test_dct4_3d(32, 32, 32);

    // Test round-trip
    std::cout << "--- Round-trip Tests ---\n\n";
    all_passed &= test_dct1_roundtrip(32);
    all_passed &= test_cuda_dct1_roundtrip(32);
    all_passed &= test_cuda_dct1_3d_roundtrip(16, 16, 16);
    all_passed &= test_dct23_inverse(32);

    // Test 2D Mixed DCT
    std::cout << "--- 2D Mixed DCT Tests ---\n\n";
    all_passed &= test_mixed_2d(16, 16, CUDA_DCT_2, CUDA_DCT_3);
    all_passed &= test_mixed_2d(16, 16, CUDA_DCT_3, CUDA_DCT_2);
    all_passed &= test_mixed_2d(16, 16, CUDA_DCT_2, CUDA_DCT_4);
    all_passed &= test_mixed_2d(16, 16, CUDA_DCT_4, CUDA_DCT_3);

    // Test 3D Mixed DCT
    std::cout << "--- 3D Mixed DCT Tests ---\n\n";
    all_passed &= test_mixed_3d(8, 8, 8, CUDA_DCT_2, CUDA_DCT_3, CUDA_DCT_4);
    all_passed &= test_mixed_3d(8, 8, 8, CUDA_DCT_4, CUDA_DCT_2, CUDA_DCT_3);
    all_passed &= test_mixed_3d(8, 8, 8, CUDA_DCT_3, CUDA_DCT_4, CUDA_DCT_2);

    if (all_passed)
        std::cout << "=== All tests PASSED! ===\n";
    else
        std::cout << "=== Some tests FAILED! ===\n";

    return all_passed ? 0 : 1;
}
