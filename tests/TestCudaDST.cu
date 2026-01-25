/**
 * @file TestCudaDST.cu
 * @brief Test CUDA DST Types 1-4 against FFTW reference.
 *
 * Compares CudaDST output with FFTW RODFT00/10/01/11 for all DST types.
 */

#include <iostream>
#include <cmath>
#include <vector>
#include <cstdlib>
#include <algorithm>
#include <cuda_runtime.h>
#include <fftw3.h>

#include "../src/platforms/cuda/CudaRealTransform.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

//==============================================================================
// Test 1D DST-1 (FFTW_RODFT00)
//==============================================================================
bool test_dst1_1d(int N)
{
    printf("Testing 1D DST-1: N=%d points\n", N);

    // Allocate host memory
    std::vector<double> h_input(N);
    std::vector<double> h_fftw(N);
    std::vector<double> h_cuda(N);

    // Initialize with test data (sine functions work well with DST)
    for (int i = 0; i < N; ++i) {
        h_input[i] = sin(M_PI * (i + 1) / (N + 1)) + 0.5 * sin(2 * M_PI * (i + 1) / (N + 1));
    }

    // === FFTW DST-1 (RODFT00) ===
    std::copy(h_input.begin(), h_input.end(), h_fftw.begin());
    fftw_plan plan = fftw_plan_r2r_1d(N, h_fftw.data(), h_fftw.data(), FFTW_RODFT00, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);

    // === CUDA DST-1 ===
    double* d_data;
    cudaMalloc(&d_data, sizeof(double) * N);
    cudaMemcpy(d_data, h_input.data(), sizeof(double) * N, cudaMemcpyHostToDevice);

    CudaDST dst(N, CUDA_DST_1);
    dst.execute(d_data);

    cudaMemcpy(h_cuda.data(), d_data, sizeof(double) * N, cudaMemcpyDeviceToHost);
    cudaFree(d_data);

    // Compare results
    double max_err = 0;
    for (int i = 0; i < N; ++i) {
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
// Test 1D DST-2 (FFTW_RODFT10)
//==============================================================================
bool test_dst2_1d(int N)
{
    printf("Testing 1D DST-2: N=%d points\n", N);

    // Allocate host memory
    std::vector<double> h_input(N);
    std::vector<double> h_fftw(N);
    std::vector<double> h_cuda(N);

    // Initialize with test data
    for (int i = 0; i < N; ++i) {
        h_input[i] = sin(M_PI * (i + 0.5) / N) + 0.5 * sin(2 * M_PI * (i + 0.5) / N);
    }

    // === FFTW DST-2 (RODFT10) ===
    std::copy(h_input.begin(), h_input.end(), h_fftw.begin());
    fftw_plan plan = fftw_plan_r2r_1d(N, h_fftw.data(), h_fftw.data(), FFTW_RODFT10, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);

    // === CUDA DST-2 ===
    double* d_data;
    cudaMalloc(&d_data, sizeof(double) * N);
    cudaMemcpy(d_data, h_input.data(), sizeof(double) * N, cudaMemcpyHostToDevice);

    CudaDST dst(N, CUDA_DST_2);
    dst.execute(d_data);

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
// Test 1D DST-3 (FFTW_RODFT01)
//==============================================================================
bool test_dst3_1d(int N)
{
    printf("Testing 1D DST-3: N=%d points\n", N);

    // Allocate host memory
    std::vector<double> h_input(N);
    std::vector<double> h_fftw(N);
    std::vector<double> h_cuda(N);

    // Initialize with test data
    for (int i = 0; i < N; ++i) {
        h_input[i] = sin(M_PI * (i + 1) / (N + 1)) + 0.5 * sin(2 * M_PI * (i + 1) / (N + 1));
    }

    // === FFTW DST-3 (RODFT01) ===
    std::copy(h_input.begin(), h_input.end(), h_fftw.begin());
    fftw_plan plan = fftw_plan_r2r_1d(N, h_fftw.data(), h_fftw.data(), FFTW_RODFT01, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);

    // === CUDA DST-3 ===
    double* d_data;
    cudaMalloc(&d_data, sizeof(double) * N);
    cudaMemcpy(d_data, h_input.data(), sizeof(double) * N, cudaMemcpyHostToDevice);

    CudaDST dst(N, CUDA_DST_3);
    dst.execute(d_data);

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
// Test 1D DST-4 (FFTW_RODFT11)
//==============================================================================
bool test_dst4_1d(int N)
{
    printf("Testing 1D DST-4: N=%d points\n", N);

    // Allocate host memory
    std::vector<double> h_input(N);
    std::vector<double> h_fftw(N);
    std::vector<double> h_cuda(N);

    // Initialize with test data
    for (int i = 0; i < N; ++i) {
        h_input[i] = sin(M_PI * (i + 0.5) / N) + 0.5 * sin(2 * M_PI * (i + 0.5) / N);
    }

    // === FFTW DST-4 (RODFT11) ===
    std::copy(h_input.begin(), h_input.end(), h_fftw.begin());
    fftw_plan plan = fftw_plan_r2r_1d(N, h_fftw.data(), h_fftw.data(), FFTW_RODFT11, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);

    // === CUDA DST-4 ===
    double* d_data;
    cudaMalloc(&d_data, sizeof(double) * N);
    cudaMemcpy(d_data, h_input.data(), sizeof(double) * N, cudaMemcpyHostToDevice);

    CudaDST dst(N, CUDA_DST_4);
    dst.execute(d_data);

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
// Test 2D DST-1
//==============================================================================
bool test_dst1_2d(int Nx, int Ny)
{
    printf("Testing 2D DST-1: %d x %d\n", Nx, Ny);

    int M = Nx * Ny;

    std::vector<double> h_input(M);
    std::vector<double> h_fftw(M);
    std::vector<double> h_cuda(M);

    // Initialize with test data
    for (int ix = 0; ix < Nx; ++ix) {
        for (int iy = 0; iy < Ny; ++iy) {
            int idx = ix * Ny + iy;
            double x = (ix + 1.0) / (Nx + 1);
            double y = (iy + 1.0) / (Ny + 1);
            h_input[idx] = sin(M_PI * x) * sin(M_PI * y);
        }
    }

    // === FFTW 2D DST-1 (RODFT00) ===
    std::copy(h_input.begin(), h_input.end(), h_fftw.begin());
    fftw_plan plan = fftw_plan_r2r_2d(Nx, Ny, h_fftw.data(), h_fftw.data(),
                                       FFTW_RODFT00, FFTW_RODFT00, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);

    // === CUDA 2D DST-1 ===
    double* d_data;
    cudaMalloc(&d_data, sizeof(double) * M);
    cudaMemcpy(d_data, h_input.data(), sizeof(double) * M, cudaMemcpyHostToDevice);

    CudaDST2D dst(Nx, Ny, CUDA_DST_1);
    dst.execute(d_data);

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
// Test 2D DST-2
//==============================================================================
bool test_dst2_2d(int Nx, int Ny)
{
    printf("Testing 2D DST-2: %d x %d\n", Nx, Ny);

    int M = Nx * Ny;

    std::vector<double> h_input(M);
    std::vector<double> h_fftw(M);
    std::vector<double> h_cuda(M);

    // Initialize with test data
    for (int ix = 0; ix < Nx; ++ix) {
        for (int iy = 0; iy < Ny; ++iy) {
            int idx = ix * Ny + iy;
            double x = (ix + 0.5) / Nx;
            double y = (iy + 0.5) / Ny;
            h_input[idx] = sin(M_PI * x) * sin(M_PI * y);
        }
    }

    // === FFTW 2D DST-2 (RODFT10) ===
    std::copy(h_input.begin(), h_input.end(), h_fftw.begin());
    fftw_plan plan = fftw_plan_r2r_2d(Nx, Ny, h_fftw.data(), h_fftw.data(),
                                       FFTW_RODFT10, FFTW_RODFT10, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);

    // === CUDA 2D DST-2 ===
    double* d_data;
    cudaMalloc(&d_data, sizeof(double) * M);
    cudaMemcpy(d_data, h_input.data(), sizeof(double) * M, cudaMemcpyHostToDevice);

    CudaDST2D dst(Nx, Ny, CUDA_DST_2);
    dst.execute(d_data);

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
// Test 2D DST-3
//==============================================================================
bool test_dst3_2d(int Nx, int Ny)
{
    printf("Testing 2D DST-3: %d x %d\n", Nx, Ny);

    int M = Nx * Ny;

    std::vector<double> h_input(M);
    std::vector<double> h_fftw(M);
    std::vector<double> h_cuda(M);

    // Initialize with test data
    for (int ix = 0; ix < Nx; ++ix) {
        for (int iy = 0; iy < Ny; ++iy) {
            int idx = ix * Ny + iy;
            double x = (ix + 1.0) / (Nx + 1);
            double y = (iy + 1.0) / (Ny + 1);
            h_input[idx] = sin(M_PI * x) * sin(M_PI * y);
        }
    }

    // === FFTW 2D DST-3 (RODFT01) ===
    std::copy(h_input.begin(), h_input.end(), h_fftw.begin());
    fftw_plan plan = fftw_plan_r2r_2d(Nx, Ny, h_fftw.data(), h_fftw.data(),
                                       FFTW_RODFT01, FFTW_RODFT01, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);

    // === CUDA 2D DST-3 ===
    double* d_data;
    cudaMalloc(&d_data, sizeof(double) * M);
    cudaMemcpy(d_data, h_input.data(), sizeof(double) * M, cudaMemcpyHostToDevice);

    CudaDST2D dst(Nx, Ny, CUDA_DST_3);
    dst.execute(d_data);

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
// Test 2D DST-4
//==============================================================================
bool test_dst4_2d(int Nx, int Ny)
{
    printf("Testing 2D DST-4: %d x %d\n", Nx, Ny);

    int M = Nx * Ny;

    std::vector<double> h_input(M);
    std::vector<double> h_fftw(M);
    std::vector<double> h_cuda(M);

    // Initialize with test data
    for (int ix = 0; ix < Nx; ++ix) {
        for (int iy = 0; iy < Ny; ++iy) {
            int idx = ix * Ny + iy;
            double x = (ix + 0.5) / Nx;
            double y = (iy + 0.5) / Ny;
            h_input[idx] = sin(M_PI * x) * sin(M_PI * y);
        }
    }

    // === FFTW 2D DST-4 (RODFT11) ===
    std::copy(h_input.begin(), h_input.end(), h_fftw.begin());
    fftw_plan plan = fftw_plan_r2r_2d(Nx, Ny, h_fftw.data(), h_fftw.data(),
                                       FFTW_RODFT11, FFTW_RODFT11, FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);

    // === CUDA 2D DST-4 ===
    double* d_data;
    cudaMalloc(&d_data, sizeof(double) * M);
    cudaMemcpy(d_data, h_input.data(), sizeof(double) * M, cudaMemcpyHostToDevice);

    CudaDST2D dst(Nx, Ny, CUDA_DST_4);
    dst.execute(d_data);

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
// Test 3D DST-1
//==============================================================================
bool test_dst1_3d(int Nx, int Ny, int Nz)
{
    printf("Testing 3D DST-1: %d x %d x %d\n", Nx, Ny, Nz);

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
                double x = (ix + 1.0) / (Nx + 1);
                double y = (iy + 1.0) / (Ny + 1);
                double z = (iz + 1.0) / (Nz + 1);
                h_input[idx] = sin(M_PI * x) * sin(M_PI * y) * sin(M_PI * z);
            }
        }
    }

    // === FFTW 3D DST-1 (RODFT00) ===
    std::copy(h_input.begin(), h_input.end(), h_fftw.begin());
    fftw_plan plan = fftw_plan_r2r_3d(Nx, Ny, Nz, h_fftw.data(), h_fftw.data(),
                                       FFTW_RODFT00, FFTW_RODFT00, FFTW_RODFT00,
                                       FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);

    // === CUDA 3D DST-1 ===
    double* d_data;
    cudaMalloc(&d_data, sizeof(double) * M);
    cudaMemcpy(d_data, h_input.data(), sizeof(double) * M, cudaMemcpyHostToDevice);

    CudaDST3D dst(Nx, Ny, Nz, CUDA_DST_1);
    dst.execute(d_data);

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
// Test 3D DST-2
//==============================================================================
bool test_dst2_3d(int Nx, int Ny, int Nz)
{
    printf("Testing 3D DST-2: %d x %d x %d\n", Nx, Ny, Nz);

    int M = Nx * Ny * Nz;

    std::vector<double> h_input(M);
    std::vector<double> h_fftw(M);
    std::vector<double> h_cuda(M);

    // Initialize with test data
    for (int ix = 0; ix < Nx; ++ix) {
        for (int iy = 0; iy < Ny; ++iy) {
            for (int iz = 0; iz < Nz; ++iz) {
                int idx = (ix * Ny + iy) * Nz + iz;
                double x = (ix + 0.5) / Nx;
                double y = (iy + 0.5) / Ny;
                double z = (iz + 0.5) / Nz;
                h_input[idx] = sin(M_PI * x) * sin(M_PI * y) * sin(M_PI * z);
            }
        }
    }

    // === FFTW 3D DST-2 (RODFT10) ===
    std::copy(h_input.begin(), h_input.end(), h_fftw.begin());
    fftw_plan plan = fftw_plan_r2r_3d(Nx, Ny, Nz, h_fftw.data(), h_fftw.data(),
                                       FFTW_RODFT10, FFTW_RODFT10, FFTW_RODFT10,
                                       FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);

    // === CUDA 3D DST-2 ===
    double* d_data;
    cudaMalloc(&d_data, sizeof(double) * M);
    cudaMemcpy(d_data, h_input.data(), sizeof(double) * M, cudaMemcpyHostToDevice);

    CudaDST3D dst(Nx, Ny, Nz, CUDA_DST_2);
    dst.execute(d_data);

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
// Test 3D DST-3
//==============================================================================
bool test_dst3_3d(int Nx, int Ny, int Nz)
{
    printf("Testing 3D DST-3: %d x %d x %d\n", Nx, Ny, Nz);

    int M = Nx * Ny * Nz;

    std::vector<double> h_input(M);
    std::vector<double> h_fftw(M);
    std::vector<double> h_cuda(M);

    // Initialize with test data
    for (int ix = 0; ix < Nx; ++ix) {
        for (int iy = 0; iy < Ny; ++iy) {
            for (int iz = 0; iz < Nz; ++iz) {
                int idx = (ix * Ny + iy) * Nz + iz;
                double x = (ix + 1.0) / (Nx + 1);
                double y = (iy + 1.0) / (Ny + 1);
                double z = (iz + 1.0) / (Nz + 1);
                h_input[idx] = sin(M_PI * x) * sin(M_PI * y) * sin(M_PI * z);
            }
        }
    }

    // === FFTW 3D DST-3 (RODFT01) ===
    std::copy(h_input.begin(), h_input.end(), h_fftw.begin());
    fftw_plan plan = fftw_plan_r2r_3d(Nx, Ny, Nz, h_fftw.data(), h_fftw.data(),
                                       FFTW_RODFT01, FFTW_RODFT01, FFTW_RODFT01,
                                       FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);

    // === CUDA 3D DST-3 ===
    double* d_data;
    cudaMalloc(&d_data, sizeof(double) * M);
    cudaMemcpy(d_data, h_input.data(), sizeof(double) * M, cudaMemcpyHostToDevice);

    CudaDST3D dst(Nx, Ny, Nz, CUDA_DST_3);
    dst.execute(d_data);

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
// Test 3D DST-4
//==============================================================================
bool test_dst4_3d(int Nx, int Ny, int Nz)
{
    printf("Testing 3D DST-4: %d x %d x %d\n", Nx, Ny, Nz);

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
                double x = (ix + 0.5) / Nx;
                double y = (iy + 0.5) / Ny;
                double z = (iz + 0.5) / Nz;
                h_input[idx] = sin(M_PI * x) * sin(M_PI * y) * sin(M_PI * z);
            }
        }
    }

    // === FFTW 3D DST-4 (RODFT11) ===
    std::copy(h_input.begin(), h_input.end(), h_fftw.begin());
    fftw_plan plan = fftw_plan_r2r_3d(Nx, Ny, Nz, h_fftw.data(), h_fftw.data(),
                                       FFTW_RODFT11, FFTW_RODFT11, FFTW_RODFT11,
                                       FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);

    // === CUDA 3D DST-4 ===
    double* d_data;
    cudaMalloc(&d_data, sizeof(double) * M);
    cudaMemcpy(d_data, h_input.data(), sizeof(double) * M, cudaMemcpyHostToDevice);

    CudaDST3D dst(Nx, Ny, Nz, CUDA_DST_4);
    dst.execute(d_data);

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
// Test round-trip (DST-1 is self-inverse)
//==============================================================================
bool test_dst1_roundtrip(int N)
{
    printf("Testing DST-1 round-trip: N=%d\n", N);

    std::vector<double> h_input(N);
    std::vector<double> h_result(N);

    // Initialize
    for (int i = 0; i < N; ++i) {
        h_input[i] = sin(M_PI * (i + 1) / (N + 1)) + 0.5 * sin(2 * M_PI * (i + 1) / (N + 1));
    }

    // FFTW: forward + backward
    std::copy(h_input.begin(), h_input.end(), h_result.begin());

    fftw_plan plan = fftw_plan_r2r_1d(N, h_result.data(), h_result.data(), FFTW_RODFT00, FFTW_ESTIMATE);
    fftw_execute(plan);  // Forward
    fftw_execute(plan);  // Backward (DST-1 is self-inverse)
    fftw_destroy_plan(plan);

    // Compute actual scaling factor
    double scale = h_result[0] / h_input[0];
    printf("  Actual scaling factor: %.4f (expected 2*(N+1)=%.4f)\n", scale, 2.0 * (N + 1));

    // FFTW DST-1: round-trip multiplies by 2*(N+1)
    double norm = 1.0 / (2.0 * (N + 1));
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

    printf("  FFTW round-trip: Max abs error: %.6e, Relative error: %.6e\n", max_err, rel_err);

    bool passed = rel_err < 1e-12;
    printf("  %s\n\n", passed ? "PASSED!" : "FAILED!");

    return passed;
}

//==============================================================================
// Test DST-2/DST-3 inverse pair
//==============================================================================
bool test_dst23_inverse(int N)
{
    printf("Testing DST-2/DST-3 inverse pair: N=%d\n", N);

    std::vector<double> h_input(N);
    std::vector<double> h_result(N);

    // Initialize
    for (int i = 0; i < N; ++i) {
        h_input[i] = sin(M_PI * (i + 0.5) / N) + 0.5 * sin(2 * M_PI * (i + 0.5) / N);
    }

    // FFTW: DST-2 then DST-3 should recover input (up to scale)
    std::copy(h_input.begin(), h_input.end(), h_result.begin());

    fftw_plan plan2 = fftw_plan_r2r_1d(N, h_result.data(), h_result.data(), FFTW_RODFT10, FFTW_ESTIMATE);
    fftw_execute(plan2);
    fftw_destroy_plan(plan2);

    fftw_plan plan3 = fftw_plan_r2r_1d(N, h_result.data(), h_result.data(), FFTW_RODFT01, FFTW_ESTIMATE);
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

    printf("  FFTW DST-2 + DST-3 round-trip: rel_err = %.6e\n", rel_err);

    bool passed = rel_err < 1e-12;
    printf("  %s\n\n", passed ? "PASSED!" : "FAILED!");

    return passed;
}

//==============================================================================
// Test CUDA DST-1 round-trip
//==============================================================================
bool test_cuda_dst1_roundtrip(int N)
{
    printf("Testing CUDA DST-1 round-trip: N=%d\n", N);

    std::vector<double> h_input(N);
    std::vector<double> h_result(N);

    // Initialize
    for (int i = 0; i < N; ++i) {
        h_input[i] = sin(M_PI * (i + 1) / (N + 1)) + 0.5 * sin(2 * M_PI * (i + 1) / (N + 1));
    }

    // CUDA: forward + backward
    double* d_data;
    cudaMalloc(&d_data, sizeof(double) * N);
    cudaMemcpy(d_data, h_input.data(), sizeof(double) * N, cudaMemcpyHostToDevice);

    CudaDST dst(N, CUDA_DST_1);
    dst.execute(d_data);  // Forward
    dst.execute(d_data);  // Backward (DST-1 is self-inverse)

    cudaMemcpy(h_result.data(), d_data, sizeof(double) * N, cudaMemcpyDeviceToHost);
    cudaFree(d_data);

    // Compute actual scaling factor
    double scale = h_result[0] / h_input[0];
    printf("  Actual scaling factor: %.4f (expected 2*(N+1)=%.4f)\n", scale, 2.0 * (N + 1));

    // Apply normalization: 1/(2*(N+1))
    double norm = 1.0 / (2.0 * (N + 1));
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

    printf("  Max abs error: %.6e, Relative error: %.6e\n", max_err, rel_err);

    bool passed = rel_err < 1e-12;
    printf("  %s\n\n", passed ? "PASSED!" : "FAILED!");

    return passed;
}

//==============================================================================
// Test 2D Mixed DST (vs FFTW)
//==============================================================================
bool test_mixed_2d(int Nx, int Ny, CudaDSTType type_x, CudaDSTType type_y)
{
    const char* type_names[] = {"DCT-1", "DCT-2", "DCT-3", "DCT-4", "DST-1", "DST-2", "DST-3", "DST-4"};
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
            h_input[idx] = sin(M_PI * (ix + 0.5) / Nx) * sin(M_PI * (iy + 0.5) / Ny)
                         + 0.5 * sin(2 * M_PI * (ix + 0.5) / Nx) * sin(2 * M_PI * (iy + 0.5) / Ny);
        }
    }

    // FFTW reference: separate 1D transforms per dimension
    std::copy(h_input.begin(), h_input.end(), h_fftw.begin());

    // Y dimension first (contiguous)
    fftw_r2r_kind kind_y = (type_y == CUDA_DST_2) ? FFTW_RODFT10 :
                           (type_y == CUDA_DST_3) ? FFTW_RODFT01 : FFTW_RODFT11;
    for (int ix = 0; ix < Nx; ++ix) {
        fftw_plan plan = fftw_plan_r2r_1d(Ny, &h_fftw[ix * Ny], &h_fftw[ix * Ny], kind_y, FFTW_ESTIMATE);
        fftw_execute(plan);
        fftw_destroy_plan(plan);
    }

    // X dimension (need transpose for FFTW or do row-by-row)
    std::vector<double> h_col(Nx);
    fftw_r2r_kind kind_x = (type_x == CUDA_DST_2) ? FFTW_RODFT10 :
                           (type_x == CUDA_DST_3) ? FFTW_RODFT01 : FFTW_RODFT11;
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

    CudaDST2D dst(Nx, Ny, type_x, type_y);
    dst.execute(d_data);

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
// Test 3D Mixed DST (vs FFTW)
//==============================================================================
bool test_mixed_3d(int Nx, int Ny, int Nz, CudaDSTType type_x, CudaDSTType type_y, CudaDSTType type_z)
{
    const char* type_names[] = {"DCT-1", "DCT-2", "DCT-3", "DCT-4", "DST-1", "DST-2", "DST-3", "DST-4"};
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
                             * sin(M_PI * (iy + 0.5) / Ny)
                             * sin(M_PI * (iz + 0.5) / Nz);
            }
        }
    }

    // FFTW reference
    std::copy(h_input.begin(), h_input.end(), h_fftw.begin());

    // Z dimension (contiguous)
    fftw_r2r_kind kind_z = (type_z == CUDA_DST_2) ? FFTW_RODFT10 :
                           (type_z == CUDA_DST_3) ? FFTW_RODFT01 : FFTW_RODFT11;
    for (int ix = 0; ix < Nx; ++ix) {
        for (int iy = 0; iy < Ny; ++iy) {
            int offset = (ix * Ny + iy) * Nz;
            fftw_plan plan = fftw_plan_r2r_1d(Nz, &h_fftw[offset], &h_fftw[offset], kind_z, FFTW_ESTIMATE);
            fftw_execute(plan);
            fftw_destroy_plan(plan);
        }
    }

    // Y dimension
    fftw_r2r_kind kind_y = (type_y == CUDA_DST_2) ? FFTW_RODFT10 :
                           (type_y == CUDA_DST_3) ? FFTW_RODFT01 : FFTW_RODFT11;
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
    fftw_r2r_kind kind_x = (type_x == CUDA_DST_2) ? FFTW_RODFT10 :
                           (type_x == CUDA_DST_3) ? FFTW_RODFT01 : FFTW_RODFT11;
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

    CudaDST3D dst(Nx, Ny, Nz, type_x, type_y, type_z);
    dst.execute(d_data);

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
    std::cout << "========== CUDA DST vs FFTW Test ==========\n\n";

    bool all_passed = true;

    // Test 1D DST-1
    std::cout << "--- 1D DST-1 Tests (vs FFTW RODFT00) ---\n\n";
    all_passed &= test_dst1_1d(16);
    all_passed &= test_dst1_1d(32);
    all_passed &= test_dst1_1d(64);

    // Test 1D DST-2
    std::cout << "--- 1D DST-2 Tests (vs FFTW RODFT10) ---\n\n";
    all_passed &= test_dst2_1d(16);
    all_passed &= test_dst2_1d(32);
    all_passed &= test_dst2_1d(64);

    // Test 1D DST-3
    std::cout << "--- 1D DST-3 Tests (vs FFTW RODFT01) ---\n\n";
    all_passed &= test_dst3_1d(16);
    all_passed &= test_dst3_1d(32);
    all_passed &= test_dst3_1d(64);

    // Test 1D DST-4
    std::cout << "--- 1D DST-4 Tests (vs FFTW RODFT11) ---\n\n";
    all_passed &= test_dst4_1d(16);
    all_passed &= test_dst4_1d(32);
    all_passed &= test_dst4_1d(64);

    // Test 2D DST-1
    std::cout << "--- 2D DST-1 Tests (vs FFTW RODFT00) ---\n\n";
    all_passed &= test_dst1_2d(8, 8);
    all_passed &= test_dst1_2d(16, 16);
    all_passed &= test_dst1_2d(32, 32);

    // Test 2D DST-2
    std::cout << "--- 2D DST-2 Tests (vs FFTW RODFT10) ---\n\n";
    all_passed &= test_dst2_2d(8, 8);
    all_passed &= test_dst2_2d(16, 16);
    all_passed &= test_dst2_2d(32, 32);

    // Test 2D DST-3
    std::cout << "--- 2D DST-3 Tests (vs FFTW RODFT01) ---\n\n";
    all_passed &= test_dst3_2d(8, 8);
    all_passed &= test_dst3_2d(16, 16);
    all_passed &= test_dst3_2d(32, 32);

    // Test 2D DST-4
    std::cout << "--- 2D DST-4 Tests (vs FFTW RODFT11) ---\n\n";
    all_passed &= test_dst4_2d(8, 8);
    all_passed &= test_dst4_2d(16, 16);
    all_passed &= test_dst4_2d(32, 32);

    // Test 3D DST-1
    std::cout << "--- 3D DST-1 Tests (vs FFTW RODFT00) ---\n\n";
    all_passed &= test_dst1_3d(8, 8, 8);
    all_passed &= test_dst1_3d(16, 16, 16);
    all_passed &= test_dst1_3d(32, 32, 32);

    // Test 3D DST-2
    std::cout << "--- 3D DST-2 Tests (vs FFTW RODFT10) ---\n\n";
    all_passed &= test_dst2_3d(8, 8, 8);
    all_passed &= test_dst2_3d(16, 16, 16);
    all_passed &= test_dst2_3d(32, 32, 32);

    // Test 3D DST-3
    std::cout << "--- 3D DST-3 Tests (vs FFTW RODFT01) ---\n\n";
    all_passed &= test_dst3_3d(8, 8, 8);
    all_passed &= test_dst3_3d(16, 16, 16);
    all_passed &= test_dst3_3d(32, 32, 32);

    // Test 3D DST-4
    std::cout << "--- 3D DST-4 Tests (vs FFTW RODFT11) ---\n\n";
    all_passed &= test_dst4_3d(8, 8, 8);
    all_passed &= test_dst4_3d(16, 16, 16);
    all_passed &= test_dst4_3d(32, 32, 32);

    // Test round-trip
    std::cout << "--- Round-trip Tests ---\n\n";
    all_passed &= test_dst1_roundtrip(32);
    all_passed &= test_cuda_dst1_roundtrip(32);
    all_passed &= test_dst23_inverse(32);

    // Test 2D Mixed DST
    std::cout << "--- 2D Mixed DST Tests ---\n\n";
    all_passed &= test_mixed_2d(16, 16, CUDA_DST_2, CUDA_DST_3);
    all_passed &= test_mixed_2d(16, 16, CUDA_DST_3, CUDA_DST_2);
    all_passed &= test_mixed_2d(16, 16, CUDA_DST_2, CUDA_DST_4);
    all_passed &= test_mixed_2d(16, 16, CUDA_DST_4, CUDA_DST_3);

    // Test 3D Mixed DST
    std::cout << "--- 3D Mixed DST Tests ---\n\n";
    all_passed &= test_mixed_3d(8, 8, 8, CUDA_DST_2, CUDA_DST_3, CUDA_DST_4);
    all_passed &= test_mixed_3d(8, 8, 8, CUDA_DST_4, CUDA_DST_2, CUDA_DST_3);
    all_passed &= test_mixed_3d(8, 8, 8, CUDA_DST_3, CUDA_DST_4, CUDA_DST_2);

    if (all_passed)
        std::cout << "=== All tests PASSED! ===\n";
    else
        std::cout << "=== Some tests FAILED! ===\n";

    return all_passed ? 0 : 1;
}
