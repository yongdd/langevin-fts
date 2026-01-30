/**
 * @file TestCrysFFTCudaBench.cu
 * @brief CUDA FFT-only benchmark for CrysFFT (Pmmm/3m) vs full FFT.
 *
 * Usage: ./TestCrysFFTCudaBench [Nx Ny Nz] [iters] [warmup]
 * Defaults: Nx=Ny=Nz=64, iters=100, warmup=10
 */

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <vector>
#include <fftw3.h>

#include <cuda_runtime.h>
#include <cufft.h>

#include "CudaCommon.h"
#include "CudaCrysFFT.h"
#include "CudaCrysFFTHex.h"
#include "CudaCrysFFTRecursive3m.h"
#include "FftwCrysFFTHex.h"
#include "FftwCrysFFTPmmm.h"
#include "FftwCrysFFTRecursive3m.h"
#include "SpaceGroup.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Simple CUDA kernel for diffusion multiplier on full FFT
__global__ void ker_apply_boltzmann(
    cufftDoubleComplex* freq, double Lx, double Ly, double Lz,
    int Nx, int Ny, int Nz_half, double coeff, double norm)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int M = Nx * Ny * Nz_half;
    if (idx >= M) return;

    int iz = idx % Nz_half;
    int iy = (idx / Nz_half) % Ny;
    int ix = idx / (Ny * Nz_half);

    int kx = (ix > Nx/2) ? (Nx - ix) : ix;
    int ky = (iy > Ny/2) ? (Ny - iy) : iy;

    double kx2 = (kx * 2.0 * M_PI / Lx);
    double ky2 = (ky * 2.0 * M_PI / Ly);
    double kz2 = (iz * 2.0 * M_PI / Lz);

    double boltz = exp(-(kx2*kx2 + ky2*ky2 + kz2*kz2) * coeff) * norm;
    freq[idx].x *= boltz;
    freq[idx].y *= boltz;
}

__global__ void ker_apply_boltzmann_lut(
    cufftDoubleComplex* freq, const double* boltz, int M)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M) return;
    const double b = boltz[idx];
    freq[idx].x *= b;
    freq[idx].y *= b;
}

static void fill_physical_field(std::vector<double>& h_phys, int Nx, int Ny, int Nz, double Lx, double Ly, double Lz)
{
    int Nx2 = Nx / 2, Ny2 = Ny / 2, Nz2 = Nz / 2;
    for (int ix = 0; ix < Nx2; ++ix)
    {
        double x = (ix + 0.5) * Lx / Nx;
        for (int iy = 0; iy < Ny2; ++iy)
        {
            double y = (iy + 0.5) * Ly / Ny;
            for (int iz = 0; iz < Nz2; ++iz)
            {
                double z = (iz + 0.5) * Lz / Nz;
                h_phys[(ix * Ny2 + iy) * Nz2 + iz] =
                    std::cos(2*M_PI*x/Lx) * std::cos(2*M_PI*y/Ly) * std::cos(2*M_PI*z/Lz);
            }
        }
    }
}

static void fill_full_field(std::vector<double>& h_full, int Nx, int Ny, int Nz, double Lx, double Ly, double Lz)
{
    for (int ix = 0; ix < Nx; ++ix)
    {
        double x = (ix + 0.5) * Lx / Nx;
        for (int iy = 0; iy < Ny; ++iy)
        {
            double y = (iy + 0.5) * Ly / Ny;
            for (int iz = 0; iz < Nz; ++iz)
            {
                double z = (iz + 0.5) * Lz / Nz;
                h_full[(ix * Ny + iy) * Nz + iz] =
                    std::cos(2*M_PI*x/Lx) * std::cos(2*M_PI*y/Ly) * std::cos(2*M_PI*z/Lz);
            }
        }
    }
}

static void fill_hex_physical_field(
    std::vector<double>& h_hex, const std::vector<double>& h_full, int Nx, int Ny, int Nz)
{
    const int Nz2 = Nz / 2;
    for (int ix = 0; ix < Nx; ++ix)
    {
        for (int iy = 0; iy < Ny; ++iy)
        {
            for (int iz = 0; iz < Nz2; ++iz)
            {
                h_hex[(ix * Ny + iy) * Nz2 + iz] =
                    h_full[(ix * Ny + iy) * Nz + iz];
            }
        }
    }
}

static std::array<double, 6> compute_recip_metric_hex(const std::array<double, 6>& cell_para)
{
    const double a = cell_para[0];
    const double b = cell_para[1];
    const double c = cell_para[2];
    const double alpha = cell_para[3];
    const double beta = cell_para[4];
    const double gamma = cell_para[5];

    const double cos_a = std::cos(alpha);
    const double cos_b = std::cos(beta);
    const double cos_g = std::cos(gamma);
    const double sin_g = std::sin(gamma);

    const double vol_factor_sq =
        1.0 - cos_a * cos_a - cos_b * cos_b - cos_g * cos_g + 2.0 * cos_a * cos_b * cos_g;
    const double vol_factor = std::sqrt(vol_factor_sq);
    const double volume = a * b * c * vol_factor;

    const double ax = a, ay = 0.0, az = 0.0;
    const double bx = b * cos_g, by = b * sin_g, bz = 0.0;
    const double cx = c * cos_b;
    const double cy = (sin_g != 0.0) ? c * (cos_a - cos_b * cos_g) / sin_g : 0.0;
    const double cz = volume / (a * b * sin_g);

    const double bc_x = by * cz - bz * cy;
    const double bc_y = bz * cx - bx * cz;
    const double bc_z = bx * cy - by * cx;

    const double ca_x = cy * az - cz * ay;
    const double ca_y = cz * ax - cx * az;
    const double ca_z = cx * ay - cy * ax;

    const double ab_x = ay * bz - az * by;
    const double ab_y = az * bx - ax * bz;
    const double ab_z = ax * by - ay * bx;

    const double a_star_x = bc_x / volume;
    const double a_star_y = bc_y / volume;
    const double a_star_z = bc_z / volume;
    const double b_star_x = ca_x / volume;
    const double b_star_y = ca_y / volume;
    const double b_star_z = ca_z / volume;
    const double c_star_x = ab_x / volume;
    const double c_star_y = ab_y / volume;
    const double c_star_z = ab_z / volume;

    std::array<double, 6> metric{};
    metric[0] = a_star_x * a_star_x + a_star_y * a_star_y + a_star_z * a_star_z;
    metric[1] = a_star_x * b_star_x + a_star_y * b_star_y + a_star_z * b_star_z;
    metric[2] = a_star_x * c_star_x + a_star_y * c_star_y + a_star_z * c_star_z;
    metric[3] = b_star_x * b_star_x + b_star_y * b_star_y + b_star_z * b_star_z;
    metric[4] = b_star_x * c_star_x + b_star_y * c_star_y + b_star_z * c_star_z;
    metric[5] = c_star_x * c_star_x + c_star_y * c_star_y + c_star_z * c_star_z;
    return metric;
}

static void fill_boltzmann_hex(
    std::vector<double>& boltz, int Nx, int Ny, int Nz,
    const std::array<double, 6>& metric, double coeff, double norm)
{
    const double G11 = metric[0];
    const double G12 = metric[1];
    const double G13 = metric[2];
    const double G22 = metric[3];
    const double G23 = metric[4];
    const double G33 = metric[5];
    const double factor = 4.0 * M_PI * M_PI;

    int idx = 0;
    for (int ix = 0; ix < Nx; ++ix)
    {
        int m1 = (ix > Nx / 2) ? (ix - Nx) : ix;
        for (int iy = 0; iy < Ny; ++iy)
        {
            int m2 = (iy > Ny / 2) ? (iy - Ny) : iy;
            for (int iz = 0; iz < Nz / 2 + 1; ++iz)
            {
                int m3 = iz;
                double gmm = G11 * m1 * m1 + G22 * m2 * m2 + G33 * m3 * m3
                           + 2.0 * (G12 * m1 * m2 + G13 * m1 * m3 + G23 * m2 * m3);
                double k2 = factor * gmm;
                boltz[idx++] = std::exp(-k2 * coeff) * norm;
            }
        }
    }
}

static float time_cuda_events(std::function<void()> fn, int iters)
{
    cudaEvent_t start{}, stop{};
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i)
        fn();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / iters;
}

static double time_cpu_iters(std::function<void()> fn, int iters)
{
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i)
        fn();
    auto t1 = std::chrono::steady_clock::now();
    std::chrono::duration<double> dt = t1 - t0;
    return (dt.count() * 1e3) / iters;  // ms/iter
}

int main(int argc, char** argv)
{
    int Nx = 64, Ny = 64, Nz = 64;
    int iters = 100;
    int warmup = 10;
    if (argc >= 4) {
        Nx = std::atoi(argv[1]);
        Ny = std::atoi(argv[2]);
        Nz = std::atoi(argv[3]);
    }
    if (argc >= 5) iters = std::atoi(argv[4]);
    if (argc >= 6) warmup = std::atoi(argv[5]);

    if (Nx <= 0 || Ny <= 0 || Nz <= 0 || (Nx % 2) || (Ny % 2) || (Nz % 2))
    {
        std::fprintf(stderr, "Grid size must be positive even numbers.\n");
        return 1;
    }

    const double Lx = 4.0, Ly = 4.0, Lz = 4.0;
    const double coeff = 0.01;  // diffusion coefficient (b^2 ds / 6)

    const int M_full = Nx * Ny * Nz;
    const int M_phys = (Nx/2) * (Ny/2) * (Nz/2);
    const int M_complex = Nx * Ny * (Nz/2 + 1);

    std::vector<double> h_phys(M_phys, 0.0);
    std::vector<double> h_full(M_full, 0.0);
    fill_physical_field(h_phys, Nx, Ny, Nz, Lx, Ly, Lz);
    fill_full_field(h_full, Nx, Ny, Nz, Lx, Ly, Lz);

    // === Pmmm DCT (CudaCrysFFT) ===
    CudaCrysFFT crys_pmmm({Nx, Ny, Nz}, {Lx, Ly, Lz, M_PI/2, M_PI/2, M_PI/2});
    crys_pmmm.set_contour_step(coeff);

    double *d_pmmm_in{}, *d_pmmm_out{};
    gpu_error_check(cudaMalloc(&d_pmmm_in, sizeof(double) * M_phys));
    gpu_error_check(cudaMalloc(&d_pmmm_out, sizeof(double) * M_phys));
    gpu_error_check(cudaMemcpy(d_pmmm_in, h_phys.data(), sizeof(double) * M_phys, cudaMemcpyHostToDevice));

    for (int i = 0; i < warmup; ++i)
    {
        crys_pmmm.diffusion(d_pmmm_in, d_pmmm_out);
        std::swap(d_pmmm_in, d_pmmm_out);
    }
    gpu_error_check(cudaDeviceSynchronize());

    float ms_pmmm = time_cuda_events([&](){
        crys_pmmm.diffusion(d_pmmm_in, d_pmmm_out);
        std::swap(d_pmmm_in, d_pmmm_out);
    }, iters);

    // === 3m recursive (CudaCrysFFTRecursive3m) ===
    std::array<double, 9> trans_part = {0,0,0, 0,0,0, 0,0,0};
    SpaceGroup sg({Nx, Ny, Nz}, "Im-3m", 529);
    if (!sg.get_m3_translations(trans_part))
    {
        std::fprintf(stderr, "Failed to obtain 3m translations from SpaceGroup.\n");
        return 2;
    }

    CudaCrysFFTRecursive3m crys_3m({Nx, Ny, Nz}, {Lx, Ly, Lz, M_PI/2, M_PI/2, M_PI/2}, trans_part);
    crys_3m.set_contour_step(coeff);

    double *d_3m_in{}, *d_3m_out{};
    gpu_error_check(cudaMalloc(&d_3m_in, sizeof(double) * M_phys));
    gpu_error_check(cudaMalloc(&d_3m_out, sizeof(double) * M_phys));
    gpu_error_check(cudaMemcpy(d_3m_in, h_phys.data(), sizeof(double) * M_phys, cudaMemcpyHostToDevice));

    for (int i = 0; i < warmup; ++i)
    {
        crys_3m.diffusion(d_3m_in, d_3m_out);
        std::swap(d_3m_in, d_3m_out);
    }
    gpu_error_check(cudaDeviceSynchronize());

    float ms_3m = time_cuda_events([&](){
        crys_3m.diffusion(d_3m_in, d_3m_out);
        std::swap(d_3m_in, d_3m_out);
    }, iters);

    // === Full FFT (cuFFT D2Z/Z2D) ===
    double* d_full{};
    cufftDoubleComplex* d_freq{};
    gpu_error_check(cudaMalloc(&d_full, sizeof(double) * M_full));
    gpu_error_check(cudaMalloc(&d_freq, sizeof(cufftDoubleComplex) * M_complex));
    gpu_error_check(cudaMemcpy(d_full, h_full.data(), sizeof(double) * M_full, cudaMemcpyHostToDevice));

    cufftHandle plan_fwd{}, plan_bwd{};
    int n[3] = {Nx, Ny, Nz};
    cufftPlanMany(&plan_fwd, 3, n, nullptr, 1, 0, nullptr, 1, 0, CUFFT_D2Z, 1);
    cufftPlanMany(&plan_bwd, 3, n, nullptr, 1, 0, nullptr, 1, 0, CUFFT_Z2D, 1);

    int blocks = (M_complex + 255) / 256;
    double norm = 1.0 / M_full;

    for (int i = 0; i < warmup; ++i)
    {
        cufftExecD2Z(plan_fwd, d_full, d_freq);
        ker_apply_boltzmann<<<blocks, 256>>>(d_freq, Lx, Ly, Lz, Nx, Ny, Nz/2+1, coeff, norm);
        cufftExecZ2D(plan_bwd, d_freq, d_full);
    }
    gpu_error_check(cudaDeviceSynchronize());

    float ms_off = time_cuda_events([&](){
        cufftExecD2Z(plan_fwd, d_full, d_freq);
        ker_apply_boltzmann<<<blocks, 256>>>(d_freq, Lx, Ly, Lz, Nx, Ny, Nz/2+1, coeff, norm);
        cufftExecZ2D(plan_bwd, d_freq, d_full);
    }, iters);

    cufftDestroy(plan_fwd);
    cufftDestroy(plan_bwd);

    cudaFree(d_pmmm_in);
    cudaFree(d_pmmm_out);
    cudaFree(d_3m_in);
    cudaFree(d_3m_out);
    cudaFree(d_full);
    cudaFree(d_freq);

    std::printf("=== CUDA FFT-only benchmark ===\n");
    std::printf("Grid: %dx%dx%d  (iters=%d, warmup=%d)\n", Nx, Ny, Nz, iters, warmup);
    std::printf("Pmmm DCT: %.4f ms/iter\n", ms_pmmm);
    std::printf("3m recursive: %.4f ms/iter\n", ms_3m);
    std::printf("Full FFT: %.4f ms/iter\n", ms_off);
    std::printf("Speedup (Pmmm/off): %.3fx\n", ms_off / ms_pmmm);
    std::printf("Speedup (3m/off): %.3fx\n", ms_off / ms_3m);

    // === Optional CPU bench (enable with FTS_BENCH_CPU=1) ===
    const bool bench_cpu = (std::getenv("FTS_BENCH_CPU") != nullptr);
    if (bench_cpu)
    {
        std::printf("\n=== CPU FFT-only benchmark ===\n");

        // Pmmm (FFTW DCT-II/III)
        std::vector<double> cpu_pmmm_in = h_phys;
        std::vector<double> cpu_pmmm_out(M_phys, 0.0);
        FftwCrysFFTPmmm crys_cpu_pmmm({Nx, Ny, Nz}, {Lx, Ly, Lz, M_PI/2, M_PI/2, M_PI/2});
        crys_cpu_pmmm.set_contour_step(coeff);

        for (int i = 0; i < warmup; ++i)
        {
            crys_cpu_pmmm.diffusion(cpu_pmmm_in.data(), cpu_pmmm_out.data());
            std::swap(cpu_pmmm_in, cpu_pmmm_out);
        }

        double cpu_ms_pmmm = time_cpu_iters([&](){
            crys_cpu_pmmm.diffusion(cpu_pmmm_in.data(), cpu_pmmm_out.data());
            std::swap(cpu_pmmm_in, cpu_pmmm_out);
        }, iters);

        // 3m recursive
        std::vector<double> cpu_m3_in = h_phys;
        std::vector<double> cpu_m3_out(M_phys, 0.0);
        FftwCrysFFTRecursive3m crys_cpu_3m({Nx, Ny, Nz}, {Lx, Ly, Lz, M_PI/2, M_PI/2, M_PI/2}, trans_part);
        crys_cpu_3m.set_contour_step(coeff);

        for (int i = 0; i < warmup; ++i)
        {
            crys_cpu_3m.diffusion(cpu_m3_in.data(), cpu_m3_out.data());
            std::swap(cpu_m3_in, cpu_m3_out);
        }

        double cpu_ms_3m = time_cpu_iters([&](){
            crys_cpu_3m.diffusion(cpu_m3_in.data(), cpu_m3_out.data());
            std::swap(cpu_m3_in, cpu_m3_out);
        }, iters);

        // Full FFT (FFTW r2c/c2r)
        std::vector<double> cpu_full_in = h_full;
        std::vector<double> cpu_full_work(M_full, 0.0);
        int M_complex_full = Nx * Ny * (Nz/2 + 1);
        fftw_complex* freq = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * M_complex_full);
        fftw_plan plan_fwd = fftw_plan_dft_r2c_3d(Nx, Ny, Nz, cpu_full_work.data(), freq, FFTW_ESTIMATE);
        fftw_plan plan_bwd = fftw_plan_dft_c2r_3d(Nx, Ny, Nz, freq, cpu_full_work.data(), FFTW_ESTIMATE);

        std::vector<double> boltz(M_complex_full, 0.0);
        for (int ix = 0; ix < Nx; ++ix)
        {
            int kx = (ix > Nx/2) ? (Nx - ix) : ix;
            double kx2 = (kx * 2.0 * M_PI / Lx) * (kx * 2.0 * M_PI / Lx);
            for (int iy = 0; iy < Ny; ++iy)
            {
                int ky = (iy > Ny/2) ? (Ny - iy) : iy;
                double ky2 = (ky * 2.0 * M_PI / Ly) * (ky * 2.0 * M_PI / Ly);
                for (int iz = 0; iz < Nz/2 + 1; ++iz)
                {
                    double kz2 = (iz * 2.0 * M_PI / Lz) * (iz * 2.0 * M_PI / Lz);
                    int idx = (ix * Ny + iy) * (Nz/2 + 1) + iz;
                    boltz[idx] = std::exp(-(kx2 + ky2 + kz2) * coeff) * (1.0 / M_full);
                }
            }
        }

        for (int i = 0; i < warmup; ++i)
        {
            std::memcpy(cpu_full_work.data(), cpu_full_in.data(), sizeof(double) * M_full);
            fftw_execute(plan_fwd);
            for (int k = 0; k < M_complex_full; ++k)
            {
                freq[k][0] *= boltz[k];
                freq[k][1] *= boltz[k];
            }
            fftw_execute(plan_bwd);
        }

        double cpu_ms_off = time_cpu_iters([&](){
            std::memcpy(cpu_full_work.data(), cpu_full_in.data(), sizeof(double) * M_full);
            fftw_execute(plan_fwd);
            for (int k = 0; k < M_complex_full; ++k)
            {
                freq[k][0] *= boltz[k];
                freq[k][1] *= boltz[k];
            }
            fftw_execute(plan_bwd);
        }, iters);

        fftw_destroy_plan(plan_fwd);
        fftw_destroy_plan(plan_bwd);
        fftw_free(freq);

        std::printf("Pmmm DCT: %.4f ms/iter\n", cpu_ms_pmmm);
        std::printf("3m recursive: %.4f ms/iter\n", cpu_ms_3m);
        std::printf("Full FFT: %.4f ms/iter\n", cpu_ms_off);
        std::printf("Speedup (Pmmm/off): %.3fx\n", cpu_ms_off / cpu_ms_pmmm);
        std::printf("Speedup (3m/off): %.3fx\n", cpu_ms_off / cpu_ms_3m);
    }

    // === Optional Hex bench (enable with FTS_BENCH_HEX=1) ===
    const bool bench_hex = (std::getenv("FTS_BENCH_HEX") != nullptr);
    if (bench_hex)
    {
        std::printf("\n=== Hexagonal (P6/mmm) FFT-only benchmark ===\n");
        const double gamma_hex = 2.0 * M_PI / 3.0;
        const std::array<double, 6> cell_hex = {Lx, Ly, Lz, M_PI/2, M_PI/2, gamma_hex};
        const int M_hex_phys = Nx * Ny * (Nz / 2);

        std::vector<double> h_hex_phys(M_hex_phys, 0.0);
        fill_hex_physical_field(h_hex_phys, h_full, Nx, Ny, Nz);

        // Hex CrysFFT (CudaCrysFFTHex)
        CudaCrysFFTHex crys_hex({Nx, Ny, Nz}, cell_hex);
        crys_hex.set_contour_step(coeff);

        double *d_hex_in{}, *d_hex_out{};
        gpu_error_check(cudaMalloc(&d_hex_in, sizeof(double) * M_hex_phys));
        gpu_error_check(cudaMalloc(&d_hex_out, sizeof(double) * M_hex_phys));
        gpu_error_check(cudaMemcpy(d_hex_in, h_hex_phys.data(), sizeof(double) * M_hex_phys, cudaMemcpyHostToDevice));

        for (int i = 0; i < warmup; ++i)
        {
            crys_hex.diffusion(d_hex_in, d_hex_out);
            std::swap(d_hex_in, d_hex_out);
        }
        gpu_error_check(cudaDeviceSynchronize());

        float ms_hex = time_cuda_events([&](){
            crys_hex.diffusion(d_hex_in, d_hex_out);
            std::swap(d_hex_in, d_hex_out);
        }, iters);

        // Full FFT baseline with hex metric
        const std::array<double, 6> metric_hex = compute_recip_metric_hex(cell_hex);
        std::vector<double> boltz_hex(M_complex, 0.0);
        fill_boltzmann_hex(boltz_hex, Nx, Ny, Nz, metric_hex, coeff, norm);

        double* d_hex_full{};
        cufftDoubleComplex* d_hex_freq{};
        double* d_boltz{};
        gpu_error_check(cudaMalloc(&d_hex_full, sizeof(double) * M_full));
        gpu_error_check(cudaMalloc(&d_hex_freq, sizeof(cufftDoubleComplex) * M_complex));
        gpu_error_check(cudaMalloc(&d_boltz, sizeof(double) * M_complex));
        gpu_error_check(cudaMemcpy(d_hex_full, h_full.data(), sizeof(double) * M_full, cudaMemcpyHostToDevice));
        gpu_error_check(cudaMemcpy(d_boltz, boltz_hex.data(), sizeof(double) * M_complex, cudaMemcpyHostToDevice));

        cufftHandle plan_hex_fwd{}, plan_hex_bwd{};
        int n_hex[3] = {Nx, Ny, Nz};
        cufftPlanMany(&plan_hex_fwd, 3, n_hex, nullptr, 1, 0, nullptr, 1, 0, CUFFT_D2Z, 1);
        cufftPlanMany(&plan_hex_bwd, 3, n_hex, nullptr, 1, 0, nullptr, 1, 0, CUFFT_Z2D, 1);

        int blocks_hex = (M_complex + 255) / 256;
        for (int i = 0; i < warmup; ++i)
        {
            cufftExecD2Z(plan_hex_fwd, d_hex_full, d_hex_freq);
            ker_apply_boltzmann_lut<<<blocks_hex, 256>>>(d_hex_freq, d_boltz, M_complex);
            cufftExecZ2D(plan_hex_bwd, d_hex_freq, d_hex_full);
        }
        gpu_error_check(cudaDeviceSynchronize());

        float ms_hex_off = time_cuda_events([&](){
            cufftExecD2Z(plan_hex_fwd, d_hex_full, d_hex_freq);
            ker_apply_boltzmann_lut<<<blocks_hex, 256>>>(d_hex_freq, d_boltz, M_complex);
            cufftExecZ2D(plan_hex_bwd, d_hex_freq, d_hex_full);
        }, iters);

        cufftDestroy(plan_hex_fwd);
        cufftDestroy(plan_hex_bwd);
        cudaFree(d_hex_in);
        cudaFree(d_hex_out);
        cudaFree(d_hex_full);
        cudaFree(d_hex_freq);
        cudaFree(d_boltz);

        std::printf("Hex DCTz+FFTxy: %.4f ms/iter\n", ms_hex);
        std::printf("Hex Full FFT: %.4f ms/iter\n", ms_hex_off);
        std::printf("Speedup (Hex/off): %.3fx\n", ms_hex_off / ms_hex);

        if (bench_cpu)
        {
            std::printf("\n=== CPU Hex FFT-only benchmark ===\n");

            std::vector<double> cpu_hex_in = h_hex_phys;
            std::vector<double> cpu_hex_out(M_hex_phys, 0.0);
            FftwCrysFFTHex crys_cpu_hex({Nx, Ny, Nz}, cell_hex);
            crys_cpu_hex.set_contour_step(coeff);

            for (int i = 0; i < warmup; ++i)
            {
                crys_cpu_hex.diffusion(cpu_hex_in.data(), cpu_hex_out.data());
                std::swap(cpu_hex_in, cpu_hex_out);
            }

            double cpu_ms_hex = time_cpu_iters([&](){
                crys_cpu_hex.diffusion(cpu_hex_in.data(), cpu_hex_out.data());
                std::swap(cpu_hex_in, cpu_hex_out);
            }, iters);

            std::vector<double> cpu_full_hex_in = h_full;
            std::vector<double> cpu_full_hex_work(M_full, 0.0);
            fftw_complex* freq_hex = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * M_complex);
            fftw_plan plan_hex_fwd_cpu = fftw_plan_dft_r2c_3d(Nx, Ny, Nz, cpu_full_hex_work.data(), freq_hex, FFTW_ESTIMATE);
            fftw_plan plan_hex_bwd_cpu = fftw_plan_dft_c2r_3d(Nx, Ny, Nz, freq_hex, cpu_full_hex_work.data(), FFTW_ESTIMATE);

            std::vector<double> boltz_hex_cpu = boltz_hex;

            for (int i = 0; i < warmup; ++i)
            {
                std::memcpy(cpu_full_hex_work.data(), cpu_full_hex_in.data(), sizeof(double) * M_full);
                fftw_execute(plan_hex_fwd_cpu);
                for (int k = 0; k < M_complex; ++k)
                {
                    freq_hex[k][0] *= boltz_hex_cpu[k];
                    freq_hex[k][1] *= boltz_hex_cpu[k];
                }
                fftw_execute(plan_hex_bwd_cpu);
            }

            double cpu_ms_hex_off = time_cpu_iters([&](){
                std::memcpy(cpu_full_hex_work.data(), cpu_full_hex_in.data(), sizeof(double) * M_full);
                fftw_execute(plan_hex_fwd_cpu);
                for (int k = 0; k < M_complex; ++k)
                {
                    freq_hex[k][0] *= boltz_hex_cpu[k];
                    freq_hex[k][1] *= boltz_hex_cpu[k];
                }
                fftw_execute(plan_hex_bwd_cpu);
            }, iters);

            fftw_destroy_plan(plan_hex_fwd_cpu);
            fftw_destroy_plan(plan_hex_bwd_cpu);
            fftw_free(freq_hex);

            std::printf("Hex DCTz+FFTxy: %.4f ms/iter\n", cpu_ms_hex);
            std::printf("Hex Full FFT: %.4f ms/iter\n", cpu_ms_hex_off);
            std::printf("Speedup (Hex/off): %.3fx\n", cpu_ms_hex_off / cpu_ms_hex);
        }
    }

    return 0;
}
