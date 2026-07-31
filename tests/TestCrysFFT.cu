/**
 * @file TestCrysFFT.cu
 * @brief Test Crystallographic FFT (DCT-II/III) for both CPU and CUDA.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>
#include <fftw3.h>
#include <cufft.h>

#include "CudaCommon.h"
#include "CudaCrysFFT.h"
#include "CudaCrysFFTRecursive3m.h"
#include "FftwCrysFFTPmmm.h"
#include "FftwCrysFFTRecursive3m.h"
#ifdef USE_CPU_MKL
#include "MklCrysFFTRecursive3m.h"
#endif
#include "SpaceGroup.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

//==============================================================================
// Helper functions
//==============================================================================

void expand_pmmm(const double* phys, double* logical, int Nx, int Ny, int Nz)
{
    int Nx2 = Nx / 2, Ny2 = Ny / 2, Nz2 = Nz / 2;
    for (int ix = 0; ix < Nx; ++ix)
    {
        int ix_phys = (ix < Nx2) ? ix : (Nx - 1 - ix);
        for (int iy = 0; iy < Ny; ++iy)
        {
            int iy_phys = (iy < Ny2) ? iy : (Ny - 1 - iy);
            for (int iz = 0; iz < Nz; ++iz)
            {
                int iz_phys = (iz < Nz2) ? iz : (Nz - 1 - iz);
                logical[(ix * Ny + iy) * Nz + iz] =
                    phys[(ix_phys * Ny2 + iy_phys) * Nz2 + iz_phys];
            }
        }
    }
}

void diffusion_standard_fft(double* q_in, double* q_out,
    int Nx, int Ny, int Nz, double Lx, double Ly, double Lz, double ds)
{
    int M = Nx * Ny * Nz;
    int M_complex = Nx * Ny * (Nz / 2 + 1);

    double* work = (double*)fftw_malloc(sizeof(double) * M);
    fftw_complex* freq = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * M_complex);

    fftw_plan plan_fwd = fftw_plan_dft_r2c_3d(Nx, Ny, Nz, work, freq, FFTW_ESTIMATE);
    fftw_plan plan_bwd = fftw_plan_dft_c2r_3d(Nx, Ny, Nz, freq, work, FFTW_ESTIMATE);

    for (int i = 0; i < M; ++i) work[i] = q_in[i];
    fftw_execute(plan_fwd);

    double norm = 1.0 / M;
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
                double boltz = std::exp(-(kx2 + ky2 + kz2) * ds) * norm;
                int idx = (ix * Ny + iy) * (Nz/2 + 1) + iz;
                freq[idx][0] *= boltz;
                freq[idx][1] *= boltz;
            }
        }
    }

    fftw_execute(plan_bwd);
    for (int i = 0; i < M; ++i) q_out[i] = work[i];

    fftw_destroy_plan(plan_fwd);
    fftw_destroy_plan(plan_bwd);
    fftw_free(work);
    fftw_free(freq);
}

// Full-grid reference for the ExpKx2 stress-multiplier cache:
// q_out = IFFT[ exp(-k^2 coeff) * kx^2 * FFT[q_in] ] / M
void multiplier_expkx2_standard_fft(double* q_in, double* q_out,
    int Nx, int Ny, int Nz, double Lx, double Ly, double Lz, double coeff)
{
    int M = Nx * Ny * Nz;
    int M_complex = Nx * Ny * (Nz / 2 + 1);

    double* work = (double*)fftw_malloc(sizeof(double) * M);
    fftw_complex* freq = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * M_complex);

    fftw_plan plan_fwd = fftw_plan_dft_r2c_3d(Nx, Ny, Nz, work, freq, FFTW_ESTIMATE);
    fftw_plan plan_bwd = fftw_plan_dft_c2r_3d(Nx, Ny, Nz, freq, work, FFTW_ESTIMATE);

    for (int i = 0; i < M; ++i) work[i] = q_in[i];
    fftw_execute(plan_fwd);

    double norm = 1.0 / M;
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
                double mul = std::exp(-(kx2 + ky2 + kz2) * coeff) * kx2 * norm;
                int idx = (ix * Ny + iy) * (Nz/2 + 1) + iz;
                freq[idx][0] *= mul;
                freq[idx][1] *= mul;
            }
        }
    }

    fftw_execute(plan_bwd);
    for (int i = 0; i < M; ++i) q_out[i] = work[i];

    fftw_destroy_plan(plan_fwd);
    fftw_destroy_plan(plan_bwd);
    fftw_free(work);
    fftw_free(freq);
}

//==============================================================================
// CUDA kernel
//==============================================================================

__global__ void ker_apply_boltzmann(
    cufftDoubleComplex* freq, double Lx, double Ly, double Lz,
    int Nx, int Ny, int Nz_half, double ds, double norm)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int M = Nx * Ny * Nz_half;
    if (idx >= M) return;

    int iz = idx % Nz_half;
    int iy = (idx / Nz_half) % Ny;
    int ix = idx / (Ny * Nz_half);

    int kx = (ix > Nx/2) ? (Nx - ix) : ix;
    int ky = (iy > Ny/2) ? (Ny - iy) : iy;

    double kx2 = kx * 2.0 * M_PI / Lx;
    double ky2 = ky * 2.0 * M_PI / Ly;
    double kz2 = iz * 2.0 * M_PI / Lz;

    double boltz = exp(-(kx2*kx2 + ky2*ky2 + kz2*kz2) * ds) * norm;
    freq[idx].x *= boltz;
    freq[idx].y *= boltz;
}

//==============================================================================
// Test functions
//==============================================================================

bool test_cpu(int Nx, int Ny, int Nz, double ds)
{
    double Lx = 4.0, Ly = 4.0, Lz = 4.0;
    int M_logical = Nx * Ny * Nz;
    int M_physical = (Nx/2) * (Ny/2) * (Nz/2);
    int Nx2 = Nx/2, Ny2 = Ny/2, Nz2 = Nz/2;

    std::vector<double> q_phys(M_physical), q_phys_out(M_physical);
    std::vector<double> q_logical(M_logical), q_logical_out(M_logical);

    for (int ix = 0; ix < Nx2; ++ix)
    {
        double x = (ix + 0.5) * Lx / Nx;
        for (int iy = 0; iy < Ny2; ++iy)
        {
            double y = (iy + 0.5) * Ly / Ny;
            for (int iz = 0; iz < Nz2; ++iz)
            {
                double z = (iz + 0.5) * Lz / Nz;
                q_phys[(ix * Ny2 + iy) * Nz2 + iz] =
                    std::cos(2*M_PI*x/Lx) * std::cos(2*M_PI*y/Ly) * std::cos(2*M_PI*z/Lz);
            }
        }
    }
    expand_pmmm(q_phys.data(), q_logical.data(), Nx, Ny, Nz);

    FftwCrysFFTPmmm crys({Nx, Ny, Nz}, {Lx, Ly, Lz, M_PI/2, M_PI/2, M_PI/2});
    crys.set_contour_step(ds);
    crys.diffusion(q_phys.data(), q_phys_out.data());

    diffusion_standard_fft(q_logical.data(), q_logical_out.data(), Nx, Ny, Nz, Lx, Ly, Lz, ds);

    double max_rel = 0.0;
    for (int ix = 0; ix < Nx2; ++ix)
        for (int iy = 0; iy < Ny2; ++iy)
            for (int iz = 0; iz < Nz2; ++iz)
            {
                double v1 = q_phys_out[(ix*Ny2+iy)*Nz2+iz];
                double v2 = q_logical_out[(ix*Ny+iy)*Nz+iz];
                double rel = std::abs(v1-v2) / (std::abs(v2)+1e-15);
                if (rel > max_rel) max_rel = rel;
            }

    bool ok = (max_rel < 1e-6);
    printf("  CPU:  %.2e %s\n", max_rel, ok ? "OK" : "FAIL");
    return ok;
}

bool test_cuda(int Nx, int Ny, int Nz, double ds)
{
    double Lx = 4.0, Ly = 4.0, Lz = 4.0;
    int M_logical = Nx * Ny * Nz;
    int M_physical = (Nx/2) * (Ny/2) * (Nz/2);
    int M_complex = Nx * Ny * (Nz/2 + 1);
    int Nx2 = Nx/2, Ny2 = Ny/2, Nz2 = Nz/2;

    std::vector<double> h_phys(M_physical), h_logical(M_logical);

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
    expand_pmmm(h_phys.data(), h_logical.data(), Nx, Ny, Nz);

    // CudaCrysFFT
    CudaCrysFFT crys({Nx, Ny, Nz}, {Lx, Ly, Lz, M_PI/2, M_PI/2, M_PI/2});
    crys.set_contour_step(ds);

    double *d_in, *d_out;
    gpu_error_check(cudaMalloc(&d_in, sizeof(double)*M_physical));
    gpu_error_check(cudaMalloc(&d_out, sizeof(double)*M_physical));
    gpu_error_check(cudaMemcpy(d_in, h_phys.data(), sizeof(double)*M_physical, cudaMemcpyHostToDevice));

    crys.diffusion(d_in, d_out);
    cudaDeviceSynchronize();

    std::vector<double> h_crys_out(M_physical);
    gpu_error_check(cudaMemcpy(h_crys_out.data(), d_out, sizeof(double)*M_physical, cudaMemcpyDeviceToHost));
    cudaFree(d_in); cudaFree(d_out);

    // Standard cuFFT
    double *d_full; cufftDoubleComplex *d_freq;
    gpu_error_check(cudaMalloc(&d_full, sizeof(double)*M_logical));
    gpu_error_check(cudaMalloc(&d_freq, sizeof(cufftDoubleComplex)*M_complex));
    gpu_error_check(cudaMemcpy(d_full, h_logical.data(), sizeof(double)*M_logical, cudaMemcpyHostToDevice));

    cufftHandle fwd, bwd;
    int n[3] = {Nx, Ny, Nz};
    cufftPlanMany(&fwd, 3, n, nullptr, 1, 0, nullptr, 1, 0, CUFFT_D2Z, 1);
    cufftPlanMany(&bwd, 3, n, nullptr, 1, 0, nullptr, 1, 0, CUFFT_Z2D, 1);

    cufftExecD2Z(fwd, d_full, d_freq);
    ker_apply_boltzmann<<<(M_complex+255)/256, 256>>>(d_freq, Lx, Ly, Lz, Nx, Ny, Nz/2+1, ds, 1.0/M_logical);
    cufftExecZ2D(bwd, d_freq, d_full);
    cudaDeviceSynchronize();

    std::vector<double> h_std_out(M_logical);
    gpu_error_check(cudaMemcpy(h_std_out.data(), d_full, sizeof(double)*M_logical, cudaMemcpyDeviceToHost));
    cufftDestroy(fwd); cufftDestroy(bwd);
    cudaFree(d_full); cudaFree(d_freq);

    double max_rel = 0.0;
    for (int ix = 0; ix < Nx2; ++ix)
        for (int iy = 0; iy < Ny2; ++iy)
            for (int iz = 0; iz < Nz2; ++iz)
            {
                double v1 = h_crys_out[(ix*Ny2+iy)*Nz2+iz];
                double v2 = h_std_out[(ix*Ny+iy)*Nz+iz];
                double rel = std::abs(v1-v2) / (std::abs(v2)+1e-15);
                if (rel > max_rel) max_rel = rel;
            }

    bool ok = (max_rel < 1e-6);
    printf("  CUDA: %.2e %s\n", max_rel, ok ? "OK" : "FAIL");
    return ok;
}

// Recursive3m (2x2y2z fold) on Fddd, whose d-glides translate by a quarter
// cell. The fold pairs twiddle r_p with the octant-transformed Boltzmann
// array S_{psi(p)} where psi depends on the parities of the generator
// translations in grid units (see Recursive3mFoldParity.h); grids with an odd
// quarter (e.g. 84/4 = 21) exercise psi != identity.
bool test_recursive3m(int Nx, int Ny, int Nz, double ds)
{
    double Lx = 3.9, Ly = 2.7, Lz = 1.3;
    int Nx2 = Nx / 2, Ny2 = Ny / 2, Nz2 = Nz / 2;
    int M_logical = Nx * Ny * Nz;
    int M_physical = Nx2 * Ny2 * Nz2;

    SpaceGroup sg({Nx, Ny, Nz}, "Fddd", 336);
    std::array<double, 9> trans{};
    if (!sg.get_m3_translations(trans))
    {
        printf("  no m3 translations FAIL\n");
        return false;
    }
    sg.enable_m3_physical_basis();

    // Random reduced field, symmetrized to a valid symmetric full field
    std::vector<double> q_phys(M_physical);
    std::mt19937_64 rng(4242 + Nx);
    std::normal_distribution<double> dist(0.0, 1.0);
    for (auto& v : q_phys) v = dist(rng);
    sg.symmetrize_reduced_basis(q_phys.data(), 1);
    std::vector<double> q_logical(M_logical);
    sg.from_reduced_basis(q_phys.data(), q_logical.data(), 1);

    // Reference: full-grid FFT diffusion restricted to the even-index subgrid
    std::vector<double> q_logical_out(M_logical);
    diffusion_standard_fft(q_logical.data(), q_logical_out.data(),
                           Nx, Ny, Nz, Lx, Ly, Lz, ds);
    std::vector<double> ref(M_physical);
    double scale = 0.0;
    for (int ix = 0; ix < Nx2; ++ix)
        for (int iy = 0; iy < Ny2; ++iy)
            for (int iz = 0; iz < Nz2; ++iz)
            {
                double v = q_logical_out[((size_t)(2 * ix) * Ny + 2 * iy) * Nz + 2 * iz];
                ref[((size_t)ix * Ny2 + iy) * Nz2 + iz] = v;
                scale = std::max(scale, std::abs(v));
            }

    auto max_rel = [&](const std::vector<double>& got) {
        double e = 0.0;
        for (int i = 0; i < M_physical; ++i)
            e = std::max(e, std::abs(got[i] - ref[i]));
        return e / scale;
    };

    std::array<double, 6> cell = {Lx, Ly, Lz, M_PI / 2, M_PI / 2, M_PI / 2};
    bool ok = true;
    std::vector<double> got(M_physical);

    {
        FftwCrysFFTRecursive3m crys({Nx, Ny, Nz}, cell, trans);
        crys.set_contour_step(ds);
        crys.diffusion(q_phys.data(), got.data());
        double rel = max_rel(got);
        bool pass = rel < 1e-10;
        printf("  CPU (FFTW 3m):  %.2e %s\n", rel, pass ? "OK" : "FAIL");
        ok &= pass;
    }
#ifdef USE_CPU_MKL
    {
        MklCrysFFTRecursive3m crys({Nx, Ny, Nz}, cell, trans);
        crys.set_contour_step(ds);
        crys.diffusion(q_phys.data(), got.data());
        double rel = max_rel(got);
        bool pass = rel < 1e-10;
        printf("  CPU (MKL 3m):   %.2e %s\n", rel, pass ? "OK" : "FAIL");
        ok &= pass;
    }
#endif
    {
        CudaCrysFFTRecursive3m crys({Nx, Ny, Nz}, cell, trans);
        crys.set_contour_step(ds);
        double *d_in, *d_out;
        gpu_error_check(cudaMalloc(&d_in, sizeof(double) * M_physical));
        gpu_error_check(cudaMalloc(&d_out, sizeof(double) * M_physical));
        gpu_error_check(cudaMemcpy(d_in, q_phys.data(), sizeof(double) * M_physical, cudaMemcpyHostToDevice));
        crys.diffusion(d_in, d_out);
        cudaDeviceSynchronize();
        gpu_error_check(cudaMemcpy(got.data(), d_out, sizeof(double) * M_physical, cudaMemcpyDeviceToHost));
        cudaFree(d_in);
        cudaFree(d_out);
        double rel = max_rel(got);
        bool pass = rel < 1e-10;
        printf("  CUDA (3m):      %.2e %s\n", rel, pass ? "OK" : "FAIL");
        ok &= pass;
    }

    // ---- ExpKx2 stress-multiplier cache (same psi pairing as diffusion; the
    // ---- k^2-multiplier caches were equally miscomputed pre-fix) ----
    std::vector<double> refm(M_physical);
    {
        std::vector<double> q_logical_mul(M_logical);
        multiplier_expkx2_standard_fft(q_logical.data(), q_logical_mul.data(),
                                       Nx, Ny, Nz, Lx, Ly, Lz, ds);
        double scale_m = 0.0;
        for (int ix = 0; ix < Nx2; ++ix)
            for (int iy = 0; iy < Ny2; ++iy)
                for (int iz = 0; iz < Nz2; ++iz)
                {
                    double v = q_logical_mul[((size_t)(2 * ix) * Ny + 2 * iy) * Nz + 2 * iz];
                    refm[((size_t)ix * Ny2 + iy) * Nz2 + iz] = v;
                    scale_m = std::max(scale_m, std::abs(v));
                }
        auto max_rel_m = [&](const std::vector<double>& g) {
            double e = 0.0;
            for (int i = 0; i < M_physical; ++i)
                e = std::max(e, std::abs(g[i] - refm[i]));
            return e / scale_m;
        };

        {
            FftwCrysFFTRecursive3m crys({Nx, Ny, Nz}, cell, trans);
            crys.apply_multiplier(q_phys.data(), got.data(),
                                  FftwCrysFFTRecursive3m::MultiplierType::ExpKx2, ds);
            double rel = max_rel_m(got);
            bool pass = rel < 1e-10;
            printf("  CPU (FFTW 3m, ExpKx2): %.2e %s\n", rel, pass ? "OK" : "FAIL");
            ok &= pass;
        }
#ifdef USE_CPU_MKL
        {
            MklCrysFFTRecursive3m crys({Nx, Ny, Nz}, cell, trans);
            crys.apply_multiplier(q_phys.data(), got.data(),
                                  MklCrysFFTRecursive3m::MultiplierType::ExpKx2, ds);
            double rel = max_rel_m(got);
            bool pass = rel < 1e-10;
            printf("  CPU (MKL 3m, ExpKx2):  %.2e %s\n", rel, pass ? "OK" : "FAIL");
            ok &= pass;
        }
#endif
        {
            CudaCrysFFTRecursive3m crys({Nx, Ny, Nz}, cell, trans);
            double *d_in, *d_out;
            gpu_error_check(cudaMalloc(&d_in, sizeof(double) * M_physical));
            gpu_error_check(cudaMalloc(&d_out, sizeof(double) * M_physical));
            gpu_error_check(cudaMemcpy(d_in, q_phys.data(), sizeof(double) * M_physical, cudaMemcpyHostToDevice));
            crys.apply_multiplier(d_in, d_out,
                                  CudaCrysFFTRecursive3m::MultiplierType::ExpKx2, ds, 0);
            cudaDeviceSynchronize();
            gpu_error_check(cudaMemcpy(got.data(), d_out, sizeof(double) * M_physical, cudaMemcpyDeviceToHost));
            cudaFree(d_in);
            cudaFree(d_out);
            double rel = max_rel_m(got);
            bool pass = rel < 1e-10;
            printf("  CUDA (3m, ExpKx2):     %.2e %s\n", rel, pass ? "OK" : "FAIL");
            ok &= pass;
        }
    }
    return ok;
}

int main()
{
    printf("=== CrysFFT Test (CPU & CUDA) ===\n\n");

    bool all = true;
    int sizes[][3] = {{16,16,16}, {32,32,32}, {64,64,64}};

    for (auto& s : sizes)
    {
        printf("Grid: %dx%dx%d\n", s[0], s[1], s[2]);
        all &= test_cpu(s[0], s[1], s[2], 0.01);
        all &= test_cuda(s[0], s[1], s[2], 0.01);
    }

    // Recursive3m fold on unequal Fddd grids:
    // - 84x48x16 (Nx/4 = 21 odd) was miscomputed before the translation-parity
    //   pairing fix (partition functions wrong at ~0.6 relative);
    // - 64x48x16 (all quarters even) is the historically working class.
    printf("\nGrid: 84x48x16 (Recursive3m, odd Nx/4)\n");
    all &= test_recursive3m(84, 48, 16, 0.01);
    printf("Grid: 64x48x16 (Recursive3m, all quarters even)\n");
    all &= test_recursive3m(64, 48, 16, 0.01);

    printf(all ? "\nAll passed!\n" : "\nFAILED!\n");
    return all ? 0 : 1;
}
