/**
 * @file TestCrysFFTHexGlide.cu
 * @brief Validate HexZ CrysFFT (glide mirror t_z=1/2) against full FFT.
 */

#include <cmath>
#include <cstdio>
#include <vector>
#include <array>
#include <fftw3.h>
#include <cufft.h>

#include "CudaCommon.h"
#include "CudaCrysFFTHex.h"
#include "FftwCrysFFTHex.h"
#include "SpaceGroup.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

//==============================================================================
// Helper functions
//==============================================================================

static std::array<double, 6> compute_recip_metric(const std::array<double, 6>& cell_para)
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

    // Direct lattice vectors
    const double ax = a, ay = 0.0, az = 0.0;
    const double bx = b * cos_g, by = b * sin_g, bz = 0.0;
    const double cx = c * cos_b;
    const double cy = (sin_g != 0.0) ? c * (cos_a - cos_b * cos_g) / sin_g : 0.0;
    const double cz = volume / (a * b * sin_g);

    // Reciprocal lattice vectors (without 2π)
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

static void expand_hex_z(const double* phys, double* logical,
    int Nx, int Ny, int Nz, int z_shift)
{
    const int Nz2 = Nz / 2;
    for (int ix = 0; ix < Nx; ++ix)
    {
        for (int iy = 0; iy < Ny; ++iy)
        {
            for (int iz = 0; iz < Nz; ++iz)
            {
                int iz_shift = iz - z_shift;
                while (iz_shift < 0)
                    iz_shift += Nz;
                while (iz_shift >= Nz)
                    iz_shift -= Nz;
                int iz2 = (iz_shift < Nz2) ? iz_shift : (Nz - 1 - iz_shift);
                logical[(ix * Ny + iy) * Nz + iz] =
                    phys[(ix * Ny + iy) * Nz2 + iz2];
            }
        }
    }
}

static void diffusion_standard_fft_hex(double* q_in, double* q_out,
    int Nx, int Ny, int Nz, const std::array<double, 6>& cell_para, double ds)
{
    const int M = Nx * Ny * Nz;
    const int M_complex = Nx * Ny * (Nz / 2 + 1);

    double* work = (double*)fftw_malloc(sizeof(double) * M);
    fftw_complex* freq = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * M_complex);

    fftw_plan plan_fwd = fftw_plan_dft_r2c_3d(Nx, Ny, Nz, work, freq, FFTW_ESTIMATE);
    fftw_plan plan_bwd = fftw_plan_dft_c2r_3d(Nx, Ny, Nz, freq, work, FFTW_ESTIMATE);

    for (int i = 0; i < M; ++i)
        work[i] = q_in[i];
    fftw_execute(plan_fwd);

    const auto metric = compute_recip_metric(cell_para);
    const double G11 = metric[0];
    const double G12 = metric[1];
    const double G13 = metric[2];
    const double G22 = metric[3];
    const double G23 = metric[4];
    const double G33 = metric[5];
    const double factor = 4.0 * M_PI * M_PI;
    const double norm = 1.0 / M;

    const int Nz_half = Nz / 2 + 1;
    for (int ix = 0; ix < Nx; ++ix)
    {
        int m1 = (ix > Nx / 2) ? (ix - Nx) : ix;
        for (int iy = 0; iy < Ny; ++iy)
        {
            int m2 = (iy > Ny / 2) ? (iy - Ny) : iy;
            for (int iz = 0; iz < Nz_half; ++iz)
            {
                int m3 = iz;
                double gmm = G11 * m1 * m1 + G22 * m2 * m2 + G33 * m3 * m3
                           + 2.0 * (G12 * m1 * m2 + G13 * m1 * m3 + G23 * m2 * m3);
                double k2 = factor * gmm;
                double boltz = std::exp(-k2 * ds) * norm;
                const int idx = (ix * Ny + iy) * Nz_half + iz;
                freq[idx][0] *= boltz;
                freq[idx][1] *= boltz;
            }
        }
    }

    fftw_execute(plan_bwd);
    for (int i = 0; i < M; ++i)
        q_out[i] = work[i];

    fftw_destroy_plan(plan_fwd);
    fftw_destroy_plan(plan_bwd);
    fftw_free(work);
    fftw_free(freq);
}

//==============================================================================
// CUDA kernel
//==============================================================================

__global__ void ker_apply_boltzmann_hex(
    cufftDoubleComplex* freq,
    int Nx, int Ny, int Nz_half,
    double ds, double norm,
    double G11, double G12, double G13, double G22, double G23, double G33)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int M = Nx * Ny * Nz_half;
    if (idx >= M)
        return;

    const int iz = idx % Nz_half;
    const int iy = (idx / Nz_half) % Ny;
    const int ix = idx / (Ny * Nz_half);

    const int m1 = (ix > Nx / 2) ? (ix - Nx) : ix;
    const int m2 = (iy > Ny / 2) ? (iy - Ny) : iy;
    const int m3 = iz;

    const double factor = 4.0 * M_PI * M_PI;
    const double gmm = G11 * m1 * m1 + G22 * m2 * m2 + G33 * m3 * m3
                     + 2.0 * (G12 * m1 * m2 + G13 * m1 * m3 + G23 * m2 * m3);
    const double k2 = factor * gmm;
    const double boltz = exp(-k2 * ds) * norm;
    freq[idx].x *= boltz;
    freq[idx].y *= boltz;
}

//==============================================================================
// Test functions
//==============================================================================

static bool test_cpu_hex_glide(int Nx, int Ny, int Nz, double ds)
{
    const double Lx = 4.0, Ly = 4.0, Lz = 4.0;
    const std::array<double, 6> cell_hex = {Lx, Ly, Lz, M_PI / 2, M_PI / 2, 2.0 * M_PI / 3.0};

    SpaceGroup sg({Nx, Ny, Nz}, "P6_3/mmc", 488);
    sg.enable_z_mirror_physical_basis();
    const int z_shift = sg.get_z_mirror_shift();
    if (z_shift != Nz / 4)
    {
        std::printf("  CPU: unexpected z-shift %d (expected %d)\n", z_shift, Nz / 4);
        return false;
    }

    const int Nz2 = Nz / 2;
    const int M_phys = Nx * Ny * Nz2;
    const int M_logical = Nx * Ny * Nz;

    std::vector<double> q_phys(M_phys), q_phys_out(M_phys);
    std::vector<double> q_logical(M_logical), q_logical_out(M_logical);

    for (int ix = 0; ix < Nx; ++ix)
    {
        double x = (ix + 0.5) * Lx / Nx;
        for (int iy = 0; iy < Ny; ++iy)
        {
            double y = (iy + 0.5) * Ly / Ny;
            for (int iz = 0; iz < Nz2; ++iz)
            {
                int iz_full = iz + z_shift;
                if (iz_full >= Nz)
                    iz_full -= Nz;
                double z = (iz_full + 0.5) * Lz / Nz;
                q_phys[(ix * Ny + iy) * Nz2 + iz] =
                    std::cos(2.0 * M_PI * x / Lx) *
                    std::cos(2.0 * M_PI * y / Ly) *
                    std::cos(2.0 * M_PI * z / Lz);
            }
        }
    }

    expand_hex_z(q_phys.data(), q_logical.data(), Nx, Ny, Nz, z_shift);

    FftwCrysFFTHex crys({Nx, Ny, Nz}, cell_hex);
    crys.set_contour_step(ds);
    crys.diffusion(q_phys.data(), q_phys_out.data());

    diffusion_standard_fft_hex(q_logical.data(), q_logical_out.data(), Nx, Ny, Nz, cell_hex, ds);

    double max_rel = 0.0;
    for (int ix = 0; ix < Nx; ++ix)
    {
        for (int iy = 0; iy < Ny; ++iy)
        {
            for (int iz = 0; iz < Nz2; ++iz)
            {
                int iz_full = iz + z_shift;
                if (iz_full >= Nz)
                    iz_full -= Nz;
                const int full_idx = (ix * Ny + iy) * Nz + iz_full;
                const double v1 = q_phys_out[(ix * Ny + iy) * Nz2 + iz];
                const double v2 = q_logical_out[full_idx];
                const double rel = std::abs(v1 - v2) / (std::abs(v2) + 1e-15);
                if (rel > max_rel)
                    max_rel = rel;
            }
        }
    }

    const bool ok = (max_rel < 1e-6);
    std::printf("  CPU:  %.2e %s\n", max_rel, ok ? "OK" : "FAIL");
    return ok;
}

static bool test_cuda_hex_glide(int Nx, int Ny, int Nz, double ds)
{
    const double Lx = 4.0, Ly = 4.0, Lz = 4.0;
    const std::array<double, 6> cell_hex = {Lx, Ly, Lz, M_PI / 2, M_PI / 2, 2.0 * M_PI / 3.0};

    SpaceGroup sg({Nx, Ny, Nz}, "P6_3/mmc", 488);
    sg.enable_z_mirror_physical_basis();
    const int z_shift = sg.get_z_mirror_shift();
    if (z_shift != Nz / 4)
    {
        std::printf("  CUDA: unexpected z-shift %d (expected %d)\n", z_shift, Nz / 4);
        return false;
    }

    const int Nz2 = Nz / 2;
    const int M_phys = Nx * Ny * Nz2;
    const int M_logical = Nx * Ny * Nz;
    const int M_complex = Nx * Ny * (Nz / 2 + 1);

    std::vector<double> h_phys(M_phys), h_logical(M_logical);
    for (int ix = 0; ix < Nx; ++ix)
    {
        double x = (ix + 0.5) * Lx / Nx;
        for (int iy = 0; iy < Ny; ++iy)
        {
            double y = (iy + 0.5) * Ly / Ny;
            for (int iz = 0; iz < Nz2; ++iz)
            {
                int iz_full = iz + z_shift;
                if (iz_full >= Nz)
                    iz_full -= Nz;
                double z = (iz_full + 0.5) * Lz / Nz;
                h_phys[(ix * Ny + iy) * Nz2 + iz] =
                    std::cos(2.0 * M_PI * x / Lx) *
                    std::cos(2.0 * M_PI * y / Ly) *
                    std::cos(2.0 * M_PI * z / Lz);
            }
        }
    }
    expand_hex_z(h_phys.data(), h_logical.data(), Nx, Ny, Nz, z_shift);

    // HexZ CrysFFT (CUDA)
    CudaCrysFFTHex crys({Nx, Ny, Nz}, cell_hex);
    crys.set_contour_step(ds);

    double *d_in{}, *d_out{};
    gpu_error_check(cudaMalloc(&d_in, sizeof(double) * M_phys));
    gpu_error_check(cudaMalloc(&d_out, sizeof(double) * M_phys));
    gpu_error_check(cudaMemcpy(d_in, h_phys.data(), sizeof(double) * M_phys, cudaMemcpyHostToDevice));

    crys.diffusion(d_in, d_out);
    cudaDeviceSynchronize();

    std::vector<double> h_crys_out(M_phys);
    gpu_error_check(cudaMemcpy(h_crys_out.data(), d_out, sizeof(double) * M_phys, cudaMemcpyDeviceToHost));
    cudaFree(d_in);
    cudaFree(d_out);

    // Full FFT (cuFFT) reference
    double* d_full{};
    cufftDoubleComplex* d_freq{};
    gpu_error_check(cudaMalloc(&d_full, sizeof(double) * M_logical));
    gpu_error_check(cudaMalloc(&d_freq, sizeof(cufftDoubleComplex) * M_complex));
    gpu_error_check(cudaMemcpy(d_full, h_logical.data(), sizeof(double) * M_logical, cudaMemcpyHostToDevice));

    cufftHandle fwd, bwd;
    int n[3] = {Nx, Ny, Nz};
    cufftPlanMany(&fwd, 3, n, nullptr, 1, 0, nullptr, 1, 0, CUFFT_D2Z, 1);
    cufftPlanMany(&bwd, 3, n, nullptr, 1, 0, nullptr, 1, 0, CUFFT_Z2D, 1);

    cufftExecD2Z(fwd, d_full, d_freq);

    const auto metric = compute_recip_metric(cell_hex);
    const double G11 = metric[0];
    const double G12 = metric[1];
    const double G13 = metric[2];
    const double G22 = metric[3];
    const double G23 = metric[4];
    const double G33 = metric[5];
    const double norm = 1.0 / M_logical;
    const int Nz_half = Nz / 2 + 1;
    const int M_freq = Nx * Ny * Nz_half;
    ker_apply_boltzmann_hex<<<(M_freq + 255) / 256, 256>>>(
        d_freq, Nx, Ny, Nz_half, ds, norm, G11, G12, G13, G22, G23, G33);

    cufftExecZ2D(bwd, d_freq, d_full);
    cudaDeviceSynchronize();

    std::vector<double> h_std_out(M_logical);
    gpu_error_check(cudaMemcpy(h_std_out.data(), d_full, sizeof(double) * M_logical, cudaMemcpyDeviceToHost));
    cufftDestroy(fwd);
    cufftDestroy(bwd);
    cudaFree(d_full);
    cudaFree(d_freq);

    double max_rel = 0.0;
    for (int ix = 0; ix < Nx; ++ix)
    {
        for (int iy = 0; iy < Ny; ++iy)
        {
            for (int iz = 0; iz < Nz2; ++iz)
            {
                int iz_full = iz + z_shift;
                if (iz_full >= Nz)
                    iz_full -= Nz;
                const int full_idx = (ix * Ny + iy) * Nz + iz_full;
                const double v1 = h_crys_out[(ix * Ny + iy) * Nz2 + iz];
                const double v2 = h_std_out[full_idx];
                const double rel = std::abs(v1 - v2) / (std::abs(v2) + 1e-15);
                if (rel > max_rel)
                    max_rel = rel;
            }
        }
    }

    const bool ok = (max_rel < 1e-6);
    std::printf("  CUDA: %.2e %s\n", max_rel, ok ? "OK" : "FAIL");
    return ok;
}

int main()
{
    std::printf("=== CrysFFT HexZ Glide Test (P6_3/mmc) ===\n\n");

    const int Nx = 24, Ny = 24, Nz = 24;
    const double ds = 0.01;

    std::printf("Grid: %dx%dx%d\n", Nx, Ny, Nz);
    bool ok = true;
    ok &= test_cpu_hex_glide(Nx, Ny, Nz, ds);
    ok &= test_cuda_hex_glide(Nx, Ny, Nz, ds);

    std::printf(ok ? "\nAll passed!\n" : "\nFAILED!\n");
    return ok ? 0 : 1;
}
