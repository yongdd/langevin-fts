/**
 * @file TestCrysFFTObliqueZCuda.cu
 * @brief CUDA ObliqueZ CrysFFT against the CPU (FFTW) ObliqueZ engine.
 *
 * The CUDA engine forms the z DCT and the xy FFT as one permuted real 3D FFT
 * (see CudaCrysFFTObliqueZ.h); the CPU engine runs them as separate transforms.
 * Both use the same Boltzmann convention, so for random fields (standard
 * deviation 5) they must agree to round-off. The grids cover odd and even
 * Nz/2 (including 1 and 2), odd in-plane sizes, rectangular, hexagonal and
 * monoclinic in-plane angles, and grid spacings 8x apart in the plane, where
 * the alias search must be exact for the Boltzmann factors to be even in
 * (kx, ky) as the CUDA kernel requires. Each case also checks an in-place call, that an
 * out-of-place call keeps its input, a clone sharing the tables, and a cell
 * change followed by a new table.
 */

#include <array>
#include <cmath>
#include <cstdio>
#include <memory>
#include <random>
#include <vector>

#include "CudaCommon.h"
#include "CudaCrysFFTObliqueZ.h"
#include "FftwCrysFFTObliqueZ.h"
#include "TestNanSafe.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static constexpr double TOLERANCE = 1e-12;   // max |difference| / max |reference|

static double relative_difference(const std::vector<double>& value, const std::vector<double>& reference)
{
    double diff = 0.0, scale = 0.0;
    for (size_t i = 0; i < reference.size(); ++i)
    {
        diff = max_or_nan(diff, std::abs(value[i] - reference[i]));   // a NaN stays
        scale = std::max(scale, std::abs(reference[i]));
    }
    return diff / scale;
}

static std::vector<double> to_host(const double* d_data, int n)
{
    std::vector<double> h(n);
    CUDA_CHECK(cudaMemcpy(h.data(), d_data, sizeof(double) * n, cudaMemcpyDeviceToHost));
    return h;
}

static bool test_case(const std::array<int, 3>& nx, const std::array<double, 6>& cell, double ds, std::mt19937_64& rng)
{
    const int M_phys = nx[0] * nx[1] * (nx[2] / 2);
    std::normal_distribution<double> normal(0.0, 5.0);
    std::vector<double> q(M_phys);
    for (double& v : q)
        v = normal(rng);

    FftwCrysFFTObliqueZ cpu(nx, cell);
    cpu.set_contour_step(ds);
    std::vector<double> q_cpu = q, reference(M_phys);
    cpu.diffusion(q_cpu.data(), reference.data());

    CudaCrysFFTObliqueZ gpu(nx, cell);
    gpu.set_contour_step(ds);
    CudaDeviceBuffer<double> d_in(M_phys), d_out(M_phys);
    CUDA_CHECK(cudaMemcpy(d_in, q.data(), sizeof(double) * M_phys, cudaMemcpyHostToDevice));

    gpu.diffusion(d_in, d_out);
    const double err_out = relative_difference(to_host(d_out, M_phys), reference);
    const double err_input = relative_difference(to_host(d_in, M_phys), q);

    CUDA_CHECK(cudaMemcpy(d_out, q.data(), sizeof(double) * M_phys, cudaMemcpyHostToDevice));
    gpu.diffusion(d_out, d_out);
    const double err_in_place = relative_difference(to_host(d_out, M_phys), reference);

    std::unique_ptr<CudaCrysFFTBase> clone = gpu.clone_sharing_tables();
    clone->set_contour_step(ds);
    clone->diffusion(d_in, d_out);
    const double err_clone = relative_difference(to_host(d_out, M_phys), reference);

    // Cell change: lengths scaled, angles kept
    std::array<double, 6> cell_new = cell;
    for (int d = 0; d < 3; ++d)
        cell_new[d] *= 1.1;
    cpu.set_cell_para(cell_new);
    cpu.set_contour_step(ds);
    q_cpu = q;
    cpu.diffusion(q_cpu.data(), reference.data());
    gpu.set_cell_para(cell_new);
    gpu.set_contour_step(ds);
    gpu.diffusion(d_in, d_out);
    const double err_cell = relative_difference(to_host(d_out, M_phys), reference);

    const double err = std::max({err_out, err_in_place, err_clone, err_cell});
    const bool ok = (err < TOLERANCE) && (err_input == 0.0);
    std::printf("  %3dx%3dx%3d gamma=%5.1f  out %.1e  in-place %.1e  clone %.1e  new cell %.1e  input kept %s  %s\n",
                nx[0], nx[1], nx[2], cell[5] * 180.0 / M_PI, err_out, err_in_place, err_clone, err_cell,
                err_input == 0.0 ? "yes" : "NO", ok ? "OK" : "FAIL");
    return ok;
}

int main()
{
    std::printf("=== CUDA ObliqueZ CrysFFT vs CPU (FFTW) ObliqueZ, random fields (std 5) ===\n");
    const double rad = M_PI / 180.0;
    const double ds = 0.02;
    struct Case
    {
        std::array<int, 3> nx;
        std::array<double, 6> cell;
    };
    const Case cases[] = {
        {{24, 24, 30}, {2.0, 2.0, 2.5, 90 * rad, 90 * rad, 120 * rad}},   // odd Nz/2 = 15
        {{9, 9, 8}, {1.8, 1.8, 2.0, 90 * rad, 90 * rad, 120 * rad}},      // odd in-plane sizes
        {{16, 16, 64}, {2.0, 2.0, 4.0, 90 * rad, 90 * rad, 120 * rad}},
        {{6, 6, 2}, {1.5, 1.5, 1.0, 90 * rad, 90 * rad, 120 * rad}},      // Nz/2 = 1
        {{11, 10, 8}, {1.5, 1.3, 1.2, 90 * rad, 90 * rad, 100 * rad}},    // monoclinic, odd nx
        {{8, 6, 4}, {1.6, 1.2, 1.0, 90 * rad, 90 * rad, 100 * rad}},      // Nz/2 = 2
        {{12, 10, 14}, {1.5, 1.4, 1.8, 90 * rad, 90 * rad, 75 * rad}},    // odd Nz/2 = 7
        {{10, 8, 12}, {1.5, 1.2, 1.6, 90 * rad, 90 * rad, 90 * rad}},     // rectangular
        // grid spacing 8x apart: a one-period alias window misses the least form
        {{4, 32, 8}, {4.0, 4.0, 2.0, 90 * rad, 90 * rad, 120 * rad}},
        {{6, 48, 8}, {2.0, 2.0, 2.0, 90 * rad, 90 * rad, 110 * rad}},
    };

    std::mt19937_64 rng(2026);
    bool ok = true;
    for (const Case& c : cases)
        ok &= test_case(c.nx, c.cell, ds, rng);

    std::printf(ok ? "\nAll passed!\n" : "\nFAILED!\n");
    return ok ? 0 : 1;
}
