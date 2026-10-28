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
 * @file TestCrysFFTSharedTables.cu
 * @brief CUDA CrysFFT clones share their tables and cell consistently.
 *
 * A solver keeps one CrysFFT object per stream; the clones share the k-space
 * tables and the cell (CudaCrysFFTTables.h). For the Pmmm DCT, recursive 3m and
 * ObliqueZ engines, with `a` and its clone `b`:
 *
 * 1. b uses the table a built.
 * 2. a changes the cell, which b is not told, and builds a table for a new
 *    coefficient in the buffers of the dropped ones. b's diffusion() without a
 *    new set_contour_step() throws instead of reading a dropped table.
 * 3. b's old and new coefficients give the diffusion of an object created for
 *    the new cell; a then uses the table b built.
 * 4. The stress multipliers of b after a's cell change are those of the new cell.
 * 5. When b's first call after a's cell change is a stress-multiplier lookup,
 *    b's set_contour_step() with its old coefficient still drops the table of
 *    the old cell (the lookup does not use up the report of the change).
 *
 * "Equal" means a relative difference of at most 1e-14 against the fresh object.
 */

#include <array>
#include <cmath>
#include <cstdio>
#include <exception>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "CudaCommon.h"
#include "CudaCrysFFT.h"
#include "CudaCrysFFTRecursive3m.h"
#include "CudaCrysFFTObliqueZ.h"

namespace {

constexpr double TOLERANCE = 1e-14;
constexpr double PI = 3.14159265358979323846;

using Make = std::function<std::unique_ptr<CudaCrysFFTBase>(const std::array<double, 6>&)>;
/// Applies a stress multiplier of the engine to d_in (does nothing for ObliqueZ)
using ApplyMultiplier = std::function<bool(CudaCrysFFTBase&, double*, double*)>;

std::vector<double> to_host(const double* d, int n)
{
    std::vector<double> h(n);
    CUDA_CHECK(cudaMemcpy(h.data(), d, sizeof(double) * n, cudaMemcpyDeviceToHost));
    return h;
}

double relative_difference(const std::vector<double>& a, const std::vector<double>& b)
{
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        num = std::max(num, std::abs(a[i] - b[i]));
        den = std::max(den, std::abs(b[i]));
    }
    return num / den;
}

bool check(const char* engine, const char* what, double err)
{
    const bool ok = std::isfinite(err) && err <= TOLERANCE;
    std::printf("  %-12s %-52s %.1e  %s\n", engine, what, err, ok ? "ok" : "FAIL");
    return ok;
}

/// diffusion(coeff) of `crysfft` on d_in, on the host
std::vector<double> diffuse(CudaCrysFFTBase& crysfft, double coeff, double* d_in, double* d_out, int M_phys)
{
    crysfft.set_contour_step(coeff);
    crysfft.diffusion(d_in, d_out);
    CUDA_CHECK(cudaDeviceSynchronize());
    return to_host(d_out, M_phys);
}

bool run(const char* engine, const Make& make, const ApplyMultiplier& apply_multiplier, int M_phys,
         const std::array<double, 6>& cell_a, const std::array<double, 6>& cell_b)
{
    const double ds1 = 0.01, ds2 = 0.015, ds3 = 0.0075;

    std::mt19937 rng(17);
    std::normal_distribution<double> normal(0.0, 5.0);
    std::vector<double> q(M_phys);
    for (double& v : q)
        v = std::exp(0.2 * normal(rng));
    CudaDeviceBuffer<double> d_q(M_phys), d_out(M_phys);
    CUDA_CHECK(cudaMemcpy(d_q, q.data(), sizeof(double) * M_phys, cudaMemcpyHostToDevice));

    // References: objects created for each cell
    std::unique_ptr<CudaCrysFFTBase> fresh_a = make(cell_a);
    std::unique_ptr<CudaCrysFFTBase> fresh_b = make(cell_b);
    const std::vector<double> ref_a1 = diffuse(*fresh_a, ds1, d_q, d_out, M_phys);
    const std::vector<double> ref_b1 = diffuse(*fresh_b, ds1, d_q, d_out, M_phys);
    const std::vector<double> ref_b2 = diffuse(*fresh_b, ds2, d_q, d_out, M_phys);
    const std::vector<double> ref_b3 = diffuse(*fresh_b, ds3, d_q, d_out, M_phys);
    std::vector<double> ref_mult_b;
    const bool has_multiplier = apply_multiplier(*fresh_b, d_q, d_out);
    if (has_multiplier)
    {
        CUDA_CHECK(cudaDeviceSynchronize());
        ref_mult_b = to_host(d_out, M_phys);
    }

    std::unique_ptr<CudaCrysFFTBase> a = make(cell_a);
    std::unique_ptr<CudaCrysFFTBase> b = a->clone_sharing_tables();
    bool ok = true;

    // 1. b uses the table a built (and a multiplier of the old cell)
    ok = check(engine, "a, old cell", relative_difference(diffuse(*a, ds1, d_q, d_out, M_phys), ref_a1)) && ok;
    ok = check(engine, "b with the table a built", relative_difference(diffuse(*b, ds1, d_q, d_out, M_phys), ref_a1)) && ok;
    if (has_multiplier)
        apply_multiplier(*b, d_q, d_out);

    // 2. a changes the cell and builds ds2 in the buffers of the dropped tables
    a->set_cell_para(cell_b);
    ok = check(engine, "a after its cell change", relative_difference(diffuse(*a, ds2, d_q, d_out, M_phys), ref_b2)) && ok;
    bool threw = false;
    try
    {
        b->diffusion(d_q, d_out);
    }
    catch (std::exception&)
    {
        threw = true;
    }
    std::printf("  %-12s %-52s %s\n", engine, "b's stale table is not used", threw ? "ok" : "FAIL");
    ok = threw && ok;

    // 3. b's coefficients are those of the new cell; a uses the table b builds
    ok = check(engine, "b, old coefficient, new cell", relative_difference(diffuse(*b, ds1, d_q, d_out, M_phys), ref_b1)) && ok;
    ok = check(engine, "b, new coefficient, new cell", relative_difference(diffuse(*b, ds3, d_q, d_out, M_phys), ref_b3)) && ok;
    ok = check(engine, "a with the table b built", relative_difference(diffuse(*a, ds3, d_q, d_out, M_phys), ref_b3)) && ok;

    // 4. b's stress multiplier after a's cell change
    if (has_multiplier)
    {
        apply_multiplier(*b, d_q, d_out);
        CUDA_CHECK(cudaDeviceSynchronize());
        ok = check(engine, "b's stress multiplier, new cell", relative_difference(to_host(d_out, M_phys), ref_mult_b)) && ok;
    }

    // 5. A multiplier lookup as the first call after the cell change
    if (has_multiplier)
    {
        std::unique_ptr<CudaCrysFFTBase> c = make(cell_a);
        std::unique_ptr<CudaCrysFFTBase> d = c->clone_sharing_tables();
        diffuse(*c, ds1, d_q, d_out, M_phys);
        diffuse(*d, ds1, d_q, d_out, M_phys);
        c->set_cell_para(cell_b);
        apply_multiplier(*d, d_q, d_out);
        ok = check(engine, "b after a multiplier lookup, old coefficient",
                   relative_difference(diffuse(*d, ds1, d_q, d_out, M_phys), ref_b1)) && ok;
    }
    return ok;
}

}  // namespace

int main()
{
    try
    {
        bool ok = true;
        const double right = PI / 2;

        // Pmmm DCT: physical grid (Nx/2, Ny/2, Nz/2)
        {
            const std::array<int, 3> nx = {8, 6, 10};
            const Make make = [nx](const std::array<double, 6>& cell) {
                return std::unique_ptr<CudaCrysFFTBase>(new CudaCrysFFT(nx, cell));
            };
            const ApplyMultiplier apply = [](CudaCrysFFTBase& c, double* d_in, double* d_out) {
                auto& pmmm = static_cast<CudaCrysFFT&>(c);
                pmmm.apply_multiplier(d_in, d_out, pmmm.get_axis_boltz_k2_multiplier(3, 0.02), 0);
                return true;
            };
            ok = run("Pmmm DCT", make, apply, 4 * 3 * 5,
                     {1.5, 1.3, 1.7, right, right, right}, {1.62, 1.25, 1.81, right, right, right}) && ok;
        }

        // Recursive 3m (no translations): physical grid (Nx/2, Ny/2, Nz/2)
        {
            const std::array<int, 3> nx = {8, 8, 8};
            const std::array<double, 9> translations = {0, 0, 0, 0, 0, 0, 0, 0, 0};
            const Make make = [nx, translations](const std::array<double, 6>& cell) {
                return std::unique_ptr<CudaCrysFFTBase>(new CudaCrysFFTRecursive3m(nx, cell, translations));
            };
            const ApplyMultiplier apply = [](CudaCrysFFTBase& c, double* d_in, double* d_out) {
                static_cast<CudaCrysFFTRecursive3m&>(c).apply_multiplier(
                    d_in, d_out, CudaCrysFFTRecursive3m::MultiplierType::ExpKx2, 0.02, 0);
                return true;
            };
            ok = run("Recursive 3m", make, apply, 4 * 4 * 4,
                     {1.6, 1.6, 1.6, right, right, right}, {1.75, 1.75, 1.75, right, right, right}) && ok;
        }

        // ObliqueZ: physical grid (Nx, Ny, Nz/2), gamma changes with the cell
        {
            const std::array<int, 3> nx = {8, 6, 8};
            const Make make = [nx](const std::array<double, 6>& cell) {
                return std::unique_ptr<CudaCrysFFTBase>(new CudaCrysFFTObliqueZ(nx, cell));
            };
            const ApplyMultiplier none = [](CudaCrysFFTBase&, double*, double*) { return false; };
            ok = run("ObliqueZ", make, none, 8 * 6 * 4,
                     {1.5, 1.3, 1.2, right, right, 100.0 * PI / 180.0},
                     {1.55, 1.28, 1.22, right, right, 104.0 * PI / 180.0}) && ok;
        }

        if (!ok)
        {
            std::printf("TestCrysFFTSharedTables FAILED\n");
            return -1;
        }
        std::printf("TestCrysFFTSharedTables passed\n");
        return 0;
    }
    catch (std::exception& exc)
    {
        std::printf("Unexpected exception: %s\n", exc.what());
        return -1;
    }
}
