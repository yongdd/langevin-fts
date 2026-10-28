/**
 * @file CudaCrysFFTObliqueZ.cu
 * @brief CUDA implementation of z-mirror CrysFFT (DCT-z + FFT-xy as one
 *        permuted real 3D FFT, see CudaCrysFFTObliqueZ.h).
 */

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <mutex>
#include <vector>

#include "CudaCrysFFTObliqueZ.h"
#include "CudaCommon.h"
#include "Exception.h"
#include "MinimalFormAlias.h"
#include "ReciprocalMetric.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace {

// Makhoul's permutation of each z line of length n: v[m] = x[2m] for
// m < (n+1)/2 and v[n-1-j] = x[2j+1]
__global__ void ker_permute_z(const double* __restrict__ in, double* __restrict__ out, int total, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < total)
    {
        const int m = idx % n;
        const int src = (m < (n + 1) / 2) ? 2 * m : 2 * (n - 1 - m) + 1;
        out[idx] = in[idx - m + src];
        idx += blockDim.x * gridDim.x;
    }
}

// Inverse of ker_permute_z
__global__ void ker_unpermute_z(const double* __restrict__ in, double* __restrict__ out, int total, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < total)
    {
        const int i = idx % n;
        const int m = (i % 2 == 0) ? i / 2 : n - 1 - i / 2;
        out[idx] = in[idx - i + m];
        idx += blockDim.x * gridDim.x;
    }
}

// S p + D w conj(p_m) for one entry, with f = (S, D)
__device__ inline cufftDoubleComplex obliquez_mode(
    cufftDoubleComplex p, cufftDoubleComplex p_m, double2 f, double2 w)
{
    // w * conj(p_m)
    const double wc_re = w.x * p_m.x + w.y * p_m.y;
    const double wc_im = w.y * p_m.x - w.x * p_m.y;

    cufftDoubleComplex out;
    out.x = f.x * p.x + f.y * wc_re;
    out.y = f.x * p.y + f.y * wc_im;
    return out;
}

// P'(k) = S(k) P(k) + D(k) w(kz) conj(P(-kx, -ky, kz)) on the half spectrum
// (nz_c = n/2 + 1 stored kz), with sd = (S, D) and w = (cos, sin)(pi kz/n).
// In place: k and its mirror (-kx, -ky, kz) need each other's old value, so the
// thread of the lower index of each pair reads both before it writes both; an
// entry that is its own mirror is updated once.
__global__ void ker_obliquez_spectral(
    cufftDoubleComplex* __restrict__ spectrum,
    const double2* __restrict__ sd,
    const double2* __restrict__ twiddle,
    int nx, int ny, int nz_c)
{
    const int total = nx * ny * nz_c;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < total)
    {
        const int kz = idx % nz_c;
        const int tmp = idx / nz_c;
        const int iy = tmp % ny;
        const int ix = tmp / ny;
        const int ix_m = (nx - ix) % nx;
        const int iy_m = (ny - iy) % ny;
        const int idx_m = (ix_m * ny + iy_m) * nz_c + kz;

        if (idx <= idx_m)
        {
            const cufftDoubleComplex p = spectrum[idx];
            const cufftDoubleComplex p_m = spectrum[idx_m];
            const double2 w = twiddle[kz];
            spectrum[idx] = obliquez_mode(p, p_m, sd[idx], w);
            if (idx_m != idx)
                spectrum[idx_m] = obliquez_mode(p_m, p, sd[idx_m], w);
        }

        idx += blockDim.x * gridDim.x;
    }
}

}  // namespace

//------------------------------------------------------------------------------
// Constructor / Destructor
//------------------------------------------------------------------------------
CudaCrysFFTObliqueZ::CudaCrysFFTObliqueZ(
    std::array<int, 3> nx_logical,
    std::array<double, 6> cell_para,
    std::array<double, 9> /*trans_part*/)
    : nx_logical_(nx_logical),
      tables_(std::make_shared<CudaCrysFFTTables<Table>>(cell_para))
{
    try
    {
        for (int d = 0; d < 3; ++d)
        {
            if (nx_logical_[d] <= 0)
                throw_with_line_number("CudaCrysFFTObliqueZ requires positive grid dimensions.");
        }
        if (nx_logical_[2] % 2 != 0)
            throw_with_line_number("CudaCrysFFTObliqueZ requires even Nz.");

        nx_physical_ = {nx_logical_[0], nx_logical_[1], nx_logical_[2] / 2};
        M_logical_ = nx_logical_[0] * nx_logical_[1] * nx_logical_[2];
        M_physical_ = nx_physical_[0] * nx_physical_[1] * nx_physical_[2];
        M_spectral_ = nx_physical_[0] * nx_physical_[1] * (nx_physical_[2] / 2 + 1);
        norm_factor_ = 1.0 / static_cast<double>(M_logical_);

        init_transforms();
    }
    catch (...)
    {
        release();
        throw;
    }
}

CudaCrysFFTObliqueZ::CudaCrysFFTObliqueZ(const CudaCrysFFTObliqueZ& other)
    : nx_logical_(other.nx_logical_),
      nx_physical_(other.nx_physical_),
      M_logical_(other.M_logical_),
      M_physical_(other.M_physical_),
      M_spectral_(other.M_spectral_),
      tables_(other.tables_.shared()),
      d_twiddle_(other.d_twiddle_),
      norm_factor_(other.norm_factor_)
{
    try
    {
        init_transforms();
    }
    catch (...)
    {
        release();
        throw;
    }
}

std::unique_ptr<CudaCrysFFTBase> CudaCrysFFTObliqueZ::clone_sharing_tables() const
{
    return std::unique_ptr<CudaCrysFFTBase>(new CudaCrysFFTObliqueZ(*this));
}

void CudaCrysFFTObliqueZ::init_transforms()
{
    // Called once, from a constructor: a second call would leak the plans
    if (plans_initialized_)
        throw_with_line_number("CudaCrysFFTObliqueZ::init_transforms() called with plans already created.");

    const int n = nx_physical_[2];
    if (!d_twiddle_)
    {
        const int nz_c = n / 2 + 1;
        std::vector<double> h_twiddle(2 * nz_c);
        for (int kz = 0; kz < nz_c; ++kz)
        {
            const double angle = M_PI * kz / n;
            h_twiddle[2 * kz] = std::cos(angle);
            h_twiddle[2 * kz + 1] = std::sin(angle);
        }
        auto d_twiddle = std::make_shared<CudaDeviceBuffer<double>>(2 * nz_c);
        CUDA_CHECK(cudaMemcpy(*d_twiddle, h_twiddle.data(), sizeof(double) * 2 * nz_c, cudaMemcpyHostToDevice));
        d_twiddle_ = std::move(d_twiddle);
    }

    d_real_.allocate(M_physical_);
    d_spectrum_.allocate(M_spectral_);

    if (count_cufft_plan(cufftPlan3d(&plan_fwd_, nx_physical_[0], nx_physical_[1], n, CUFFT_D2Z)) != CUFFT_SUCCESS)
        throw_with_line_number("CudaCrysFFTObliqueZ failed to create its cuFFT D2Z plan.");
    if (count_cufft_plan(cufftPlan3d(&plan_bwd_, nx_physical_[0], nx_physical_[1], n, CUFFT_Z2D)) != CUFFT_SUCCESS)
    {
        // plans_initialized_ covers both plans; destroy the first one here
        release_cufft_plan(plan_fwd_);
        throw_with_line_number("CudaCrysFFTObliqueZ failed to create its cuFFT Z2D plan.");
    }
    plans_initialized_ = true;
    CUFFT_CHECK(cufftSetStream(plan_fwd_, stream_));
    CUFFT_CHECK(cufftSetStream(plan_bwd_, stream_));
}

CudaCrysFFTObliqueZ::~CudaCrysFFTObliqueZ()
{
    release();
}

// Idempotent; tolerates a partly built object. The cuFFT plans need the
// explicit free; the tables and buffers are owning members.
void CudaCrysFFTObliqueZ::release()
{
    d_boltz_current_ = nullptr;
    tables_.reset();
    free_plans();
    d_twiddle_.reset();
    d_real_.release();
    d_spectrum_.release();
}

void CudaCrysFFTObliqueZ::free_plans()
{
    if (plans_initialized_)
    {
        release_cufft_plan(plan_fwd_);
        release_cufft_plan(plan_bwd_);
        plans_initialized_ = false;
    }
}

//------------------------------------------------------------------------------
// Boltzmann cache
//------------------------------------------------------------------------------
CudaCrysFFTObliqueZ::Table CudaCrysFFTObliqueZ::generateBoltzmann(double ds, const std::array<double, 6>& cell, Table buffers)
{
    const int Nx = nx_physical_[0];
    const int Ny = nx_physical_[1];
    const int n = nx_physical_[2];
    const int nz_c = n / 2 + 1;
    std::vector<double> h_table(2 * static_cast<size_t>(M_spectral_));

    const std::array<double, 6> recip_metric = reciprocal_metric_of_cell(cell);
    const double G11 = recip_metric[0];
    const double G12 = recip_metric[1];
    const double G13 = recip_metric[2];
    const double G22 = recip_metric[3];
    const double G23 = recip_metric[4];
    const double G33 = recip_metric[5];
    const double factor = 4.0 * M_PI * M_PI;

    // For oblique in-plane cells (gamma != 90), pick the in-plane alias
    // representative minimizing the metric form, matching the invariant
    // (first-Brillouin-zone) convention used by Pseudo for the standard
    // full-grid path (see CrysFFTObliqueZBase::generateBoltzmann on CPU).
    // The z (DCT) index is a true non-negative frequency and needs no
    // re-aliasing (ObliqueZ requires alpha = beta = 90, i.e. G13 = G23 = 0).
    // The minimal form is the same for (kx, ky) and (-kx, -ky), which makes
    // the factors even in (kx, ky) as the spectral kernel requires; the search
    // is exact (MinimalFormAlias.h), since a window of one period misses the
    // minimum on cells with strongly anisotropic grid spacing.
    // Relative-tolerance gate: gamma of exactly 90 degrees gives
    // cos(pi/2) ~ 6e-17; do not flip rectangular cells onto the
    // minimal-form alias path.
    const bool oblique_xy = (std::abs(G12) > 1e-12 * std::max(std::abs(G11), std::abs(G22)));
    const std::array<int, 3> grid_xy = {Nx, Ny, 1};
    const IndexMetric metric_xy{G11, G22, 0.0, G12, 0.0, 0.0};
    auto form_xy = [&](double a, double b, double) {
        return G11 * a * a + 2.0 * G12 * a * b + G22 * b * b;
    };

    // (S, D) = (B(kz) + B(n - kz), B(kz) - B(n - kz)) with B(n) = 0, per half-spectrum entry
    size_t idx = 0;
    for (int ix = 0; ix < Nx; ++ix)
    {
        for (int iy = 0; iy < Ny; ++iy)
        {
            std::array<int, 3> rep = {(ix > Nx / 2) ? (ix - Nx) : ix, (iy > Ny / 2) ? (iy - Ny) : iy, 0};
            if (oblique_xy)
                min_form_alias_rep(rep, grid_xy, metric_xy, form_xy);
            const int m1 = rep[0], m2 = rep[1];
            auto boltzmann = [&](int m3) {
                const double gmm = G11 * m1 * m1 + G22 * m2 * m2 + G33 * m3 * m3
                                 + 2.0 * (G12 * m1 * m2 + G13 * m1 * m3 + G23 * m2 * m3);
                return std::exp(-factor * gmm * ds) * norm_factor_;
            };
            for (int kz = 0; kz < nz_c; ++kz)
            {
                const double b = boltzmann(kz);
                const double b_mirror = (kz == 0) ? 0.0 : boltzmann(n - kz);
                h_table[2 * idx] = b + b_mirror;
                h_table[2 * idx + 1] = b - b_mirror;
                ++idx;
            }
        }
    }

    // `buffers` hold a dropped table when one is left; upload on this object's stream
    if (buffers.empty())
        buffers.allocate(2 * M_spectral_);
    CUDA_CHECK(cudaMemcpyAsync(buffers, h_table.data(), sizeof(double) * h_table.size(), cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));   // h_table is about to go out of scope
    return buffers;
}

//------------------------------------------------------------------------------
// Public API
//------------------------------------------------------------------------------
void CudaCrysFFTObliqueZ::set_cell_para(const std::array<double, 6>& cell_para)
{
    // Changes the cell of every object sharing the tables (no-op for the current cell)
    tables_.shared()->set_cell(cell_para);
    if (tables_.refresh())
        d_boltz_current_ = nullptr;
}

void CudaCrysFFTObliqueZ::set_contour_step(double ds)
{
    if (tables_.refresh())   // another object changed the cell
        d_boltz_current_ = nullptr;
    if (d_boltz_current_ != nullptr && ds == ds_current_)
        return;

    d_boltz_current_ = tables_.get({kBoltzmann, 0, ds},
        [&](const std::array<double, 6>& cell, Table buffers) {
            return generateBoltzmann(ds, cell, std::move(buffers));
        }).get();
    ds_current_ = ds;
}

void CudaCrysFFTObliqueZ::set_stream(cudaStream_t stream)
{
    // The plans already run on stream_ (set at construction or by the last
    // call); diffusion() calls this every time
    if (stream == stream_)
        return;
    stream_ = stream;
    if (plans_initialized_)
    {
        CUFFT_CHECK(cufftSetStream(plan_fwd_, stream_));
        CUFFT_CHECK(cufftSetStream(plan_bwd_, stream_));
    }
}

void CudaCrysFFTObliqueZ::diffusion(double* d_q_in, double* d_q_out)
{
    diffusion(d_q_in, d_q_out, stream_);
}

void CudaCrysFFTObliqueZ::diffusion(double* d_q_in, double* d_q_out, cudaStream_t stream)
{
    set_stream(stream);
    if (tables_.refresh())   // another object changed the cell
        d_boltz_current_ = nullptr;
    if (d_boltz_current_ == nullptr)
        throw_with_line_number("CudaCrysFFTObliqueZ::set_contour_step must be called before diffusion() and after a cell change.");

    // FTS_PROFILE_CRYSFFT_HEX_DETAIL times the stages of the first call
    static const bool do_profile = (std::getenv("FTS_PROFILE_CRYSFFT_HEX_DETAIL") != nullptr);
    static std::atomic<bool> prof_once{false};
    std::vector<CudaEvent> events(do_profile && !prof_once.exchange(true) ? 6 : 0);   // destroyed on every exit
    auto mark = [&](size_t i) {
        if (!events.empty())
            CUDA_CHECK(cudaEventRecord(events[i], stream_));
    };

    const int threads = 256;
    const int n = nx_physical_[2];

    // The input is only read, so an out-of-place call preserves it
    mark(0);
    ker_permute_z<<<grid_blocks(M_physical_, threads), threads, 0, stream_>>>(d_q_in, d_real_, M_physical_, n);
    CUDA_CHECK_LAUNCH();
    mark(1);
    CUFFT_CHECK(cufftExecD2Z(plan_fwd_, d_real_, d_spectrum_));
    mark(2);
    // Boltzmann factors in the table include the normalization
    ker_obliquez_spectral<<<grid_blocks(M_spectral_, threads), threads, 0, stream_>>>(
        d_spectrum_, reinterpret_cast<const double2*>(d_boltz_current_),
        reinterpret_cast<const double2*>(d_twiddle_->get()), nx_physical_[0], nx_physical_[1], n / 2 + 1);
    CUDA_CHECK_LAUNCH();
    mark(3);
    // Z2D overwrites its input d_spectrum_, which the next step recomputes
    CUFFT_CHECK(cufftExecZ2D(plan_bwd_, d_spectrum_, d_real_));
    mark(4);
    ker_unpermute_z<<<grid_blocks(M_physical_, threads), threads, 0, stream_>>>(d_real_, d_q_out, M_physical_, n);
    CUDA_CHECK_LAUNCH();
    mark(5);

    if (!events.empty())
    {
        CUDA_CHECK(cudaEventSynchronize(events[5]));
        const char* stages[5] = {"permute", "fft", "spectral", "ifft", "unpermute"};
        std::printf("[CrysFFT-ObliqueZ]");
        for (int i = 0; i < 5; ++i)
        {
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, events[i], events[i + 1]));
            std::printf(" %s(ms)=%.4f", stages[i], ms);
        }
        float ms_total = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms_total, events[0], events[5]));
        std::printf(" total(ms)=%.4f\n", ms_total);
    }
}
