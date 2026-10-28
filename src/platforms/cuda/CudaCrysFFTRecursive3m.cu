/**
 * @file CudaCrysFFTRecursive3m.cu
 * @brief CUDA crystallographic FFT using 2x2x2 (3m) algorithm.
 */

#include "CudaCrysFFTRecursive3m.h"

#include "CudaCommon.h"
#include "Exception.h"
#include "Recursive3mFoldParity.h"
#include "Recursive3mHelpers.h"

#include <cstring>
#include <cmath>
#include <stdexcept>
#include <tuple>

// Host-side fold, twiddles and multipliers are shared with the CPU engine
using namespace crys_recursive_3m;

namespace {

__global__ void ker_apply_k_3m(
    cufftDoubleComplex* __restrict__ dst,
    const cufftDoubleComplex* __restrict__ src,
    const double* __restrict__ k_re0,
    const double* __restrict__ k_re1,
    const double* __restrict__ k_re2,
    const double* __restrict__ k_re3,
    const double* __restrict__ k_re4,
    const double* __restrict__ k_re5,
    const double* __restrict__ k_re6,
    const double* __restrict__ k_re7,
    const double* __restrict__ k_im0,
    const double* __restrict__ k_im1,
    const double* __restrict__ k_im2,
    const double* __restrict__ k_im3,
    const double* __restrict__ k_im4,
    const double* __restrict__ k_im5,
    const double* __restrict__ k_im6,
    const double* __restrict__ k_im7,
    int Nx2, int Ny2, int Nz2, int Nz2c)
{
    const int total = Nx2 * Ny2 * Nz2c;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    while (idx < total)
    {
        int iz = idx % Nz2c;
        int tmp = idx / Nz2c;
        int iy = tmp % Ny2;
        int ix = tmp / Ny2;

        int ix_m = (Nx2 - ix) % Nx2;
        int iy_m = (Ny2 - iy) % Ny2;

        size_t base_c = static_cast<size_t>(ix * Ny2 + iy) * Nz2c;
        size_t base_k = static_cast<size_t>(ix * Ny2 + iy) * Nz2;
        size_t base_c_110 = static_cast<size_t>(ix_m * Ny2 + iy_m) * Nz2c;
        size_t base_c_010 = static_cast<size_t>(ix * Ny2 + iy_m) * Nz2c;
        size_t base_c_100 = static_cast<size_t>(ix_m * Ny2 + iy) * Nz2c;
        size_t idx_k = base_k + iz;

        cufftDoubleComplex s000 = src[base_c + iz];
        cufftDoubleComplex s110 = src[base_c_110 + iz];
        cufftDoubleComplex s010 = src[base_c_010 + iz];
        cufftDoubleComplex s100 = src[base_c_100 + iz];

        double out_re =
            s000.x * k_re0[idx_k] - s000.y * k_im7[idx_k] +
            s110.x * k_re6[idx_k] - s110.y * k_im1[idx_k] +
            s010.x * k_re2[idx_k] - s010.y * k_im5[idx_k] +
            s100.x * k_re4[idx_k] - s100.y * k_im3[idx_k];

        double out_im =
            s000.x * k_im0[idx_k] + s000.y * k_re7[idx_k] +
            s110.x * k_im6[idx_k] + s110.y * k_re1[idx_k] +
            s010.x * k_im2[idx_k] + s010.y * k_re5[idx_k] +
            s100.x * k_im4[idx_k] + s100.y * k_re3[idx_k];

        dst[base_c + iz].x = out_re;
        dst[base_c + iz].y = out_im;

        idx += blockDim.x * gridDim.x;
    }
}

}  // namespace

CudaCrysFFTRecursive3m::CudaCrysFFTRecursive3m(
    std::array<int, 3> nx_logical,
    std::array<double, 6> cell_para,
    std::array<double, 9> translational_part)
    : nx_logical_(nx_logical),
      translational_part_(translational_part),
      tables_(std::make_shared<CudaCrysFFTTables<KCacheDevice>>(cell_para))
{
    try
    {
        for (int d = 0; d < 3; ++d)
        {
            if (nx_logical_[d] <= 0 || (nx_logical_[d] % 2) != 0)
            {
                throw_with_line_number("CudaCrysFFTRecursive3m requires even, positive grid dimensions.");
            }
        }
        // The kernels here are size-agnostic (ker_apply_k_3m iterates the packed
        // z range Nz2c = Nz2/2+1 with exact bounds; cuFFT D2Z/Z2D handles any
        // length). The nz/2 >= 2 bound mirrors the guard in
        // CrysFFTRecursive3mBase.h so CPU and CUDA stay constructible for exactly
        // the same grids.
        if ((nx_logical_[2] / 2) < 2)
        {
            throw_with_line_number("CudaCrysFFTRecursive3m requires Nz/2 >= 2.");
        }

        // The fold pairs twiddle r_p with the octant-transformed Boltzmann array
        // S_{psi(p)}, where psi depends on the parities of the m3 generator
        // translations in grid units (see Recursive3mFoldParity.h). When psi is
        // not a bijection the even-index physical basis does not exist on this
        // grid and the fold cannot represent the group; CrysFFTSelector rejects
        // such grids, but guard direct construction.
        if (!compute_m3_fold_permutation(translational_part_, nx_logical_, fold_perm_))
        {
            throw_with_line_number("CudaCrysFFTRecursive3m: m3 translations are "
                "incompatible with this grid (translation-parity permutation is "
                "not a bijection; no even-index physical basis exists).");
        }

        nx_physical_ = { nx_logical_[0] / 2, nx_logical_[1] / 2, nx_logical_[2] / 2 };
        M_logical_ = nx_logical_[0] * nx_logical_[1] * nx_logical_[2];
        M_physical_ = nx_physical_[0] * nx_physical_[1] * nx_physical_[2];
        M_complex_ = nx_physical_[0] * nx_physical_[1] * (nx_physical_[2] / 2 + 1);

        d_step1_.allocate(M_complex_);
        d_step2_.allocate(M_complex_);

        init_plans();
        twiddles_ = generate_twiddle_factors();
    }
    catch (...)
    {
        release();
        throw;
    }
}

CudaCrysFFTRecursive3m::CudaCrysFFTRecursive3m(const CudaCrysFFTRecursive3m& other)
    : nx_logical_(other.nx_logical_),
      nx_physical_(other.nx_physical_),
      M_logical_(other.M_logical_),
      M_physical_(other.M_physical_),
      M_complex_(other.M_complex_),
      translational_part_(other.translational_part_),
      fold_perm_(other.fold_perm_),
      twiddles_(other.twiddles_),
      tables_(other.tables_.shared())
{
    try
    {
        d_step1_.allocate(M_complex_);
        d_step2_.allocate(M_complex_);
        init_plans();
    }
    catch (...)
    {
        release();
        throw;
    }
}

std::unique_ptr<CudaCrysFFTBase> CudaCrysFFTRecursive3m::clone_sharing_tables() const
{
    return std::unique_ptr<CudaCrysFFTBase>(new CudaCrysFFTRecursive3m(*this));
}

CudaCrysFFTRecursive3m::~CudaCrysFFTRecursive3m()
{
    release();
}

// Idempotent; tolerates a partly built object (null buffers, plans not created).
void CudaCrysFFTRecursive3m::release()
{
    // The caches and buffers are owning; the cuFFT plans need the explicit free
    k_current_ = nullptr;
    tables_.reset();
    twiddles_.reset();
    d_step1_.release();
    d_step2_.release();

    free_plans();
}

void CudaCrysFFTRecursive3m::init_plans()
{
    if (plans_initialized_)
        return;
    if (count_cufft_plan(cufftPlan3d(&plan_r2c_, nx_physical_[0], nx_physical_[1], nx_physical_[2], CUFFT_D2Z)) != CUFFT_SUCCESS)
        throw_with_line_number("Failed to create cuFFT D2Z plan for CudaCrysFFTRecursive3m.");
    if (count_cufft_plan(cufftPlan3d(&plan_c2r_, nx_physical_[0], nx_physical_[1], nx_physical_[2], CUFFT_Z2D)) != CUFFT_SUCCESS)
    {
        // plans_initialized_ covers both plans; destroy the first one here
        release_cufft_plan(plan_r2c_);
        throw_with_line_number("Failed to create cuFFT Z2D plan for CudaCrysFFTRecursive3m.");
    }
    plans_initialized_ = true;
}

void CudaCrysFFTRecursive3m::free_plans()
{
    if (!plans_initialized_)
        return;
    release_cufft_plan(plan_r2c_);
    release_cufft_plan(plan_c2r_);
    plans_initialized_ = false;
}

std::shared_ptr<const CudaCrysFFTRecursive3m::Twiddles> CudaCrysFFTRecursive3m::generate_twiddle_factors() const
{
    auto twiddles = std::make_shared<Twiddles>();
    auto& r_re = twiddles->re;
    auto& r_im = twiddles->im;
    for (int i = 0; i < 8; ++i)
    {
        r_re[i].assign(M_physical_, 0.0);
        r_im[i].assign(M_physical_, 0.0);
    }

    auto operations = generate_m3_operations(translational_part_, nx_logical_);
    std::vector<std::tuple<double*, double*, std::array<double, 3>>> twiddle;
    for (size_t i = 0; i < operations.size(); ++i)
    {
        twiddle.emplace_back(r_re[i + 1].data(), r_im[i + 1].data(), operations[i]);
    }

    std::fill(r_re[0].begin(), r_re[0].end(), 1.0);
    std::fill(r_im[0].begin(), r_im[0].end(), 0.0);

    twiddle_factor(twiddle, nx_physical_);
    return twiddles;
}

CudaCrysFFTRecursive3m::KCacheDevice CudaCrysFFTRecursive3m::generate_k_cache(
    double coeff, const std::array<double, 6>& cell, KCacheDevice cache)
{
    const std::vector<double> tempmat = logical_k_table(nx_logical_, cell, [coeff](double kx2, double ky2, double kz2) {
        return std::exp(-(kx2 + ky2 + kz2) * coeff);
    });
    return build_k_cache_from_tempmat(tempmat, std::move(cache));
}

CudaCrysFFTRecursive3m::KCacheDevice CudaCrysFFTRecursive3m::generate_k_cache_from_multiplier(
    MultiplierType type, double coeff, const std::array<double, 6>& cell, KCacheDevice cache)
{
    const int axes = multiplier_axes(type);
    const bool with_bond = multiplier_has_bond(type);
    const std::vector<double> tempmat = logical_k_table(nx_logical_, cell, [axes, with_bond, coeff](double kx2, double ky2, double kz2) {
        const double k_axes = ((axes & 1) ? kx2 : 0.0) + ((axes & 2) ? ky2 : 0.0) + ((axes & 4) ? kz2 : 0.0);
        return with_bond ? std::exp(-(kx2 + ky2 + kz2) * coeff) * k_axes : k_axes;
    });
    return build_k_cache_from_tempmat(tempmat, std::move(cache));
}

CudaCrysFFTRecursive3m::KCacheDevice CudaCrysFFTRecursive3m::build_k_cache_from_tempmat(
    const std::vector<double>& tempmat, KCacheDevice cache)
{
    std::array<std::vector<double>, 8> k_split;
    std::array<double*, 8> k_split_ptr{};
    for (int i = 0; i < 8; ++i)
    {
        k_split[i].assign(M_physical_, 0.0);
        k_split_ptr[i] = k_split[i].data();
    }

    mat_split(tempmat.data(), k_split_ptr, nx_logical_);

    std::array<int, 3> halfsize = nx_physical_;
    constexpr std::array<std::pair<int, int>, 4> seq0{{{0,4},{1,5},{2,6},{3,7}}};
    constexpr std::array<std::pair<int, int>, 4> seq1{{{0,2},{1,3},{4,6},{5,7}}};
    constexpr std::array<std::pair<int, int>, 4> seq2{{{0,1},{2,3},{4,5},{6,7}}};
    add_sub_by_seq(k_split_ptr, seq0, halfsize);
    add_sub_by_seq(k_split_ptr, seq1, halfsize);
    add_sub_by_seq(k_split_ptr, seq2, halfsize);

    std::array<std::vector<double>, 8> k_re_host;
    std::array<std::vector<double>, 8> k_im_host;
    std::array<double*, 8> k_re_ptr{};
    std::array<double*, 8> k_im_ptr{};
    std::array<const double*, 8> r_re_ptr{};
    std::array<const double*, 8> r_im_ptr{};
    std::array<const double*, 8> k_split_perm_ptr{};

    for (int i = 0; i < 8; ++i)
    {
        k_re_host[i].assign(M_physical_, 0.0);
        k_im_host[i].assign(M_physical_, 0.0);
        k_re_ptr[i] = k_re_host[i].data();
        k_im_ptr[i] = k_im_host[i].data();
        r_re_ptr[i] = twiddles_->re[i].data();
        r_im_ptr[i] = twiddles_->im[i].data();
        // Twiddle r_p pairs with S_{fold_perm_[p]} (translation-parity
        // permutation; identity when all generator translations are even in
        // grid units).
        k_split_perm_ptr[i] = k_split[fold_perm_[i]].data();
    }

    constexpr std::array<std::pair<int, int>, 4> seq3{{{0,7},{6,1},{2,5},{4,3}}};
    mul_add_sub_by_seq(k_re_ptr, k_im_ptr, r_re_ptr, r_im_ptr, k_split_perm_ptr, seq3, halfsize);

    // `cache` holds the buffers of a cache dropped by a cell change when one is
    // left; allocating and copying on the default stream would stall the other
    // streams. On failure the local cache releases its tables
    for (int i = 0; i < 8; ++i)
    {
        if (cache.re[i].empty())
            cache.re[i].allocate(M_physical_);
        if (cache.im[i].empty())
            cache.im[i].allocate(M_physical_);
        CUDA_CHECK(cudaMemcpyAsync(cache.re[i], k_re_host[i].data(), sizeof(double) * M_physical_,
                                        cudaMemcpyHostToDevice, stream_));
        CUDA_CHECK(cudaMemcpyAsync(cache.im[i], k_im_host[i].data(), sizeof(double) * M_physical_,
                                        cudaMemcpyHostToDevice, stream_));
    }
    // Complete before the host tables go away and before any stream uses the cache
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    return cache;
}

void CudaCrysFFTRecursive3m::set_cell_para(const std::array<double, 6>& cell_para)
{
    // Changes the cell of every object sharing the caches (no-op for the current cell)
    tables_.shared()->set_cell(cell_para);
    if (tables_.refresh())
        k_current_ = nullptr;
}

void CudaCrysFFTRecursive3m::set_contour_step(double coeff)
{
    if (tables_.refresh())   // another object changed the cell
        k_current_ = nullptr;
    if (k_current_ && coeff == coeff_current_)
        return;

    k_current_ = &tables_.get({kBoltzmann, 0, coeff},
        [&](const std::array<double, 6>& cell, KCacheDevice cache) {
            return generate_k_cache(coeff, cell, std::move(cache));
        });
    coeff_current_ = coeff;
}

void CudaCrysFFTRecursive3m::diffusion(double* d_q_in, double* d_q_out)
{
    diffusion(d_q_in, d_q_out, stream_);
}

void CudaCrysFFTRecursive3m::diffusion(double* d_q_in, double* d_q_out, cudaStream_t stream)
{
    if (tables_.refresh())   // another object changed the cell
        k_current_ = nullptr;
    if (!k_current_)
        throw_with_line_number("CudaCrysFFTRecursive3m::set_contour_step must be called before diffusion() and after a cell change.");

    apply_with_cache(*k_current_, d_q_in, d_q_out, stream);
}

void CudaCrysFFTRecursive3m::apply_with_cache(
    const KCacheDevice& cache, double* d_q_in, double* d_q_out, cudaStream_t stream)
{
    set_stream(stream);

    double* d_work = (d_q_in == d_q_out) ? d_q_in : d_q_out;
    if (d_q_in != d_q_out)
    {
        CUDA_CHECK(cudaMemcpyAsync(d_q_out, d_q_in, sizeof(double) * M_physical_, cudaMemcpyDeviceToDevice, stream_));
    }

    CUFFT_CHECK(cufftExecD2Z(plan_r2c_, d_work, d_step1_));

    const int threads = 256;
    const int blocks = grid_blocks(M_complex_, threads);

    ker_apply_k_3m<<<blocks, threads, 0, stream_>>>(
        d_step2_, d_step1_,
        cache.re[0], cache.re[1], cache.re[2], cache.re[3],
        cache.re[4], cache.re[5], cache.re[6], cache.re[7],
        cache.im[0], cache.im[1], cache.im[2], cache.im[3],
        cache.im[4], cache.im[5], cache.im[6], cache.im[7],
        nx_physical_[0], nx_physical_[1], nx_physical_[2], nx_physical_[2] / 2 + 1);
    CUDA_CHECK_LAUNCH();

    CUFFT_CHECK(cufftExecZ2D(plan_c2r_, d_step2_, d_work));
}

const CudaCrysFFTRecursive3m::KCacheDevice& CudaCrysFFTRecursive3m::multiplier_cache(MultiplierType type, double coeff)
{
    // Types without the bond factor do not depend on coeff; normalize the cache key
    const double key_coeff = multiplier_has_bond(type) ? coeff : 0.0;
    return tables_.get({kMultiplier, static_cast<int>(type), key_coeff},
        [&](const std::array<double, 6>& cell, KCacheDevice cache) {
            return generate_k_cache_from_multiplier(type, coeff, cell, std::move(cache));
        });
}

void CudaCrysFFTRecursive3m::apply_multiplier(
    double* d_q_in, double* d_q_out, MultiplierType type, double coeff, cudaStream_t stream)
{
    apply_with_cache(multiplier_cache(type, coeff), d_q_in, d_q_out, stream);
}

void CudaCrysFFTRecursive3m::prepare_multiplier(MultiplierType type, double coeff)
{
    multiplier_cache(type, coeff);
}

void CudaCrysFFTRecursive3m::set_stream(cudaStream_t stream)
{
    if (stream_ == stream)
        return;
    stream_ = stream;
    CUFFT_CHECK(cufftSetStream(plan_r2c_, stream_));
    CUFFT_CHECK(cufftSetStream(plan_c2r_, stream_));
}
