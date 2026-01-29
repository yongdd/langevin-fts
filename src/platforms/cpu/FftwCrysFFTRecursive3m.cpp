/**
 * @file FftwCrysFFTRecursive3m.cpp
 * @brief CPU recursive crystallographic FFT (2x2y2z) implementation.
 */

#include "FftwCrysFFTRecursive3m.h"
#include "Exception.h"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace {
constexpr size_t kAlignment = 64;

inline double* alloc_aligned_double(size_t count)
{
    return static_cast<double*>(fftw_malloc(sizeof(double) * count));
}

inline fftw_complex* alloc_aligned_complex(size_t count)
{
    return static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * count));
}

inline size_t align_up(size_t value, size_t alignment)
{
    return ((value + alignment - 1) / alignment) * alignment;
}

std::vector<std::array<double, 3>> generate_m3_operations(const std::array<double, 9>& g, const std::array<int, 3>& s)
{
    return {
        {-2. * M_PI * (g[0]),                               -2. * M_PI * (g[1]),                               -2. * M_PI * (g[2] - 1.0 / s[2])},
        {-2. * M_PI * (g[3]),                               -2. * M_PI * (g[4] - 1.0 / s[1]),                  -2. * M_PI * (g[5])},
        {-2. * M_PI * (g[0] + g[3]),                        -2. * M_PI * (-g[1] + g[4] - 1.0 / s[1]),          -2. * M_PI * (g[2] + g[5] - 1.0 / s[2])},
        {-2. * M_PI * (g[6] - 1.0 / s[0]),                  -2. * M_PI * (g[7]),                               -2. * M_PI * (g[8])},
        {-2. * M_PI * (g[6] - g[0] - 1.0 / s[0]),           -2. * M_PI * (g[1] + g[7]),                        -2. * M_PI * (g[2] + g[8] - 1.0 / s[2])},
        {-2. * M_PI * (-g[3] + g[6] - 1.0 / s[0]),          -2. * M_PI * (g[4] + g[7] - 1.0 / s[1]),           -2. * M_PI * (g[5] + g[8])},
        {-2. * M_PI * (-g[0] - g[3] + g[6] - 1.0 / s[0]),   -2. * M_PI * (-g[1] + g[4] + g[7] - 1.0 / s[1]),   -2. * M_PI * (g[2] + g[5] + g[8] - 1.0 / s[2])}
    };
}

void twiddle_factor(
    const std::vector<std::tuple<double*, double*, std::array<double, 3>>>& data,
    const std::array<int, 3>& range)
{
    const int Nx = range[0];
    const int Ny = range[1];
    const int Nz = range[2];

    for (size_t itr_task = 0; itr_task < data.size(); ++itr_task)
    {
        auto dstcos = std::get<0>(data[itr_task]);
        auto dstsin = std::get<1>(data[itr_task]);
        auto paras = std::get<2>(data[itr_task]);
        for (int ix = 0; ix < Nx; ++ix)
        {
            for (int iy = 0; iy < Ny; ++iy)
            {
                size_t base = static_cast<size_t>(ix * Ny + iy) * Nz;
                for (int iz = 0; iz < Nz; ++iz)
                {
                    double angle = ix * paras[0] + iy * paras[1] + iz * paras[2];
                    dstcos[base + iz] = std::cos(angle);
                    dstsin[base + iz] = std::sin(angle);
                }
            }
        }
    }
}

void mat_split(
    const double* src,
    const std::array<double*, 8>& dst,
    const std::array<int, 3>& size)
{
    const int Nx = size[0];
    const int Ny = size[1];
    const int Nz = size[2];

    const int Nx2 = Nx / 2;
    const int Ny2 = Ny / 2;
    const int Nz2 = Nz / 2;

    for (int part = 0; part < 8; ++part)
    {
        const double* local_src = src
            + ((part & 4) >> 2) * Nx2 * Ny * Nz
            + ((part & 2) >> 1) * Ny2 * Nz
            + (part & 1) * Nz2;
        double* local_dst = dst[part];

        for (int ix = 0; ix < Nx2; ++ix)
        {
            for (int iy = 0; iy < Ny2; ++iy)
            {
                const double* src_row = local_src + (ix * Ny + iy) * Nz;
                double* dst_row = local_dst + (ix * Ny2 + iy) * Nz2;
                std::memcpy(dst_row, src_row, sizeof(double) * Nz2);
            }
        }
    }
}

void add_sub_by_seq(
    const std::array<double*, 8>& dst,
    const std::array<std::pair<int, int>, 4>& seq,
    const std::array<int, 3>& halfsize)
{
    const int Nx = halfsize[0];
    const int Ny = halfsize[1];
    const int Nz = halfsize[2];
    const int NxNy = Nx * Ny;

    for (size_t itr_seq = 0; itr_seq < seq.size(); ++itr_seq)
    {
        double* dst0 = dst[seq[itr_seq].first];
        double* dst1 = dst[seq[itr_seq].second];
        for (int itr_xy = 0; itr_xy < NxNy; ++itr_xy)
        {
            size_t base = static_cast<size_t>(itr_xy) * Nz;
            for (int itr_z = 0; itr_z < Nz; ++itr_z)
            {
                double temp = dst0[base + itr_z] + dst1[base + itr_z];
                dst1[base + itr_z] = dst0[base + itr_z] - dst1[base + itr_z];
                dst0[base + itr_z] = temp;
            }
        }
    }
}

void mul_add_sub_by_seq(
    const std::array<double*, 8>& dst_re,
    const std::array<double*, 8>& dst_im,
    const std::array<double*, 8>& src_re,
    const std::array<double*, 8>& src_im,
    const std::array<double*, 8>& src_mul,
    const std::array<std::pair<int, int>, 4>& seq,
    const std::array<int, 3>& halfsize)
{
    const int Nx = halfsize[0];
    const int Ny = halfsize[1];
    const int Nz = halfsize[2];
    const int NxNy = Nx * Ny;

    for (size_t itr_seq = 0; itr_seq < seq.size(); ++itr_seq)
    {
        int i0 = seq[itr_seq].first;
        int i1 = seq[itr_seq].second;
        double* dst_re0 = dst_re[i0];
        double* dst_re1 = dst_re[i1];
        double* dst_im0 = dst_im[i0];
        double* dst_im1 = dst_im[i1];
        const double* src_re0 = src_re[i0];
        const double* src_re1 = src_re[i1];
        const double* src_im0 = src_im[i0];
        const double* src_im1 = src_im[i1];
        const double* mul0 = src_mul[i0];
        const double* mul1 = src_mul[i1];

        for (int itr_xy = 0; itr_xy < NxNy; ++itr_xy)
        {
            size_t base = static_cast<size_t>(itr_xy) * Nz;
            for (int itr_z = 0; itr_z < Nz; ++itr_z)
            {
                size_t idx = base + itr_z;
                dst_re0[idx] = mul0[idx] * src_re0[idx];
                dst_im0[idx] = mul0[idx] * src_im0[idx];
                dst_re1[idx] = mul1[idx] * src_re1[idx];
                dst_im1[idx] = mul1[idx] * src_im1[idx];

                double tempre = dst_re0[idx] + dst_re1[idx];
                dst_re1[idx] = dst_re0[idx] - dst_re1[idx];
                dst_re0[idx] = tempre;

                double tempim = dst_im0[idx] + dst_im1[idx];
                dst_im1[idx] = dst_im0[idx] - dst_im1[idx];
                dst_im0[idx] = tempim;
            }
        }
    }
}

}  // namespace

FftwCrysFFTRecursive3m::FftwCrysFFTRecursive3m(
    std::array<int, 3> nx_logical,
    std::array<double, 6> cell_para,
    std::array<double, 9> translational_part)
    : nx_logical_(nx_logical),
      cell_para_(cell_para),
      translational_part_(translational_part),
      instance_id_(next_instance_id_.fetch_add(1, std::memory_order_relaxed))
{
    for (int d = 0; d < 3; ++d)
    {
        if (nx_logical_[d] <= 0 || (nx_logical_[d] % 2) != 0)
        {
            throw_with_line_number("FftwCrysFFTRecursive3m requires even, positive grid dimensions.");
        }
    }
    if ((nx_logical_[2] / 2) % (kAlignment / 8) != 0)
    {
        throw_with_line_number("FftwCrysFFTRecursive3m requires Nz/2 to be aligned to 64-byte SIMD lanes.");
    }

    nx_physical_ = { nx_logical_[0] / 2, nx_logical_[1] / 2, nx_logical_[2] / 2 };
    M_logical_ = nx_logical_[0] * nx_logical_[1] * nx_logical_[2];
    M_physical_ = nx_physical_[0] * nx_physical_[1] * nx_physical_[2];

    init_fft_plans();
    generate_twiddle_factors();

}

FftwCrysFFTRecursive3m::~FftwCrysFFTRecursive3m()
{
    if (plan_forward_) fftw_destroy_plan(plan_forward_);
    if (plan_backward_) fftw_destroy_plan(plan_backward_);
}

void FftwCrysFFTRecursive3m::set_cell_para(const std::array<double, 6>& cell_para)
{
    if (cell_para[0] == cell_para_[0] &&
        cell_para[1] == cell_para_[1] &&
        cell_para[2] == cell_para_[2])
    {
        return;
    }

    cell_para_ = cell_para;
    cache_epoch_.fetch_add(1, std::memory_order_acq_rel);
}

void FftwCrysFFTRecursive3m::set_contour_step(double coeff)
{
    ThreadState& state = get_thread_state();
    if (state.k_current != nullptr && state.coeff_current == coeff)
        return;

    const KCache* cache = nullptr;
    auto it = state.k_cache.find(coeff);
    if (it == state.k_cache.end())
    {
        auto inserted = state.k_cache.emplace(coeff, generate_k_cache(coeff));
        it = inserted.first;
    }
    cache = &it->second;
    state.coeff_current = coeff;
    state.k_current = cache;
}

void FftwCrysFFTRecursive3m::diffusion(const double* q_in, double* q_out) const
{
    ThreadState& state = get_thread_state();
    if (!state.k_current)
        throw_with_line_number("FftwCrysFFTRecursive3m::set_contour_step must be called before diffusion().");

    apply_with_cache(*state.k_current, q_in, q_out);
}

FftwCrysFFTRecursive3m::ThreadState& FftwCrysFFTRecursive3m::get_thread_state() const
{
    struct ThreadLocalStates
    {
        std::unordered_map<const FftwCrysFFTRecursive3m*, ThreadState> states;
    };
    thread_local ThreadLocalStates tls;

    ThreadState& state = tls.states[this];
    const uint64_t epoch = cache_epoch_.load(std::memory_order_acquire);
    if (state.instance_id != instance_id_ || state.epoch != epoch)
    {
        state.k_current = nullptr;
        state.coeff_current = std::numeric_limits<double>::quiet_NaN();
        state.k_cache.clear();
        state.exp_kx2_cache.clear();
        state.exp_ky2_cache.clear();
        state.exp_kz2_cache.clear();
        state.kx2_cache.reset();
        state.ky2_cache.reset();
        state.kz2_cache.reset();
        state.epoch = epoch;
        state.instance_id = instance_id_;
    }
    return state;
}

void FftwCrysFFTRecursive3m::apply_multiplier(
    const double* q_in, double* q_out, MultiplierType type, double coeff) const
{
    const KCache& cache = get_multiplier_cache(type, coeff);
    apply_with_cache(cache, q_in, q_out);
}

void FftwCrysFFTRecursive3m::apply_with_cache(
    const KCache& cache, const double* q_in, double* q_out) const
{
    struct ThreadBuffers {
        std::vector<double> iomat;
        std::unique_ptr<fftw_complex, AlignedDeleter> step1;
        std::unique_ptr<fftw_complex, AlignedDeleter> step2;
        int size = 0;
    };

    thread_local ThreadBuffers tls_buffers;
    ThreadBuffers& buffers = tls_buffers;
    if (buffers.size != M_physical_)
    {
        buffers.iomat.resize(M_physical_);
        buffers.step1.reset(alloc_aligned_complex(M_physical_));
        buffers.step2.reset(alloc_aligned_complex(M_physical_));
        if (!buffers.step1 || !buffers.step2)
        {
            throw_with_line_number("Failed to allocate FFTW work buffers for FftwCrysFFTRecursive3m.");
        }
        buffers.size = M_physical_;
    }

    std::memcpy(buffers.iomat.data(), q_in, sizeof(double) * M_physical_);

    fftw_execute_dft_r2c(plan_forward_, buffers.iomat.data(), buffers.step1.get());

    const int Nx2 = nx_physical_[0];
    const int Ny2 = nx_physical_[1];
    const int Nz2 = nx_physical_[2];
    const size_t Nz_qua = align_up(static_cast<size_t>(Nz2 / 2 + 1), kAlignment / 8);

    const auto& k_re = cache.re;
    const auto& k_im = cache.im;

    for (int ix = 0; ix < Nx2; ++ix)
    {
        for (int iy = 0; iy < Ny2; ++iy)
        {
            const size_t base_shift = static_cast<size_t>(ix * Ny2 + iy) * Nz2;
            const fftw_complex* src000 = buffers.step1.get() + base_shift;
            const fftw_complex* src110 = buffers.step1.get() + (static_cast<size_t>((Nx2 - ix) % Nx2) * Ny2 + (Ny2 - iy) % Ny2) * Nz2;
            const fftw_complex* src010 = buffers.step1.get() + (static_cast<size_t>(ix) * Ny2 + (Ny2 - iy) % Ny2) * Nz2;
            const fftw_complex* src100 = buffers.step1.get() + (static_cast<size_t>((Nx2 - ix) % Nx2) * Ny2 + iy) * Nz2;
            fftw_complex* dst = buffers.step2.get() + base_shift;

            for (size_t iz = 0; iz < Nz_qua; ++iz)
            {
                size_t idx = base_shift + iz;
                dst[iz][0]
                    = src000[iz][0] * k_re[0].get()[idx] - src000[iz][1] * k_im[7].get()[idx]
                    + src110[iz][0] * k_re[6].get()[idx] - src110[iz][1] * k_im[1].get()[idx]
                    + src010[iz][0] * k_re[2].get()[idx] - src010[iz][1] * k_im[5].get()[idx]
                    + src100[iz][0] * k_re[4].get()[idx] - src100[iz][1] * k_im[3].get()[idx];

                dst[iz][1]
                    = src000[iz][0] * k_im[0].get()[idx] + src000[iz][1] * k_re[7].get()[idx]
                    + src110[iz][0] * k_im[6].get()[idx] + src110[iz][1] * k_re[1].get()[idx]
                    + src010[iz][0] * k_im[2].get()[idx] + src010[iz][1] * k_re[5].get()[idx]
                    + src100[iz][0] * k_im[4].get()[idx] + src100[iz][1] * k_re[3].get()[idx];
            }
        }
    }

    fftw_execute_dft_c2r(plan_backward_, buffers.step2.get(), buffers.iomat.data());

    std::memcpy(q_out, buffers.iomat.data(), sizeof(double) * M_physical_);
}

const FftwCrysFFTRecursive3m::KCache& FftwCrysFFTRecursive3m::get_multiplier_cache(
    MultiplierType type, double coeff) const
{
    ThreadState& state = get_thread_state();
    switch (type)
    {
        case MultiplierType::Kx2:
            if (!state.kx2_cache)
                state.kx2_cache = std::make_unique<KCache>(generate_k_cache_from_multiplier(type, 0.0));
            return *state.kx2_cache;
        case MultiplierType::Ky2:
            if (!state.ky2_cache)
                state.ky2_cache = std::make_unique<KCache>(generate_k_cache_from_multiplier(type, 0.0));
            return *state.ky2_cache;
        case MultiplierType::Kz2:
            if (!state.kz2_cache)
                state.kz2_cache = std::make_unique<KCache>(generate_k_cache_from_multiplier(type, 0.0));
            return *state.kz2_cache;
        case MultiplierType::ExpKx2:
        {
            auto it = state.exp_kx2_cache.find(coeff);
            if (it == state.exp_kx2_cache.end())
                it = state.exp_kx2_cache.emplace(coeff, generate_k_cache_from_multiplier(type, coeff)).first;
            return it->second;
        }
        case MultiplierType::ExpKy2:
        {
            auto it = state.exp_ky2_cache.find(coeff);
            if (it == state.exp_ky2_cache.end())
                it = state.exp_ky2_cache.emplace(coeff, generate_k_cache_from_multiplier(type, coeff)).first;
            return it->second;
        }
        case MultiplierType::ExpKz2:
        {
            auto it = state.exp_kz2_cache.find(coeff);
            if (it == state.exp_kz2_cache.end())
                it = state.exp_kz2_cache.emplace(coeff, generate_k_cache_from_multiplier(type, coeff)).first;
            return it->second;
        }
        default:
            throw_with_line_number("Unsupported multiplier type for FftwCrysFFTRecursive3m.");
    }
}

void FftwCrysFFTRecursive3m::init_fft_plans()
{
    std::unique_ptr<double, AlignedDeleter> dummy_in(alloc_aligned_double(M_physical_));
    std::unique_ptr<fftw_complex, AlignedDeleter> dummy_out(alloc_aligned_complex(M_physical_));

    fftw_iodim dims[3];
    dims[0].n = nx_physical_[0];
    dims[0].is = nx_physical_[1] * nx_physical_[2];
    dims[0].os = nx_physical_[1] * nx_physical_[2];
    dims[1].n = nx_physical_[1];
    dims[1].is = nx_physical_[2];
    dims[1].os = nx_physical_[2];
    dims[2].n = nx_physical_[2];
    dims[2].is = 1;
    dims[2].os = 1;

    plan_forward_ = fftw_plan_guru_dft_r2c(
        3, dims, 0, nullptr, dummy_in.get(), dummy_out.get(), FFTW_PATIENT);
    plan_backward_ = fftw_plan_guru_dft_c2r(
        3, dims, 0, nullptr, dummy_out.get(), dummy_in.get(), FFTW_PATIENT);

    if (!plan_forward_ || !plan_backward_)
    {
        throw_with_line_number("Failed to create FFTW plans for FftwCrysFFTRecursive3m.");
    }
}

void FftwCrysFFTRecursive3m::generate_twiddle_factors()
{
    std::array<int, 3> size = nx_logical_;
    std::array<int, 3> halfsize = nx_physical_;

    for (int i = 0; i < 8; ++i)
    {
        r_re_[i].reset(alloc_aligned_double(M_physical_));
        r_im_[i].reset(alloc_aligned_double(M_physical_));
        if (!r_re_[i] || !r_im_[i])
        {
            throw_with_line_number("Failed to allocate twiddle factor buffers.");
        }
    }

    std::vector<std::tuple<double*, double*, std::array<double, 3>>> twiddle;
    auto operations = generate_m3_operations(translational_part_, size);
    for (size_t i = 0; i < operations.size(); ++i)
    {
        twiddle.emplace_back(r_re_[i + 1].get(), r_im_[i + 1].get(), operations[i]);
    }

    std::fill_n(r_re_[0].get(), M_physical_, 1.0);
    std::fill_n(r_im_[0].get(), M_physical_, 0.0);

    twiddle_factor(twiddle, halfsize);
}

FftwCrysFFTRecursive3m::KCache FftwCrysFFTRecursive3m::generate_k_cache(double coeff) const
{
    KCache cache;
    for (int i = 0; i < 8; ++i)
    {
        cache.re[i].reset(alloc_aligned_double(M_physical_));
        cache.im[i].reset(alloc_aligned_double(M_physical_));
        if (!cache.re[i] || !cache.im[i])
        {
            throw_with_line_number("Failed to allocate K matrix buffers.");
        }
    }

    const double Lx = cell_para_[0];
    const double Ly = cell_para_[1];
    const double Lz = cell_para_[2];

    std::vector<double> kx(nx_logical_[0]);
    std::vector<double> ky(nx_logical_[1]);
    std::vector<double> kz(nx_logical_[2]);

    double factor_Lx = 2.0 * M_PI / Lx;
    factor_Lx *= factor_Lx;
    double factor_Ly = 2.0 * M_PI / Ly;
    factor_Ly *= factor_Ly;
    double factor_Lz = 2.0 * M_PI / Lz;
    factor_Lz *= factor_Lz;

    for (int ix = 0; ix < nx_logical_[0]; ++ix)
    {
        int temp = (ix > nx_logical_[0] / 2) ? (nx_logical_[0] - ix) : ix;
        kx[ix] = temp * temp * factor_Lx;
    }
    for (int iy = 0; iy < nx_logical_[1]; ++iy)
    {
        int temp = (iy > nx_logical_[1] / 2) ? (nx_logical_[1] - iy) : iy;
        ky[iy] = temp * temp * factor_Ly;
    }
    for (int iz = 0; iz < nx_logical_[2]; ++iz)
    {
        int temp = (iz > nx_logical_[2] / 2) ? (nx_logical_[2] - iz) : iz;
        kz[iz] = temp * temp * factor_Lz;
    }

    std::vector<double> tempmat(static_cast<size_t>(M_logical_));
    double factor = 1.0 / static_cast<double>(M_logical_);

    for (int ix = 0; ix < nx_logical_[0]; ++ix)
    {
        for (int iy = 0; iy < nx_logical_[1]; ++iy)
        {
            size_t base = static_cast<size_t>(ix * nx_logical_[1] + iy) * nx_logical_[2];
            for (int iz = 0; iz < nx_logical_[2]; ++iz)
            {
                tempmat[base + iz] = std::exp(-(kx[ix] + ky[iy] + kz[iz]) * coeff) * factor;
            }
        }
    }

    std::array<double*, 8> k_split;
    for (int i = 0; i < 8; ++i)
    {
        k_split[i] = alloc_aligned_double(M_physical_);
        if (!k_split[i])
        {
            throw_with_line_number("Failed to allocate K split buffers.");
        }
    }

    mat_split(tempmat.data(), k_split, nx_logical_);

    std::array<int, 3> halfsize = nx_physical_;
    constexpr std::array<std::pair<int, int>, 4> seq0{{{0,4},{1,5},{2,6},{3,7}}};
    constexpr std::array<std::pair<int, int>, 4> seq1{{{0,2},{1,3},{4,6},{5,7}}};
    constexpr std::array<std::pair<int, int>, 4> seq2{{{0,1},{2,3},{4,5},{6,7}}};
    add_sub_by_seq(k_split, seq0, halfsize);
    add_sub_by_seq(k_split, seq1, halfsize);
    add_sub_by_seq(k_split, seq2, halfsize);

    constexpr std::array<std::pair<int, int>, 4> seq3{{{0,7},{6,1},{2,5},{4,3}}};
    mul_add_sub_by_seq(
        { cache.re[0].get(), cache.re[1].get(), cache.re[2].get(), cache.re[3].get(),
          cache.re[4].get(), cache.re[5].get(), cache.re[6].get(), cache.re[7].get() },
        { cache.im[0].get(), cache.im[1].get(), cache.im[2].get(), cache.im[3].get(),
          cache.im[4].get(), cache.im[5].get(), cache.im[6].get(), cache.im[7].get() },
        { r_re_[0].get(), r_re_[1].get(), r_re_[2].get(), r_re_[3].get(),
          r_re_[4].get(), r_re_[5].get(), r_re_[6].get(), r_re_[7].get() },
        { r_im_[0].get(), r_im_[1].get(), r_im_[2].get(), r_im_[3].get(),
          r_im_[4].get(), r_im_[5].get(), r_im_[6].get(), r_im_[7].get() },
        { k_split[0], k_split[1], k_split[2], k_split[3],
          k_split[4], k_split[5], k_split[6], k_split[7] },
        seq3,
        halfsize);

    for (int i = 0; i < 8; ++i)
    {
        fftw_free(k_split[i]);
    }

    return cache;
}

FftwCrysFFTRecursive3m::KCache FftwCrysFFTRecursive3m::generate_k_cache_from_multiplier(
    MultiplierType type, double coeff) const
{
    KCache cache;
    for (int i = 0; i < 8; ++i)
    {
        cache.re[i].reset(alloc_aligned_double(M_physical_));
        cache.im[i].reset(alloc_aligned_double(M_physical_));
        if (!cache.re[i] || !cache.im[i])
        {
            throw_with_line_number("Failed to allocate K matrix buffers.");
        }
    }

    const double Lx = cell_para_[0];
    const double Ly = cell_para_[1];
    const double Lz = cell_para_[2];

    std::vector<double> kx(nx_logical_[0]);
    std::vector<double> ky(nx_logical_[1]);
    std::vector<double> kz(nx_logical_[2]);

    double factor_Lx = 2.0 * M_PI / Lx;
    factor_Lx *= factor_Lx;
    double factor_Ly = 2.0 * M_PI / Ly;
    factor_Ly *= factor_Ly;
    double factor_Lz = 2.0 * M_PI / Lz;
    factor_Lz *= factor_Lz;

    for (int ix = 0; ix < nx_logical_[0]; ++ix)
    {
        int temp = (ix > nx_logical_[0] / 2) ? (nx_logical_[0] - ix) : ix;
        kx[ix] = temp * temp * factor_Lx;
    }
    for (int iy = 0; iy < nx_logical_[1]; ++iy)
    {
        int temp = (iy > nx_logical_[1] / 2) ? (nx_logical_[1] - iy) : iy;
        ky[iy] = temp * temp * factor_Ly;
    }
    for (int iz = 0; iz < nx_logical_[2]; ++iz)
    {
        int temp = (iz > nx_logical_[2] / 2) ? (nx_logical_[2] - iz) : iz;
        kz[iz] = temp * temp * factor_Lz;
    }

    std::vector<double> tempmat(static_cast<size_t>(M_logical_));
    double factor = 1.0 / static_cast<double>(M_logical_);

    for (int ix = 0; ix < nx_logical_[0]; ++ix)
    {
        for (int iy = 0; iy < nx_logical_[1]; ++iy)
        {
            size_t base = static_cast<size_t>(ix * nx_logical_[1] + iy) * nx_logical_[2];
            for (int iz = 0; iz < nx_logical_[2]; ++iz)
            {
                double kx2 = kx[ix];
                double ky2 = ky[iy];
                double kz2 = kz[iz];
                double k2 = kx2 + ky2 + kz2;
                double value = 0.0;

                switch (type)
                {
                    case MultiplierType::Kx2:
                        value = kx2;
                        break;
                    case MultiplierType::Ky2:
                        value = ky2;
                        break;
                    case MultiplierType::Kz2:
                        value = kz2;
                        break;
                    case MultiplierType::ExpKx2:
                        value = std::exp(-k2 * coeff) * kx2;
                        break;
                    case MultiplierType::ExpKy2:
                        value = std::exp(-k2 * coeff) * ky2;
                        break;
                    case MultiplierType::ExpKz2:
                        value = std::exp(-k2 * coeff) * kz2;
                        break;
                    default:
                        value = 0.0;
                        break;
                }

                tempmat[base + iz] = value * factor;
            }
        }
    }

    std::array<double*, 8> k_split;
    for (int i = 0; i < 8; ++i)
    {
        k_split[i] = alloc_aligned_double(M_physical_);
        if (!k_split[i])
        {
            throw_with_line_number("Failed to allocate K split buffers.");
        }
    }

    mat_split(tempmat.data(), k_split, nx_logical_);

    std::array<int, 3> halfsize = nx_physical_;
    constexpr std::array<std::pair<int, int>, 4> seq0{{{0,4},{1,5},{2,6},{3,7}}};
    constexpr std::array<std::pair<int, int>, 4> seq1{{{0,2},{1,3},{4,6},{5,7}}};
    constexpr std::array<std::pair<int, int>, 4> seq2{{{0,1},{2,3},{4,5},{6,7}}};
    add_sub_by_seq(k_split, seq0, halfsize);
    add_sub_by_seq(k_split, seq1, halfsize);
    add_sub_by_seq(k_split, seq2, halfsize);

    constexpr std::array<std::pair<int, int>, 4> seq3{{{0,7},{6,1},{2,5},{4,3}}};
    mul_add_sub_by_seq(
        { cache.re[0].get(), cache.re[1].get(), cache.re[2].get(), cache.re[3].get(),
          cache.re[4].get(), cache.re[5].get(), cache.re[6].get(), cache.re[7].get() },
        { cache.im[0].get(), cache.im[1].get(), cache.im[2].get(), cache.im[3].get(),
          cache.im[4].get(), cache.im[5].get(), cache.im[6].get(), cache.im[7].get() },
        { r_re_[0].get(), r_re_[1].get(), r_re_[2].get(), r_re_[3].get(),
          r_re_[4].get(), r_re_[5].get(), r_re_[6].get(), r_re_[7].get() },
        { r_im_[0].get(), r_im_[1].get(), r_im_[2].get(), r_im_[3].get(),
          r_im_[4].get(), r_im_[5].get(), r_im_[6].get(), r_im_[7].get() },
        { k_split[0], k_split[1], k_split[2], k_split[3],
          k_split[4], k_split[5], k_split[6], k_split[7] },
        seq3,
        halfsize);

    for (int i = 0; i < 8; ++i)
    {
        fftw_free(k_split[i]);
    }

    return cache;
}
