/**
 * @file CrysFFTRecursive3mBase.h
 * @brief CRTP base class for recursive crystallographic FFT (2x2y2z).
 *
 * Implements the recursive 2x2y2z algorithm described in
 * "Accelerated pseudo-spectral method of self-consistent field theory via
 * crystallographic fast Fourier transform", Macromolecules (2020).
 *
 * @see FftwCrysFFTRecursive3m for FFTW implementation
 * @see MklCrysFFTRecursive3m for MKL implementation
 */

#ifndef CRYS_FFT_RECURSIVE_3M_BASE_H_
#define CRYS_FFT_RECURSIVE_3M_BASE_H_

#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "Exception.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

//==============================================================================
// Common helper functions (shared between FFTW and MKL)
//==============================================================================
namespace crys_recursive_3m {

inline std::vector<std::array<double, 3>> generate_m3_operations(
    const std::array<double, 9>& g, const std::array<int, 3>& s)
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

inline void twiddle_factor(
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

inline void mat_split(
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

inline void add_sub_by_seq(
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

inline void mul_add_sub_by_seq(
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

} // namespace crys_recursive_3m

//==============================================================================
// CRTP base class
//==============================================================================

/**
 * @class CrysFFTRecursive3mBase
 * @brief CRTP base class for recursive crystallographic FFT.
 *
 * Provides common functionality for the 2x2y2z recursive algorithm.
 * Derived classes must implement:
 * - init_fft_plans(): Create platform-specific FFT plans
 * - apply_with_cache(): Execute FFT with cached Boltzmann factors
 *
 * @tparam Derived The derived class type (CRTP pattern)
 */
template <typename Derived>
class CrysFFTRecursive3mBase
{
public:
    enum class MultiplierType
    {
        Kx2,
        Ky2,
        Kz2,
        ExpKx2,
        ExpKy2,
        ExpKz2
    };

    struct KCache {
        std::array<std::unique_ptr<double[]>, 8> re;
        std::array<std::unique_ptr<double[]>, 8> im;
    };

protected:
    std::array<int, 3> nx_logical_;
    std::array<int, 3> nx_physical_;
    int M_logical_;
    int M_physical_;
    std::array<double, 6> cell_para_;
    std::array<double, 9> translational_part_;

    std::array<std::unique_ptr<double[]>, 8> r_re_;
    std::array<std::unique_ptr<double[]>, 8> r_im_;

    struct ThreadState {
        std::map<double, KCache> k_cache;
        const KCache* k_current = nullptr;
        double coeff_current = std::numeric_limits<double>::quiet_NaN();
        std::map<double, KCache> exp_kx2_cache;
        std::map<double, KCache> exp_ky2_cache;
        std::map<double, KCache> exp_kz2_cache;
        std::unique_ptr<KCache> kx2_cache;
        std::unique_ptr<KCache> ky2_cache;
        std::unique_ptr<KCache> kz2_cache;
        uint64_t epoch = 0;
        uint64_t instance_id = 0;
    };

    inline static std::atomic<uint64_t> next_instance_id_{1};
    uint64_t instance_id_{0};
    mutable std::atomic<uint64_t> cache_epoch_{1};

    Derived& derived() { return static_cast<Derived&>(*this); }
    const Derived& derived() const { return static_cast<const Derived&>(*this); }

    ThreadState& get_thread_state() const
    {
        struct ThreadLocalStates {
            std::unordered_map<const CrysFFTRecursive3mBase*, ThreadState> states;
        };
        thread_local ThreadLocalStates tls;

        ThreadState& state = tls.states[this];
        uint64_t epoch = cache_epoch_.load(std::memory_order_acquire);
        if (state.instance_id != instance_id_ || state.epoch != epoch)
        {
            state.k_cache.clear();
            state.k_current = nullptr;
            state.coeff_current = std::numeric_limits<double>::quiet_NaN();
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

    void generate_twiddle_factors()
    {
        using namespace crys_recursive_3m;

        for (int i = 0; i < 8; ++i)
        {
            r_re_[i] = std::make_unique<double[]>(M_physical_);
            r_im_[i] = std::make_unique<double[]>(M_physical_);
        }

        // First twiddle factor is identity
        std::fill_n(r_re_[0].get(), M_physical_, 1.0);
        std::fill_n(r_im_[0].get(), M_physical_, 0.0);

        // Generate remaining twiddle factors
        auto ops = generate_m3_operations(translational_part_, nx_logical_);
        std::vector<std::tuple<double*, double*, std::array<double, 3>>> tasks;
        for (size_t i = 0; i < ops.size(); ++i)
        {
            tasks.emplace_back(r_re_[i + 1].get(), r_im_[i + 1].get(), ops[i]);
        }
        twiddle_factor(tasks, nx_physical_);
    }

    KCache generate_k_cache(double coeff) const
    {
        using namespace crys_recursive_3m;

        KCache cache;
        for (int i = 0; i < 8; ++i)
        {
            cache.re[i] = std::make_unique<double[]>(M_physical_);
            cache.im[i] = std::make_unique<double[]>(M_physical_);
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

        // Build full-size tempmat with exp(-k^2 * coeff)
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

        // Split into 8 octants
        std::array<std::unique_ptr<double[]>, 8> k_split;
        for (int i = 0; i < 8; ++i)
        {
            k_split[i] = std::make_unique<double[]>(M_physical_);
        }

        std::array<double*, 8> k_split_ptr;
        for (int i = 0; i < 8; ++i) k_split_ptr[i] = k_split[i].get();

        mat_split(tempmat.data(), k_split_ptr, nx_logical_);

        // Apply 2x2y2z decomposition
        std::array<int, 3> halfsize = nx_physical_;
        constexpr std::array<std::pair<int, int>, 4> seq0{{{0,4},{1,5},{2,6},{3,7}}};
        constexpr std::array<std::pair<int, int>, 4> seq1{{{0,2},{1,3},{4,6},{5,7}}};
        constexpr std::array<std::pair<int, int>, 4> seq2{{{0,1},{2,3},{4,5},{6,7}}};
        add_sub_by_seq(k_split_ptr, seq0, halfsize);
        add_sub_by_seq(k_split_ptr, seq1, halfsize);
        add_sub_by_seq(k_split_ptr, seq2, halfsize);

        // Multiply with twiddle factors
        constexpr std::array<std::pair<int, int>, 4> seq3{{{0,7},{6,1},{2,5},{4,3}}};
        std::array<double*, 8> cache_re_ptr, cache_im_ptr, r_re_ptr, r_im_ptr;
        for (int i = 0; i < 8; ++i)
        {
            cache_re_ptr[i] = cache.re[i].get();
            cache_im_ptr[i] = cache.im[i].get();
            r_re_ptr[i] = r_re_[i].get();
            r_im_ptr[i] = r_im_[i].get();
        }

        mul_add_sub_by_seq(
            cache_re_ptr, cache_im_ptr,
            r_re_ptr, r_im_ptr,
            k_split_ptr,
            seq3, halfsize);

        return cache;
    }

    KCache generate_k_cache_from_multiplier(MultiplierType type, double coeff) const
    {
        using namespace crys_recursive_3m;

        KCache cache;
        for (int i = 0; i < 8; ++i)
        {
            cache.re[i] = std::make_unique<double[]>(M_physical_);
            cache.im[i] = std::make_unique<double[]>(M_physical_);
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

        // Split into 8 octants
        std::array<std::unique_ptr<double[]>, 8> k_split;
        for (int i = 0; i < 8; ++i)
        {
            k_split[i] = std::make_unique<double[]>(M_physical_);
        }

        std::array<double*, 8> k_split_ptr;
        for (int i = 0; i < 8; ++i) k_split_ptr[i] = k_split[i].get();

        mat_split(tempmat.data(), k_split_ptr, nx_logical_);

        // Apply 2x2y2z decomposition
        std::array<int, 3> halfsize = nx_physical_;
        constexpr std::array<std::pair<int, int>, 4> seq0{{{0,4},{1,5},{2,6},{3,7}}};
        constexpr std::array<std::pair<int, int>, 4> seq1{{{0,2},{1,3},{4,6},{5,7}}};
        constexpr std::array<std::pair<int, int>, 4> seq2{{{0,1},{2,3},{4,5},{6,7}}};
        add_sub_by_seq(k_split_ptr, seq0, halfsize);
        add_sub_by_seq(k_split_ptr, seq1, halfsize);
        add_sub_by_seq(k_split_ptr, seq2, halfsize);

        // Multiply with twiddle factors
        constexpr std::array<std::pair<int, int>, 4> seq3{{{0,7},{6,1},{2,5},{4,3}}};
        std::array<double*, 8> cache_re_ptr, cache_im_ptr, r_re_ptr, r_im_ptr;
        for (int i = 0; i < 8; ++i)
        {
            cache_re_ptr[i] = cache.re[i].get();
            cache_im_ptr[i] = cache.im[i].get();
            r_re_ptr[i] = r_re_[i].get();
            r_im_ptr[i] = r_im_[i].get();
        }

        mul_add_sub_by_seq(
            cache_re_ptr, cache_im_ptr,
            r_re_ptr, r_im_ptr,
            k_split_ptr,
            seq3, halfsize);

        return cache;
    }

    const KCache& get_multiplier_cache(MultiplierType type, double coeff) const
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
                throw_with_line_number("Unsupported multiplier type.");
        }
    }

public:
    CrysFFTRecursive3mBase(
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
            if (nx_logical_[d] % 2 != 0)
            {
                throw_with_line_number("CrysFFTRecursive3m requires even grid dimensions.");
            }
            if (nx_logical_[d] <= 0)
            {
                throw_with_line_number("CrysFFTRecursive3m requires positive grid dimensions.");
            }
        }

        nx_physical_ = {
            nx_logical_[0] / 2,
            nx_logical_[1] / 2,
            nx_logical_[2] / 2
        };

        M_logical_ = nx_logical_[0] * nx_logical_[1] * nx_logical_[2];
        M_physical_ = nx_physical_[0] * nx_physical_[1] * nx_physical_[2];
    }

    virtual ~CrysFFTRecursive3mBase() = default;

    void set_cell_para(const std::array<double, 6>& cell_para)
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

    void set_contour_step(double coeff)
    {
        ThreadState& state = get_thread_state();
        if (state.k_current != nullptr && state.coeff_current == coeff)
            return;

        auto it = state.k_cache.find(coeff);
        if (it == state.k_cache.end())
        {
            it = state.k_cache.emplace(coeff, generate_k_cache(coeff)).first;
        }

        state.coeff_current = coeff;
        state.k_current = &it->second;
    }

    void diffusion(const double* q_in, double* q_out) const
    {
        ThreadState& state = get_thread_state();
        if (!state.k_current)
            throw_with_line_number("set_contour_step must be called before diffusion().");

        derived().apply_with_cache(*state.k_current, q_in, q_out);
    }

    void apply_multiplier(const double* q_in, double* q_out, MultiplierType type, double coeff) const
    {
        const KCache& cache = get_multiplier_cache(type, coeff);
        derived().apply_with_cache(cache, q_in, q_out);
    }

    const std::array<int, 3>& get_nx_logical() const { return nx_logical_; }
    const std::array<int, 3>& get_nx_physical() const { return nx_physical_; }
    int get_M_logical() const { return M_logical_; }
    int get_M_physical() const { return M_physical_; }
};

#endif  // CRYS_FFT_RECURSIVE_3M_BASE_H_
