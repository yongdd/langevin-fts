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
#include <cmath>
#include <cstring>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include "CrysFFTSharedTables.h"
#include "Exception.h"
#include "Recursive3mFoldParity.h"
#include "Recursive3mHelpers.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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
    /// Stress multipliers (see Recursive3mHelpers.h)
    using MultiplierType = crys_recursive_3m::MultiplierType;

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

    /// Pairing permutation of the fold: twiddle r_p multiplies the
    /// octant-transformed Boltzmann array S_{fold_perm_[p]} (identity when
    /// all generator translations are an even number of grid cells).
    /// See Recursive3mFoldParity.h for the derivation.
    std::array<int, 8> fold_perm_{};

    std::array<std::unique_ptr<double[]>, 8> r_re_;
    std::array<std::unique_ptr<double[]>, 8> r_im_;

    /// Boltzmann factors per coefficient and stress multipliers per (type,
    /// coefficient), shared by the threads that call the engine
    SharedTableCache<double, KCache> k_cache_;
    SharedTableCache<std::pair<MultiplierType, double>, KCache> multiplier_cache_;

    Derived& derived() { return static_cast<Derived&>(*this); }
    const Derived& derived() const { return static_cast<const Derived&>(*this); }

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

    /**
     * @brief Build the folded 2x2y2z k-space cache for a per-mode value.
     *
     * Shared body of generate_k_cache() and generate_k_cache_from_multiplier().
     * Tabulates the squared wavevector components kx^2, ky^2, kz^2 of the
     * logical grid, fills the full grid with value_at(kx2, ky2, kz2) / M_logical,
     * splits it into 8 octants, applies the 2x2y2z butterflies and multiplies
     * with the (parity-permuted) twiddle factors.
     *
     * @tparam ValueFn Callable with signature double(double kx2, double ky2, double kz2).
     * @param value_at Per-mode value before the 1/M_logical normalization.
     */
    template <typename ValueFn>
    KCache generate_k_cache_impl(ValueFn&& value_at) const
    {
        using namespace crys_recursive_3m;

        KCache cache;
        for (int i = 0; i < 8; ++i)
        {
            cache.re[i] = std::make_unique<double[]>(M_physical_);
            cache.im[i] = std::make_unique<double[]>(M_physical_);
        }

        // value_at(kx^2, ky^2, kz^2) / M_logical on the logical grid
        std::vector<double> tempmat = logical_k_table(nx_logical_, cell_para_, value_at);

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

        // Multiply with twiddle factors. Twiddle r_p pairs with
        // S_{fold_perm_[p]} (translation-parity permutation; identity when all
        // generator translations are even in grid units).
        constexpr std::array<std::pair<int, int>, 4> seq3{{{0,7},{6,1},{2,5},{4,3}}};
        std::array<double*, 8> cache_re_ptr, cache_im_ptr;
        std::array<const double*, 8> r_re_ptr, r_im_ptr, k_split_perm_ptr;
        for (int i = 0; i < 8; ++i)
        {
            cache_re_ptr[i] = cache.re[i].get();
            cache_im_ptr[i] = cache.im[i].get();
            r_re_ptr[i] = r_re_[i].get();
            r_im_ptr[i] = r_im_[i].get();
            k_split_perm_ptr[i] = k_split[fold_perm_[i]].get();
        }

        mul_add_sub_by_seq(
            cache_re_ptr, cache_im_ptr,
            r_re_ptr, r_im_ptr,
            k_split_perm_ptr,
            seq3, halfsize);

        return cache;
    }

    /// Boltzmann cache exp(-k^2 coeff) used by diffusion().
    KCache generate_k_cache(double coeff) const
    {
        return generate_k_cache_impl([coeff](double kx2, double ky2, double kz2) {
            return std::exp(-(kx2 + ky2 + kz2) * coeff);
        });
    }

    /// Multiplier cache (k_d² summed over the axes of the type, times
    /// exp(-k² coeff) for the Exp types) used by apply_multiplier() for the stress.
    KCache generate_k_cache_from_multiplier(MultiplierType type, double coeff) const
    {
        const int axes = crys_recursive_3m::multiplier_axes(type);
        const bool bond = crys_recursive_3m::multiplier_has_bond(type);
        return generate_k_cache_impl([axes, bond, coeff](double kx2, double ky2, double kz2) {
            const double k_axes = ((axes & 1) ? kx2 : 0.0) + ((axes & 2) ? ky2 : 0.0) + ((axes & 4) ? kz2 : 0.0);
            return bond ? std::exp(-(kx2 + ky2 + kz2) * coeff) * k_axes : k_axes;
        });
    }

    const KCache& get_multiplier_cache(MultiplierType type, double coeff) const
    {
        // Types without the bond factor do not depend on coeff
        const auto key = std::make_pair(type, crys_recursive_3m::multiplier_has_bond(type) ? coeff : 0.0);
        return multiplier_cache_.get(key, [&] { return generate_k_cache_from_multiplier(type, coeff); });
    }

public:
    CrysFFTRecursive3mBase(
        std::array<int, 3> nx_logical,
        std::array<double, 6> cell_para,
        std::array<double, 9> translational_part)
        : nx_logical_(nx_logical),
          cell_para_(cell_para),
          translational_part_(translational_part)
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

        // The k-space multiply loops in the MKL/FFTW implementations pad the
        // packed z iteration count to align_up(Nz2/2+1, 8) clamped to Nz2, so
        // any Nz2 = nz/2 >= 2 is safe (the clamp keeps the padded loop inside
        // each row while still covering the packed range [0, Nz2/2]).
        if (nx_logical_[2] / 2 < 2)
        {
            throw_with_line_number("CrysFFTRecursive3m requires Nz/2 >= 2.");
        }

        // The fold pairs twiddle r_p with the octant-transformed Boltzmann
        // array S_{psi(p)}, where psi depends on the parities of the m3
        // generator translations in grid units (see Recursive3mFoldParity.h).
        // When psi is not a bijection the even-index physical basis does not
        // exist on this grid and the fold cannot represent the group;
        // CrysFFTSelector rejects such grids, but guard direct construction.
        if (!compute_m3_fold_permutation(translational_part_, nx_logical_, fold_perm_))
        {
            throw_with_line_number("CrysFFTRecursive3m: m3 translations are "
                "incompatible with this grid (translation-parity permutation "
                "is not a bijection; no even-index physical basis exists).");
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
        k_cache_.invalidate();
        multiplier_cache_.invalidate();
    }

    void set_contour_step(double coeff)
    {
        k_cache_.select(coeff, [&] { return generate_k_cache(coeff); });
    }

    void diffusion(const double* q_in, double* q_out) const
    {
        const KCache* cache = k_cache_.current();
        if (!cache)
            throw_with_line_number("set_contour_step must be called before diffusion().");

        derived().apply_with_cache(*cache, q_in, q_out);
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
