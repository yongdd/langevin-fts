/**
 * @file CudaCrysFFTRecursive3m.h
 * @brief CUDA crystallographic FFT using 2x2x2 (3m) algorithm.
 *
 * This implements the generalized 3m symmetry using only cuFFT and
 * precomputed k-matrix factors. It mirrors the CPU FftwCrysFFTRecursive3m
 * algorithm but runs fully on GPU.
 */

#ifndef CUDA_CRYS_FFT_RECURSIVE_3M_H_
#define CUDA_CRYS_FFT_RECURSIVE_3M_H_

#include <array>
#include <map>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <cufft.h>

#include "CudaCrysFFT.h"
#include "Recursive3mHelpers.h"

class CudaCrysFFTRecursive3m : public CudaCrysFFTBase
{
public:
    /// Multiplier selection for the stress fast path, shared with the CPU
    /// engine (crys_recursive_3m::MultiplierType in Recursive3mHelpers.h)
    using MultiplierType = crys_recursive_3m::MultiplierType;

    CudaCrysFFTRecursive3m(
        std::array<int, 3> nx_logical,
        std::array<double, 6> cell_para,
        std::array<double, 9> translational_part);

    ~CudaCrysFFTRecursive3m() override;

    void set_cell_para(const std::array<double, 6>& cell_para) override;
    void set_contour_step(double coeff) override;
    void diffusion(double* d_q_in, double* d_q_out) override;
    void diffusion(double* d_q_in, double* d_q_out, cudaStream_t stream) override;
    void set_stream(cudaStream_t stream) override;
    std::unique_ptr<CudaCrysFFTBase> clone_sharing_tables() const override;

    /**
     * @brief Apply a k-space multiplier via the 3m pipeline (for stress).
     *
     * Uses the same D2Z -> k-multiply -> Z2D pipeline as diffusion() but with
     * a multiplier cache built from MultiplierType instead of the Boltzmann
     * factor. Round-trip normalization is included. The input array is
     * preserved when d_q_in != d_q_out.
     *
     * @param d_q_in  Input field on device (physical grid)
     * @param d_q_out Output field on device (physical grid)
     * @param type    Multiplier type (per-axis k² with optional bond factor)
     * @param coeff   Bond factor coefficient (used for Exp* types only)
     * @param stream  CUDA stream for execution
     */
    void apply_multiplier(double* d_q_in, double* d_q_out, MultiplierType type, double coeff, cudaStream_t stream);

    /// Build the multiplier cache of apply_multiplier(type, coeff) now (outside the stress loop).
    void prepare_multiplier(MultiplierType type, double coeff);

private:
    /// Device tables of one k-space multiplier (owning; the struct is move-only)
    struct KCacheDevice
    {
        std::array<CudaDeviceBuffer<double>, 8> re;
        std::array<CudaDeviceBuffer<double>, 8> im;
    };

    std::array<int, 3> nx_logical_;
    std::array<int, 3> nx_physical_;
    int M_logical_{0};
    int M_physical_{0};
    int M_complex_{0};

    std::array<double, 9> translational_part_;

    /// Pairing permutation of the fold: twiddle r_p multiplies the
    /// octant-transformed Boltzmann array S_{fold_perm_[p]} (identity when
    /// all generator translations are an even number of grid cells).
    /// See Recursive3mFoldParity.h for the derivation.
    std::array<int, 8> fold_perm_{};

    /// Twiddle factors of the fold, fixed by the grid and the translations (shared with the clones)
    struct Twiddles
    {
        std::array<std::vector<double>, 8> re;
        std::array<std::vector<double>, 8> im;
    };
    std::shared_ptr<const Twiddles> twiddles_;

    /// k-space caches: Boltzmann factors per coefficient and the stress
    /// multipliers per (type, coeff), shared with the clones together with the
    /// cell (see CudaCrysFFTTables.h); all caches have one shape
    enum TableKind { kBoltzmann, kMultiplier };
    CudaCrysFFTTableView<KCacheDevice> tables_;
    const KCacheDevice* k_current_{nullptr};
    double coeff_current_{0.0};

    cufftHandle plan_r2c_{};
    cufftHandle plan_c2r_{};
    bool plans_initialized_{false};

    CudaDeviceBuffer<cufftDoubleComplex> d_step1_;
    CudaDeviceBuffer<cufftDoubleComplex> d_step2_;

    cudaStream_t stream_{0};

    /// Clone of `other` sharing its twiddle factors and tables (see clone_sharing_tables()).
    explicit CudaCrysFFTRecursive3m(const CudaCrysFFTRecursive3m& other);

    void init_plans();
    void free_plans();
    std::shared_ptr<const Twiddles> generate_twiddle_factors() const;
    // The cache builders make the cache of the given cell in `cache` (a table
    // build, see CudaCrysFFTTables::get())
    KCacheDevice generate_k_cache(double coeff, const std::array<double, 6>& cell, KCacheDevice cache);
    KCacheDevice generate_k_cache_from_multiplier(MultiplierType type, double coeff,
                                                  const std::array<double, 6>& cell, KCacheDevice cache);
    KCacheDevice build_k_cache_from_tempmat(const std::vector<double>& tempmat, KCacheDevice cache);
    /// The multiplier cache of (type, coeff), built when missing.
    const KCacheDevice& multiplier_cache(MultiplierType type, double coeff);
    /// Free everything acquired so far (idempotent; also used when the constructor throws).
    void release();
    void apply_with_cache(const KCacheDevice& cache, double* d_q_in, double* d_q_out, cudaStream_t stream);
};

#endif
