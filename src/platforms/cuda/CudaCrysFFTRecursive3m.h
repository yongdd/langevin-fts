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
#include <unordered_map>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <cufft.h>

#include "CudaCrysFFT.h"

class CudaCrysFFTRecursive3m : public CudaCrysFFTBase
{
public:
    /**
     * @brief Multiplier selection for the stress fast path.
     *
     * Mirrors the CPU CrysFFTRecursive3mBase::MultiplierType:
     * - Kx2/Ky2/Kz2:       Cartesian k_axis² (continuous chains)
     * - ExpKx2/ExpKy2/ExpKz2: exp(-k²·coeff)·k_axis² (discrete chains)
     */
    enum class MultiplierType
    {
        Kx2,
        Ky2,
        Kz2,
        ExpKx2,
        ExpKy2,
        ExpKz2
    };

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

private:
    struct KCacheDevice
    {
        std::array<double*, 8> re{};
        std::array<double*, 8> im{};
    };

    std::array<int, 3> nx_logical_;
    std::array<int, 3> nx_physical_;
    int M_logical_{0};
    int M_physical_{0};
    int M_complex_{0};

    std::array<double, 6> cell_para_;
    std::array<double, 9> translational_part_;

    /// Pairing permutation of the fold: twiddle r_p multiplies the
    /// octant-transformed Boltzmann array S_{fold_perm_[p]} (identity when
    /// all generator translations are an even number of grid cells).
    /// See Recursive3mFoldParity.h for the derivation.
    std::array<int, 8> fold_perm_{};

    std::array<std::vector<double>, 8> r_re_;
    std::array<std::vector<double>, 8> r_im_;

    std::unordered_map<double, KCacheDevice> k_cache_;
    const KCacheDevice* k_current_{nullptr};
    double coeff_current_{0.0};

    /// Multiplier caches for the stress fast path, keyed by (type, coeff)
    std::map<std::pair<int, double>, KCacheDevice> multiplier_cache_;

    cufftHandle plan_r2c_{};
    cufftHandle plan_c2r_{};
    bool plans_initialized_{false};

    cufftDoubleComplex* d_step1_{nullptr};
    cufftDoubleComplex* d_step2_{nullptr};
    double* d_work_{nullptr};

    cudaStream_t stream_{0};

    void init_plans();
    void free_plans();
    void generate_twiddle_factors();
    KCacheDevice generate_k_cache(double coeff);
    KCacheDevice generate_k_cache_from_multiplier(MultiplierType type, double coeff);
    KCacheDevice build_k_cache_from_tempmat(const std::vector<double>& tempmat);
    void free_cache_device(KCacheDevice& cache);
    void apply_with_cache(const KCacheDevice& cache, double* d_q_in, double* d_q_out, cudaStream_t stream);
};

#endif
