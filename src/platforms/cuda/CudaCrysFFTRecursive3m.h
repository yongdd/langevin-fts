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
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>
#include <cufft.h>

#include "CudaCrysFFT.h"

class CudaCrysFFTRecursive3m : public CudaCrysFFTBase
{
public:
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

    std::array<std::vector<double>, 8> r_re_;
    std::array<std::vector<double>, 8> r_im_;

    std::unordered_map<double, KCacheDevice> k_cache_;
    const KCacheDevice* k_current_{nullptr};
    double coeff_current_{0.0};

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
};

#endif
