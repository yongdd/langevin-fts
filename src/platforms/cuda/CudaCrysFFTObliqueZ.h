/**
 * @file CudaCrysFFTObliqueZ.h
 * @brief CUDA CrysFFT for z-mirror symmetry (DCT-z + FFT-xy, gamma arbitrary).
 */

#ifndef CUDA_CRYS_FFT_OBLIQUEZ_H_
#define CUDA_CRYS_FFT_OBLIQUEZ_H_

#include <array>
#include <map>
#include <cuda_runtime.h>
#include <cufft.h>

#include "CudaCrysFFT.h"

class CudaRealTransform3D;

class CudaCrysFFTObliqueZ : public CudaCrysFFTBase
{
private:
    std::array<int, 3> nx_logical_;
    std::array<int, 3> nx_physical_;
    int M_logical_{0};
    int M_physical_{0};
    int M_complex_xy_{0};

    std::array<double, 6> cell_para_;
    std::array<double, 6> recip_metric_;

    std::map<double, double*> d_boltzmann_;
    double* d_boltz_current_{nullptr};
    double ds_current_{0.0};

    CudaRealTransform3D* dct_forward_z_{nullptr};
    CudaRealTransform3D* dct_backward_z_{nullptr};

    cufftHandle plan_xy_fwd_{};
    cufftHandle plan_xy_bwd_{};
    bool plan_xy_initialized_{false};

    cufftDoubleComplex* d_complex_{nullptr};
    double* d_real_xy_{nullptr};

    double norm_factor_{1.0};
    cudaStream_t stream_{0};

    static std::array<double, 6> compute_recip_metric(const std::array<double, 6>& cell_para);
    void init_fft_xy();
    void free_fft_xy();
    void generateBoltzmann(double ds);
    void freeBoltzmann();

public:
    CudaCrysFFTObliqueZ(
        std::array<int, 3> nx_logical,
        std::array<double, 6> cell_para,
        std::array<double, 9> trans_part = {0,0,0, 0,0,0, 0,0,0});

    ~CudaCrysFFTObliqueZ() override;

    void set_cell_para(const std::array<double, 6>& cell_para) override;
    void set_contour_step(double ds) override;
    void diffusion(double* d_q_in, double* d_q_out) override;
    void diffusion(double* d_q_in, double* d_q_out, cudaStream_t stream) override;
    void set_stream(cudaStream_t stream) override;

    int get_M_physical() const { return M_physical_; }
};

#endif  // CUDA_CRYS_FFT_OBLIQUEZ_H_
