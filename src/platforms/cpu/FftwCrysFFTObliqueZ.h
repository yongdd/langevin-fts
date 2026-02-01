/**
 * @file FftwCrysFFTObliqueZ.h
 * @brief FFTW implementation of crystallographic FFT for z-mirror symmetry.
 *
 * Uses native FFTW3 DCT-II/III (REDFT10/REDFT01) with optional FFT-based DCT.
 *
 * @see CrysFFTObliqueZBase for common functionality
 * @see MklCrysFFTObliqueZ for MKL implementation
 */

#ifndef FFTW_CRYS_FFT_OBLIQUEZ_H_
#define FFTW_CRYS_FFT_OBLIQUEZ_H_

#include <fftw3.h>
#include <vector>
#include "CrysFFTObliqueZBase.h"

class FftwCrysFFTObliqueZ : public CrysFFTObliqueZBase<FftwCrysFFTObliqueZ>
{
private:
    using Base = CrysFFTObliqueZBase<FftwCrysFFTObliqueZ>;
    friend Base;

    fftw_plan plan_dct_forward_z_{nullptr};
    fftw_plan plan_dct_backward_z_{nullptr};
    fftw_plan plan_fft_forward_xy_{nullptr};
    fftw_plan plan_fft_backward_xy_{nullptr};
    fftw_plan plan_fft_z_forward_{nullptr};
    fftw_plan plan_fft_z_backward_{nullptr};

    double* io_buffer_{nullptr};
    double* temp_buffer_{nullptr};
    fftw_complex* complex_buffer_{nullptr};
    double* fft_z_real_{nullptr};
    fftw_complex* fft_z_complex_{nullptr};

    bool use_fft_dct_{false};
    int M_complex_z_{0};
    double fft_dct_scale_fwd_{1.0};
    double fft_dct_scale_bwd_{1.0};
    std::vector<double> dct_fft_cos_;
    std::vector<double> dct_fft_sin_;

    void initFFTPlans();
    void initFFTPlansZ(unsigned plan_flags);
    void calibrate_fft_dct_scale();

public:
    FftwCrysFFTObliqueZ(
        std::array<int, 3> nx_logical,
        std::array<double, 6> cell_para,
        std::array<double, 9> trans_part = {0,0,0, 0,0,0, 0,0,0});

    ~FftwCrysFFTObliqueZ();

    void diffusion(double* q_in, double* q_out);
    void apply_multiplier(const double* q_in, double* q_out, const double* multiplier);
};

#endif  // FFTW_CRYS_FFT_OBLIQUEZ_H_
