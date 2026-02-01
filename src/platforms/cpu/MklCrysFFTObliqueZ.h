/**
 * @file MklCrysFFTObliqueZ.h
 * @brief MKL implementation of crystallographic FFT for z-mirror symmetry.
 *
 * Uses FFT-based DCT-II/III implementation with mirrored sequences.
 *
 * @see CrysFFTObliqueZBase for common functionality
 * @see FftwCrysFFTObliqueZ for FFTW implementation
 */

#ifndef MKL_CRYS_FFT_OBLIQUEZ_H_
#define MKL_CRYS_FFT_OBLIQUEZ_H_

#include <complex>
#include <vector>
#include "mkl_dfti.h"
#include "CrysFFTObliqueZBase.h"

class MklCrysFFTObliqueZ : public CrysFFTObliqueZBase<MklCrysFFTObliqueZ>
{
private:
    using Base = CrysFFTObliqueZBase<MklCrysFFTObliqueZ>;
    friend Base;

    DFTI_DESCRIPTOR_HANDLE plan_dct_forward_z_{nullptr};
    DFTI_DESCRIPTOR_HANDLE plan_dct_backward_z_{nullptr};
    DFTI_DESCRIPTOR_HANDLE plan_fft_forward_xy_{nullptr};
    DFTI_DESCRIPTOR_HANDLE plan_fft_backward_xy_{nullptr};

    double* io_buffer_{nullptr};
    double* temp_buffer_{nullptr};
    std::complex<double>* complex_buffer_{nullptr};

    std::vector<double> dct_cos_;
    std::vector<double> dct_sin_;
    double dct_scale_fwd_{1.0};
    double dct_scale_bwd_{1.0};

    void initFFTPlans();
    void calibrateDCTScale();

    void applyDCT2ForwardZ(const double* in, double* out) const;
    void applyDCT3BackwardZ(const double* in, double* out) const;

public:
    MklCrysFFTObliqueZ(
        std::array<int, 3> nx_logical,
        std::array<double, 6> cell_para,
        std::array<double, 9> trans_part = {0,0,0, 0,0,0, 0,0,0});

    ~MklCrysFFTObliqueZ();

    void diffusion(double* q_in, double* q_out);
    void apply_multiplier(const double* q_in, double* q_out, const double* multiplier);
};

#endif  // MKL_CRYS_FFT_OBLIQUEZ_H_
