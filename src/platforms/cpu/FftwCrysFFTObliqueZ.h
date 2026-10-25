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
#include <memory>
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

    bool use_fft_dct_{false};
    int M_complex_z_{0};
    /// The FFT of the mirrored 2N sequence gives 0.5*(Re cos + Im sin) = half of
    /// FFTW's unnormalized REDFT10, so the forward coefficients are scaled by 2;
    /// the unnormalized 2N-point inverse of the twiddled spectrum is REDFT01
    /// itself (no backward factor).
    static constexpr double fft_dct_scale_fwd_ = 2.0;
    std::vector<double> dct_fft_cos_;
    std::vector<double> dct_fft_sin_;

    /// Per-thread FFTW work buffers shared by diffusion() and
    /// apply_multiplier(); see get_thread_buffers().
    struct RealDeleter { void operator()(double* ptr) const { if (ptr) fftw_free(ptr); } };
    struct ComplexDeleter { void operator()(fftw_complex* ptr) const { if (ptr) fftw_free(ptr); } };
    struct ThreadBuffers
    {
        std::unique_ptr<double, RealDeleter> real_in;
        std::unique_ptr<double, RealDeleter> real_tmp;
        std::unique_ptr<fftw_complex, ComplexDeleter> complex;
        std::unique_ptr<double, RealDeleter> z_real;
        std::unique_ptr<fftw_complex, ComplexDeleter> z_complex;
        int size_real = 0;
        int size_complex = 0;
        int size_z_real = 0;
        int size_z_complex = 0;
    };

    void initFFTPlans();
    void initFFTPlansZ(unsigned plan_flags);

    /// Destroy the plans (also after a failed construction); idempotent.
    void release();

    /// Thread-local work buffers, (re)allocated to the current grid size.
    ThreadBuffers& get_thread_buffers() const;
    /// DCT-II along z (native REDFT10, or the FFT-based path when
    /// FTS_CRYSFFT_HEX_DCT_FFT is set).
    void dct2_execute(ThreadBuffers& buffers, const double* in, double* out) const;
    /// DCT-III along z (native REDFT01, or the FFT-based path).
    void dct3_execute(ThreadBuffers& buffers, const double* in, double* out) const;
    /// Shared pipeline of diffusion() and apply_multiplier():
    /// DCT-II(z) -> r2c FFT(xy) -> multiply -> c2r FFT(xy) -> DCT-III(z) -> normalize.
    void apply_spectral_multiplier(const double* q_in, double* q_out, const double* multiplier) const;

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
