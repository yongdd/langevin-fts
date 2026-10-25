/**
 * @file FftwCrysFFTObliqueZ.h
 * @brief FFTW implementation of crystallographic FFT for z-mirror symmetry.
 *
 * Uses native FFTW3 DCT-II/III (REDFT10/REDFT01).
 *
 * @see CrysFFTObliqueZBase for common functionality
 * @see MklCrysFFTObliqueZ for MKL implementation
 */

#ifndef FFTW_CRYS_FFT_OBLIQUEZ_H_
#define FFTW_CRYS_FFT_OBLIQUEZ_H_

#include <fftw3.h>
#include <memory>
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

    /// Per-thread FFTW work buffers shared by diffusion() and
    /// apply_multiplier(); see get_thread_buffers().
    struct RealDeleter { void operator()(double* ptr) const { if (ptr) fftw_free(ptr); } };
    struct ComplexDeleter { void operator()(fftw_complex* ptr) const { if (ptr) fftw_free(ptr); } };
    struct ThreadBuffers
    {
        std::unique_ptr<double, RealDeleter> real_in;
        std::unique_ptr<double, RealDeleter> real_tmp;
        std::unique_ptr<fftw_complex, ComplexDeleter> complex;
        int size_real = 0;
        int size_complex = 0;
    };

    void initFFTPlans();

    /// Destroy the plans (also after a failed construction); idempotent.
    void release();

    /// Thread-local work buffers, (re)allocated to the current grid size.
    ThreadBuffers& get_thread_buffers() const;
    /// DCT-II along z (REDFT10)
    void dct2_execute(const double* in, double* out) const;
    /// DCT-III along z (REDFT01)
    void dct3_execute(const double* in, double* out) const;
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
