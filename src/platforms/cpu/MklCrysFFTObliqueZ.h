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

#include <array>
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

    std::vector<double> dct_cos_;
    std::vector<double> dct_sin_;

    void initFFTPlans();

    /// Free the DFTI descriptors (also after a failed construction); idempotent.
    void release();

    void applyDCT2ForwardZ(const double* in, double* out) const;
    void applyDCT3BackwardZ(const double* in, double* out) const;

    /// Shared pipeline of diffusion() and apply_multiplier():
    /// DCT-II(z) -> r2c FFT(xy) -> multiply -> c2r FFT(xy) -> DCT-III(z) -> normalize.
    void apply_spectral_multiplier(const double* q_in, double* q_out, const double* multiplier) const;

public:
    MklCrysFFTObliqueZ(
        std::array<int, 3> nx_logical,
        std::array<double, 6> cell_para,
        std::array<double, 9> trans_part = {0,0,0, 0,0,0, 0,0,0});

    ~MklCrysFFTObliqueZ();

    void diffusion(double* q_in, double* q_out);
    void apply_multiplier(const double* q_in, double* q_out, const double* multiplier);

    /// Elements of this thread's z work arrays {mirrored real, spectrum}, as sized
    /// by the last transform on the thread (tests check them against the grid).
    static std::array<size_t, 2> thread_z_work_sizes();
};

#endif  // MKL_CRYS_FFT_OBLIQUEZ_H_
