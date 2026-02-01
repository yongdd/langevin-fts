/**
 * @file FftwCrysFFTRecursive3m.h
 * @brief FFTW implementation of recursive crystallographic FFT (2x2y2z).
 *
 * @see CrysFFTRecursive3mBase for common functionality
 * @see MklCrysFFTRecursive3m for MKL implementation
 */

#ifndef FFTW_CRYS_FFT_RECURSIVE_3M_H_
#define FFTW_CRYS_FFT_RECURSIVE_3M_H_

#include <fftw3.h>
#include "CrysFFTRecursive3mBase.h"

class FftwCrysFFTRecursive3m : public CrysFFTRecursive3mBase<FftwCrysFFTRecursive3m>
{
private:
    using Base = CrysFFTRecursive3mBase<FftwCrysFFTRecursive3m>;
    friend Base;

    fftw_plan plan_forward_ = nullptr;
    fftw_plan plan_backward_ = nullptr;

    void init_fft_plans();

public:
    FftwCrysFFTRecursive3m(
        std::array<int, 3> nx_logical,
        std::array<double, 6> cell_para,
        std::array<double, 9> translational_part = {0,0,0, 0,0,0, 0,0,0});

    ~FftwCrysFFTRecursive3m();

    void apply_with_cache(const KCache& cache, const double* q_in, double* q_out) const;
};

#endif  // FFTW_CRYS_FFT_RECURSIVE_3M_H_
