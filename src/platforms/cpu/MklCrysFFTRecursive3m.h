/**
 * @file MklCrysFFTRecursive3m.h
 * @brief MKL implementation of recursive crystallographic FFT (2x2y2z).
 *
 * @see CrysFFTRecursive3mBase for common functionality
 * @see FftwCrysFFTRecursive3m for FFTW implementation
 */

#ifndef MKL_CRYS_FFT_RECURSIVE_3M_H_
#define MKL_CRYS_FFT_RECURSIVE_3M_H_

#include <complex>
#include "mkl_dfti.h"
#include "CrysFFTRecursive3mBase.h"

class MklCrysFFTRecursive3m : public CrysFFTRecursive3mBase<MklCrysFFTRecursive3m>
{
private:
    using Base = CrysFFTRecursive3mBase<MklCrysFFTRecursive3m>;
    friend Base;

    DFTI_DESCRIPTOR_HANDLE plan_forward_ = nullptr;
    DFTI_DESCRIPTOR_HANDLE plan_backward_ = nullptr;

    void init_fft_plans();

public:
    MklCrysFFTRecursive3m(
        std::array<int, 3> nx_logical,
        std::array<double, 6> cell_para,
        std::array<double, 9> translational_part = {0,0,0, 0,0,0, 0,0,0});

    ~MklCrysFFTRecursive3m();

    void apply_with_cache(const KCache& cache, const double* q_in, double* q_out) const;
};

#endif  // MKL_CRYS_FFT_RECURSIVE_3M_H_
