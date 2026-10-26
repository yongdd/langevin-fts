/**
 * @file CpuFourierTransform.h
 * @brief FourierTransform with the CPU transforms (FFTW or MKL).
 *
 * The fields of a call are transformed concurrently (OpenMP over fields, the
 * transforms themselves single-threaded as in the propagator solvers) and the
 * multipliers are applied with OpenMP over wavevectors.
 *
 * @see FourierTransform for the interface and layouts
 */

#ifndef CPU_FOURIER_TRANSFORM_H_
#define CPU_FOURIER_TRANSFORM_H_

#include <complex>
#include <memory>
#include <vector>

#include "FFT.h"
#include "FourierTransform.h"

/// Multiplier values in host memory
template <typename T>
class CpuFourierMultiplier : public FourierMultiplier<T>
{
public:
    CpuFourierMultiplier(const FourierTransform<T>* owner, int n_out, int n_in, bool diagonal)
        : FourierMultiplier<T>(owner, n_out, n_in, diagonal) {}

    std::vector<std::complex<double>> complex_values;   ///< complex coefficients
    std::vector<double> real_values;                    ///< real coefficients
};

template <typename T>
class CpuFourierTransform : public FourierTransform<T>
{
public:
    /**
     * @param cb      Computation box; must outlive the transform
     * @param bc      Transform of each axis (see FourierTransform)
     * @param backend FFTW or MKL
     */
    CpuFourierTransform(ComputationBox<T>* cb, std::vector<BoundaryCondition> bc, FFTBackend backend);

protected:
    std::unique_ptr<FourierMultiplier<T>> store_multiplier(
        const std::complex<double>* values, long long n_values, int n_out, int n_in, bool diagonal) override;
    void apply_impl(const std::vector<FourierTerm<T>>& terms, int n_out, T* out) override;
    void forward_impl(const T* fields, int n_fields, double* coefficients) override;
    void backward_impl(const double* coefficients, int n_fields, T* fields) override;

private:
    std::unique_ptr<FFT<T>> fft_;
    std::vector<double> coefficients_in_;    ///< coefficients of the inputs of a term
    std::vector<double> coefficients_out_;   ///< accumulated output coefficients
};

#endif
