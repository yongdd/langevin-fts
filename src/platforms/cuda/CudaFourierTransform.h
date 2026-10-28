/**
 * @file CudaFourierTransform.h
 * @brief FourierTransform on the GPU (cuFFT and the CUDA cosine/sine transforms).
 *
 * The multipliers stay in device memory. A call uploads its fields, transforms
 * and multiplies them on the device and downloads the result, all on the
 * transform's own stream; the call returns when the result is on the host.
 * The device buffers grow to the largest call and are reused. On the device the
 * fields and coefficients of a call start an even number of doubles apart, so
 * that every field handed to cuFFT is aligned to 16 bytes (an odd number of
 * real grid points would misalign every second field).
 *
 * @see FourierTransform for the interface and layouts
 */

#ifndef CUDA_FOURIER_TRANSFORM_H_
#define CUDA_FOURIER_TRANSFORM_H_

#include <complex>
#include <memory>
#include <vector>

#include "CudaCommon.h"
#include "CudaFFT.h"
#include "FourierTransform.h"

/// Multiplier values in device memory
template <typename T>
class CudaFourierMultiplier : public FourierMultiplier<T>
{
public:
    CudaFourierMultiplier(const FourierTransform<T>* owner, int n_out, int n_in, bool diagonal)
        : FourierMultiplier<T>(owner, n_out, n_in, diagonal) {}

    CudaDeviceBuffer<cuDoubleComplex> complex_values;   ///< complex coefficients
    CudaDeviceBuffer<double> real_values;               ///< real coefficients
};

template <typename T>
class CudaFourierTransform : public FourierTransform<T>
{
public:
    /**
     * @param cb Computation box; must outlive the transform
     * @param bc Transform of each axis (see FourierTransform)
     */
    CudaFourierTransform(ComputationBox<T>* cb, std::vector<BoundaryCondition> bc);
    ~CudaFourierTransform();

protected:
    std::unique_ptr<FourierMultiplier<T>> store_multiplier(
        const std::complex<double>* values, long long n_values, int n_out, int n_in, bool diagonal) override;
    void apply_impl(const std::vector<FourierTerm<T>>& terms, int n_out, T* out) override;
    void forward_impl(const T* fields, int n_fields, double* coefficients) override;
    void backward_impl(const double* coefficients, int n_fields, T* fields) override;

private:
    using DeviceT = CuDeviceData<T>;

    std::unique_ptr<CudaFFTBase<T>> fft_;
    cudaStream_t stream_ = nullptr;

    CudaDeviceBuffer<DeviceT> d_fields_;           ///< fields of a call (inputs, then outputs)
    CudaDeviceBuffer<double> d_coefficients_in_;   ///< coefficients of the inputs of a term
    CudaDeviceBuffer<double> d_coefficients_out_;  ///< accumulated output coefficients
    long long field_pitch_ = 0;                    ///< elements of T from one device field to the next
    long long coefficient_pitch_ = 0;              ///< doubles from one field's device coefficients to the next

    /// Upload n_fields packed host fields to d_fields_
    void upload_fields(const T* fields, int n_fields);
    /// Download the first n_fields fields of d_fields_ to packed host fields
    void download_fields(T* fields, int n_fields);
    /// Copy n_fields coefficient arrays between the packed host layout and the device pitch
    void copy_coefficients(double* destination, const double* source, int n_fields, bool to_device);
    /// Transform the first n_fields fields of d_fields_ into `coefficients` (device)
    void transform_forward(int n_fields, double* d_coefficients);
    /// Inverse-transform n_fields device coefficients into d_fields_
    void transform_backward(double* d_coefficients, int n_fields);
    /// Multiply n_fields device coefficients by `factor`. The CUDA cosine and sine
    /// series are 2 Σ_j f_j φ_m(x_j) per axis (their inverse divides by 2N), the CPU
    /// ones Σ_j f_j φ_m(x_j); forward() and backward() return and take the latter.
    /// apply() needs no scaling: the factor cancels.
    void scale_wall_coefficients(double* d_coefficients, int n_fields, double factor);
};

#endif
