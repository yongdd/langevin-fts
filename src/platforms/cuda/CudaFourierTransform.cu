/**
 * @file CudaFourierTransform.cu
 * @brief Fourier multipliers with cuFFT and the CUDA cosine/sine transforms.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cuComplex.h>

#include "CudaFourierTransform.h"

namespace
{
__device__ inline double fourier_mul(double a, double b) { return a * b; }
__device__ inline cuDoubleComplex fourier_mul(cuDoubleComplex a, cuDoubleComplex b) { return cuCmul(a, b); }
__device__ inline double fourier_add(double a, double b) { return a + b; }
__device__ inline cuDoubleComplex fourier_add(cuDoubleComplex a, cuDoubleComplex b) { return cuCadd(a, b); }
__device__ inline void fourier_zero(double& a) { a = 0.0; }
__device__ inline void fourier_zero(cuDoubleComplex& a) { a = make_cuDoubleComplex(0.0, 0.0); }

/// out_i (+)= Σ_j M_ij · in_j at every coefficient (diagonal: out_i (+)= M · in_i).
/// The n coefficients of a field are `pitch` apart in `in` and `out`; m is packed.
template <typename C>
__global__ void ker_fourier_multiply(C* out, const C* m, const C* in, int n_out, int n_in,
                                     bool diagonal, long long n, long long pitch, bool accumulate)
{
    for (long long k = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; k < n;
         k += static_cast<long long>(blockDim.x) * gridDim.x)
    {
        for (int i = 0; i < n_out; ++i)
        {
            C sum;
            if (diagonal)
            {
                sum = fourier_mul(m[k], in[i * pitch + k]);
            }
            else
            {
                fourier_zero(sum);
                for (int j = 0; j < n_in; ++j)
                    sum = fourier_add(sum, fourier_mul(m[(static_cast<long long>(i) * n_in + j) * n + k], in[j * pitch + k]));
            }
            out[i * pitch + k] = accumulate ? fourier_add(out[i * pitch + k], sum) : sum;
        }
    }
}

/// Allocate at least `count` elements, keeping a large enough buffer
template <typename X>
void reserve(CudaDeviceBuffer<X>& buffer, long long count)
{
    if (static_cast<long long>(buffer.size()) < count)
        buffer.allocate(count);
}
}  // namespace

template <typename T>
CudaFourierTransform<T>::CudaFourierTransform(ComputationBox<T>* cb, std::vector<BoundaryCondition> bc)
    : FourierTransform<T>(cb, bc)
{
    const std::vector<int> nx = cb->get_nx();
    if (this->dim_ == 3)
        fft_ = std::make_unique<CudaFFT<T, 3>>(std::array<int, 3>{nx[0], nx[1], nx[2]},
                                                std::array<BoundaryCondition, 3>{bc[0], bc[1], bc[2]});
    else if (this->dim_ == 2)
        fft_ = std::make_unique<CudaFFT<T, 2>>(std::array<int, 2>{nx[0], nx[1]},
                                                std::array<BoundaryCondition, 2>{bc[0], bc[1]});
    else if (this->dim_ == 1)
        fft_ = std::make_unique<CudaFFT<T, 1>>(std::array<int, 1>{nx[0]},
                                                std::array<BoundaryCondition, 1>{bc[0]});
    else
        throw_with_line_number("A Fourier transform supports 1D, 2D or 3D grids, got dim = "
                               + std::to_string(this->dim_) + ".");
    // cuFFT needs 16-byte aligned real data: give every field an even number of doubles
    const long long doubles_per_value = static_cast<long long>(sizeof(T) / sizeof(double));
    const long long field_doubles = this->n_grid_ * doubles_per_value;
    field_pitch_ = (field_doubles + field_doubles % 2) / doubles_per_value;
    coefficient_pitch_ = this->coefficient_stride() + this->coefficient_stride() % 2;
    // Created last: nothing else can throw after it
    CUDA_CHECK(count_cuda_stream(cudaStreamCreate(&stream_)));
}

template <typename T>
CudaFourierTransform<T>::~CudaFourierTransform()
{
    release_cuda_stream(stream_);
}

template <typename T>
std::unique_ptr<FourierMultiplier<T>> CudaFourierTransform<T>::store_multiplier(
    const std::complex<double>* values, long long n_values, int n_out, int n_in, bool diagonal)
{
    auto multiplier = std::make_unique<CudaFourierMultiplier<T>>(this, n_out, n_in, diagonal);
    if (this->complex_coefficients_)
    {
        multiplier->complex_values.allocate(n_values);
        CUDA_CHECK(cudaMemcpy(multiplier->complex_values, values, sizeof(cuDoubleComplex) * n_values,
                              cudaMemcpyHostToDevice));
    }
    else
    {
        std::vector<double> real_values(n_values);
        for (long long i = 0; i < n_values; ++i)
            real_values[i] = values[i].real();
        multiplier->real_values.allocate(n_values);
        CUDA_CHECK(cudaMemcpy(multiplier->real_values, real_values.data(), sizeof(double) * n_values,
                              cudaMemcpyHostToDevice));
    }
    return multiplier;
}

template <typename T>
void CudaFourierTransform<T>::upload_fields(const T* fields, int n_fields)
{
    // The fields are packed on the host and field_pitch_ apart on the device
    const long long n_grid = this->n_grid_;
    T* d_fields = reinterpret_cast<T*>(d_fields_.get());
    for (int f = 0; f < n_fields; ++f)
        CUDA_CHECK(cudaMemcpyAsync(d_fields + f * field_pitch_, fields + f * n_grid, sizeof(T) * n_grid,
                                   cudaMemcpyHostToDevice, stream_));
}

template <typename T>
void CudaFourierTransform<T>::download_fields(T* fields, int n_fields)
{
    const long long n_grid = this->n_grid_;
    const T* d_fields = reinterpret_cast<const T*>(d_fields_.get());
    for (int f = 0; f < n_fields; ++f)
        CUDA_CHECK(cudaMemcpyAsync(fields + f * n_grid, d_fields + f * field_pitch_, sizeof(T) * n_grid,
                                   cudaMemcpyDeviceToHost, stream_));
}

template <typename T>
void CudaFourierTransform<T>::copy_coefficients(double* destination, const double* source, int n_fields,
                                                bool to_device)
{
    // Packed on the host, coefficient_pitch_ apart on the device
    const long long stride = this->coefficient_stride();
    for (int f = 0; f < n_fields; ++f)
    {
        if (to_device)
            CUDA_CHECK(cudaMemcpyAsync(destination + f * coefficient_pitch_, source + f * stride,
                                       sizeof(double) * stride, cudaMemcpyHostToDevice, stream_));
        else
            CUDA_CHECK(cudaMemcpyAsync(destination + f * stride, source + f * coefficient_pitch_,
                                       sizeof(double) * stride, cudaMemcpyDeviceToHost, stream_));
    }
}

template <typename T>
void CudaFourierTransform<T>::transform_forward(int n_fields, double* d_coefficients)
{
    T* d_fields = reinterpret_cast<T*>(d_fields_.get());
    for (int f = 0; f < n_fields; ++f)
    {
        if (this->periodic_)
            fft_->forward_stream(d_fields + f * field_pitch_,
                                 reinterpret_cast<std::complex<double>*>(d_coefficients + f * coefficient_pitch_), stream_);
        else
            fft_->forward_stream(d_fields + f * field_pitch_, d_coefficients + f * coefficient_pitch_, stream_);
    }
}

template <typename T>
void CudaFourierTransform<T>::transform_backward(double* d_coefficients, int n_fields)
{
    T* d_fields = reinterpret_cast<T*>(d_fields_.get());
    for (int f = 0; f < n_fields; ++f)
    {
        if (this->periodic_)
            fft_->backward_stream(reinterpret_cast<std::complex<double>*>(d_coefficients + f * coefficient_pitch_),
                                  d_fields + f * field_pitch_, stream_);
        else
            fft_->backward_stream(d_coefficients + f * coefficient_pitch_, d_fields + f * field_pitch_, stream_);
    }
}

template <typename T>
void CudaFourierTransform<T>::scale_wall_coefficients(double* d_coefficients, int n_fields, double factor)
{
    // One launch per field: a field's coefficients fit the int index of the kernel
    const long long stride = this->coefficient_stride();
    const int n_threads = CudaCommon::get_instance().get_n_threads();
    for (int f = 0; f < n_fields; ++f)
    {
        double* d_field = d_coefficients + f * coefficient_pitch_;
        ker_linear_scaling<<<grid_blocks(stride, n_threads), n_threads, 0, stream_>>>(
            d_field, d_field, factor, 0.0, static_cast<int>(stride));
        CUDA_CHECK_LAUNCH();
    }
}

template <typename T>
void CudaFourierTransform<T>::forward_impl(const T* fields, int n_fields, double* coefficients)
{
    reserve(d_fields_, n_fields * field_pitch_);
    reserve(d_coefficients_in_, n_fields * coefficient_pitch_);
    upload_fields(fields, n_fields);
    transform_forward(n_fields, d_coefficients_in_);
    if (!this->periodic_)
        scale_wall_coefficients(d_coefficients_in_, n_fields, std::ldexp(1.0, -this->dim_));
    copy_coefficients(coefficients, d_coefficients_in_, n_fields, false);
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

template <typename T>
void CudaFourierTransform<T>::backward_impl(const double* coefficients, int n_fields, T* fields)
{
    reserve(d_coefficients_out_, n_fields * coefficient_pitch_);
    reserve(d_fields_, n_fields * field_pitch_);
    copy_coefficients(d_coefficients_out_, coefficients, n_fields, true);
    if (!this->periodic_)
        scale_wall_coefficients(d_coefficients_out_, n_fields, std::ldexp(1.0, this->dim_));
    transform_backward(d_coefficients_out_, n_fields);
    download_fields(fields, n_fields);
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

template <typename T>
void CudaFourierTransform<T>::apply_impl(const std::vector<FourierTerm<T>>& terms, int n_out, T* out)
{
    const long long n = this->n_coefficients_;
    // Pitch in coefficients: complex coefficients take two doubles each
    const long long pitch = this->complex_coefficients_ ? coefficient_pitch_ / 2 : coefficient_pitch_;
    int max_in = 0;
    for (const FourierTerm<T>& term : terms)
        max_in = std::max(max_in, term.n_fields);
    // Reserved before any work is queued: a reallocation frees memory in use
    reserve(d_fields_, std::max(max_in, n_out) * field_pitch_);
    reserve(d_coefficients_in_, max_in * coefficient_pitch_);
    reserve(d_coefficients_out_, n_out * coefficient_pitch_);

    const int n_threads = CudaCommon::get_instance().get_n_threads();
    const int n_blocks = grid_blocks(n, n_threads);
    for (size_t t = 0; t < terms.size(); ++t)
    {
        const FourierTerm<T>& term = terms[t];
        const auto* m = static_cast<const CudaFourierMultiplier<T>*>(term.multiplier);
        upload_fields(term.fields, term.n_fields);
        transform_forward(term.n_fields, d_coefficients_in_);
        if (this->complex_coefficients_)
        {
            ker_fourier_multiply<<<n_blocks, n_threads, 0, stream_>>>(
                reinterpret_cast<cuDoubleComplex*>(d_coefficients_out_.get()), m->complex_values.get(),
                reinterpret_cast<const cuDoubleComplex*>(d_coefficients_in_.get()),
                n_out, m->get_n_in(), m->is_diagonal(), n, pitch, t > 0);
        }
        else
        {
            ker_fourier_multiply<<<n_blocks, n_threads, 0, stream_>>>(
                d_coefficients_out_.get(), m->real_values.get(), d_coefficients_in_.get(),
                n_out, m->get_n_in(), m->is_diagonal(), n, pitch, t > 0);
        }
        CUDA_CHECK_LAUNCH();
    }
    transform_backward(d_coefficients_out_, n_out);
    download_fields(out, n_out);
    CUDA_CHECK(cudaStreamSynchronize(stream_));
}

template class CudaFourierTransform<double>;
template class CudaFourierTransform<std::complex<double>>;
