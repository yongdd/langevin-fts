/**
 * @file CudaTypeDispatch.h
 * @brief Scalar-type dispatch helpers for the CUDA solvers.
 *
 * The solvers are instantiated for real (T = double) and complex
 * (T = std::complex<double>) fields. Most places where the two differ are
 * purely mechanical: real fields use the half-complex cuFFT transforms
 * (D2Z/Z2D) and a plain CUB sum, complex fields use the full Z2Z transforms
 * and a CUB reduce with ComplexSumOp. The overloads below let the solver
 * code call one function and leave the choice to overload resolution, so an
 * `if constexpr` in a solver now marks a genuine algorithmic difference.
 */

#ifndef CUDA_TYPE_DISPATCH_H_
#define CUDA_TYPE_DISPATCH_H_

#include <complex>
#include <cufft.h>
#include <cub/device/device_reduce.cuh>

#include "CudaCommon.h"   // CuDeviceData<T>, ComplexSumOp

/// cuFFT plan types for a given scalar type.
template <typename T>
struct CufftPlanType
{
    static constexpr cufftType forward  = CUFFT_D2Z;   ///< real -> half-complex
    static constexpr cufftType backward = CUFFT_Z2D;   ///< half-complex -> real
};

template <>
struct CufftPlanType<std::complex<double>>
{
    static constexpr cufftType forward  = CUFFT_Z2Z;
    static constexpr cufftType backward = CUFFT_Z2Z;
};

/// Forward transform: real input uses D2Z, complex input uses Z2Z.
inline cufftResult cufft_exec_forward(cufftHandle plan, double* in, cuDoubleComplex* out)
{
    return cufftExecD2Z(plan, in, out);
}
inline cufftResult cufft_exec_forward(cufftHandle plan, cuDoubleComplex* in, cuDoubleComplex* out)
{
    return cufftExecZ2Z(plan, in, out, CUFFT_FORWARD);
}

/// Backward transform: real output uses Z2D, complex output uses Z2Z.
inline cufftResult cufft_exec_backward(cufftHandle plan, cuDoubleComplex* in, double* out)
{
    return cufftExecZ2D(plan, in, out);
}
inline cufftResult cufft_exec_backward(cufftHandle plan, cuDoubleComplex* in, cuDoubleComplex* out)
{
    return cufftExecZ2Z(plan, in, out, CUFFT_INVERSE);
}

/// Sum of a device array into d_out[0] (CUB). Pass d_temp_storage == nullptr
/// to query the required temp_storage_bytes, as with cub::DeviceReduce.
inline cudaError_t cuda_reduce_sum(void* d_temp_storage, size_t& temp_storage_bytes,
                                   const double* d_in, double* d_out, int num_items, cudaStream_t stream)
{
    return cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, stream);
}
inline cudaError_t cuda_reduce_sum(void* d_temp_storage, size_t& temp_storage_bytes,
                                   const cuDoubleComplex* d_in, cuDoubleComplex* d_out, int num_items, cudaStream_t stream)
{
    return cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items,
                                     ComplexSumOp(), cuDoubleComplex{0.0, 0.0}, stream);
}

#endif  // CUDA_TYPE_DISPATCH_H_
