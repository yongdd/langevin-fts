/*-------------------------------------------------------------
* This class defines simulation grids and Lengths parameters and provide
* methods that compute inner product in a given geometry.
*--------------------------------------------------------------*/
#include <iostream>
#include <complex>
#include <cuda_runtime.h>
#include <cub/device/device_reduce.cuh>
#include <cuComplex.h>

#include "CudaComputationBox.h"
#include "CudaCommon.h"

// Define custom binary operator for complex addition
struct CufftComplexSum {
    __device__ __forceinline__
    cufftDoubleComplex operator()(const cufftDoubleComplex& a, const cufftDoubleComplex& b) const {
        cufftDoubleComplex result;
        result.x = a.x + b.x;
        result.y = a.y + b.y;
        return result;
    }
};

//----------------- Constructor -----------------------------
template <typename T>
CudaComputationBox<T>::CudaComputationBox(
    std::vector<int> nx, std::vector<double> lx, std::vector<std::string> bc, const double* mask)
    : ComputationBox<T>(nx, lx, bc, mask)
{
    initialize();
}
template <typename T>
void CudaComputationBox<T>::initialize()
{
    gpu_error_check(cudaMalloc((void**)&d_dv,  sizeof(T)*this->total_grid));
    for(int i = 0; i < this->total_grid; i++)
    {
        T temp_dv;
        if constexpr (std::is_same<T, double>::value)
            temp_dv = this->dv[i];
        else
        {
            temp_dv.x = this->dv[i];
            temp_dv.y = 0.0;
        }
        gpu_error_check(cudaMemcpy(&d_dv[i], &temp_dv, sizeof(T), cudaMemcpyHostToDevice));
    }
    // Temporal storage
    gpu_error_check(cudaMalloc((void**)&d_multiple, sizeof(T)*this->total_grid));

    gpu_error_check(cudaMalloc((void**)&d_g, sizeof(T)*this->total_grid));
    gpu_error_check(cudaMalloc((void**)&d_h, sizeof(T)*this->total_grid));
    gpu_error_check(cudaMalloc((void**)&d_w, sizeof(T)*this->total_grid));

    // Allocate memory for cub reduction sum
    gpu_error_check(cudaMalloc((void**)&d_sum, sizeof(T)*this->total_grid));
    gpu_error_check(cudaMalloc((void**)&d_sum_out, sizeof(T)));

    // Determine temporary storage size for cub reduction
    if constexpr (std::is_same<T, std::complex<double>>::value) {
        ftsComplex identity = make_cuDoubleComplex(0.0, 0.0);
        cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_sum, d_sum_out, this->total_grid, CufftComplexSum(), identity);
    } else {
        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_sum, d_sum_out, this->total_grid);
    }
    gpu_error_check(cudaMalloc(&d_temp_storage, temp_storage_bytes));
}
//----------------- Destructor -----------------------------
template <typename T>
CudaComputationBox<T>::~CudaComputationBox()
{
    cudaFree(d_dv);

    cudaFree(d_multiple);
    cudaFree(d_g);
    cudaFree(d_h);
    cudaFree(d_w);

    cudaFree(d_sum);
    cudaFree(d_sum_out);
    cudaFree(d_temp_storage);
}
//-----------------------------------------------------------
template <typename T>
void CudaComputationBox<T>::set_lx(std::vector<double> new_lx)
{
    ComputationBox<T>::set_lx(new_lx);
    gpu_error_check(cudaMemcpy(d_dv, this->dv, sizeof(double)*this->total_grid,cudaMemcpyHostToDevice));
}
//-----------------------------------------------------------
template <typename T>
T CudaComputationBox<T>::integral_device(const T *d_g)
{
    const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();
    T sum{0};

    ker_multi<T><<<N_BLOCKS, N_THREADS>>>(d_sum, d_g, d_dv, 1.0, this->total_grid);
    if constexpr (std::is_same<T, std::complex<double>>::value) {
        ftsComplex identity = make_cuDoubleComplex(0.0, 0.0);
        cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_sum, d_sum_out, this->total_grid, CufftComplexSum(), identity);
    } else {
        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_sum, d_sum_out, this->total_grid);
    }
    gpu_error_check(cudaMemcpy(&sum, d_sum_out, sizeof(T),cudaMemcpyDeviceToHost));
    return sum;
}
//-----------------------------------------------------------
template <typename T>
T CudaComputationBox<T>::inner_product_device(const T* d_g, const T* d_h)
{
    const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();
    T sum{0.0};

    ker_multi<T><<<N_BLOCKS, N_THREADS>>>(d_sum, d_g, d_h, 1.0, this->total_grid);
    ker_multi<T><<<N_BLOCKS, N_THREADS>>>(d_sum, d_dv, d_sum, 1.0, this->total_grid);
    if constexpr (std::is_same<T, std::complex<double>>::value) {
        ftsComplex identity = make_cuDoubleComplex(0.0, 0.0);
        cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_sum, d_sum_out, this->total_grid, CufftComplexSum(), identity);
    } else {
        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_sum, d_sum_out, this->total_grid);
    }
    gpu_error_check(cudaMemcpy(&sum, d_sum_out, sizeof(T), cudaMemcpyDeviceToHost));
    return sum;
}
//-----------------------------------------------------------
template <typename T>
T CudaComputationBox<T>::inner_product_inverse_weight_device(const T* d_g, const T* d_h, const T* d_w)
{
    const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();
    T sum{0.0};

    ker_multi<T><<<N_BLOCKS, N_THREADS>>>(d_sum, d_g, d_h, 1.0, this->total_grid);
    ker_multi<T><<<N_BLOCKS, N_THREADS>>>(d_sum, d_dv, d_sum, 1.0, this->total_grid);
    ker_divide<T><<<N_BLOCKS, N_THREADS>>>(d_sum, d_sum, d_w, 1.0, this->total_grid);
    if constexpr (std::is_same<T, std::complex<double>>::value) {
        ftsComplex identity = make_cuDoubleComplex(0.0, 0.0);
        cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_sum, d_sum_out, this->total_grid, CufftComplexSum(), identity);
    } else {
        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_sum, d_sum_out, this->total_grid);
    }
    gpu_error_check(cudaMemcpy(&sum, d_sum_out, sizeof(T), cudaMemcpyDeviceToHost));
    return sum;
}
//-----------------------------------------------------------
template <typename T>
T CudaComputationBox<T>::multi_inner_product_device(int n_comp, const T* d_g, const T* d_h)
{
    const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();
    T sum{0.0};

    ker_mutiple_multi<T><<<N_BLOCKS, N_THREADS>>>(n_comp, d_sum, d_g, d_h, 1.0, this->total_grid);
    ker_multi<T><<<N_BLOCKS, N_THREADS>>>(d_sum, d_dv, d_sum, 1.0, this->total_grid);
    if constexpr (std::is_same<T, std::complex<double>>::value) {
        ftsComplex identity = make_cuDoubleComplex(0.0, 0.0);
        cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_sum, d_sum_out, this->total_grid, CufftComplexSum(), identity);
    } else {
        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_sum, d_sum_out, this->total_grid);
    }
    gpu_error_check(cudaMemcpy(&sum, d_sum_out, sizeof(T), cudaMemcpyDeviceToHost));
    return sum;
}
//-----------------------------------------------------------
template <typename T>
void CudaComputationBox<T>::zero_mean_device(T* d_g)
{
    const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();
    T sum{0.0};

    ker_multi<T><<<N_BLOCKS, N_THREADS>>>(d_sum, d_dv, d_g, 1.0, this->total_grid);
    if constexpr (std::is_same<T, std::complex<double>>::value) {
        ftsComplex identity = make_cuDoubleComplex(0.0, 0.0);
        cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_sum, d_sum_out, this->total_grid, CufftComplexSum(), identity);
    } else {
        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_sum, d_sum_out, this->total_grid);
    }
    gpu_error_check(cudaMemcpy(&sum, d_sum_out, sizeof(T), cudaMemcpyDeviceToHost));
    ker_linear_scaling<T><<<N_BLOCKS, N_THREADS>>>(d_g, d_g, 1.0, -sum/this->volume, this->total_grid);
}

// Explicit template instantiation for double and std::complex<double>
template class CudaComputationBox<double>;
// template class CudaComputationBox<ftsComplex>;