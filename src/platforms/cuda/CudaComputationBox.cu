/**
 * @file CudaComputationBox.cu
 * @brief CUDA implementation of computation box with GPU operations.
 *
 * Extends ComputationBox with GPU-accelerated integral and inner product
 * operations using CUB parallel reductions.
 *
 * **GPU Operations:**
 *
 * - integral_device(): Volume-weighted sum on GPU
 * - inner_product_device(): <f,g> = ∫ f(r)g(r) dv
 * - zero_mean_device(): Remove mean from field
 *
 * **CUB Reductions:**
 *
 * Uses CUB library for high-performance parallel reductions:
 * - DeviceReduce::Sum for real fields
 * - Custom ComplexSumOp for complex fields
 *
 * **Memory Layout:**
 *
 * - d_dv: Device copy of volume elements
 * - d_sum, d_sum_out: Temporary arrays for reduction
 * - d_temp_storage: CUB workspace
 *
 * **Template Instantiations:**
 *
 * - CudaComputationBox<double>: Real fields
 * - CudaComputationBox<std::complex<double>>: Complex fields
 *
 * @see ComputationBox for base class interface
 */

#include <iostream>
#include <complex>
#include <cuda_runtime.h>
#include <cub/device/device_reduce.cuh>
#include <cuComplex.h>

#include "CudaComputationBox.h"
#include "CudaCommon.h"

/**
 * @brief Construct CUDA computation box with device memory.
 *
 * @param nx   Grid points per dimension
 * @param lx   Box lengths per dimension
 * @param bc   Boundary conditions
 * @param mask Optional mask for impenetrable regions
 */
template <typename T>
CudaComputationBox<T>::CudaComputationBox(
    std::vector<int> nx, std::vector<double> lx, std::vector<std::string> bc, const double* mask)
    : ComputationBox<T>(nx, lx, bc, mask)
{
    initialize();
}
template <typename T>
CudaComputationBox<T>::CudaComputationBox(
    std::vector<int> nx, std::vector<double> lx, std::vector<std::string> bc,
    std::vector<double> angles, const double* mask)
    : ComputationBox<T>(nx, lx, bc, angles, mask)
{
    initialize();
}
template <typename T>
void CudaComputationBox<T>::initialize()
{
    gpu_error_check(cudaMalloc((void**)&d_dv, sizeof(double)*this->total_grid));
    gpu_error_check(cudaMemcpy(d_dv, this->dv, sizeof(double)*this->total_grid, cudaMemcpyHostToDevice));

    // Temporal storage
    gpu_error_check(cudaMalloc((void**)&d_multiple, sizeof(T)*this->total_grid));

    // Allocate memory for cub reduction sum
    gpu_error_check(cudaMalloc((void**)&d_sum,      sizeof(T)*this->total_grid));
    gpu_error_check(cudaMalloc((void**)&d_sum_out,  sizeof(T)));

    // Determine temporary storage size for cub reduction
    if constexpr (std::is_same<T, double>::value) 
        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_sum, d_sum_out, this->total_grid);
    else
        cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_sum, d_sum_out, this->total_grid, ComplexSumOp(), CuDeviceData<T>{0.0,0.0});
    gpu_error_check(cudaMalloc(&d_temp_storage, temp_storage_bytes));
}
//----------------- Destructor -----------------------------
template <typename T>
CudaComputationBox<T>::~CudaComputationBox()
{
    cudaFree(d_dv);
    cudaFree(d_multiple);
    cudaFree(d_sum);
    cudaFree(d_sum_out);
    cudaFree(d_temp_storage);
}
//-----------------------------------------------------------
template <typename T>
void CudaComputationBox<T>::set_lx(std::vector<double> new_lx)
{
    ComputationBox<T>::set_lx(new_lx);
    gpu_error_check(cudaMemcpy(d_dv, this->dv, sizeof(double)*this->total_grid, cudaMemcpyHostToDevice));
}
//-----------------------------------------------------------
template <typename T>
void CudaComputationBox<T>::set_lattice_parameters(std::vector<double> new_lx, std::vector<double> new_angles)
{
    ComputationBox<T>::set_lattice_parameters(new_lx, new_angles);
    gpu_error_check(cudaMemcpy(d_dv, this->dv, sizeof(double)*this->total_grid, cudaMemcpyHostToDevice));
}
//-----------------------------------------------------------
template <typename T>
T CudaComputationBox<T>::integral_device(const CuDeviceData<T> *d_g)
{
    const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();
    T sum{0.0};

    ker_multi<<<N_BLOCKS, N_THREADS>>>(d_sum, d_g, d_dv, 1.0, this->total_grid);

    if constexpr (std::is_same<T, double>::value)
        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_sum, d_sum_out, this->total_grid);
    else
        cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_sum, d_sum_out, this->total_grid, ComplexSumOp(), CuDeviceData<T>{0.0,0.0});
    gpu_error_check(cudaMemcpy(&sum, d_sum_out, sizeof(T), cudaMemcpyDeviceToHost));
    return sum;
}
//-----------------------------------------------------------
template <typename T>
T CudaComputationBox<T>::inner_product_device(const CuDeviceData<T>* d_g, const CuDeviceData<T>* d_h)
{
    const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();
    T sum{0.0};

    ker_multi<<<N_BLOCKS, N_THREADS>>>(d_sum, d_g, d_h, 1.0, this->total_grid);
    ker_multi<<<N_BLOCKS, N_THREADS>>>(d_sum, d_sum, d_dv, 1.0, this->total_grid);

    if constexpr (std::is_same<T, double>::value)
        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_sum, d_sum_out, this->total_grid);
    else
        cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_sum, d_sum_out, this->total_grid, ComplexSumOp(), CuDeviceData<T>{0.0,0.0});

    gpu_error_check(cudaMemcpy(&sum, d_sum_out, sizeof(T), cudaMemcpyDeviceToHost));
    return sum;
}
//-----------------------------------------------------------
template <typename T>
T CudaComputationBox<T>::inner_product_inverse_weight_device(const CuDeviceData<T>* d_g, const CuDeviceData<T>* d_h, const CuDeviceData<T>* d_w)
{
    const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();
    T sum{0.0};

    ker_multi<<<N_BLOCKS, N_THREADS>>>(d_sum, d_g, d_h, 1.0, this->total_grid);
    ker_multi<<<N_BLOCKS, N_THREADS>>>(d_sum, d_sum, d_dv, 1.0, this->total_grid);
    ker_divide<<<N_BLOCKS, N_THREADS>>>(d_sum, d_sum, d_w, 1.0, this->total_grid);
    if constexpr (std::is_same<T, double>::value)
        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_sum, d_sum_out, this->total_grid);
    else
        cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_sum, d_sum_out, this->total_grid, ComplexSumOp(), CuDeviceData<T>{0.0,0.0});

    gpu_error_check(cudaMemcpy(&sum, d_sum_out, sizeof(T), cudaMemcpyDeviceToHost));
    return sum;
}
//-----------------------------------------------------------
template <typename T>
T CudaComputationBox<T>::multi_inner_product_device(int n_comp, const CuDeviceData<T>* d_g, const CuDeviceData<T>* d_h)
{
    const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();
    T sum{0.0};

    ker_mutiple_multi<<<N_BLOCKS, N_THREADS>>>(n_comp, d_sum, d_g, d_h, 1.0, this->total_grid);
    ker_multi<<<N_BLOCKS, N_THREADS>>>(d_sum, d_sum, d_dv, 1.0, this->total_grid);
    if constexpr (std::is_same<T, double>::value)
        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_sum, d_sum_out, this->total_grid);
    else
        cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_sum, d_sum_out, this->total_grid, ComplexSumOp(), CuDeviceData<T>{0.0,0.0});
    gpu_error_check(cudaMemcpy(&sum, d_sum_out, sizeof(T), cudaMemcpyDeviceToHost));
    return sum;
}
//-----------------------------------------------------------
template <typename T>
void CudaComputationBox<T>::zero_mean_device(CuDeviceData<T>* d_g)
{
    const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();
    CuDeviceData<T> sum{0.0};

    ker_multi<<<N_BLOCKS, N_THREADS>>>(d_sum, d_g, d_dv, 1.0, this->total_grid);

    if constexpr (std::is_same<T, double>::value)
    {
        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_sum, d_sum_out, this->total_grid);
        gpu_error_check(cudaMemcpy(&sum, d_sum_out, sizeof(T), cudaMemcpyDeviceToHost));
    }
    else
    {
        cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_sum, d_sum_out, this->total_grid, ComplexSumOp(), CuDeviceData<T>{0.0,0.0});
        gpu_error_check(cudaMemcpy(&sum, d_sum_out, sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
        sum.x = -sum.x/this->volume;
        sum.y = -sum.y/this->volume;
    }
    ker_linear_scaling<<<N_BLOCKS, N_THREADS>>>(d_g, d_g, 1.0, sum, this->total_grid);
}

// Explicit template instantiation

template class CudaComputationBox<double>;
template class CudaComputationBox<std::complex<double>>;