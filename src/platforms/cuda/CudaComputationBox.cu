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
CudaComputationBox::CudaComputationBox(
    std::vector<int> nx, std::vector<double> lx, std::vector<std::string> bc, const double* mask)
    : ComputationBox(nx, lx, bc, mask)
{
    initialize();
}
void CudaComputationBox::initialize()
{
    gpu_error_check(cudaMalloc((void**)&d_dv,  sizeof(double)*this->total_grid));
    gpu_error_check(cudaMemcpy(d_dv, dv, sizeof(double)*this->total_grid, cudaMemcpyHostToDevice));

    // Temporal storage
    gpu_error_check(cudaMalloc((void**)&d_multiple, sizeof(double)*this->total_grid));

    gpu_error_check(cudaMalloc((void**)&d_g, sizeof(double)*this->total_grid));
    gpu_error_check(cudaMalloc((void**)&d_h, sizeof(double)*this->total_grid));
    gpu_error_check(cudaMalloc((void**)&d_w, sizeof(double)*this->total_grid));

    // Allocate memory for cub reduction sum
    gpu_error_check(cudaMalloc((void**)&d_sum, sizeof(double)*this->total_grid));
    gpu_error_check(cudaMalloc((void**)&d_sum_out, sizeof(double)));

    // Determine temporary storage size for cub reduction
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_sum, d_sum_out, this->total_grid);
    gpu_error_check(cudaMalloc(&d_temp_storage, temp_storage_bytes));
}
//----------------- Destructor -----------------------------
CudaComputationBox::~CudaComputationBox()
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
void CudaComputationBox::set_lx(std::vector<double> new_lx)
{
    ComputationBox::set_lx(new_lx);
    gpu_error_check(cudaMemcpy(d_dv, this->dv, sizeof(double)*this->total_grid,cudaMemcpyHostToDevice));
}
//-----------------------------------------------------------
double CudaComputationBox::integral_device(const double *d_g)
{
    const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();
    double sum{0};
    ker_multi<double><<<N_BLOCKS, N_THREADS>>>(d_sum, d_g, d_dv, 1.0, this->total_grid);
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_sum, d_sum_out, this->total_grid);
    gpu_error_check(cudaMemcpy(&sum, d_sum_out, sizeof(double), cudaMemcpyDeviceToHost));
    return sum;
}
//-----------------------------------------------------------
double CudaComputationBox::inner_product_device(const double* d_g, const double* d_h)
{
    const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();
    double sum{0.0};

    ker_multi<double><<<N_BLOCKS, N_THREADS>>>(d_sum, d_g, d_h, 1.0, this->total_grid);
    ker_multi<double><<<N_BLOCKS, N_THREADS>>>(d_sum, d_dv, d_sum, 1.0, this->total_grid);
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_sum, d_sum_out, this->total_grid);
    gpu_error_check(cudaMemcpy(&sum, d_sum_out, sizeof(double), cudaMemcpyDeviceToHost));
    return sum;
}
//-----------------------------------------------------------
double CudaComputationBox::inner_product_inverse_weight_device(const double* d_g, const double* d_h, const double* d_w)
{
    const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();
    double sum{0.0};

    ker_multi<double><<<N_BLOCKS, N_THREADS>>>(d_sum, d_g, d_h, 1.0, this->total_grid);
    ker_multi<double><<<N_BLOCKS, N_THREADS>>>(d_sum, d_dv, d_sum, 1.0, this->total_grid);
    ker_divide<double><<<N_BLOCKS, N_THREADS>>>(d_sum, d_sum, d_w, 1.0, this->total_grid);
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_sum, d_sum_out, this->total_grid);
    gpu_error_check(cudaMemcpy(&sum, d_sum_out, sizeof(double), cudaMemcpyDeviceToHost));
    return sum;
}
//-----------------------------------------------------------
double CudaComputationBox::multi_inner_product_device(int n_comp, const double* d_g, const double* d_h)
{
    const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();
    double sum{0.0};

    ker_mutiple_multi<double><<<N_BLOCKS, N_THREADS>>>(n_comp, d_sum, d_g, d_h, 1.0, this->total_grid);
    ker_multi<double><<<N_BLOCKS, N_THREADS>>>(d_sum, d_dv, d_sum, 1.0, this->total_grid);
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_sum, d_sum_out, this->total_grid);
    gpu_error_check(cudaMemcpy(&sum, d_sum_out, sizeof(double), cudaMemcpyDeviceToHost));
    return sum;
}
//-----------------------------------------------------------
void CudaComputationBox::zero_mean_device(double* d_g)
{
    const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();
    double sum{0.0};

    ker_multi<double><<<N_BLOCKS, N_THREADS>>>(d_sum, d_dv, d_g, 1.0, this->total_grid);
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_sum, d_sum_out, this->total_grid);
    gpu_error_check(cudaMemcpy(&sum, d_sum_out, sizeof(double), cudaMemcpyDeviceToHost));
    ker_linear_scaling<double><<<N_BLOCKS, N_THREADS>>>(d_g, d_g, 1.0, -sum/this->volume, this->total_grid);
}