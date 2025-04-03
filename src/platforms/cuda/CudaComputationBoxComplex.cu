/*-------------------------------------------------------------
* This class defines simulation grids and Lengths parameters and provide
* methods that compute inner product in a given geometry.
*--------------------------------------------------------------*/
#include <iostream>
#include <complex>
#include <cuda_runtime.h>
#include <cub/device/device_reduce.cuh>
#include <cuComplex.h>

#include "CudaComputationBoxComplex.h"
#include "CudaCommon.h"


// Define custom reduction operator for ftsComplex
struct ComplexSumOp {
    __device__ __forceinline__
    ftsComplex operator()(const ftsComplex& a, const ftsComplex& b) const {
        ftsComplex result;
        result.x = a.x + b.x;
        result.y = a.y + b.y;
        return result;
    }
};

//----------------- Constructor -----------------------------
CudaComputationBoxComplex::CudaComputationBoxComplex(
    std::vector<int> nx, std::vector<double> lx, std::vector<std::string> bc, const double* mask)
    : ComputationBox(nx, lx, bc, mask)
{
    initialize();
}
void CudaComputationBoxComplex::initialize()
{
    gpu_error_check(cudaMalloc((void**)&d_dv,  sizeof(double)*this->total_grid));
    gpu_error_check(cudaMemcpy(d_dv, dv, sizeof(double)*this->total_grid, cudaMemcpyHostToDevice));

    // Temporal storage
    gpu_error_check(cudaMalloc((void**)&d_multiple, sizeof(ftsComplex)*this->total_grid));

    // Allocate memory for cub reduction sum
    gpu_error_check(cudaMalloc((void**)&d_sum, sizeof(ftsComplex)*this->total_grid));
    gpu_error_check(cudaMalloc((void**)&d_sum_out, sizeof(ftsComplex)));

    // Determine temporary storage size for cub reduction
    // Query the required temporary storage size
    ftsComplex zero = {0.0, 0.0};
    cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_sum, d_sum_out, this->total_grid, ComplexSumOp(), zero);
    gpu_error_check(cudaMalloc(&d_temp_storage, temp_storage_bytes));
}
//----------------- Destructor -----------------------------
CudaComputationBoxComplex::~CudaComputationBoxComplex()
{
    cudaFree(d_dv);
    cudaFree(d_multiple);
    cudaFree(d_sum);
    cudaFree(d_sum_out);
    cudaFree(d_temp_storage);
}
//-----------------------------------------------------------
void CudaComputationBoxComplex::set_lx(std::vector<double> new_lx)
{
    ComputationBox::set_lx(new_lx);
    gpu_error_check(cudaMemcpy(d_dv, this->dv, sizeof(double)*this->total_grid,cudaMemcpyHostToDevice));
}
//-----------------------------------------------------------
std::complex<double> CudaComputationBoxComplex::integral_device(const ftsComplex *d_g)
{
    const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();
    ftsComplex sum{0};
    
    ker_multi_complex_real<<<N_BLOCKS, N_THREADS>>>(d_sum, d_g, d_dv, 1.0, this->total_grid);
    ftsComplex zero = {0.0, 0.0};
    cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_sum, d_sum_out, this->total_grid, ComplexSumOp(), zero);
    gpu_error_check(cudaMemcpy(&sum, d_sum_out, sizeof(ftsComplex), cudaMemcpyDeviceToHost));
    return std::complex<double>{sum.x, sum.y};
}
//-----------------------------------------------------------
std::complex<double> CudaComputationBoxComplex::inner_product_device(const ftsComplex* d_g, const ftsComplex* d_h)
{
    const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();
    ftsComplex sum{0.0};

    ker_multi<ftsComplex><<<N_BLOCKS, N_THREADS>>>(d_sum, d_g, d_h, 1.0, this->total_grid);
    ker_multi_complex_real<<<N_BLOCKS, N_THREADS>>>(d_sum, d_sum, d_dv, 1.0, this->total_grid);
    ftsComplex zero = {0.0, 0.0};
    cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_sum, d_sum_out, this->total_grid, ComplexSumOp(), zero);
    gpu_error_check(cudaMemcpy(&sum, d_sum_out, sizeof(ftsComplex), cudaMemcpyDeviceToHost));
    return std::complex<double>{sum.x, sum.y};
}
//-----------------------------------------------------------
std::complex<double> CudaComputationBoxComplex::inner_product_inverse_weight_device(const ftsComplex* d_g, const ftsComplex* d_h, const ftsComplex* d_w)
{
    const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();
    ftsComplex sum{0.0};

    ker_multi<ftsComplex><<<N_BLOCKS, N_THREADS>>>(d_sum, d_g, d_h, 1.0, this->total_grid);
    ker_divide<ftsComplex><<<N_BLOCKS, N_THREADS>>>(d_sum, d_sum, d_w, 1.0, this->total_grid);
    ker_multi_complex_real<<<N_BLOCKS, N_THREADS>>>(d_sum, d_sum, d_dv, 1.0, this->total_grid);
    ftsComplex zero = {0.0, 0.0};
    cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_sum, d_sum_out, this->total_grid, ComplexSumOp(), zero);
    gpu_error_check(cudaMemcpy(&sum, d_sum_out, sizeof(ftsComplex), cudaMemcpyDeviceToHost));
    return std::complex<double>{sum.x, sum.y};
}
//-----------------------------------------------------------
std::complex<double> CudaComputationBoxComplex::multi_inner_product_device(int n_comp, const ftsComplex* d_g, const ftsComplex* d_h)
{
    const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();
    ftsComplex sum{0.0};

    ker_mutiple_multi<ftsComplex><<<N_BLOCKS, N_THREADS>>>(n_comp, d_sum, d_g, d_h, 1.0, this->total_grid);
    ker_multi_complex_real<<<N_BLOCKS, N_THREADS>>>(d_sum, d_sum, d_dv, 1.0, this->total_grid);
    ftsComplex zero = {0.0, 0.0};
    cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_sum, d_sum_out, this->total_grid, ComplexSumOp(), zero);
    gpu_error_check(cudaMemcpy(&sum, d_sum_out, sizeof(ftsComplex), cudaMemcpyDeviceToHost));
    return std::complex<double>{sum.x, sum.y};
}
//-----------------------------------------------------------
void CudaComputationBoxComplex::zero_mean_device(ftsComplex* d_g)
{
    const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();
    ftsComplex sum{0.0};

    ker_multi_complex_real<<<N_BLOCKS, N_THREADS>>>(d_sum, d_sum, d_dv, 1.0, this->total_grid);
    ftsComplex zero = {0.0, 0.0};
    cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_sum, d_sum_out, this->total_grid, ComplexSumOp(), zero);
    gpu_error_check(cudaMemcpy(&sum, d_sum_out, sizeof(ftsComplex), cudaMemcpyDeviceToHost));
    sum.x = -sum.x/this->volume;
    sum.y = -sum.y/this->volume;
    ker_linear_scaling<ftsComplex><<<N_BLOCKS, N_THREADS>>>(d_g, d_g, 1.0, sum, this->total_grid);
}