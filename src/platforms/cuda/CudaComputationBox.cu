/*-------------------------------------------------------------
* This class defines Simulation Grids and Lengths parameters and provide
* methods that compute inner product in a given geometry.
*--------------------------------------------------------------*/
#define THRUST_IGNORE_DEPRECATED_CPP_DIALECT
#define CUB_IGNORE_DEPRECATED_CPP_DIALECT

#include <iostream>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include "CudaComputationBox.h"
#include "CudaCommon.h"

//----------------- Constructor -----------------------------
CudaComputationBox::CudaComputationBox(
    std::vector<int> nx, std::vector<double> lx)
    : ComputationBox(nx, lx)
{
    initialize();
}
void CudaComputationBox::initialize()
{
    sum = new double[n_grid];
    gpu_error_check(cudaMalloc((void**)&d_dv, sizeof(double)*n_grid));
    gpu_error_check(cudaMemcpy(d_dv, dv,      sizeof(double)*n_grid,cudaMemcpyHostToDevice));

    // temporal storage
    gpu_error_check(cudaMalloc((void**)&d_sum, sizeof(double)*n_grid));
    gpu_error_check(cudaMalloc((void**)&d_multiple, sizeof(double)*n_grid));
}
//----------------- Destructor -----------------------------
CudaComputationBox::~CudaComputationBox()
{
    delete[] sum;
    cudaFree(d_dv);
    cudaFree(d_sum);
    cudaFree(d_multiple);
}
//-----------------------------------------------------------
void CudaComputationBox::set_lx(std::vector<double> new_lx)
{
    ComputationBox::set_lx(new_lx);
    gpu_error_check(cudaMemcpy(d_dv, dv,  sizeof(double)*n_grid,cudaMemcpyHostToDevice));
}
//-----------------------------------------------------------
double CudaComputationBox::integral_gpu(double *d_g)
{
    const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();
    thrust::device_ptr<double> temp_gpu_ptr(d_sum);
    
    multi_real<<<N_BLOCKS, N_THREADS>>>(d_sum, d_g, d_dv, 1.0, n_grid);
    return thrust::reduce(temp_gpu_ptr, temp_gpu_ptr + n_grid);
}
//-----------------------------------------------------------
double CudaComputationBox::inner_product_gpu(double *d_g, double *d_h)
{
    const int N_BLOCKS = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();

    multi_real<<<N_BLOCKS, N_THREADS>>>(d_multiple, d_g, d_h, 1.0, n_grid);
    return CudaComputationBox::integral_gpu(d_multiple);
}
//-----------------------------------------------------------
double CudaComputationBox::mutiple_inner_product_gpu(int n_comp, double *d_g, double *d_h)
{
    const int N_BLOCKS = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();

    mutiple_multi_real<<<N_BLOCKS, N_THREADS>>>(n_comp, d_multiple, d_g, d_h, 1.0, n_grid);
    return CudaComputationBox::integral_gpu(d_multiple);
}