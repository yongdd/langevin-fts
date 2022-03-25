/*-------------------------------------------------------------
* This class defines simulation box parameters and provide
* methods that compute inner product in a given geometry.
*--------------------------------------------------------------*/
#include <iostream>
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include "CudaSimulationBox.h"
#include "CudaCommon.h"

//----------------- Constructor -----------------------------
CudaSimulationBox::CudaSimulationBox(
    std::vector<int> nx, std::vector<double> lx)
    : SimulationBox(nx, lx)
{
    initialize();
}
void CudaSimulationBox::initialize()
{
    cudaMalloc((void**)&dv_d, sizeof(double)*n_grid);
    cudaMemcpy(dv_d, dv,      sizeof(double)*n_grid,cudaMemcpyHostToDevice);

    // temporal storage
    sum = new double[n_grid];
    cudaMalloc((void**)&sum_d, sizeof(double)*n_grid);
    cudaMalloc((void**)&multiple_d, sizeof(double)*n_grid);
}
//----------------- Destructor -----------------------------
CudaSimulationBox::~CudaSimulationBox()
{
    delete[] sum;
    cudaFree(dv_d);
    cudaFree(sum_d);
    cudaFree(multiple_d);
}
//-----------------------------------------------------------
void CudaSimulationBox::set_lx(std::vector<double> new_lx)
{
    SimulationBox::set_lx(new_lx);
    cudaMemcpy(dv_d, dv,  sizeof(double)*n_grid,cudaMemcpyHostToDevice);
}
//-----------------------------------------------------------
double CudaSimulationBox::integral_gpu(double *g_d)
{
    const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();
    thrust::device_ptr<double> temp_gpu_ptr(sum_d);
    
    multi_real<<<N_BLOCKS, N_THREADS>>>(sum_d, g_d, dv_d, 1.0, n_grid);
    return thrust::reduce(temp_gpu_ptr, temp_gpu_ptr + n_grid);
}
//-----------------------------------------------------------
double CudaSimulationBox::inner_product_gpu(double *g_d, double *h_d)
{
    const int N_BLOCKS = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();

    multi_real<<<N_BLOCKS, N_THREADS>>>(multiple_d, g_d, h_d, 1.0, n_grid);
    return CudaSimulationBox::integral_gpu(multiple_d);
}
//-----------------------------------------------------------
double CudaSimulationBox::mutiple_inner_product_gpu(int n_comp, double *g_d, double *h_d)
{
    const int N_BLOCKS = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();

    mutiple_multi_real<<<N_BLOCKS, N_THREADS>>>(n_comp, multiple_d, g_d, h_d, 1.0, n_grid);
    return CudaSimulationBox::integral_gpu(multiple_d);
}
