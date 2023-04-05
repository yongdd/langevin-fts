/*-------------------------------------------------------------
* This class defines Simulation Grids and Lengths parameters and provide
* methods that compute inner product in a given geometry.
*--------------------------------------------------------------*/
#include <iostream>
#include <thrust/reduce.h>
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
    gpu_error_check(cudaMalloc((void**)&d_dv, sizeof(double)*n_grid));
    gpu_error_check(cudaMemcpy(d_dv, dv,      sizeof(double)*n_grid,cudaMemcpyHostToDevice));

    // temporal storage
    gpu_error_check(cudaMalloc((void**)&d_multiple, sizeof(double)*n_grid));

    gpu_error_check(cudaMalloc((void**)&d_g, sizeof(double)*n_grid));
    gpu_error_check(cudaMalloc((void**)&d_h, sizeof(double)*n_grid));
    gpu_error_check(cudaMalloc((void**)&d_w, sizeof(double)*n_grid));

    // allocate memory for cub reduction sum
    gpu_error_check(cudaMalloc((void**)&d_sum, sizeof(double)*n_grid));
    gpu_error_check(cudaMalloc((void**)&d_sum_out, sizeof(double)));
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_sum, d_sum_out, n_grid);
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
    gpu_error_check(cudaMemcpy(d_dv, dv,  sizeof(double)*n_grid,cudaMemcpyHostToDevice));
}
// //-----------------------------------------------------------
// double CudaComputationBox::integral(double *g)
// {
//     const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
//     const int N_THREADS = CudaCommon::get_instance().get_n_threads();
//     double sum{0.0};

//     gpu_error_check(cudaMemcpy(d_g, g,  sizeof(double)*n_grid, cudaMemcpyHostToDevice));
//     multi_real<<<N_BLOCKS, N_THREADS>>>(d_sum, d_dv, d_g, 1.0, n_grid);
//     cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_sum, d_sum_out, n_grid);
//     gpu_error_check(cudaMemcpy(&sum, d_sum_out, sizeof(double), cudaMemcpyDeviceToHost));
//     return sum;
// }
// //-----------------------------------------------------------
// double CudaComputationBox::inner_product(double *g, double *h)
// {
//     const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
//     const int N_THREADS = CudaCommon::get_instance().get_n_threads();
//     double sum{0.0};

//     gpu_error_check(cudaMemcpy(d_g, g,  sizeof(double)*n_grid, cudaMemcpyHostToDevice));
//     gpu_error_check(cudaMemcpy(d_h, h,  sizeof(double)*n_grid, cudaMemcpyHostToDevice));
//     multi_real<<<N_BLOCKS, N_THREADS>>>(d_sum, d_g, d_h, 1.0, n_grid);
//     multi_real<<<N_BLOCKS, N_THREADS>>>(d_sum, d_dv, d_sum, 1.0, n_grid);
//     cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_sum, d_sum_out, n_grid);
//     gpu_error_check(cudaMemcpy(&sum, d_sum_out, sizeof(double), cudaMemcpyDeviceToHost));
//     return sum;
// }
// //-----------------------------------------------------------
// double CudaComputationBox::inner_product_inverse_weight(double *g, double *h, double *w)
// {
//     const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
//     const int N_THREADS = CudaCommon::get_instance().get_n_threads();
//     double sum{0.0};

//     gpu_error_check(cudaMemcpy(d_g, g,  sizeof(double)*n_grid, cudaMemcpyHostToDevice));
//     gpu_error_check(cudaMemcpy(d_h, h,  sizeof(double)*n_grid, cudaMemcpyHostToDevice));
//     gpu_error_check(cudaMemcpy(d_w, w,  sizeof(double)*n_grid, cudaMemcpyHostToDevice));
//     multi_real<<<N_BLOCKS, N_THREADS>>>(d_sum, d_g, d_h, 1.0, n_grid);
//     multi_real<<<N_BLOCKS, N_THREADS>>>(d_sum, d_dv, d_sum, 1.0, n_grid);
//     divide_real<<<N_BLOCKS, N_THREADS>>>(d_sum, d_sum, d_w, 1.0, n_grid);
//     cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_sum, d_sum_out, n_grid);
//     gpu_error_check(cudaMemcpy(&sum, d_sum_out, sizeof(double), cudaMemcpyDeviceToHost));
//     return sum;
// }
// //-----------------------------------------------------------
// double CudaComputationBox::multi_inner_product(int n_comp, double *g, double *h)
// {
//     double sum{0.0};
//     for(int n=0; n < n_comp; n++)
//     {
//         for(int i=0; i<n_grid; i++)
//             sum += dv[i]*g[i+n*n_grid]*h[i+n*n_grid];
//     }
//     return sum;
// }
// //-----------------------------------------------------------
// void CudaComputationBox::zero_mean(double *g)
// {
//     const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
//     const int N_THREADS = CudaCommon::get_instance().get_n_threads();
//     double sum{0.0};

//     gpu_error_check(cudaMemcpy(d_g, g,  sizeof(double)*n_grid, cudaMemcpyHostToDevice));
//     multi_real<<<N_BLOCKS, N_THREADS>>>(d_sum, d_dv, d_g, 1.0, n_grid);
//     cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_sum, d_sum_out, n_grid);
//     gpu_error_check(cudaMemcpy(&sum, d_sum_out, sizeof(double), cudaMemcpyDeviceToHost));

//     add_scalar_constant<<<N_BLOCKS, N_THREADS>>>(d_g, -sum/volume, n_grid);
//     gpu_error_check(cudaMemcpy(g, d_g, sizeof(double)*n_grid, cudaMemcpyDeviceToHost));
// }
//-----------------------------------------------------------
double CudaComputationBox::integral_gpu(double *d_g)
{
    const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();
    double sum{0};

    multi_real<<<N_BLOCKS, N_THREADS>>>(d_sum, d_g, d_dv, 1.0, n_grid);
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_sum, d_sum_out, n_grid);
    gpu_error_check(cudaMemcpy(&sum, d_sum_out, sizeof(double),cudaMemcpyDeviceToHost));
    return sum;
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
double CudaComputationBox::inner_product_inverse_weight_gpu(double *d_g, double *d_h, double *d_w)
{
    const int N_BLOCKS = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();

    multi_real <<<N_BLOCKS, N_THREADS>>>(d_multiple, d_g,        d_h, 1.0, n_grid);
    divide_real<<<N_BLOCKS, N_THREADS>>>(d_multiple, d_multiple, d_w, 1.0, n_grid);
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
