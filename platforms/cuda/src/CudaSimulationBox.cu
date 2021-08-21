/*-------------------------------------------------------------
* This class defines simulation box parameters and provide
* methods that compute inner product in a given geometry.
*--------------------------------------------------------------*/
#include <iostream>
#include "CudaSimulationBox.h"
#include "CudaCommon.h"

//----------------- Constructor -----------------------------
CudaSimulationBox::CudaSimulationBox(
    std::array<int,3> nx, std::array<double,3> lx)
    : SimulationBox(nx, lx)
{
    cudaMalloc((void**)&dv_d, sizeof(double)*MM);
    cudaMemcpy(dv_d, dv,      sizeof(double)*MM,cudaMemcpyHostToDevice);

    // temporal storage
    sum = new double[MM];
    cudaMalloc((void**)&sum_d, sizeof(double)*MM);
    cudaMalloc((void**)&multiple_d, sizeof(double)*MM);
    
    const int N_BLOCKS = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();

    if (N_BLOCKS > MM)
    {
        std::cout<< "'the number of grids'{" <<N_BLOCKS ;
        std::cout<< "} should not be less 'the number of grids'{";
        std::cout<< MM << "}." << std::endl;
        exit(-1);
    }
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
double CudaSimulationBox::integral_gpu(double *g_d)
{
    const int N_BLOCKS = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();

    switch(N_THREADS)
    {
    case 1024:
        multiplyReductionKernel<1024>
        <<<N_BLOCKS, N_THREADS, N_THREADS*sizeof(double)>>>
        (dv_d, g_d, sum_d, MM);
        break;
    case 512:
        multiplyReductionKernel<512>
        <<<N_BLOCKS, N_THREADS, N_THREADS*sizeof(double)>>>
        (dv_d, g_d, sum_d, MM);
        break;
    case 256:
        multiplyReductionKernel<256>
        <<<N_BLOCKS, N_THREADS, N_THREADS*sizeof(double)>>>
        (dv_d, g_d, sum_d, MM);
        break;
    case 128:
        multiplyReductionKernel<128>
        <<<N_BLOCKS, N_THREADS, N_THREADS*sizeof(double)>>>
        (dv_d, g_d, sum_d, MM);
        break;
    case 64:
        multiplyReductionKernel<64>
        <<<N_BLOCKS, N_THREADS, N_THREADS*sizeof(double)>>>
        (dv_d, g_d, sum_d, MM);
        break;
    case 32:
        multiplyReductionKernel<32>
        <<<N_BLOCKS, N_THREADS, N_THREADS*sizeof(double)>>>
        (dv_d, g_d, sum_d, MM);
        break;
    case 16:
        multiplyReductionKernel<16>
        <<<N_BLOCKS, N_THREADS, N_THREADS*sizeof(double)>>>
        (dv_d, g_d, sum_d, MM);
        break;
    case 8:
        multiplyReductionKernel<8>
        <<<N_BLOCKS, N_THREADS, N_THREADS*sizeof(double)>>>
        (dv_d, g_d, sum_d, MM);
        break;
    case 4:
        multiplyReductionKernel<4>
        <<<N_BLOCKS, N_THREADS, N_THREADS*sizeof(double)>>>
        (dv_d, g_d, sum_d, MM);
        break;
    case 2:
        multiplyReductionKernel<2>
        <<<N_BLOCKS, N_THREADS, N_THREADS*sizeof(double)>>>
        (dv_d, g_d, sum_d, MM);
        break;
    case 1:
        multiplyReductionKernel<1>
        <<<N_BLOCKS, N_THREADS, N_THREADS*sizeof(double)>>>
        (dv_d, g_d, sum_d, MM);
        break;
    }
    cudaMemcpy(sum, sum_d, sizeof(double)*N_BLOCKS,cudaMemcpyDeviceToHost);
    double total{0.0};
    for(int i=0; i<N_BLOCKS; i++)
    {
        total += sum[i];
    }
    return total;
}
//-----------------------------------------------------------
double CudaSimulationBox::inner_product_gpu(double *g_d, double *h_d)
{
    const int N_BLOCKS = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads(); 
    
    multiReal<<<N_BLOCKS, N_THREADS>>>(multiple_d, g_d, h_d, 1.0, MM);
    return CudaSimulationBox::integral_gpu(multiple_d);
}
//-----------------------------------------------------------
double CudaSimulationBox::mutiple_inner_product_gpu(int n_comp, double *g_d, double *h_d)
{
    const int N_BLOCKS = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads(); 
    
    mutipleMultiReal<<<N_BLOCKS, N_THREADS>>>(n_comp, multiple_d, g_d, h_d, 1.0, MM);
    return CudaSimulationBox::integral_gpu(multiple_d);
}
