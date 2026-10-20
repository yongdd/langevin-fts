#include <map>
#include "CudaCommon.h"

int main()
{
    try{
        const int M{32*32*32};

        double *d_progpa_temp;
        double * propagator;
        std::string key = "A";
        int max_n_segment = 10;

        // Allocate gpu memory
        CUDA_CHECK(cudaMalloc((void**)&d_progpa_temp, sizeof(double)*M));
        // Allocate pinned host memory
        CUDA_CHECK(cudaMallocHost((void**)&propagator, sizeof(double)*(max_n_segment+1)*M));

        for(int n=0; n<=max_n_segment; n++)
        {
            if(n%20==0)
            {
                std::cout << n << ", " << max_n_segment << std::endl;
            }

            CUDA_CHECK(cudaMemcpy(&propagator[n*M], d_progpa_temp, sizeof(double)*M,
                cudaMemcpyDeviceToHost));
        }

        cudaFreeHost(propagator);
        cudaFree(d_progpa_temp);

        return 0;
    }
    catch(std::exception& exc)
    {
        std::cout << exc.what() << std::endl;
        return -1;
    }
}