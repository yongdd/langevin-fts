#include <map>
#include "CudaCommon.h"

int main()
{
    try{
        const int M{128*128*128};

        double *d_progpa_temp;
        double * propagator;
        std::string key = "A";
        int max_n_segment = 200;

        // allocate gpu memory
        gpu_error_check(cudaMalloc((void**)&d_progpa_temp, sizeof(double)*M));
        // allocate pinned host memory
        gpu_error_check(cudaMallocHost((void**)&propagator, sizeof(double)*(max_n_segment+1)*M));

        for(int n=0; n<=max_n_segment; n++)
        {
            if(n%20==0)
            {
                std::cout << n << ", " << max_n_segment << std::endl;
            }

            gpu_error_check(cudaMemcpy(&propagator[n*M], d_progpa_temp, sizeof(double)*M,
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