#include <thrust/reduce.h>
#include "CudaCommon.h"

int main()
{
    try{
        const int M{32*32*32};
        const int N_GPUS = CudaCommon::get_instance().get_n_gpus();

        double array[32*32*32];
        for (int i=0; i<M; i++)
            array[i] = i;

        // Variables for cub reduction sum
        size_t temp_storage_bytes;
        double *d_temp_storage;
        double *d_array;
        double *d_array_sum;
        double array_sum;

        for(int gpu=0; gpu<N_GPUS; gpu++)
        {
            array_sum = 0.0;
            gpu_error_check(cudaSetDevice(gpu));
            gpu_error_check(cudaMalloc((void**)&d_array, sizeof(double)*M));
            gpu_error_check(cudaMalloc((void**)&d_array_sum, sizeof(double)*1));
            gpu_error_check(cudaMemcpy(d_array, array, sizeof(double)*M, cudaMemcpyHostToDevice));

            // It seems that cub::DeviceReduce::Sum changes temp_storage_bytes[gpu],
            // If d_temp_storage[gpu]=nullptr and temp_storage_bytes[gpu]=0.
            d_temp_storage = nullptr;
            temp_storage_bytes = 0;
            cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_array, d_array_sum, M);
            gpu_error_check(cudaMalloc(&d_temp_storage, temp_storage_bytes));

            gpu_error_check(cudaMemcpy(&array_sum, d_array_sum, sizeof(double), cudaMemcpyDeviceToHost));
            std::cout << "gpu, array_sum 0: " << gpu << ", " << array_sum << std::endl;

            cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_array, d_array_sum, M);
            gpu_error_check(cudaMemcpy(&array_sum, d_array_sum, sizeof(double), cudaMemcpyDeviceToHost));
            std::cout << "gpu, array_sum 1: " << gpu << ", " << array_sum << std::endl;

            cudaFree(d_array);
            cudaFree(d_array_sum);
            cudaFree(d_temp_storage);
        }
        return 0;
    }
    catch(std::exception& exc)
    {
        std::cout << exc.what() << std::endl;
        return -1;
    }
}