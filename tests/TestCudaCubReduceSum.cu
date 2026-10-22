#include <cmath>
#include <vector>
#include <cub/device/device_reduce.cuh>
#include "CudaCommon.h"

int main()
{
    try
    {
        const int M = 32*32*32;
        std::vector<double> input(M);
        for (int i = 0; i < M; ++i)
            input[i] = i;

        CudaDeviceBuffer<double> d_input(M), d_sum(1);
        CUDA_CHECK(cudaMemcpy(d_input.get(), input.data(), sizeof(double)*M, cudaMemcpyHostToDevice));
        size_t bytes = 0;
        CUDA_CHECK(cub::DeviceReduce::Sum(nullptr, bytes, d_input.get(), d_sum.get(), M));
        CudaDeviceBuffer<unsigned char> scratch(bytes);
        CUDA_CHECK(cub::DeviceReduce::Sum(scratch.get(), bytes, d_input.get(), d_sum.get(), M));
        double result;
        CUDA_CHECK(cudaMemcpy(&result, d_sum.get(), sizeof(double), cudaMemcpyDeviceToHost));
        const double expected = static_cast<double>(M)*(M-1)/2.0;
        if (!std::isfinite(result) || result != expected)
        {
            std::cerr << "Reduction: " << result << ", expected " << expected << std::endl;
            return -1;
        }
        return 0;
    }
    catch (std::exception& exc)
    {
        std::cerr << exc.what() << std::endl;
        return -1;
    }
}
