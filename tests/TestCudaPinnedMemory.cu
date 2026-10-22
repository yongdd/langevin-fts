#include <vector>
#include "CudaCommon.h"

int main()
{
    try
    {
        const int M = 32*32*32;
        const int max_n_segment = 10;
        std::vector<double> input(M);
        for (int i = 0; i < M; ++i)
            input[i] = (i % 97) - 48.0;
        CudaDeviceBuffer<double> device(M);
        CudaPinnedBuffer<double> pinned((max_n_segment+1)*M);
        CUDA_CHECK(cudaMemcpy(device.get(), input.data(), sizeof(double)*M, cudaMemcpyHostToDevice));
        for (int n = 0; n <= max_n_segment; ++n)
        {
            double* slice = pinned.get() + n*M;
            CUDA_CHECK(cudaMemcpy(slice, device.get(), sizeof(double)*M, cudaMemcpyDeviceToHost));
            for (int i = 0; i < M; ++i)
                if (slice[i] != input[i])
                {
                    std::cerr << "Pinned copy differs at segment " << n << ", point " << i << std::endl;
                    return -1;
                }
        }
        return 0;
    }
    catch (std::exception& exc)
    {
        std::cerr << exc.what() << std::endl;
        return -1;
    }
}
