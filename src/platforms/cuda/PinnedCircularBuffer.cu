/**
 * @file PinnedCircularBuffer.cu
 * @brief CUDA pinned memory circular buffer implementation.
 *
 * Provides a circular buffer using CUDA pinned (page-locked) host memory
 * for efficient asynchronous host-device transfers. Used in memory-saving
 * mode to store propagator history on host while enabling fast GPU access.
 *
 * **Memory Benefits:**
 *
 * - cudaMallocHost allocates page-locked memory
 * - Enables DMA transfers without CPU involvement
 * - Allows overlapping computation with memory transfer
 *
 * **Operations:**
 *
 * - insert(): Add new array to buffer (circular overwrite)
 * - get_array(): Retrieve by relative index (0 = most recent)
 * - reset(): Clear buffer state
 *
 * **Template Instantiations:**
 *
 * - PinnedCircularBuffer<double>: Real field history
 * - PinnedCircularBuffer<std::complex<double>>: Complex field history
 *
 * @see CudaAndersonMixingReduceMemory for memory-saving mode
 * @see CudaCircularBuffer for device memory version
 */

#include <algorithm>
#include "CudaCommon.h"
#include "PinnedCircularBuffer.h"

template <typename T>
PinnedCircularBuffer<T>::PinnedCircularBuffer(int length, int width)
{
    this->length = length;
    this->width = width;
    this->start = 0;
    this->n_items = 0;

    elems = new T*[length];
    for(int i=0; i<length; i++)
    {
        gpu_error_check(cudaMallocHost((void**)&elems[i], sizeof(T)*width));
        for(int j=0; j<width; j++)
            elems[i][j] = 0.0;
    }
}
template <typename T>
PinnedCircularBuffer<T>::~PinnedCircularBuffer()
{
    for(int i=0; i<length; i++)
        cudaFreeHost(elems[i]);
    delete[] elems;
}
template <typename T>
void PinnedCircularBuffer<T>::reset()
{
    start = 0;
    n_items = 0;
}
template <typename T>
void PinnedCircularBuffer<T>::insert(T* new_arr)
{
    int i = (start+n_items)%length;
    for(int m=0; m<width; m++){
        elems[i][m] = new_arr[m];
    }
    if (n_items == length)
        start = (start+1)%length;
    n_items = std::min(n_items+1, length);
}
template <typename T>
T* PinnedCircularBuffer<T>::get_array(int n)
{
    int i = (start+n_items-n-1+length)%length;
    return elems[i];
}
// double* PinnedCircularBuffer<T>::operator[] (int n)
// {
//     int i = (start+n_items-n-1+length)%length;
//     return elems[i];
// }
// double PinnedCircularBuffer<T>::get(int n, int m)
// {
//     int i = (start+n_items-n-1+length)%length;
//     return elems[i][m];
// }


// Explicit template instantiation
template class PinnedCircularBuffer<double>;
template class PinnedCircularBuffer<std::complex<double>>;