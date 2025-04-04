#include <algorithm>

#include "CudaCommon.h"
#include "CudaCircularBuffer.h"

template<typename T>
CudaCircularBuffer<T>::CudaCircularBuffer(int length, int width)
{
    this->length = length;
    this->width = width;
    this->start = 0;
    this->n_items = 0;

    d_elems = new CuDeviceData<T>*[length];
    for(int i=0; i<length; i++)
    {
        gpu_error_check(cudaMalloc((void**)&d_elems[i], sizeof(CuDeviceData<T>)*width));
        gpu_error_check(cudaMemset(d_elems[i], 0, sizeof(CuDeviceData<T>)*width));
    }
}
template<typename T>
CudaCircularBuffer<T>::~CudaCircularBuffer<double>()
{
    for(int i=0; i<length; i++)
        cudaFree(d_elems[i]);
    delete[] d_elems;
}
template<typename T>
void CudaCircularBuffer<T>::reset()
{
    start = 0;
    n_items = 0;
}
template<typename T>
void CudaCircularBuffer<T>::insert(CuDeviceData<T>* d_new_arr)
{
    int i = (start+n_items)%length;
    gpu_error_check(cudaMemcpy(d_elems[i], d_new_arr, sizeof(CuDeviceData<T>)*width, cudaMemcpyDeviceToDevice));
    if (n_items == length)
        start = (start+1)%length;
    n_items = min(n_items+1, length);
}
template<typename T>
CuDeviceData<T>* CudaCircularBuffer<T>::get_array(int n)
{
    int i = (start+n_items-n-1+length)%length;
    return d_elems[i];
}

// Explicit template instantiation
template class CudaCircularBuffer<double>;
template class CudaCircularBuffer<std::complex<double>>;