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