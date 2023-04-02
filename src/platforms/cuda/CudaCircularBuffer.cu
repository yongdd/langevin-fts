#include <algorithm>
#include "CudaCommon.h"
#include "CudaCircularBuffer.h"

CudaCircularBuffer::CudaCircularBuffer(int length, int width)
{
    this->length = length;
    this->width = width;
    this->start = 0;
    this->n_items = 0;

    d_elems = new double*[length];
    for(int i=0; i<length; i++)
    {
        gpu_error_check(cudaMalloc((void**)&d_elems[i], sizeof(double)*width));
        gpu_error_check(cudaMemset(d_elems[i], 0, sizeof(double)*width));
    }
}
CudaCircularBuffer::~CudaCircularBuffer()
{
    for(int i=0; i<length; i++)
        cudaFree(d_elems[i]);
    delete[] d_elems;
}
void CudaCircularBuffer::reset()
{
    start = 0;
    n_items = 0;
}
void CudaCircularBuffer::insert(double* new_arr)
{
    int i = (start+n_items)%length;
    gpu_error_check(cudaMemcpy(d_elems[i], new_arr, sizeof(double)*width, cudaMemcpyHostToDevice));
    if (n_items == length)
        start = (start+1)%length;
    n_items = min(n_items+1, length);
}
double* CudaCircularBuffer::get_array(int n)
{
    int i = (start+n_items-n-1+length)%length;
    return d_elems[i];
}