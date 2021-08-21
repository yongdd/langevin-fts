#include <algorithm>
#include "CudaCircularBuffer.h"

CudaCircularBuffer::CudaCircularBuffer(int length, int width)
{
    this->length = length;
    this->width = width;
    this->start = 0;
    this->n_items = 0;

    cudaMalloc((void**)&elems_d, sizeof(double)*length*width);
}
CudaCircularBuffer::~CudaCircularBuffer()
{
    cudaFree(elems_d);
}
void CudaCircularBuffer::reset()
{
    start = 0;
    n_items = 0;
}
void CudaCircularBuffer::insert(double* new_arr)
{
    int i = (start+n_items)%length;
    cudaMemcpy(&elems_d[i*width], new_arr, sizeof(double)*width, cudaMemcpyHostToDevice);
    if (n_items == length)
        start = (start+1)%length;
    n_items = min(n_items+1, length);
}
double* CudaCircularBuffer::get_array(int n)
{
    int i = (start+n_items-n-1+length)%length;
    return &elems_d[i*width];
}
