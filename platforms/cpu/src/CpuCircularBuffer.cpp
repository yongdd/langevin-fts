#include <algorithm>
#include "CpuCircularBuffer.h"

CpuCircularBuffer::CpuCircularBuffer(int length, int width)
{
    this->length = length;
    this->width = width;
    this->start = 0;
    this->n_items = 0;

    elems = new double[length*width];
}
CpuCircularBuffer::~CpuCircularBuffer()
{
    delete[] elems;
}
void CpuCircularBuffer::reset()
{
    start = 0;
    n_items = 0;
}
void CpuCircularBuffer::insert(double* new_arr)
{
    int i = (start+n_items)%length;
    for(int m=0; m<width; m++){
        elems[i*width + m] = new_arr[m];
    }
    if (n_items == length)
        start = (start+1)%length;
    n_items = std::min(n_items+1, length);
}
double* CpuCircularBuffer::get_array(int n)
{
    int i = (start+n)%length;
    return &elems[i*width];
}
double CpuCircularBuffer::get(int n, int m)
{
    int i = (start+n)%length;
    return elems[i*width + m];
}
double CpuCircularBuffer::get_sym(int n, int m)
{
    int i = (start+std::max(n,m))%length ;
    return elems[i*width + abs(n-m)];
}
