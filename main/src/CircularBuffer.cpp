#include <algorithm>
#include "CircularBuffer.h"

CircularBuffer::CircularBuffer(int length, int width)
{
    this->length = length;
    this->width = width;
    this->start = 0;
    this->n_items = 0;

    elems = new double[length*width];
}
CircularBuffer::~CircularBuffer()
{
    delete[] elems;
}
void CircularBuffer::reset()
{
    start = 0;
    n_items = 0;
}
void CircularBuffer::insert(double* new_arr)
{
    int i = (start+n_items)%length;
    for(int m=0; m<width; m++){
        elems[i*width + m] = new_arr[m];
    }
    if (n_items == length)
        start = (start+1)%length;
    n_items = std::min(n_items+1, length);
}
double* CircularBuffer::get_array(int n)
{
    int i = (start+n)%length;
    return &elems[i*width];
}
double CircularBuffer::get(int n, int m)
{
    int i = (start+n)%length;
    return elems[i*width + m];
}
double CircularBuffer::get_sym(int n, int m)
{
    int i = (start+std::max(n,m))%length ;
    return elems[i*width + abs(n-m)];
}
