#include <algorithm>
#include "CircularBuffer.h"

CircularBuffer::CircularBuffer(int length, int width)
{
    this->length = length;
    this->width = width;
    this->start = 0;
    this->n_items = 0;

    elems = new double[length*width];
    for(int i=0; i<length*width; i++){
        elems[i] = 0.0;
    }
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
    int i = (start+n_items-n-1+length)%length;
    return &elems[i*width];
}
double* CircularBuffer::operator[] (int n)
{
    int i = (start+n_items-n-1+length)%length;
    return &elems[i*width];
}
double CircularBuffer::get(int n, int m)
{
    int i = (start+n_items-n-1+length)%length;
    return elems[i*width + m];
}
