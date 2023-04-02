#include <algorithm>
#include "CircularBuffer.h"

CircularBuffer::CircularBuffer(int length, int width)
{
    this->length = length;
    this->width = width;
    this->start = 0;
    this->n_items = 0;

    elems = new double*[length];
    for(int i=0; i<length; i++)
    {
        elems[i] = new double[width];
        for(int j=0; j<width; j++)
            elems[i][j] = 0.0;
    }
}
CircularBuffer::~CircularBuffer()
{
    for(int i=0; i<length; i++)
        delete[] elems[i];
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
        elems[i][m] = new_arr[m];
    }
    if (n_items == length)
        start = (start+1)%length;
    n_items = std::min(n_items+1, length);
}
double* CircularBuffer::get_array(int n)
{
    int i = (start+n_items-n-1+length)%length;
    return elems[i];
}
double* CircularBuffer::operator[] (int n)
{
    int i = (start+n_items-n-1+length)%length;
    return elems[i];
}
double CircularBuffer::get(int n, int m)
{
    int i = (start+n_items-n-1+length)%length;
    return elems[i][m];
}
