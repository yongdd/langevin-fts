/*-----------------------------------------------------------------
! A circular buffer is a data structure that uses a single,
! fixed-size buffer as if it were connected end-to-end.
! Each elements are 1-dimensional real array.
!-----------------------------------------------------------------*/

#ifndef CIRCULAR_BUFFER_H_
#define CIRCULAR_BUFFER_H_

template <typename T>
class CircularBuffer
{
private:
    int length; // maximum number of elements
    int width;  // size of each elements
    int start;  // index of oldest elements
    int n_items;   // index at which to write new element
    T** elems;

public:
    CircularBuffer(int length, int width);
    ~CircularBuffer();
    void reset();
    void insert(T* new_arr);
    T* get_array(int n);
    T* operator[] (int n);
    T get(int n, int m);
};
#endif
