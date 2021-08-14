/*-----------------------------------------------------------------
! A circular buffer is a data structure that uses a single,
! fixed-size buffer as if it were connected end-to-end.
! Each elements are 1-dmensional real array.
!-----------------------------------------------------------------*/

#ifndef CPU_CIRCULAR_BUFFER_H_
#define CPU_CIRCULAR_BUFFER_H_

class CircularBuffer
{
private:
    int length; // maximum number of elements
    int width;  // size of each elements
    int start;  // index of oldest elements
    int n_items;   // index at which to write new element
    double* elems;

public:
    CircularBuffer(int length, int width);
    ~CircularBuffer();
    void reset();
    void insert(double* new_arr);
    double* get_array(int n);
    double get(int n, int m);
    double get_sym(int n, int m);
};
#endif
