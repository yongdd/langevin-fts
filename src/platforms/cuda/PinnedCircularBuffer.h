#ifndef PINNED_CIRCULAR_BUFFER_H_
#define PINNED_CIRCULAR_BUFFER_H_

/*-----------------------------------------------------------------
! A circular buffer stores data in the pinned host memory.
!-----------------------------------------------------------------*/
class PinnedCircularBuffer
{
private:
    int length; // maximum number of elements
    int width;  // size of each elements
    int start;  // index of oldest elements
    int n_items;   // index at which to write new element
    double** elems;

public:
    PinnedCircularBuffer(int length, int width);
    ~PinnedCircularBuffer();
    void reset();
    void insert(double* new_arr);
    double* get_array(int n);
};

#endif



