#ifndef CUDA_CIRCULAR_BUFFER_H_
#define CUDA_CIRCULAR_BUFFER_H_

/*-----------------------------------------------------------------
! A circular buffer stores data in the GPU memory.
!-----------------------------------------------------------------*/
class CudaCircularBuffer
{
private:
    int length; // maximum number of elements
    int width;  // size of each elements
    int start;  // index of oldest elements
    int n_items;   // index at which to write new element
    double** d_elems;

public:
    CudaCircularBuffer(int length, int width);
    ~CudaCircularBuffer();
    void reset();
    void insert(double* d_new_arr);
    double* get_array(int n);
};

#endif



