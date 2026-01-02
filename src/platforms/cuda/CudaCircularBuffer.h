/**
 * @file CudaCircularBuffer.h
 * @brief Circular buffer for storing arrays in GPU device memory.
 *
 * This header provides CudaCircularBuffer, a ring buffer data structure
 * that stores fixed-size arrays in GPU device memory. Used by Anderson
 * mixing to maintain history of field values on the GPU.
 *
 * **GPU Memory Usage:**
 *
 * Allocates length × width × sizeof(T) bytes of GPU memory upfront.
 * Arrays are stored contiguously in device memory for efficient access.
 *
 * **Usage Pattern:**
 *
 * @code
 * CudaCircularBuffer<double> buffer(10, 1024);  // 10 arrays of 1024 elements
 *
 * // Insert new array (copies to GPU)
 * buffer.insert(d_new_field);
 *
 * // Access history
 * double* d_prev = buffer.get_array(0);  // Most recent
 * double* d_prev2 = buffer.get_array(1); // Second most recent
 * @endcode
 *
 * @see CircularBuffer for CPU version
 * @see PinnedCircularBuffer for pinned host memory version
 * @see CudaAndersonMixing for usage in SCFT iteration
 */

#ifndef CUDA_CIRCULAR_BUFFER_H_
#define CUDA_CIRCULAR_BUFFER_H_

#include "CudaCommon.h"

/**
 * @class CudaCircularBuffer
 * @brief Ring buffer storing arrays in GPU device memory.
 *
 * Maintains a fixed-capacity circular buffer of GPU arrays for
 * efficient history storage in iterative algorithms.
 *
 * @tparam T Numeric type (double or std::complex<double>)
 *
 * **Memory Layout:**
 *
 * - d_elems: Array of device pointers
 * - Each element points to a width-sized array in device memory
 * - When full, oldest element is overwritten
 *
 * **Performance:**
 *
 * - Insert: O(1) device-to-device copy
 * - Access: O(1) pointer lookup
 * - No host-device transfers during normal operation
 */
template <typename T>
class CudaCircularBuffer
{
private:
    int length;    ///< Maximum number of arrays (history depth)
    int width;     ///< Size of each array (n_grid)
    int start;     ///< Index of oldest element
    int n_items;   ///< Current number of items stored
    CuDeviceData<T>** d_elems;  ///< Array of device pointers

public:
    /**
     * @brief Construct circular buffer with given capacity.
     *
     * @param length Maximum number of arrays to store
     * @param width  Size of each array (number of elements)
     */
    CudaCircularBuffer(int length, int width);

    /**
     * @brief Destructor. Frees all GPU memory.
     */
    ~CudaCircularBuffer();

    /**
     * @brief Clear buffer (logical reset, memory retained).
     *
     * Resets indices but does not deallocate memory.
     */
    void reset();

    /**
     * @brief Insert new array at front of buffer.
     *
     * If buffer is full, overwrites oldest entry.
     * Performs device-to-device copy.
     *
     * @param d_new_arr Source array in device memory (size width)
     */
    void insert(CuDeviceData<T>* d_new_arr);

    /**
     * @brief Get array at position n in history.
     *
     * @param n History index (0 = most recent, 1 = second most recent, ...)
     * @return Device pointer to array
     */
    CuDeviceData<T>* get_array(int n);
};

#endif



