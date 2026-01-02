/**
 * @file PinnedCircularBuffer.h
 * @brief Circular buffer using CUDA pinned (page-locked) host memory.
 *
 * This header provides PinnedCircularBuffer, a ring buffer that stores
 * arrays in pinned host memory for efficient asynchronous transfers
 * between host and GPU device.
 *
 * **Pinned Memory Benefits:**
 *
 * - Enables asynchronous cudaMemcpyAsync transfers
 * - Overlaps data transfer with kernel execution
 * - Higher bandwidth than pageable memory
 *
 * **Memory-Saving Mode:**
 *
 * Used by CudaAndersonMixingReduceMemory and
 * CudaComputationReduceMemoryContinuous/Discrete to store propagator
 * history in host memory when GPU memory is limited.
 *
 * **Usage Pattern:**
 *
 * @code
 * PinnedCircularBuffer<double> buffer(10, 1024);
 *
 * // Insert from device (async copy to host)
 * buffer.insert(host_copy_of_field);
 *
 * // Get for async transfer back to device
 * double* pinned_ptr = buffer.get_array(0);
 * cudaMemcpyAsync(d_dest, pinned_ptr, size, cudaMemcpyHostToDevice, stream);
 * @endcode
 *
 * @see CudaCircularBuffer for device memory version
 * @see CudaAndersonMixingReduceMemory for usage
 */

#ifndef PINNED_CIRCULAR_BUFFER_H_
#define PINNED_CIRCULAR_BUFFER_H_

/**
 * @class PinnedCircularBuffer
 * @brief Ring buffer using CUDA pinned (page-locked) host memory.
 *
 * Stores arrays in pinned memory for efficient async transfers.
 * Used in memory-saving mode to keep data on host while minimizing
 * transfer overhead through overlapping.
 *
 * @tparam T Numeric type (double or std::complex<double>)
 *
 * **Memory Allocation:**
 *
 * Uses cudaMallocHost() for pinned memory allocation:
 * - Page-locked memory not swapped to disk
 * - DMA-capable for direct GPU access
 * - Limited resource (use judiciously)
 *
 * **Performance vs Device Memory:**
 *
 * - Slower than CudaCircularBuffer for access
 * - Enables hiding transfer latency with computation
 * - Reduces GPU memory footprint significantly
 */
template <typename T>
class PinnedCircularBuffer
{
private:
    int length;   ///< Maximum number of arrays
    int width;    ///< Size of each array
    int start;    ///< Index of oldest element
    int n_items;  ///< Current number of items
    T** elems;    ///< Array of pinned memory pointers

public:
    /**
     * @brief Construct pinned buffer with given capacity.
     *
     * Allocates length arrays in pinned host memory using cudaMallocHost.
     *
     * @param length Maximum number of arrays to store
     * @param width  Size of each array (number of elements)
     */
    PinnedCircularBuffer(int length, int width);

    /**
     * @brief Destructor. Frees pinned memory using cudaFreeHost.
     */
    ~PinnedCircularBuffer();

    /**
     * @brief Clear buffer (logical reset).
     */
    void reset();

    /**
     * @brief Insert new array at front of buffer.
     *
     * Copies data into pinned memory. If buffer is full,
     * overwrites oldest entry.
     *
     * @param new_arr Source array (can be host or device pointer)
     */
    void insert(T* new_arr);

    /**
     * @brief Get array at position n in history.
     *
     * @param n History index (0 = most recent)
     * @return Pinned host memory pointer
     */
    T* get_array(int n);
};

#endif



