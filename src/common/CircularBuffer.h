/**
 * @file CircularBuffer.h
 * @brief Circular buffer data structure for fixed-size history storage.
 *
 * This header provides a CircularBuffer class that stores a fixed number of
 * array elements in a ring buffer. When the buffer is full, the oldest element
 * is overwritten by new insertions. This is used in Anderson Mixing to store
 * the history of previous field values and residuals.
 *
 * **Data Structure:**
 *
 * The buffer stores 'length' arrays, each of size 'width'. Memory layout:
 * ```
 * elems[0] -> [v0_0, v0_1, ..., v0_{width-1}]  // oldest (if full)
 * elems[1] -> [v1_0, v1_1, ..., v1_{width-1}]
 * ...
 * elems[length-1] -> [...]                      // newest (after length inserts)
 * ```
 *
 * After the buffer is full, new insertions overwrite the oldest entry,
 * and the 'start' pointer advances circularly.
 *
 * @see AndersonMixing for primary use case (storing field history)
 *
 * @example
 * @code
 * // Create buffer for 5 history vectors of length 1000
 * CircularBuffer<double> buffer(5, 1000);
 *
 * // Insert new vectors
 * double* vec1 = new double[1000];
 * // ... fill vec1 ...
 * buffer.insert(vec1);  // Copies vec1 into buffer
 *
 * // Access elements (0 = oldest, n_items-1 = newest)
 * double* oldest = buffer[0];
 * double* newest = buffer[buffer.n_items - 1];
 *
 * // Access single element
 * double value = buffer.get(2, 500);  // element 500 of vector 2
 *
 * // Reset buffer (clear all entries)
 * buffer.reset();
 * @endcode
 */

#ifndef CIRCULAR_BUFFER_H_
#define CIRCULAR_BUFFER_H_

#include <vector>

/**
 * @class CircularBuffer
 * @brief Fixed-size circular buffer storing arrays of elements.
 *
 * Implements a ring buffer that stores up to 'length' arrays, each of size
 * 'width'. When full, new insertions overwrite the oldest entry. Used by
 * Anderson Mixing to maintain a history of previous iterations.
 *
 * @tparam T Element type (typically double)
 *
 * **Indexing:**
 *
 * - get_array(0) or operator[](0) returns the oldest stored array
 * - get_array(n_items-1) returns the newest stored array
 * - Indices are mapped internally to handle wrap-around
 *
 * **Memory Management:**
 *
 * - Constructor allocates length * width elements
 * - Destructor frees all memory
 * - insert() copies the input array (does not take ownership)
 */
template <typename T>
class CircularBuffer
{
private:
    int length;    ///< Maximum number of arrays that can be stored
    int width;     ///< Size of each array (number of elements)
    int start;     ///< Index of the oldest element in the buffer
    int n_items;   ///< Current number of items stored (0 to length)
    std::vector<std::vector<T>> elems;  ///< Storage for arrays (length x width)

public:
    /**
     * @brief Construct a CircularBuffer with specified capacity.
     *
     * Allocates memory for 'length' arrays of 'width' elements each.
     *
     * @param length Maximum number of arrays to store
     * @param width  Size of each array
     *
     * @example
     * @code
     * // Buffer for 20 history vectors of 32768 grid points
     * CircularBuffer<double> history(20, 32768);
     * @endcode
     */
    CircularBuffer(int length, int width);

    /**
     * @brief Destructor. Frees all allocated memory.
     */
    ~CircularBuffer();

    /**
     * @brief Reset the buffer to empty state.
     *
     * Sets n_items = 0 and start = 0. Does not deallocate memory.
     * Call this before starting a new SCFT run to clear previous history.
     */
    void reset();

    /**
     * @brief Insert a new array into the buffer.
     *
     * Copies the contents of new_arr into the buffer. If the buffer is full,
     * the oldest entry is overwritten.
     *
     * @param new_arr Array to insert (must have 'width' elements)
     *
     * @note The input array is copied; the buffer does not take ownership.
     *
     * @example
     * @code
     * double* field = new double[width];
     * // ... compute field values ...
     * buffer.insert(field);  // Copies field into buffer
     * delete[] field;        // Safe to delete original
     * @endcode
     */
    void insert(T* new_arr);

    /**
     * @brief Get pointer to stored array by logical index.
     *
     * @param n Logical index (0 = oldest, n_items-1 = newest)
     * @return Pointer to the stored array
     *
     * @note Returns pointer to internal storage; do not free or reallocate.
     */
    T* get_array(int n);

    /**
     * @brief Array subscript operator (same as get_array).
     *
     * @param n Logical index (0 = oldest, n_items-1 = newest)
     * @return Pointer to the stored array
     */
    T* operator[] (int n);

    /**
     * @brief Get single element from stored array.
     *
     * @param n Logical array index (0 = oldest)
     * @param m Element index within array (0 to width-1)
     * @return Value at position m in array n
     *
     * @example
     * @code
     * double val = buffer.get(2, 100);  // Element 100 of 3rd oldest array
     * @endcode
     */
    T get(int n, int m);
};
#endif
