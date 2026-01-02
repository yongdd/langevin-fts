/**
 * @file CircularBuffer.cpp
 * @brief Implementation of CircularBuffer ring buffer template.
 *
 * Provides a fixed-size ring buffer for storing field history in
 * Anderson mixing. When the buffer is full, new insertions overwrite
 * the oldest entries.
 *
 * **Memory Layout:**
 *
 * Stores `length` arrays of `width` elements each:
 * - elems[i][j]: Element j of history entry i
 * - Indexing wraps around using modular arithmetic
 *
 * **Template Instantiations:**
 *
 * - CircularBuffer<double>: Real field history
 * - CircularBuffer<std::complex<double>>: Complex field history
 */

#include <algorithm>
#include <complex>

#include "CircularBuffer.h"

/**
 * @brief Construct circular buffer with given dimensions.
 *
 * Allocates length × width array and initializes to zero.
 *
 * @param length Number of history entries (buffer capacity)
 * @param width  Size of each entry (typically n_var for fields)
 */
template <typename T>
CircularBuffer<T>::CircularBuffer(int length, int width)
{
    this->length = length;
    this->width = width;
    this->start = 0;
    this->n_items = 0;

    elems = new T*[length];
    for(int i=0; i<length; i++)
    {
        elems[i] = new T[width];
        for(int j=0; j<width; j++)
            elems[i][j] = 0.0;
    }
}
/**
 * @brief Destructor. Frees all allocated memory.
 */
template <typename T>
CircularBuffer<T>::~CircularBuffer()
{
    for(int i=0; i<length; i++)
        delete[] elems[i];
    delete[] elems;
}

/**
 * @brief Reset buffer to empty state.
 *
 * Clears logical contents without deallocating memory.
 * Next insertion will start at position 0.
 */
template <typename T>
void CircularBuffer<T>::reset()
{
    start = 0;
    n_items = 0;
}

/**
 * @brief Insert new array into the buffer.
 *
 * Copies new_arr into the next available slot. If buffer is full,
 * overwrites the oldest entry and advances start pointer.
 *
 * @param new_arr Array of size `width` to insert
 */
template <typename T>
void CircularBuffer<T>::insert(T* new_arr)
{
    int i = (start+n_items)%length;
    for(int m=0; m<width; m++){
        elems[i][m] = new_arr[m];
    }
    if (n_items == length)
        start = (start+1)%length;
    n_items = std::min(n_items+1, length);
}

/**
 * @brief Get pointer to history entry n.
 *
 * Index 0 is the most recent entry, 1 is second most recent, etc.
 *
 * @param n History index (0 = most recent)
 * @return Pointer to array of size `width`
 */
template <typename T>
T* CircularBuffer<T>::get_array(int n)
{
    int i = (start+n_items-n-1+length)%length;
    return elems[i];
}

/**
 * @brief Array subscript operator (alias for get_array).
 *
 * @param n History index (0 = most recent)
 * @return Pointer to array of size `width`
 */
template <typename T>
T* CircularBuffer<T>::operator[] (int n)
{
    int i = (start+n_items-n-1+length)%length;
    return elems[i];
}

/**
 * @brief Get single element from history.
 *
 * @param n History index (0 = most recent)
 * @param m Element index within array
 * @return Value at position (n, m)
 */
template <typename T>
T CircularBuffer<T>::get(int n, int m)
{
    int i = (start+n_items-n-1+length)%length;
    return elems[i][m];
}

// Explicit template instantiation
template class CircularBuffer<double>;
template class CircularBuffer<std::complex<double>>;