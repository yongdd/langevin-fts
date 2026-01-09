/**
 * @file MemoryUtils.h
 * @brief Memory management utilities for safe array allocation.
 *
 * Provides RAII wrappers for dynamic arrays that automatically manage memory
 * while maintaining compatibility with raw pointer interfaces required by
 * FFT libraries and CUDA.
 *
 * **Design Philosophy:**
 *
 * This library uses high-performance computing patterns that require raw
 * pointer access for FFT libraries (MKL, cuFFT) and CUDA kernels. The utilities
 * here provide:
 *
 * 1. RAII wrappers that own memory and automatically deallocate
 * 2. Raw pointer access for library interoperability
 * 3. Clear ownership semantics
 *
 * **Usage:**
 *
 * @code
 * #include "MemoryUtils.h"
 *
 * // Allocate a 1D array
 * auto array = memory::make_array<double>(1000);
 * double* ptr = array.get();  // Raw pointer for FFT/CUDA
 *
 * // Allocate a 2D array (row-major)
 * auto array2d = memory::make_array_2d<double>(100, 64);
 * double** ptrs = array2d.row_pointers();  // Row pointers
 * double* data = array2d.data();           // Contiguous data
 *
 * // Memory automatically freed when array goes out of scope
 * @endcode
 *
 * **Thread Safety:**
 *
 * Array allocation is thread-safe. Array access follows standard C++ rules.
 */

#ifndef MEMORY_UTILS_H_
#define MEMORY_UTILS_H_

#include <memory>
#include <vector>
#include <cstddef>
#include <stdexcept>

namespace memory {

/**
 * @brief RAII wrapper for 1D arrays with raw pointer access.
 *
 * This class owns a dynamically allocated array and provides:
 * - Automatic deallocation via RAII
 * - Raw pointer access for library interoperability
 * - Move semantics (no copy)
 *
 * @tparam T Element type
 */
template<typename T>
class Array {
public:
    /**
     * @brief Construct an empty array.
     */
    Array() noexcept : data_(nullptr), size_(0) {}

    /**
     * @brief Construct and allocate an array.
     * @param size Number of elements
     */
    explicit Array(size_t size) : data_(new T[size]()), size_(size) {}

    /**
     * @brief Destructor - automatically frees memory.
     */
    ~Array() {
        delete[] data_;
    }

    // Move semantics
    Array(Array&& other) noexcept : data_(other.data_), size_(other.size_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }

    Array& operator=(Array&& other) noexcept {
        if (this != &other) {
            delete[] data_;
            data_ = other.data_;
            size_ = other.size_;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    // No copy
    Array(const Array&) = delete;
    Array& operator=(const Array&) = delete;

    /**
     * @brief Get raw pointer to data.
     *
     * Use this for FFT libraries, CUDA kernels, or other C interfaces.
     *
     * @return Raw pointer to first element, or nullptr if empty
     */
    T* get() noexcept { return data_; }
    const T* get() const noexcept { return data_; }

    /**
     * @brief Get size of array.
     */
    size_t size() const noexcept { return size_; }

    /**
     * @brief Check if array is allocated.
     */
    explicit operator bool() const noexcept { return data_ != nullptr; }

    /**
     * @brief Element access.
     */
    T& operator[](size_t i) { return data_[i]; }
    const T& operator[](size_t i) const { return data_[i]; }

    /**
     * @brief Release ownership and return raw pointer.
     *
     * After calling this, the Array no longer owns the memory.
     * The caller is responsible for deallocation.
     *
     * @return Raw pointer to data
     */
    T* release() noexcept {
        T* ptr = data_;
        data_ = nullptr;
        size_ = 0;
        return ptr;
    }

    /**
     * @brief Reset to empty or reallocate.
     */
    void reset(size_t new_size = 0) {
        delete[] data_;
        if (new_size > 0) {
            data_ = new T[new_size]();
            size_ = new_size;
        } else {
            data_ = nullptr;
            size_ = 0;
        }
    }

private:
    T* data_;
    size_t size_;
};

/**
 * @brief RAII wrapper for 2D arrays with contiguous storage.
 *
 * Stores data contiguously for cache efficiency, with separate row pointers
 * for 2D indexing. This layout is optimal for:
 * - Cache-friendly row-major access
 * - Passing to functions expecting T**
 * - Single-allocation efficiency
 *
 * @tparam T Element type
 */
template<typename T>
class Array2D {
public:
    /**
     * @brief Construct an empty 2D array.
     */
    Array2D() noexcept : data_(nullptr), rows_(nullptr), n_rows_(0), n_cols_(0) {}

    /**
     * @brief Construct and allocate a 2D array.
     * @param rows Number of rows
     * @param cols Number of columns
     */
    Array2D(size_t rows, size_t cols)
        : n_rows_(rows), n_cols_(cols)
    {
        if (rows == 0 || cols == 0) {
            data_ = nullptr;
            rows_ = nullptr;
            n_rows_ = 0;
            n_cols_ = 0;
            return;
        }

        // Allocate contiguous data
        data_ = new T[rows * cols]();

        // Allocate and set up row pointers
        rows_ = new T*[rows];
        for (size_t i = 0; i < rows; ++i) {
            rows_[i] = data_ + i * cols;
        }
    }

    /**
     * @brief Destructor - automatically frees memory.
     */
    ~Array2D() {
        delete[] rows_;
        delete[] data_;
    }

    // Move semantics
    Array2D(Array2D&& other) noexcept
        : data_(other.data_), rows_(other.rows_),
          n_rows_(other.n_rows_), n_cols_(other.n_cols_)
    {
        other.data_ = nullptr;
        other.rows_ = nullptr;
        other.n_rows_ = 0;
        other.n_cols_ = 0;
    }

    Array2D& operator=(Array2D&& other) noexcept {
        if (this != &other) {
            delete[] rows_;
            delete[] data_;
            data_ = other.data_;
            rows_ = other.rows_;
            n_rows_ = other.n_rows_;
            n_cols_ = other.n_cols_;
            other.data_ = nullptr;
            other.rows_ = nullptr;
            other.n_rows_ = 0;
            other.n_cols_ = 0;
        }
        return *this;
    }

    // No copy
    Array2D(const Array2D&) = delete;
    Array2D& operator=(const Array2D&) = delete;

    /**
     * @brief Get raw pointer to contiguous data.
     */
    T* data() noexcept { return data_; }
    const T* data() const noexcept { return data_; }

    /**
     * @brief Get row pointer array (T**).
     *
     * Use this for functions expecting T** parameter.
     */
    T** row_pointers() noexcept { return rows_; }
    T* const* row_pointers() const noexcept { return rows_; }

    /**
     * @brief Get pointer to specific row.
     */
    T* row(size_t i) noexcept { return rows_[i]; }
    const T* row(size_t i) const noexcept { return rows_[i]; }

    /**
     * @brief Get dimensions.
     */
    size_t rows() const noexcept { return n_rows_; }
    size_t cols() const noexcept { return n_cols_; }
    size_t total_size() const noexcept { return n_rows_ * n_cols_; }

    /**
     * @brief Check if array is allocated.
     */
    explicit operator bool() const noexcept { return data_ != nullptr; }

    /**
     * @brief 2D element access.
     */
    T& operator()(size_t row, size_t col) { return rows_[row][col]; }
    const T& operator()(size_t row, size_t col) const { return rows_[row][col]; }

private:
    T* data_;      // Contiguous data storage
    T** rows_;     // Row pointers
    size_t n_rows_;
    size_t n_cols_;
};

/**
 * @brief Create a 1D array.
 *
 * @tparam T Element type
 * @param size Number of elements
 * @return Array owning the allocated memory
 *
 * @code
 * auto arr = memory::make_array<double>(1000);
 * double* ptr = arr.get();  // Use with FFT
 * @endcode
 */
template<typename T>
Array<T> make_array(size_t size) {
    return Array<T>(size);
}

/**
 * @brief Create a 2D array with contiguous storage.
 *
 * @tparam T Element type
 * @param rows Number of rows
 * @param cols Number of columns
 * @return Array2D owning the allocated memory
 *
 * @code
 * auto arr = memory::make_array_2d<double>(100, 64);
 * double** ptrs = arr.row_pointers();
 * @endcode
 */
template<typename T>
Array2D<T> make_array_2d(size_t rows, size_t cols) {
    return Array2D<T>(rows, cols);
}

/**
 * @brief Aligned array for SIMD operations.
 *
 * Allocates memory with specified alignment for vectorized operations.
 * Uses standard aligned_alloc when available.
 *
 * @tparam T Element type
 */
template<typename T>
class AlignedArray {
public:
    static constexpr size_t DEFAULT_ALIGNMENT = 64;  // Cache line size

    /**
     * @brief Construct an empty array.
     */
    AlignedArray() noexcept : data_(nullptr), size_(0) {}

    /**
     * @brief Construct and allocate an aligned array.
     * @param size Number of elements
     * @param alignment Alignment in bytes (must be power of 2)
     */
    explicit AlignedArray(size_t size, size_t alignment = DEFAULT_ALIGNMENT)
        : size_(size)
    {
        if (size == 0) {
            data_ = nullptr;
            return;
        }

        // Round up size to be multiple of alignment
        size_t byte_size = size * sizeof(T);
        size_t aligned_size = ((byte_size + alignment - 1) / alignment) * alignment;

#if defined(_MSC_VER)
        data_ = static_cast<T*>(_aligned_malloc(aligned_size, alignment));
#else
        data_ = static_cast<T*>(std::aligned_alloc(alignment, aligned_size));
#endif

        if (!data_) {
            throw std::bad_alloc();
        }

        // Zero-initialize
        for (size_t i = 0; i < size; ++i) {
            new (&data_[i]) T();
        }
    }

    /**
     * @brief Destructor - automatically frees memory.
     */
    ~AlignedArray() {
        if (data_) {
#if defined(_MSC_VER)
            _aligned_free(data_);
#else
            std::free(data_);
#endif
        }
    }

    // Move semantics
    AlignedArray(AlignedArray&& other) noexcept
        : data_(other.data_), size_(other.size_)
    {
        other.data_ = nullptr;
        other.size_ = 0;
    }

    AlignedArray& operator=(AlignedArray&& other) noexcept {
        if (this != &other) {
            if (data_) {
#if defined(_MSC_VER)
                _aligned_free(data_);
#else
                std::free(data_);
#endif
            }
            data_ = other.data_;
            size_ = other.size_;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    // No copy
    AlignedArray(const AlignedArray&) = delete;
    AlignedArray& operator=(const AlignedArray&) = delete;

    T* get() noexcept { return data_; }
    const T* get() const noexcept { return data_; }
    size_t size() const noexcept { return size_; }
    explicit operator bool() const noexcept { return data_ != nullptr; }
    T& operator[](size_t i) { return data_[i]; }
    const T& operator[](size_t i) const { return data_[i]; }

private:
    T* data_;
    size_t size_;
};

/**
 * @brief Create an aligned array.
 *
 * @tparam T Element type
 * @param size Number of elements
 * @param alignment Alignment in bytes
 * @return AlignedArray owning the allocated memory
 */
template<typename T>
AlignedArray<T> make_aligned_array(size_t size, size_t alignment = AlignedArray<T>::DEFAULT_ALIGNMENT) {
    return AlignedArray<T>(size, alignment);
}

} // namespace memory

#endif // MEMORY_UTILS_H_
