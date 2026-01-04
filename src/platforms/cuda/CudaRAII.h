/**
 * @file CudaRAII.h
 * @brief RAII wrappers for CUDA resources.
 *
 * Provides exception-safe, automatic resource management for CUDA resources:
 * - CudaDeviceMemory: Device memory (cudaMalloc/cudaFree)
 * - CudaPinnedMemory: Pinned host memory (cudaMallocHost/cudaFreeHost)
 * - CudaStream: CUDA streams (cudaStreamCreate/cudaStreamDestroy)
 *
 * These wrappers ensure proper cleanup even when exceptions are thrown,
 * following the RAII (Resource Acquisition Is Initialization) pattern.
 *
 * @example
 * @code
 * // Device memory - automatically freed when out of scope
 * CudaDeviceMemory<double> d_array(1000);
 * kernel<<<blocks, threads>>>(d_array.get(), 1000);
 *
 * // Pinned memory for efficient host-device transfers
 * CudaPinnedMemory<double> h_pinned(1000);
 * cudaMemcpy(d_array.get(), h_pinned.get(), 1000*sizeof(double), cudaMemcpyHostToDevice);
 *
 * // CUDA stream - automatically destroyed
 * CudaStream stream;
 * kernel<<<blocks, threads, 0, stream.get()>>>(d_array.get());
 * @endcode
 */

#ifndef CUDA_RAII_H_
#define CUDA_RAII_H_

#include <cuda_runtime.h>
#include "CudaCommon.h"

/**
 * @class CudaDeviceMemory
 * @brief RAII wrapper for CUDA device memory.
 *
 * Manages device memory allocation and deallocation automatically.
 * Memory is freed when the object goes out of scope.
 *
 * @tparam T Element type (double, cuDoubleComplex, etc.)
 */
template<typename T>
class CudaDeviceMemory
{
private:
    T* ptr_ = nullptr;
    size_t size_ = 0;

public:
    /**
     * @brief Default constructor (no allocation).
     */
    CudaDeviceMemory() = default;

    /**
     * @brief Allocate device memory.
     * @param n Number of elements of type T to allocate
     * @throws Exception if cudaMalloc fails
     */
    explicit CudaDeviceMemory(size_t n) : size_(n)
    {
        if (n > 0)
        {
            gpu_error_check(cudaMalloc((void**)&ptr_, n * sizeof(T)));
        }
    }

    /**
     * @brief Destructor - frees device memory.
     */
    ~CudaDeviceMemory()
    {
        if (ptr_ != nullptr)
        {
            cudaFree(ptr_);
            ptr_ = nullptr;
        }
    }

    // Disable copy
    CudaDeviceMemory(const CudaDeviceMemory&) = delete;
    CudaDeviceMemory& operator=(const CudaDeviceMemory&) = delete;

    // Enable move
    CudaDeviceMemory(CudaDeviceMemory&& other) noexcept
        : ptr_(other.ptr_), size_(other.size_)
    {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    CudaDeviceMemory& operator=(CudaDeviceMemory&& other) noexcept
    {
        if (this != &other)
        {
            if (ptr_ != nullptr)
            {
                cudaFree(ptr_);
            }
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    /**
     * @brief Get raw pointer.
     * @return Pointer to device memory
     */
    T* get() const { return ptr_; }

    /**
     * @brief Get number of elements.
     * @return Size in elements (not bytes)
     */
    size_t size() const { return size_; }

    /**
     * @brief Check if memory is allocated.
     * @return true if ptr_ is not null
     */
    explicit operator bool() const { return ptr_ != nullptr; }

    /**
     * @brief Release ownership of the pointer.
     * @return The raw pointer (caller takes ownership)
     */
    T* release()
    {
        T* tmp = ptr_;
        ptr_ = nullptr;
        size_ = 0;
        return tmp;
    }

    /**
     * @brief Reset to new allocation.
     * @param n New size (0 to just free)
     */
    void reset(size_t n = 0)
    {
        if (ptr_ != nullptr)
        {
            cudaFree(ptr_);
            ptr_ = nullptr;
        }
        size_ = n;
        if (n > 0)
        {
            gpu_error_check(cudaMalloc((void**)&ptr_, n * sizeof(T)));
        }
    }
};

/**
 * @class CudaPinnedMemory
 * @brief RAII wrapper for CUDA pinned (page-locked) host memory.
 *
 * Manages pinned memory allocation and deallocation automatically.
 * Pinned memory enables faster host-device transfers and async operations.
 *
 * @tparam T Element type
 */
template<typename T>
class CudaPinnedMemory
{
private:
    T* ptr_ = nullptr;
    size_t size_ = 0;

public:
    /**
     * @brief Default constructor (no allocation).
     */
    CudaPinnedMemory() = default;

    /**
     * @brief Allocate pinned memory.
     * @param n Number of elements of type T to allocate
     * @throws Exception if cudaMallocHost fails
     */
    explicit CudaPinnedMemory(size_t n) : size_(n)
    {
        if (n > 0)
        {
            gpu_error_check(cudaMallocHost((void**)&ptr_, n * sizeof(T)));
        }
    }

    /**
     * @brief Destructor - frees pinned memory.
     */
    ~CudaPinnedMemory()
    {
        if (ptr_ != nullptr)
        {
            cudaFreeHost(ptr_);
            ptr_ = nullptr;
        }
    }

    // Disable copy
    CudaPinnedMemory(const CudaPinnedMemory&) = delete;
    CudaPinnedMemory& operator=(const CudaPinnedMemory&) = delete;

    // Enable move
    CudaPinnedMemory(CudaPinnedMemory&& other) noexcept
        : ptr_(other.ptr_), size_(other.size_)
    {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    CudaPinnedMemory& operator=(CudaPinnedMemory&& other) noexcept
    {
        if (this != &other)
        {
            if (ptr_ != nullptr)
            {
                cudaFreeHost(ptr_);
            }
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    /**
     * @brief Get raw pointer.
     * @return Pointer to pinned memory
     */
    T* get() const { return ptr_; }

    /**
     * @brief Get number of elements.
     * @return Size in elements (not bytes)
     */
    size_t size() const { return size_; }

    /**
     * @brief Array access operator.
     * @param i Index
     * @return Reference to element
     */
    T& operator[](size_t i) { return ptr_[i]; }
    const T& operator[](size_t i) const { return ptr_[i]; }

    /**
     * @brief Check if memory is allocated.
     * @return true if ptr_ is not null
     */
    explicit operator bool() const { return ptr_ != nullptr; }

    /**
     * @brief Release ownership of the pointer.
     * @return The raw pointer (caller takes ownership)
     */
    T* release()
    {
        T* tmp = ptr_;
        ptr_ = nullptr;
        size_ = 0;
        return tmp;
    }

    /**
     * @brief Reset to new allocation.
     * @param n New size (0 to just free)
     */
    void reset(size_t n = 0)
    {
        if (ptr_ != nullptr)
        {
            cudaFreeHost(ptr_);
            ptr_ = nullptr;
        }
        size_ = n;
        if (n > 0)
        {
            gpu_error_check(cudaMallocHost((void**)&ptr_, n * sizeof(T)));
        }
    }
};

/**
 * @class CudaStream
 * @brief RAII wrapper for CUDA streams.
 *
 * Manages CUDA stream creation and destruction automatically.
 * The stream is destroyed when the object goes out of scope.
 */
class CudaStream
{
private:
    cudaStream_t stream_ = nullptr;

public:
    /**
     * @brief Create a new CUDA stream.
     * @throws Exception if cudaStreamCreate fails
     */
    CudaStream()
    {
        gpu_error_check(cudaStreamCreate(&stream_));
    }

    /**
     * @brief Destructor - destroys the stream.
     */
    ~CudaStream()
    {
        if (stream_ != nullptr)
        {
            cudaStreamDestroy(stream_);
            stream_ = nullptr;
        }
    }

    // Disable copy
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;

    // Enable move
    CudaStream(CudaStream&& other) noexcept : stream_(other.stream_)
    {
        other.stream_ = nullptr;
    }

    CudaStream& operator=(CudaStream&& other) noexcept
    {
        if (this != &other)
        {
            if (stream_ != nullptr)
            {
                cudaStreamDestroy(stream_);
            }
            stream_ = other.stream_;
            other.stream_ = nullptr;
        }
        return *this;
    }

    /**
     * @brief Get the CUDA stream handle.
     * @return cudaStream_t handle
     */
    cudaStream_t get() const { return stream_; }

    /**
     * @brief Implicit conversion to cudaStream_t.
     * @return cudaStream_t handle
     */
    operator cudaStream_t() const { return stream_; }

    /**
     * @brief Synchronize the stream.
     * @throws Exception if cudaStreamSynchronize fails
     */
    void synchronize()
    {
        gpu_error_check(cudaStreamSynchronize(stream_));
    }
};

#endif // CUDA_RAII_H_
