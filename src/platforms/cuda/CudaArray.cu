/**
 * @file CudaArray.cu
 * @brief CUDA implementation of GPU memory array management.
 *
 * Provides GPU-side memory allocation and arithmetic operations using
 * CUDA kernels. Implements the abstract Array interface for device memory.
 *
 * **Memory Management:**
 *
 * - Uses cudaMalloc/cudaFree for device memory
 * - Tracks device ID for multi-GPU validation
 * - Supports host-device and device-device transfers
 *
 * **Arithmetic Operations:**
 *
 * Uses CUDA kernels from CudaCommon for element-wise operations:
 * - ker_lin_comb: Linear combination dst = a*src1 + b*src2
 * - ker_multi: Element-wise multiplication
 * - ker_divide: Element-wise division
 * - ker_linear_scaling: dst = a*src + b
 *
 * @see Array for abstract interface
 * @see CpuArray for CPU equivalent
 * @see CudaCommon for kernel definitions
 */

#include <iostream>
#include "CudaCommon.h"
#include "CudaArray.h"

/**
 * @brief Construct CUDA array with given size on current device.
 *
 * @param new_size Number of elements to allocate
 */
CudaArray::CudaArray(unsigned new_size)
{
    gpu_error_check(cudaMalloc((void**)&this->d_data, sizeof(double)*new_size));
    this->size = new_size;
    this->device="cuda";

    int device_id;
    gpu_error_check(cudaGetDevice(&device_id));
    this->device_id= device_id;
}
CudaArray::CudaArray(double* new_data, unsigned int new_size) : CudaArray(new_size)
{
    gpu_error_check(cudaMemcpy(d_data, new_data, sizeof(double)*new_size, cudaMemcpyHostToDevice));
}
CudaArray::CudaArray(const Array& array) : CudaArray(array.get_size())
{
    unsigned int new_size = array.get_size();
    double *d_new_data = array.get_ptr();
    gpu_error_check(cudaMemcpy(d_data, d_new_data, sizeof(double)*new_size, cudaMemcpyDeviceToDevice));
}
CudaArray::~CudaArray()
{
    if (this->d_data != nullptr)
    {
        cudaFree(this->d_data);
        this->size = 0;
        // std::cout << "Memory has been deallocated" << std::endl;
    }
}
void CudaArray::operator=(const Array& arr)
{
    unsigned int arr_size = arr.get_size();
    double *d_arr_data = arr.get_ptr();

    if (this->size != arr_size)
    {
        throw_with_line_number("Sizes of arrays ("
            + std::to_string(this->size) + ", "
            + std::to_string(arr_size) + ") do not match.");
    }
    gpu_error_check(cudaMemcpy(d_data, d_arr_data, sizeof(double)*arr_size, cudaMemcpyDeviceToDevice));
}
void CudaArray::set_data(double * arr_data, unsigned int arr_size)
{
    if (this->size != arr_size)
    {
        throw_with_line_number("Sizes of arrays ("
            + std::to_string(this->size) + ", "
            + std::to_string(arr_size) + ") do not match.");
    }
    gpu_error_check(cudaMemcpy(d_data, arr_data, sizeof(double)*arr_size, cudaMemcpyHostToDevice));
}
std::vector<double> CudaArray::to_vector() const
{
    double temp_arr[size];
    gpu_error_check(cudaMemcpy(temp_arr, d_data, sizeof(double)*size, cudaMemcpyDeviceToHost));

    std::vector<double> vec(temp_arr, temp_arr + size); 
    return vec;
}
double* CudaArray::get_ptr() const
{
    return this->d_data;
}
unsigned int CudaArray::get_size() const
{
    return this->size;
}
double CudaArray::operator[](unsigned int i) const
{
    if (size <= i)
        throw_with_line_number("Index [" + std::to_string(i) + "] is out of bound.");
    double element;
    gpu_error_check(cudaMemcpy(&element, &d_data[i], sizeof(double), cudaMemcpyDeviceToHost));
    return element;
}

// Arithmetic operations with two arrays
void CudaArray::add(const Array& src_1, const Array& src_2)
{
    unsigned int src1_size = src_1.get_size();
    unsigned int src2_size = src_2.get_size();
    double *d_src1_data = src_1.get_ptr();
    double *d_src2_data = src_2.get_ptr();

    if (this->size != src1_size || this->size != src2_size)
    {
        throw_with_line_number("Sizes of arrays ("
            + std::to_string(this->size) + ", "
            + std::to_string(src1_size)  + ", "
            + std::to_string(src2_size)  + ") do not match.");
    }

    const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();
    int device_id;
    gpu_error_check(cudaGetDevice(&device_id));
    if (this->device_id != device_id)
        throw_with_line_number("Device id does not match.");

    ker_lin_comb<<<N_BLOCKS, N_THREADS>>>(this->d_data, 1.0, d_src1_data, 1.0, d_src2_data, this->size);
}
void CudaArray::subtract(const Array& src_1, const Array& src_2)
{
    unsigned int src1_size = src_1.get_size();
    unsigned int src2_size = src_2.get_size();
    double *d_src1_data = src_1.get_ptr();
    double *d_src2_data = src_2.get_ptr();

    if (this->size != src1_size || this->size != src2_size)
    {
        throw_with_line_number("Sizes of arrays ("
            + std::to_string(this->size) + ", "
            + std::to_string(src1_size)  + ", "
            + std::to_string(src2_size)  + ") do not match.");
    }

    const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();
    int device_id;
    gpu_error_check(cudaGetDevice(&device_id));
    if (this->device_id != device_id)
        throw_with_line_number("Device id does not match.");

    ker_lin_comb<<<N_BLOCKS, N_THREADS>>>(this->d_data, 1.0, d_src1_data, -1.0, d_src2_data, this->size);
}
void CudaArray::multiply(const Array& src_1, const Array& src_2)
{
    unsigned int src1_size = src_1.get_size();
    unsigned int src2_size = src_2.get_size();
    double *d_src1_data = src_1.get_ptr();
    double *d_src2_data = src_2.get_ptr();

    if (this->size != src1_size || this->size != src2_size)
    {
        throw_with_line_number("Sizes of arrays ("
            + std::to_string(this->size) + ", "
            + std::to_string(src1_size)  + ", "
            + std::to_string(src2_size)  + ") do not match.");
    }

    const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();
    int device_id;
    gpu_error_check(cudaGetDevice(&device_id));
    if (this->device_id != device_id)
        throw_with_line_number("Device id does not match.");

    ker_multi<<<N_BLOCKS, N_THREADS>>>(this->d_data, d_src1_data, d_src2_data, 1.0, this->size);
}
void CudaArray::divide(const Array& src_1, const Array& src_2)
{
    unsigned int src1_size = src_1.get_size();
    unsigned int src2_size = src_2.get_size();
    double *d_src1_data = src_1.get_ptr();
    double *d_src2_data = src_2.get_ptr();

    if (this->size != src1_size || this->size != src2_size)
    {
        throw_with_line_number("Sizes of arrays ("
            + std::to_string(this->size) + ", "
            + std::to_string(src1_size)  + ", "
            + std::to_string(src2_size)  + ") do not match.");
    }

    const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();
    int device_id;
    gpu_error_check(cudaGetDevice(&device_id));
    if (this->device_id != device_id)
        throw_with_line_number("Device id does not match.");

    ker_divide<<<N_BLOCKS, N_THREADS>>>(this->d_data, d_src1_data, d_src2_data, 1.0, this->size);
}
// Arithmetic operations with an array and a float number
void CudaArray::linear_scaling(const Array& src, const double a, const double b)
{
    unsigned int src_size = src.get_size();
    double *d_src_data = src.get_ptr();

    if (this->size != src_size)
    {
        throw_with_line_number("Sizes of arrays ("
            + std::to_string(this->size) + ", "
            + std::to_string(src_size) + ") do not match.");
    }

    const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();
    int device_id;
    gpu_error_check(cudaGetDevice(&device_id));
    if (this->device_id != device_id)
        throw_with_line_number("Device id does not match.");

    ker_linear_scaling<<<N_BLOCKS, N_THREADS>>>(this->d_data, d_src_data, a, b, this->size);
}