/**
 * @file CudaComputationBox.cu
 * @brief CUDA implementation of computation box with GPU operations.
 *
 * Extends ComputationBox with GPU-accelerated integral and inner product
 * operations using CUB parallel reductions.
 *
 * **GPU Operations:**
 *
 * - integral_device(): Volume-weighted sum on GPU
 * - inner_product_device(): <f,g> = ∫ f(r)g(r) dv
 * - zero_mean_device(): Remove mean from field
 *
 * **CUB Reductions:**
 *
 * Uses CUB library for high-performance parallel reductions:
 * - DeviceReduce::Sum for real fields
 * - Custom ComplexSumOp for complex fields
 *
 * **Memory Layout:**
 *
 * - d_dv: Device copy of volume elements
 * - d_sum, d_sum_out: Temporary arrays for reduction
 * - d_temp_storage: CUB workspace
 *
 * **Template Instantiations:**
 *
 * - CudaComputationBox<double>: Real fields
 * - CudaComputationBox<std::complex<double>>: Complex fields
 *
 * @see ComputationBox for base class interface
 */

#include <iostream>
#include <complex>
#include <cuda_runtime.h>
#include <cub/device/device_reduce.cuh>
#include <cuComplex.h>

#include "CudaComputationBox.h"
#include "CudaCommon.h"
#include "SpaceGroup.h"

/**
 * @brief Construct CUDA computation box with device memory.
 *
 * @param nx   Grid points per dimension
 * @param lx   Box lengths per dimension
 * @param bc   Boundary conditions
 * @param mask Optional mask for impenetrable regions
 */
template <typename T>
CudaComputationBox<T>::CudaComputationBox(
    std::vector<int> nx, std::vector<double> lx, std::vector<std::string> bc, const double* mask)
    : ComputationBox<T>(nx, lx, bc, mask)
{
    initialize();   // the owning buffers release what was acquired if this throws
}
template <typename T>
CudaComputationBox<T>::CudaComputationBox(
    std::vector<int> nx, std::vector<double> lx, std::vector<std::string> bc,
    std::vector<double> angles, const double* mask)
    : ComputationBox<T>(nx, lx, bc, angles, mask)
{
    initialize();   // the owning buffers release what was acquired if this throws
}
template <typename T>
void CudaComputationBox<T>::initialize()
{
    d_dv.allocate(this->total_grid);
    CUDA_CHECK(cudaMemcpy(d_dv, this->dv.data(), sizeof(double)*this->total_grid, cudaMemcpyHostToDevice));

    // Temporal storage
    d_multiple.allocate(this->total_grid);

    // Allocate memory for cub reduction sum
    d_sum.allocate(this->total_grid);
    d_sum_out.allocate(1);

    // Determine temporary storage size for cub reduction (the first call only queries the size)
    if constexpr (std::is_same<T, double>::value)
        CUDA_CHECK(cub::DeviceReduce::Sum(nullptr, temp_storage_bytes, d_sum.get(), d_sum_out.get(), this->total_grid));
    else
        CUDA_CHECK(cub::DeviceReduce::Reduce(nullptr, temp_storage_bytes, d_sum.get(), d_sum_out.get(), this->total_grid, ComplexSumOp(), CuDeviceData<T>{0.0,0.0}));
    d_temp_storage.allocate(temp_storage_bytes);
}
//----------------- Destructor -----------------------------
template <typename T>
CudaComputationBox<T>::~CudaComputationBox() = default;   // the owning buffers release themselves
//-----------------------------------------------------------
template <typename T>
void CudaComputationBox<T>::set_lx(std::vector<double> new_lx)
{
    ComputationBox<T>::set_lx(new_lx);
    CUDA_CHECK(cudaMemcpy(d_dv, this->dv.data(), sizeof(double)*this->total_grid, cudaMemcpyHostToDevice));
}
//-----------------------------------------------------------
template <typename T>
void CudaComputationBox<T>::set_lattice_parameters(std::vector<double> new_lx, std::vector<double> new_angles)
{
    ComputationBox<T>::set_lattice_parameters(new_lx, new_angles);
    CUDA_CHECK(cudaMemcpy(d_dv, this->dv.data(), sizeof(double)*this->total_grid, cudaMemcpyHostToDevice));
}
//-----------------------------------------------------------
template <typename T>
void CudaComputationBox<T>::set_space_group(SpaceGroup* sg)
{
    // Build the device orbit counts first (empty without a space group): if the
    // allocation or the copy fails, or the base class rejects the group, the box
    // keeps its previous group and state (the base class commits only at its end)
    CudaDeviceBuffer<int> orbit_counts_new;
    if (sg != nullptr)
    {
        const auto& orbit_counts = sg->get_orbit_counts();
        int N = sg->get_n_reduced_basis();
        orbit_counts_new.allocate(N);
        CUDA_CHECK(cudaMemcpy(orbit_counts_new, orbit_counts.data(), sizeof(int)*N, cudaMemcpyHostToDevice));
    }

    // Base class: checks the cell, builds the basis size, orbit counts and mask,
    // then commits them
    ComputationBox<T>::set_space_group(sg);
    d_orbit_counts_ = std::move(orbit_counts_new);
}
//-----------------------------------------------------------
template <typename T>
T CudaComputationBox<T>::integral_device(const CuDeviceData<T> *d_g)
{
    const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();
    T sum{0.0};

    if (this->space_group_ != nullptr)
    {
        // Reduced basis mode: sum = Σ_j g[j] * orbit_counts[j] * dv_base
        int N = this->n_basis_;
        double dv_base = this->volume / this->total_grid;

        ker_multi_weight<<<N_BLOCKS, N_THREADS>>>(d_sum, d_g, d_orbit_counts_, N);
        CUDA_CHECK_LAUNCH();

        if constexpr (std::is_same<T, double>::value)
            CUDA_CHECK(cub::DeviceReduce::Sum(d_temp_storage.get(), temp_storage_bytes, d_sum.get(), d_sum_out.get(), N));
        else
            CUDA_CHECK(cub::DeviceReduce::Reduce(d_temp_storage.get(), temp_storage_bytes, d_sum.get(), d_sum_out.get(), N, ComplexSumOp(), CuDeviceData<T>{0.0,0.0}));

        CUDA_CHECK(cudaMemcpy(&sum, d_sum_out, sizeof(T), cudaMemcpyDeviceToHost));
        return sum * static_cast<T>(dv_base);
    }
    else
    {
        // Full grid mode: sum = Σ_i g[i] * dv[i]
        ker_multi<<<N_BLOCKS, N_THREADS>>>(d_sum, d_g, d_dv, 1.0, this->total_grid);
        CUDA_CHECK_LAUNCH();

        if constexpr (std::is_same<T, double>::value)
            CUDA_CHECK(cub::DeviceReduce::Sum(d_temp_storage.get(), temp_storage_bytes, d_sum.get(), d_sum_out.get(), this->total_grid));
        else
            CUDA_CHECK(cub::DeviceReduce::Reduce(d_temp_storage.get(), temp_storage_bytes, d_sum.get(), d_sum_out.get(), this->total_grid, ComplexSumOp(), CuDeviceData<T>{0.0,0.0}));

        CUDA_CHECK(cudaMemcpy(&sum, d_sum_out, sizeof(T), cudaMemcpyDeviceToHost));
        return sum;
    }
}
//-----------------------------------------------------------
template <typename T>
T CudaComputationBox<T>::mean_device(const CuDeviceData<T> *d_g)
{
    return integral_device(d_g) / static_cast<T>(this->volume);
}
//-----------------------------------------------------------
template <typename T>
T CudaComputationBox<T>::inner_product_device(const CuDeviceData<T>* d_g, const CuDeviceData<T>* d_h)
{
    const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();
    T sum{0.0};

    if (this->space_group_ != nullptr)
    {
        // Reduced basis mode: sum = Σ_j g[j] * h[j] * orbit_counts[j] * dv_base
        int N = this->n_basis_;
        double dv_base = this->volume / this->total_grid;

        ker_multi<<<N_BLOCKS, N_THREADS>>>(d_sum, d_g, d_h, 1.0, N);
        CUDA_CHECK_LAUNCH();
        ker_multi_weight<<<N_BLOCKS, N_THREADS>>>(d_sum, d_sum, d_orbit_counts_, N);
        CUDA_CHECK_LAUNCH();

        if constexpr (std::is_same<T, double>::value)
            CUDA_CHECK(cub::DeviceReduce::Sum(d_temp_storage.get(), temp_storage_bytes, d_sum.get(), d_sum_out.get(), N));
        else
            CUDA_CHECK(cub::DeviceReduce::Reduce(d_temp_storage.get(), temp_storage_bytes, d_sum.get(), d_sum_out.get(), N, ComplexSumOp(), CuDeviceData<T>{0.0,0.0}));

        CUDA_CHECK(cudaMemcpy(&sum, d_sum_out, sizeof(T), cudaMemcpyDeviceToHost));
        return sum * static_cast<T>(dv_base);
    }
    else
    {
        ker_multi<<<N_BLOCKS, N_THREADS>>>(d_sum, d_g, d_h, 1.0, this->total_grid);
        CUDA_CHECK_LAUNCH();
        ker_multi<<<N_BLOCKS, N_THREADS>>>(d_sum, d_sum, d_dv, 1.0, this->total_grid);
        CUDA_CHECK_LAUNCH();

        if constexpr (std::is_same<T, double>::value)
            CUDA_CHECK(cub::DeviceReduce::Sum(d_temp_storage.get(), temp_storage_bytes, d_sum.get(), d_sum_out.get(), this->total_grid));
        else
            CUDA_CHECK(cub::DeviceReduce::Reduce(d_temp_storage.get(), temp_storage_bytes, d_sum.get(), d_sum_out.get(), this->total_grid, ComplexSumOp(), CuDeviceData<T>{0.0,0.0}));

        CUDA_CHECK(cudaMemcpy(&sum, d_sum_out, sizeof(T), cudaMemcpyDeviceToHost));
        return sum;
    }
}
//-----------------------------------------------------------
template <typename T>
T CudaComputationBox<T>::inner_product_inverse_weight_device(const CuDeviceData<T>* d_g, const CuDeviceData<T>* d_h, const CuDeviceData<T>* d_w)
{
    const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();
    T sum{0.0};

    if (this->space_group_ != nullptr)
    {
        int N = this->n_basis_;
        double dv_base = this->volume / this->total_grid;

        ker_multi<<<N_BLOCKS, N_THREADS>>>(d_sum, d_g, d_h, 1.0, N);
        CUDA_CHECK_LAUNCH();
        ker_divide<<<N_BLOCKS, N_THREADS>>>(d_sum, d_sum, d_w, 1.0, N);
        CUDA_CHECK_LAUNCH();
        ker_multi_weight<<<N_BLOCKS, N_THREADS>>>(d_sum, d_sum, d_orbit_counts_, N);
        CUDA_CHECK_LAUNCH();
        if constexpr (std::is_same<T, double>::value)
            CUDA_CHECK(cub::DeviceReduce::Sum(d_temp_storage.get(), temp_storage_bytes, d_sum.get(), d_sum_out.get(), N));
        else
            CUDA_CHECK(cub::DeviceReduce::Reduce(d_temp_storage.get(), temp_storage_bytes, d_sum.get(), d_sum_out.get(), N, ComplexSumOp(), CuDeviceData<T>{0.0,0.0}));

        CUDA_CHECK(cudaMemcpy(&sum, d_sum_out, sizeof(T), cudaMemcpyDeviceToHost));
        return sum * static_cast<T>(dv_base);
    }
    else
    {
        ker_multi<<<N_BLOCKS, N_THREADS>>>(d_sum, d_g, d_h, 1.0, this->total_grid);
        CUDA_CHECK_LAUNCH();
        ker_multi<<<N_BLOCKS, N_THREADS>>>(d_sum, d_sum, d_dv, 1.0, this->total_grid);
        CUDA_CHECK_LAUNCH();
        ker_divide<<<N_BLOCKS, N_THREADS>>>(d_sum, d_sum, d_w, 1.0, this->total_grid);
        CUDA_CHECK_LAUNCH();
        if constexpr (std::is_same<T, double>::value)
            CUDA_CHECK(cub::DeviceReduce::Sum(d_temp_storage.get(), temp_storage_bytes, d_sum.get(), d_sum_out.get(), this->total_grid));
        else
            CUDA_CHECK(cub::DeviceReduce::Reduce(d_temp_storage.get(), temp_storage_bytes, d_sum.get(), d_sum_out.get(), this->total_grid, ComplexSumOp(), CuDeviceData<T>{0.0,0.0}));

        CUDA_CHECK(cudaMemcpy(&sum, d_sum_out, sizeof(T), cudaMemcpyDeviceToHost));
        return sum;
    }
}
//-----------------------------------------------------------
template <typename T>
T CudaComputationBox<T>::multi_inner_product_device(int n_comp, const CuDeviceData<T>* d_g, const CuDeviceData<T>* d_h)
{
    const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();
    T sum{0.0};

    ker_mutiple_multi<<<N_BLOCKS, N_THREADS>>>(n_comp, d_sum, d_g, d_h, 1.0, this->total_grid);
    CUDA_CHECK_LAUNCH();
    ker_multi<<<N_BLOCKS, N_THREADS>>>(d_sum, d_sum, d_dv, 1.0, this->total_grid);
    CUDA_CHECK_LAUNCH();
    if constexpr (std::is_same<T, double>::value)
        CUDA_CHECK(cub::DeviceReduce::Sum(d_temp_storage.get(), temp_storage_bytes, d_sum.get(), d_sum_out.get(), this->total_grid));
    else
        CUDA_CHECK(cub::DeviceReduce::Reduce(d_temp_storage.get(), temp_storage_bytes, d_sum.get(), d_sum_out.get(), this->total_grid, ComplexSumOp(), CuDeviceData<T>{0.0,0.0}));
    CUDA_CHECK(cudaMemcpy(&sum, d_sum_out, sizeof(T), cudaMemcpyDeviceToHost));
    return sum;
}
//-----------------------------------------------------------
template <typename T>
void CudaComputationBox<T>::zero_mean_device(CuDeviceData<T>* d_g)
{
    const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();
    CuDeviceData<T> sum{0.0};

    ker_multi<<<N_BLOCKS, N_THREADS>>>(d_sum, d_g, d_dv, 1.0, this->total_grid);
    CUDA_CHECK_LAUNCH();

    if constexpr (std::is_same<T, double>::value)
    {
        CUDA_CHECK(cub::DeviceReduce::Sum(d_temp_storage.get(), temp_storage_bytes, d_sum.get(), d_sum_out.get(), this->total_grid));
        CUDA_CHECK(cudaMemcpy(&sum, d_sum_out, sizeof(T), cudaMemcpyDeviceToHost));
    }
    else
    {
        CUDA_CHECK(cub::DeviceReduce::Reduce(d_temp_storage.get(), temp_storage_bytes, d_sum.get(), d_sum_out.get(), this->total_grid, ComplexSumOp(), CuDeviceData<T>{0.0,0.0}));
        CUDA_CHECK(cudaMemcpy(&sum, d_sum_out, sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
        sum.x = -sum.x/this->volume;
        sum.y = -sum.y/this->volume;
    }
    ker_linear_scaling<<<N_BLOCKS, N_THREADS>>>(d_g, d_g, 1.0, sum, this->total_grid);
    CUDA_CHECK_LAUNCH();
}

// Explicit template instantiation

template class CudaComputationBox<double>;
template class CudaComputationBox<std::complex<double>>;
