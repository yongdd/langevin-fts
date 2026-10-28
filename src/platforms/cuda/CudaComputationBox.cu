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
#include <numeric>
#include <cuda_runtime.h>
#include <cuComplex.h>

#include "CudaComputationBox.h"
#include "CudaCommon.h"
#include "CudaTypeDispatch.h"
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

    // Allocate memory for cub reduction sum
    d_sum.allocate(this->total_grid);
    d_sum_out.allocate(1);

    // Determine temporary storage size for cub reduction (the first call only queries the size)
    CUDA_CHECK(cuda_reduce_sum(nullptr, temp_storage_bytes, d_sum.get(), d_sum_out.get(), this->total_grid, 0));
    d_temp_storage.allocate(temp_storage_bytes);
}
//-----------------------------------------------------------
template <typename T>
T CudaComputationBox<T>::reduce_sum_to_host(int n)
{
    T sum{0.0};
    CUDA_CHECK(cuda_reduce_sum(d_temp_storage.get(), temp_storage_bytes, d_sum.get(), d_sum_out.get(), n, 0));
    CUDA_CHECK(cudaMemcpy(&sum, d_sum_out, sizeof(T), cudaMemcpyDeviceToHost));
    return sum;
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
    // Check the group against the box first (the weights read the mask through the
    // group's maps), then build the device integration weights (empty without a space
    // group): if the allocation or the copy fails, or the base class rejects the
    // group, the box keeps its previous group and state (the base class commits only
    // at its end)
    this->check_space_group(sg);
    CudaDeviceBuffer<double> basis_weights_new;
    if (sg != nullptr)
    {
        const std::vector<double> weights = this->make_basis_weights(sg);
        basis_weights_new.allocate(weights.size());
        CUDA_CHECK(cudaMemcpy(basis_weights_new, weights.data(), sizeof(double)*weights.size(), cudaMemcpyHostToDevice));
    }

    // Base class: checks the cell and the mask, builds the basis size, orbit counts,
    // mask and weights, then commits them
    ComputationBox<T>::set_space_group(sg);
    d_basis_weights_ = std::move(basis_weights_new);
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
        // Reduced basis mode: sum = Σ_j g[j] * weight[j] * dv_base (orbit counts times the mask)
        int N = this->n_basis_;
        double dv_base = this->volume / this->total_grid;

        ker_multi<<<N_BLOCKS, N_THREADS>>>(d_sum, d_g, d_basis_weights_, 1.0, N);
        CUDA_CHECK_LAUNCH();

        sum = reduce_sum_to_host(N);
        return sum * static_cast<T>(dv_base);
    }
    else
    {
        // Full grid mode: sum = Σ_i g[i] * dv[i]
        ker_multi<<<N_BLOCKS, N_THREADS>>>(d_sum, d_g, d_dv, 1.0, this->total_grid);
        CUDA_CHECK_LAUNCH();

        sum = reduce_sum_to_host(this->total_grid);
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
        // Reduced basis mode: sum = Σ_j g[j] * h[j] * weight[j] * dv_base
        int N = this->n_basis_;
        double dv_base = this->volume / this->total_grid;

        ker_multi<<<N_BLOCKS, N_THREADS>>>(d_sum, d_g, d_h, 1.0, N);
        CUDA_CHECK_LAUNCH();
        ker_multi<<<N_BLOCKS, N_THREADS>>>(d_sum, d_sum, d_basis_weights_, 1.0, N);
        CUDA_CHECK_LAUNCH();

        sum = reduce_sum_to_host(N);
        return sum * static_cast<T>(dv_base);
    }
    else
    {
        ker_multi<<<N_BLOCKS, N_THREADS>>>(d_sum, d_g, d_h, 1.0, this->total_grid);
        CUDA_CHECK_LAUNCH();
        ker_multi<<<N_BLOCKS, N_THREADS>>>(d_sum, d_sum, d_dv, 1.0, this->total_grid);
        CUDA_CHECK_LAUNCH();

        sum = reduce_sum_to_host(this->total_grid);
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
        ker_multi<<<N_BLOCKS, N_THREADS>>>(d_sum, d_sum, d_basis_weights_, 1.0, N);
        CUDA_CHECK_LAUNCH();
        sum = reduce_sum_to_host(N);
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
        sum = reduce_sum_to_host(this->total_grid);
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

    // Reduced basis mode: components of n_basis points, weights = orbit counts times the mask
    const bool reduced = this->space_group_ != nullptr;
    const int N = reduced ? this->n_basis_ : this->total_grid;
    const double* d_weight = reduced ? d_basis_weights_.get() : d_dv.get();

    ker_mutiple_multi<<<N_BLOCKS, N_THREADS>>>(n_comp, d_sum, d_g, d_h, 1.0, N);
    CUDA_CHECK_LAUNCH();
    ker_multi<<<N_BLOCKS, N_THREADS>>>(d_sum, d_sum, d_weight, 1.0, N);
    CUDA_CHECK_LAUNCH();
    sum = reduce_sum_to_host(N);
    if (reduced)
        return sum * static_cast<T>(this->volume / this->total_grid);
    return sum;
}
//-----------------------------------------------------------
template <typename T>
void CudaComputationBox<T>::zero_mean_device(CuDeviceData<T>* d_g)
{
    const int N_BLOCKS  = CudaCommon::get_instance().get_n_blocks();
    const int N_THREADS = CudaCommon::get_instance().get_n_threads();

    // The mean over the accessible volume, as in ComputationBox::zero_mean: Σ dv g / V_accessible;
    // in reduced basis mode Σ weight g / Σ weight (total_grid without a mask)
    const bool reduced = this->space_group_ != nullptr;
    const bool masked = !this->mask_full_.empty();
    const int N = reduced ? this->n_basis_ : this->total_grid;
    const double* d_weight = reduced ? d_basis_weights_.get() : d_dv.get();
    double norm = reduced ? static_cast<double>(this->total_grid) : this->volume;
    if (masked && reduced)
        norm = std::accumulate(this->basis_weights_.begin(), this->basis_weights_.end(), 0.0);
    else if (masked)
        norm = this->accessible_volume;

    ker_multi<<<N_BLOCKS, N_THREADS>>>(d_sum, d_g, d_weight, 1.0, N);
    CUDA_CHECK_LAUNCH();

    const T shift = -reduce_sum_to_host(N)/norm;
    CuDeviceData<T> d_shift;
    if constexpr (std::is_same<T, double>::value)
        d_shift = shift;
    else
        d_shift = stdToCuDoubleComplex(shift);
    ker_linear_scaling<<<N_BLOCKS, N_THREADS>>>(d_g, d_g, 1.0, d_shift, N);
    CUDA_CHECK_LAUNCH();
}

// Explicit template instantiation

template class CudaComputationBox<double>;
template class CudaComputationBox<std::complex<double>>;
