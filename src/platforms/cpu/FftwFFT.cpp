/**
 * @file FftwFFT.cpp
 * @brief FFTW3 implementation of spectral transforms with mixed BC support.
 *
 * Provides Fast Fourier Transform functionality using FFTW3 for both
 * periodic and non-periodic (reflecting/absorbing) boundary conditions.
 *
 * **Transform Types by Boundary Condition:**
 *
 * - PERIODIC: FFTW r2c/c2r transforms
 * - REFLECTING: DCT-II (forward), DCT-III (backward) via FFTW r2r
 * - ABSORBING: DST-II (forward), DST-III (backward) via FFTW r2r
 *
 * **Performance Advantage:**
 *
 * Unlike naive O(N²) matrix implementations, FFTW provides O(N log N)
 * algorithms for all transform types including DCT/DST.
 *
 * **Normalization:**
 *
 * Backward transforms include 1/N scaling so forward(backward(x)) = x.
 *
 * @see FftwFFT.h for class documentation
 */

#include <iostream>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include "FftwFFT.h"

//------------------------------------------------------------------------------
// Compute complex grid size based on periodicity and input type
//------------------------------------------------------------------------------
template <typename T, int DIM>
void FftwFFT<T, DIM>::computeComplexGridSize()
{
    if (is_all_periodic_)
    {
        if constexpr (std::is_same<T, double>::value)
        {
            // r2c FFT: exploit Hermitian symmetry
            if constexpr (DIM == 3)
                total_complex_grid_ = nx_[0] * nx_[1] * (nx_[2] / 2 + 1);
            else if constexpr (DIM == 2)
                total_complex_grid_ = nx_[0] * (nx_[1] / 2 + 1);
            else
                total_complex_grid_ = nx_[0] / 2 + 1;
        }
        else
        {
            // c2c FFT: full complex grid
            total_complex_grid_ = total_grid_;
        }
    }
    else
    {
        // DCT/DST: real-to-real, same size
        total_complex_grid_ = total_grid_;
    }
}

//------------------------------------------------------------------------------
// Constructor (periodic BC, backward compatible)
//------------------------------------------------------------------------------
template <typename T, int DIM>
FftwFFT<T, DIM>::FftwFFT(std::array<int, DIM> nx)
    : nx_(nx), is_all_periodic_(true)
{
    // Initialize all BCs to periodic and plans to nullptr
    for (int d = 0; d < DIM; ++d)
    {
        bc_[d] = BoundaryCondition::PERIODIC;
        plan_dct_forward_[d] = nullptr;
        plan_dct_backward_[d] = nullptr;
        plan_dst_forward_[d] = nullptr;
        plan_dst_backward_[d] = nullptr;
    }

    // Compute grid sizes
    total_grid_ = 1;
    for (int d = 0; d < DIM; ++d)
        total_grid_ *= nx_[d];

    computeComplexGridSize();
    initPeriodicFFT();
}

//------------------------------------------------------------------------------
// Constructor (with BC specification)
//------------------------------------------------------------------------------
template <typename T, int DIM>
FftwFFT<T, DIM>::FftwFFT(std::array<int, DIM> nx, std::array<BoundaryCondition, DIM> bc)
    : nx_(nx), bc_(bc)
{
    // Initialize plan pointers to nullptr
    for (int d = 0; d < DIM; ++d)
    {
        plan_dct_forward_[d] = nullptr;
        plan_dct_backward_[d] = nullptr;
        plan_dst_forward_[d] = nullptr;
        plan_dst_backward_[d] = nullptr;
    }

    // Compute total grid size
    total_grid_ = 1;
    for (int d = 0; d < DIM; ++d)
        total_grid_ *= nx_[d];

    // Check if all periodic
    is_all_periodic_ = true;
    for (int d = 0; d < DIM; ++d)
    {
        if (bc_[d] != BoundaryCondition::PERIODIC)
        {
            is_all_periodic_ = false;
            break;
        }
    }

    // Require either all periodic or all non-periodic (no mixed)
    if (!is_all_periodic_)
    {
        for (int d = 0; d < DIM; ++d)
        {
            if (bc_[d] == BoundaryCondition::PERIODIC)
            {
                throw_with_line_number("Mixed periodic and non-periodic BCs not yet supported. "
                                      "Use all periodic or all non-periodic (reflecting/absorbing).");
            }
        }
    }

    computeComplexGridSize();

    if (is_all_periodic_)
        initPeriodicFFT();
    else
        initNonPeriodicFFT();
}

//------------------------------------------------------------------------------
// Destructor
//------------------------------------------------------------------------------
template <typename T, int DIM>
FftwFFT<T, DIM>::~FftwFFT()
{
    if (plan_forward_ != nullptr)
        fftw_destroy_plan(plan_forward_);
    if (plan_backward_ != nullptr)
        fftw_destroy_plan(plan_backward_);

    for (int d = 0; d < DIM; ++d)
    {
        if (plan_dct_forward_[d] != nullptr)
            fftw_destroy_plan(plan_dct_forward_[d]);
        if (plan_dct_backward_[d] != nullptr)
            fftw_destroy_plan(plan_dct_backward_[d]);
        if (plan_dst_forward_[d] != nullptr)
            fftw_destroy_plan(plan_dst_forward_[d]);
        if (plan_dst_backward_[d] != nullptr)
            fftw_destroy_plan(plan_dst_backward_[d]);
    }

    if (work_buffer_ != nullptr)
        fftw_free(work_buffer_);
    if (complex_buffer_ != nullptr)
        fftw_free(complex_buffer_);
}

//------------------------------------------------------------------------------
// Initialize periodic FFT (FFTW r2c/c2r)
//------------------------------------------------------------------------------
template <typename T, int DIM>
void FftwFFT<T, DIM>::initPeriodicFFT()
{
    try
    {
        // Allocate aligned buffers
        // For complex input (T = std::complex<double>), work_buffer_ needs
        // to hold complex values for c2c transform, so allocate 2x the size
        if constexpr (std::is_same<T, double>::value)
            work_buffer_ = fftw_alloc_real(total_grid_);
        else
            work_buffer_ = fftw_alloc_real(total_grid_ * 2);  // For complex c2c FFT
        complex_buffer_ = fftw_alloc_complex(total_complex_grid_);

        if (work_buffer_ == nullptr || complex_buffer_ == nullptr)
            throw_with_line_number("FFTW memory allocation failed");

        // Create FFTW plans
        // FFTW_MEASURE: runs actual transforms to find optimal algorithm
        // This takes longer to create plans but results in faster execution
        unsigned int flags = FFTW_MEASURE;

        if constexpr (std::is_same<T, double>::value)
        {
            if (DIM == 1)
            {
                plan_forward_ = fftw_plan_dft_r2c_1d(nx_[0], work_buffer_, complex_buffer_, flags);
                plan_backward_ = fftw_plan_dft_c2r_1d(nx_[0], complex_buffer_, work_buffer_, flags);
            }
            else if (DIM == 2)
            {
                plan_forward_ = fftw_plan_dft_r2c_2d(nx_[0], nx_[1], work_buffer_, complex_buffer_, flags);
                plan_backward_ = fftw_plan_dft_c2r_2d(nx_[0], nx_[1], complex_buffer_, work_buffer_, flags);
            }
            else // DIM == 3
            {
                plan_forward_ = fftw_plan_dft_r2c_3d(nx_[0], nx_[1], nx_[2], work_buffer_, complex_buffer_, flags);
                plan_backward_ = fftw_plan_dft_c2r_3d(nx_[0], nx_[1], nx_[2], complex_buffer_, work_buffer_, flags);
            }
        }
        else // Complex input (c2c transform)
        {
            int n[DIM];
            for (int d = 0; d < DIM; ++d)
                n[d] = nx_[d];

            plan_forward_ = fftw_plan_dft(DIM, n,
                reinterpret_cast<fftw_complex*>(work_buffer_),
                complex_buffer_, FFTW_FORWARD, flags);
            plan_backward_ = fftw_plan_dft(DIM, n,
                complex_buffer_,
                reinterpret_cast<fftw_complex*>(work_buffer_),
                FFTW_BACKWARD, flags);
        }

        if (plan_forward_ == nullptr || plan_backward_ == nullptr)
            throw_with_line_number("FFTW plan creation failed");
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

//------------------------------------------------------------------------------
// Initialize non-periodic FFT (DCT/DST via FFTW r2r)
//------------------------------------------------------------------------------
template <typename T, int DIM>
void FftwFFT<T, DIM>::initNonPeriodicFFT()
{
    try
    {
        // Allocate aligned work buffer
        work_buffer_ = fftw_alloc_real(total_grid_ * 2); // Extra space for temp
        if (work_buffer_ == nullptr)
            throw_with_line_number("FFTW memory allocation failed");

        unsigned int flags = FFTW_MEASURE;

        // Create 1D r2r plans for each dimension
        // For multi-dimensional transforms, we apply 1D transforms sequentially
        for (int dim = 0; dim < DIM; ++dim)
        {
            int n = nx_[dim];

            // Calculate howmany, stride, and dist for batched 1D transforms
            int stride, num_transforms;
            getStrides(dim, stride, num_transforms);

            // For dimension-by-dimension transform, we need to handle strided data
            // FFTW guru interface allows this, but for simplicity we'll do it manually
            // Just create a single 1D plan and apply it in loops

            if (bc_[dim] == BoundaryCondition::REFLECTING)
            {
                // DCT-II forward: FFTW_REDFT10
                plan_dct_forward_[dim] = fftw_plan_r2r_1d(n, work_buffer_, work_buffer_,
                                                          FFTW_REDFT10, flags);
                // DCT-III backward: FFTW_REDFT01
                plan_dct_backward_[dim] = fftw_plan_r2r_1d(n, work_buffer_, work_buffer_,
                                                           FFTW_REDFT01, flags);

                if (plan_dct_forward_[dim] == nullptr || plan_dct_backward_[dim] == nullptr)
                    throw_with_line_number("FFTW DCT plan creation failed for dimension " + std::to_string(dim));
            }
            else if (bc_[dim] == BoundaryCondition::ABSORBING)
            {
                // DST-II forward: FFTW_RODFT10
                plan_dst_forward_[dim] = fftw_plan_r2r_1d(n, work_buffer_, work_buffer_,
                                                          FFTW_RODFT10, flags);
                // DST-III backward: FFTW_RODFT01
                plan_dst_backward_[dim] = fftw_plan_r2r_1d(n, work_buffer_, work_buffer_,
                                                           FFTW_RODFT01, flags);

                if (plan_dst_forward_[dim] == nullptr || plan_dst_backward_[dim] == nullptr)
                    throw_with_line_number("FFTW DST plan creation failed for dimension " + std::to_string(dim));
            }
        }
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

//------------------------------------------------------------------------------
// Get strides for dimension-by-dimension transform
//------------------------------------------------------------------------------
template <typename T, int DIM>
void FftwFFT<T, DIM>::getStrides(int dim, int& stride, int& num_transforms) const
{
    stride = 1;
    for (int d = dim + 1; d < DIM; ++d)
        stride *= nx_[d];

    num_transforms = 1;
    for (int d = 0; d < dim; ++d)
        num_transforms *= nx_[d];
}

//------------------------------------------------------------------------------
// Apply forward transform for one dimension (non-periodic)
//------------------------------------------------------------------------------
template <typename T, int DIM>
void FftwFFT<T, DIM>::applyForward1D(double* data, double* temp, int dim)
{
    int n = nx_[dim];
    int stride, num_transforms;
    getStrides(dim, stride, num_transforms);

    fftw_plan plan = (bc_[dim] == BoundaryCondition::REFLECTING)
                     ? plan_dct_forward_[dim]
                     : plan_dst_forward_[dim];

    // Thread-local buffer to avoid reallocation in OpenMP parallel regions
    thread_local std::vector<double> slice;
    if (static_cast<int>(slice.size()) < n)
        slice.resize(n);

    for (int batch = 0; batch < num_transforms; ++batch)
    {
        for (int s = 0; s < stride; ++s)
        {
            int offset = batch * n * stride + s;

            // Extract 1D slice
            for (int j = 0; j < n; ++j)
                slice[j] = data[offset + j * stride];

            // Execute FFTW plan (in-place)
            fftw_execute_r2r(plan, slice.data(), slice.data());

            // Store result back with scaling
            // FFTW's DCT-II/DST-II output is 2x the textbook definition
            // Scale by 0.5 to match standard mathematical definition
            for (int j = 0; j < n; ++j)
                temp[offset + j * stride] = slice[j] * 0.5;
        }
    }

    // Copy temp to data
    std::memcpy(data, temp, total_grid_ * sizeof(double));
}

//------------------------------------------------------------------------------
// Apply backward transform for one dimension (non-periodic)
//------------------------------------------------------------------------------
template <typename T, int DIM>
void FftwFFT<T, DIM>::applyBackward1D(double* data, double* temp, int dim)
{
    int n = nx_[dim];
    int stride, num_transforms;
    getStrides(dim, stride, num_transforms);

    fftw_plan plan = (bc_[dim] == BoundaryCondition::REFLECTING)
                     ? plan_dct_backward_[dim]
                     : plan_dst_backward_[dim];

    // Normalization factor for FFTW DCT/DST
    // Forward scaled by 0.5, so backward scales by 1/n for round-trip = identity
    // (FFTW_DCT3(FFTW_DCT2(x)*0.5) = 0.5*2n*x = n*x, so divide by n)
    double scale = 1.0 / n;

    // Thread-local buffer to avoid reallocation in OpenMP parallel regions
    thread_local std::vector<double> slice;
    if (static_cast<int>(slice.size()) < n)
        slice.resize(n);

    for (int batch = 0; batch < num_transforms; ++batch)
    {
        for (int s = 0; s < stride; ++s)
        {
            int offset = batch * n * stride + s;

            // Extract 1D slice
            for (int j = 0; j < n; ++j)
                slice[j] = data[offset + j * stride];

            // Execute FFTW plan (in-place)
            fftw_execute_r2r(plan, slice.data(), slice.data());

            // Store result back with normalization
            for (int j = 0; j < n; ++j)
                temp[offset + j * stride] = slice[j] * scale;
        }
    }

    // Copy temp to data
    std::memcpy(data, temp, total_grid_ * sizeof(double));
}

//------------------------------------------------------------------------------
// Forward (complex output interface - FFT<T> compatibility)
//------------------------------------------------------------------------------
template <typename T, int DIM>
void FftwFFT<T, DIM>::forward(T *rdata, std::complex<double> *cdata)
{
    if (!is_all_periodic_)
    {
        throw_with_line_number("Complex output interface requires all periodic BC. "
                              "Use forward(T*, double*) for non-periodic BC.");
    }

    // Thread-safe FFT using thread-local buffers
    thread_local std::vector<double> work_local;
    thread_local std::vector<std::complex<double>> complex_local;

    if constexpr (std::is_same<T, double>::value)
    {
        if (static_cast<int>(work_local.size()) < total_grid_)
            work_local.resize(total_grid_);

        std::memcpy(work_local.data(), rdata, total_grid_ * sizeof(double));
        fftw_execute_dft_r2c(plan_forward_, work_local.data(),
                             reinterpret_cast<fftw_complex*>(cdata));
    }
    else
    {
        if (static_cast<int>(work_local.size()) < total_grid_ * 2)
            work_local.resize(total_grid_ * 2);

        std::memcpy(work_local.data(), rdata, total_grid_ * sizeof(std::complex<double>));
        fftw_execute_dft(plan_forward_,
                         reinterpret_cast<fftw_complex*>(work_local.data()),
                         reinterpret_cast<fftw_complex*>(cdata));
    }
}

//------------------------------------------------------------------------------
// Backward (complex input interface - FFT<T> compatibility)
//------------------------------------------------------------------------------
template <typename T, int DIM>
void FftwFFT<T, DIM>::backward(std::complex<double> *cdata, T *rdata)
{
    if (!is_all_periodic_)
    {
        throw_with_line_number("Complex input interface requires all periodic BC. "
                              "Use backward(double*, T*) for non-periodic BC.");
    }

    // Thread-safe IFFT using thread-local buffers
    // CRITICAL: fftw_execute_dft_c2r destroys its input
    thread_local std::vector<double> work_local;
    thread_local std::vector<std::complex<double>> complex_local;

    double scale = 1.0 / static_cast<double>(total_grid_);

    if constexpr (std::is_same<T, double>::value)
    {
        if (static_cast<int>(work_local.size()) < total_grid_)
            work_local.resize(total_grid_);
        if (static_cast<int>(complex_local.size()) < total_complex_grid_)
            complex_local.resize(total_complex_grid_);

        std::memcpy(complex_local.data(), cdata, total_complex_grid_ * sizeof(std::complex<double>));
        fftw_execute_dft_c2r(plan_backward_,
                             reinterpret_cast<fftw_complex*>(complex_local.data()),
                             work_local.data());
        for (int i = 0; i < total_grid_; ++i)
            rdata[i] = work_local[i] * scale;
    }
    else
    {
        if (static_cast<int>(work_local.size()) < total_grid_ * 2)
            work_local.resize(total_grid_ * 2);
        if (static_cast<int>(complex_local.size()) < total_complex_grid_)
            complex_local.resize(total_complex_grid_);

        std::memcpy(complex_local.data(), cdata, total_complex_grid_ * sizeof(std::complex<double>));
        fftw_execute_dft(plan_backward_,
                         reinterpret_cast<fftw_complex*>(complex_local.data()),
                         reinterpret_cast<fftw_complex*>(work_local.data()));
        std::complex<double>* work_complex = reinterpret_cast<std::complex<double>*>(work_local.data());
        for (int i = 0; i < total_grid_; ++i)
            rdata[i] = work_complex[i] * scale;
    }
}

//------------------------------------------------------------------------------
// Forward (real coefficient interface - all BCs)
//------------------------------------------------------------------------------
template <typename T, int DIM>
void FftwFFT<T, DIM>::forward(T *rdata, double *cdata)
{
    if (is_all_periodic_)
    {
        // Thread-safe FFT using thread-local aligned buffers
        // Each thread gets its own work and complex buffers to avoid race conditions
        // when multiple threads call FFT simultaneously
        thread_local std::vector<double> work_local;
        thread_local std::vector<std::complex<double>> complex_local;

        if constexpr (std::is_same<T, double>::value)
        {
            if (static_cast<int>(work_local.size()) < total_grid_)
                work_local.resize(total_grid_);
            if (static_cast<int>(complex_local.size()) < total_complex_grid_)
                complex_local.resize(total_complex_grid_);

            // r2c: copy to local buffer, execute with new-array interface, copy output
            std::memcpy(work_local.data(), rdata, total_grid_ * sizeof(double));
            fftw_execute_dft_r2c(plan_forward_, work_local.data(),
                                 reinterpret_cast<fftw_complex*>(complex_local.data()));
            std::memcpy(cdata, complex_local.data(), total_complex_grid_ * sizeof(std::complex<double>));
        }
        else
        {
            // c2c case - both work and complex are full grid size
            if (static_cast<int>(work_local.size()) < total_grid_ * 2)
                work_local.resize(total_grid_ * 2);
            if (static_cast<int>(complex_local.size()) < total_complex_grid_)
                complex_local.resize(total_complex_grid_);

            // c2c: copy to local buffer, execute with new-array interface, copy output
            std::memcpy(work_local.data(), rdata, total_grid_ * sizeof(std::complex<double>));
            fftw_execute_dft(plan_forward_,
                             reinterpret_cast<fftw_complex*>(work_local.data()),
                             reinterpret_cast<fftw_complex*>(complex_local.data()));
            std::memcpy(cdata, complex_local.data(), total_complex_grid_ * sizeof(std::complex<double>));
        }
    }
    else
    {
        // DCT/DST: dimension-by-dimension transform
        // Use thread-local buffers for thread safety in OpenMP parallel regions
        thread_local std::vector<double> work;
        thread_local std::vector<double> temp;

        if (static_cast<int>(work.size()) < total_grid_)
        {
            work.resize(total_grid_);
            temp.resize(total_grid_);
        }

        // Copy input to work buffer
        if constexpr (std::is_same<T, double>::value)
        {
            std::memcpy(work.data(), rdata, total_grid_ * sizeof(double));
        }
        else
        {
            for (int i = 0; i < total_grid_; ++i)
                work[i] = std::real(rdata[i]);
        }

        // Apply transforms dimension by dimension
        for (int dim = 0; dim < DIM; ++dim)
        {
            applyForward1D(work.data(), temp.data(), dim);
        }

        // Copy to output
        std::memcpy(cdata, work.data(), total_complex_grid_ * sizeof(double));
    }
}

//------------------------------------------------------------------------------
// Backward (real coefficient interface - all BCs)
//------------------------------------------------------------------------------
template <typename T, int DIM>
void FftwFFT<T, DIM>::backward(double *cdata, T *rdata)
{
    if (is_all_periodic_)
    {
        // Thread-safe IFFT using thread-local aligned buffers
        // CRITICAL: fftw_execute_dft_c2r destroys its input, so we must copy first
        thread_local std::vector<double> work_local;
        thread_local std::vector<std::complex<double>> complex_local;

        double scale = 1.0 / static_cast<double>(total_grid_);

        if constexpr (std::is_same<T, double>::value)
        {
            if (static_cast<int>(work_local.size()) < total_grid_)
                work_local.resize(total_grid_);
            if (static_cast<int>(complex_local.size()) < total_complex_grid_)
                complex_local.resize(total_complex_grid_);

            // c2r: copy input to local buffer (preserving original), execute, copy with scaling
            std::memcpy(complex_local.data(), cdata, total_complex_grid_ * sizeof(std::complex<double>));
            fftw_execute_dft_c2r(plan_backward_,
                                 reinterpret_cast<fftw_complex*>(complex_local.data()),
                                 work_local.data());
            for (int i = 0; i < total_grid_; ++i)
                rdata[i] = work_local[i] * scale;
        }
        else
        {
            // c2c case
            if (static_cast<int>(work_local.size()) < total_grid_ * 2)
                work_local.resize(total_grid_ * 2);
            if (static_cast<int>(complex_local.size()) < total_complex_grid_)
                complex_local.resize(total_complex_grid_);

            // c2c: copy input to local buffer, execute, copy with scaling
            std::memcpy(complex_local.data(), cdata, total_complex_grid_ * sizeof(std::complex<double>));
            fftw_execute_dft(plan_backward_,
                             reinterpret_cast<fftw_complex*>(complex_local.data()),
                             reinterpret_cast<fftw_complex*>(work_local.data()));
            std::complex<double>* work_complex = reinterpret_cast<std::complex<double>*>(work_local.data());
            for (int i = 0; i < total_grid_; ++i)
                rdata[i] = work_complex[i] * scale;
        }
    }
    else
    {
        // DCT/DST: dimension-by-dimension inverse transform
        // Use thread-local buffers for thread safety in OpenMP parallel regions
        thread_local std::vector<double> work;
        thread_local std::vector<double> temp;

        if (static_cast<int>(work.size()) < total_grid_)
        {
            work.resize(total_grid_);
            temp.resize(total_grid_);
        }

        // Copy input to work buffer
        std::memcpy(work.data(), cdata, total_complex_grid_ * sizeof(double));

        // Apply inverse transforms dimension by dimension (reverse order)
        for (int dim = DIM - 1; dim >= 0; --dim)
        {
            applyBackward1D(work.data(), temp.data(), dim);
        }

        // Copy to output
        if constexpr (std::is_same<T, double>::value)
        {
            std::memcpy(rdata, work.data(), total_grid_ * sizeof(double));
        }
        else
        {
            for (int i = 0; i < total_grid_; ++i)
                rdata[i] = T(work[i], 0.0);
        }
    }
}

// Explicit template instantiations
template class FftwFFT<double, 1>;
template class FftwFFT<double, 2>;
template class FftwFFT<double, 3>;
template class FftwFFT<std::complex<double>, 1>;
template class FftwFFT<std::complex<double>, 2>;
template class FftwFFT<std::complex<double>, 3>;
