/**
 * @file MklFFT.cpp
 * @brief Unified Intel MKL FFT implementation with mixed BC support.
 *
 * Provides Fast Fourier Transform functionality using Intel MKL DFTI for
 * periodic BC, and DCT/DST for non-periodic (reflecting/absorbing) BC.
 *
 * **Transform Types by Boundary Condition:**
 *
 * - PERIODIC: MKL DFTI r2c/c2r transforms
 * - REFLECTING: DCT-II (forward), DCT-III (backward)
 * - ABSORBING: DST-II (forward), DST-III (backward)
 *
 * **Normalization:**
 *
 * Backward transforms include 1/N scaling so forward(backward(x)) = x.
 *
 * @see MklFFT.h for class documentation
 */

#include <iostream>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include "MklFFT.h"

//------------------------------------------------------------------------------
// Constructor (periodic BC, backward compatible)
//------------------------------------------------------------------------------
template <typename T, int DIM>
MklFFT<T, DIM>::MklFFT(std::array<int, DIM> nx)
    : nx_(nx), is_all_periodic_(true)
{
    // Initialize all BCs to periodic
    for (int d = 0; d < DIM; ++d)
        bc_[d] = BoundaryCondition::PERIODIC;

    // Compute grid sizes
    total_grid_ = 1;
    for (int d = 0; d < DIM; ++d)
        total_grid_ *= nx_[d];

    // Compute complex grid size for r2c FFT
    if constexpr (std::is_same<T, double>::value)
    {
        if (DIM == 3)
            total_complex_grid_ = nx_[0] * nx_[1] * (nx_[2] / 2 + 1);
        else if (DIM == 2)
            total_complex_grid_ = nx_[0] * (nx_[1] / 2 + 1);
        else
            total_complex_grid_ = nx_[0] / 2 + 1;
    }
    else
    {
        total_complex_grid_ = total_grid_;
    }

    initPeriodicFFT();
}

//------------------------------------------------------------------------------
// Constructor (with BC specification)
//------------------------------------------------------------------------------
template <typename T, int DIM>
MklFFT<T, DIM>::MklFFT(std::array<int, DIM> nx, std::array<BoundaryCondition, DIM> bc)
    : nx_(nx), bc_(bc)
{
    try
    {
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

        // For simplicity, require either all periodic or all non-periodic
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

        // Compute complex grid size
        if (is_all_periodic_)
        {
            if constexpr (std::is_same<T, double>::value)
            {
                if (DIM == 3)
                    total_complex_grid_ = nx_[0] * nx_[1] * (nx_[2] / 2 + 1);
                else if (DIM == 2)
                    total_complex_grid_ = nx_[0] * (nx_[1] / 2 + 1);
                else
                    total_complex_grid_ = nx_[0] / 2 + 1;
            }
            else
            {
                total_complex_grid_ = total_grid_;
            }
            initPeriodicFFT();
        }
        else
        {
            // DCT/DST: real-to-real, same size
            total_complex_grid_ = total_grid_;
            initNonPeriodicFFT();
        }
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

//------------------------------------------------------------------------------
// Destructor
//------------------------------------------------------------------------------
template <typename T, int DIM>
MklFFT<T, DIM>::~MklFFT()
{
    if (hand_forward_ != nullptr)
        DftiFreeDescriptor(&hand_forward_);
    if (hand_backward_ != nullptr)
        DftiFreeDescriptor(&hand_backward_);
}

//------------------------------------------------------------------------------
// Initialize periodic FFT (MKL DFTI)
//------------------------------------------------------------------------------
template <typename T, int DIM>
void MklFFT<T, DIM>::initPeriodicFFT()
{
    try
    {
        MKL_LONG NX[DIM];
        for (int i = 0; i < DIM; i++)
            NX[i] = nx_[i];

        MKL_LONG status{0};

        // Strides describe data layout in real and conjugate-even domain
        MKL_LONG rs[DIM + 1] = {0};
        MKL_LONG cs[DIM + 1] = {0};
        rs[DIM] = 1;
        cs[DIM] = 1;

        // Determine if we're using real or complex data type
        constexpr bool is_real = std::is_same<T, double>::value;
        constexpr DFTI_CONFIG_VALUE dtype = is_real ? DFTI_REAL : DFTI_COMPLEX;

        // Calculate strides for real domain
        for (int i = DIM - 1; i > 0; i--)
            rs[i] = rs[i + 1] * nx_[i];

        // Calculate strides for complex domain
        if constexpr (std::is_same<T, double>::value)
        {
            for (int i = DIM - 1; i > 0; i--)
                cs[i] = cs[i + 1] * (i == DIM - 1 ? nx_[i] / 2 + 1 : nx_[i]);
        }
        else
        {
            for (int i = DIM - 1; i > 0; i--)
                cs[i] = cs[i + 1] * nx_[i];
        }

        if (DIM == 1)
        {
            status = DftiCreateDescriptor(&hand_forward_, DFTI_DOUBLE, dtype, 1, NX[0]);
            status = DftiCreateDescriptor(&hand_backward_, DFTI_DOUBLE, dtype, 1, NX[0]);
        }
        else
        {
            status = DftiCreateDescriptor(&hand_forward_, DFTI_DOUBLE, dtype, DIM, NX);
            status = DftiSetValue(hand_forward_, DFTI_INPUT_STRIDES, rs);
            status = DftiSetValue(hand_forward_, DFTI_OUTPUT_STRIDES, cs);

            status = DftiCreateDescriptor(&hand_backward_, DFTI_DOUBLE, dtype, DIM, NX);
            status = DftiSetValue(hand_backward_, DFTI_INPUT_STRIDES, cs);
            status = DftiSetValue(hand_backward_, DFTI_OUTPUT_STRIDES, rs);
        }

        status = DftiSetValue(hand_forward_, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
        status = DftiSetValue(hand_backward_, DFTI_PLACEMENT, DFTI_NOT_INPLACE);

        if (is_real)
        {
            status = DftiSetValue(hand_forward_, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
            status = DftiSetValue(hand_backward_, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
        }

        status = DftiCommitDescriptor(hand_forward_);
        status = DftiSetValue(hand_backward_, DFTI_BACKWARD_SCALE, 1.0 / static_cast<double>(total_grid_));
        status = DftiCommitDescriptor(hand_backward_);

        if (status != 0)
            std::cerr << "MKL FFT init, status: " << status << std::endl;
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

//------------------------------------------------------------------------------
// Initialize non-periodic FFT (DCT/DST)
//------------------------------------------------------------------------------
template <typename T, int DIM>
void MklFFT<T, DIM>::initNonPeriodicFFT()
{
    // Allocate work buffers
    work_buffer_.resize(total_grid_);
    temp_buffer_.resize(total_grid_);

    // Precompute trig tables
    precomputeTrigTables();
}

//------------------------------------------------------------------------------
// Precompute trig tables
//------------------------------------------------------------------------------
template <typename T, int DIM>
void MklFFT<T, DIM>::precomputeTrigTables()
{
    const double PI = std::numbers::pi;
    sin_tables_.resize(DIM);
    cos_tables_.resize(DIM);

    for (int dim = 0; dim < DIM; ++dim)
    {
        int n = nx_[dim];

        if (bc_[dim] == BoundaryCondition::REFLECTING)
        {
            // DCT-II: cos(π*k*(j+0.5)/n) for k=0..n-1, j=0..n-1
            cos_tables_[dim].resize(n * n);
            for (int k = 0; k < n; ++k)
            {
                for (int j = 0; j < n; ++j)
                {
                    cos_tables_[dim][k * n + j] = std::cos(PI * k * (j + 0.5) / n);
                }
            }
        }
        else if (bc_[dim] == BoundaryCondition::ABSORBING)
        {
            // DST-II: sin(π*(k+1)*(j+0.5)/n) for k=0..n-1, j=0..n-1
            sin_tables_[dim].resize(n * n);
            for (int k = 0; k < n; ++k)
            {
                for (int j = 0; j < n; ++j)
                {
                    sin_tables_[dim][k * n + j] = std::sin(PI * (k + 1) * (j + 0.5) / n);
                }
            }
        }
    }
}

//------------------------------------------------------------------------------
// Get strides for dimension-by-dimension transform
//------------------------------------------------------------------------------
template <typename T, int DIM>
void MklFFT<T, DIM>::getStrides(int dim, int& stride, int& num_transforms) const
{
    stride = 1;
    for (int d = dim + 1; d < DIM; ++d)
        stride *= nx_[d];

    num_transforms = 1;
    for (int d = 0; d < dim; ++d)
        num_transforms *= nx_[d];
}

//------------------------------------------------------------------------------
// DCT-II Forward
//------------------------------------------------------------------------------
template <typename T, int DIM>
void MklFFT<T, DIM>::applyDCT2Forward(double* data, int dim)
{
    int n = nx_[dim];
    int stride, num_transforms;
    getStrides(dim, stride, num_transforms);

    const double* cos_table = cos_tables_[dim].data();

    for (int batch = 0; batch < num_transforms; ++batch)
    {
        for (int s = 0; s < stride; ++s)
        {
            int offset = batch * n * stride + s;

            std::vector<double> slice(n);
            for (int j = 0; j < n; ++j)
                slice[j] = data[offset + j * stride];

            for (int k = 0; k < n; ++k)
            {
                double sum = 0.0;
                for (int j = 0; j < n; ++j)
                    sum += slice[j] * cos_table[k * n + j];
                temp_buffer_[offset + k * stride] = sum;
            }
        }
    }

    for (int i = 0; i < total_grid_; ++i)
        data[i] = temp_buffer_[i];
}

//------------------------------------------------------------------------------
// DCT-III Backward
//------------------------------------------------------------------------------
template <typename T, int DIM>
void MklFFT<T, DIM>::applyDCT3Backward(double* data, int dim)
{
    int n = nx_[dim];
    int stride, num_transforms;
    getStrides(dim, stride, num_transforms);

    const double* cos_table = cos_tables_[dim].data();

    for (int batch = 0; batch < num_transforms; ++batch)
    {
        for (int s = 0; s < stride; ++s)
        {
            int offset = batch * n * stride + s;

            std::vector<double> slice(n);
            for (int k = 0; k < n; ++k)
                slice[k] = data[offset + k * stride];

            for (int j = 0; j < n; ++j)
            {
                double sum = slice[0] / n;
                for (int k = 1; k < n; ++k)
                    sum += (2.0 / n) * slice[k] * cos_table[k * n + j];
                temp_buffer_[offset + j * stride] = sum;
            }
        }
    }

    for (int i = 0; i < total_grid_; ++i)
        data[i] = temp_buffer_[i];
}

//------------------------------------------------------------------------------
// DST-II Forward
//------------------------------------------------------------------------------
template <typename T, int DIM>
void MklFFT<T, DIM>::applyDST2Forward(double* data, int dim)
{
    int n = nx_[dim];
    int stride, num_transforms;
    getStrides(dim, stride, num_transforms);

    const double* sin_table = sin_tables_[dim].data();

    for (int batch = 0; batch < num_transforms; ++batch)
    {
        for (int s = 0; s < stride; ++s)
        {
            int offset = batch * n * stride + s;

            std::vector<double> slice(n);
            for (int j = 0; j < n; ++j)
                slice[j] = data[offset + j * stride];

            for (int k = 0; k < n; ++k)
            {
                double sum = 0.0;
                for (int j = 0; j < n; ++j)
                    sum += slice[j] * sin_table[k * n + j];
                temp_buffer_[offset + k * stride] = sum;
            }
        }
    }

    for (int i = 0; i < total_grid_; ++i)
        data[i] = temp_buffer_[i];
}

//------------------------------------------------------------------------------
// DST-III Backward
//------------------------------------------------------------------------------
template <typename T, int DIM>
void MklFFT<T, DIM>::applyDST3Backward(double* data, int dim)
{
    int n = nx_[dim];
    int stride, num_transforms;
    getStrides(dim, stride, num_transforms);

    // Use precomputed sin table for thread safety
    // sin_table[k * n + j] = sin(π*(k+1)*(j+0.5)/n)
    const double* sin_table = sin_tables_[dim].data();

    for (int batch = 0; batch < num_transforms; ++batch)
    {
        for (int s = 0; s < stride; ++s)
        {
            int offset = batch * n * stride + s;

            std::vector<double> slice(n);
            for (int k = 0; k < n; ++k)
                slice[k] = data[offset + k * stride];

            for (int j = 0; j < n; ++j)
            {
                double sign = (j % 2 == 0) ? 1.0 : -1.0;
                double sum = sign * slice[n - 1];

                for (int k = 0; k < n - 1; ++k)
                    sum += 2.0 * slice[k] * sin_table[k * n + j];

                temp_buffer_[offset + j * stride] = sum / n;
            }
        }
    }

    for (int i = 0; i < total_grid_; ++i)
        data[i] = temp_buffer_[i];
}

//------------------------------------------------------------------------------
// Forward (complex output interface - FFT<T> compatibility)
//------------------------------------------------------------------------------
template <typename T, int DIM>
void MklFFT<T, DIM>::forward(T *rdata, std::complex<double> *cdata)
{
    if (!is_all_periodic_)
    {
        throw_with_line_number("Complex output interface requires all periodic BC. "
                              "Use forward(T*, double*) for non-periodic BC.");
    }

    int status = DftiComputeForward(hand_forward_, rdata, cdata);
    if (status != 0)
        throw_with_line_number("MKL forward, status: " + std::to_string(status));
}

//------------------------------------------------------------------------------
// Backward (complex input interface - FFT<T> compatibility)
//------------------------------------------------------------------------------
template <typename T, int DIM>
void MklFFT<T, DIM>::backward(std::complex<double> *cdata, T *rdata)
{
    if (!is_all_periodic_)
    {
        throw_with_line_number("Complex input interface requires all periodic BC. "
                              "Use backward(double*, T*) for non-periodic BC.");
    }

    int status = DftiComputeBackward(hand_backward_, cdata, rdata);
    if (status != 0)
        throw_with_line_number("MKL backward, status: " + std::to_string(status));
}

//------------------------------------------------------------------------------
// Forward (real coefficient interface - all BCs)
//------------------------------------------------------------------------------
template <typename T, int DIM>
void MklFFT<T, DIM>::forward(T *rdata, double *cdata)
{
    try
    {
        if (is_all_periodic_)
        {
            // Use MKL DFTI (treat double* as complex<double>*)
            std::complex<double>* cdata_complex = reinterpret_cast<std::complex<double>*>(cdata);
            int status = DftiComputeForward(hand_forward_, rdata, cdata_complex);
            if (status != 0)
                throw_with_line_number("MKL forward, status: " + std::to_string(status));
        }
        else
        {
            // Copy input to work buffer
            for (int i = 0; i < total_grid_; ++i)
            {
                if constexpr (std::is_same<T, double>::value)
                    work_buffer_[i] = rdata[i];
                else
                    work_buffer_[i] = std::real(rdata[i]);
            }

            // Apply transforms dimension by dimension
            for (int dim = 0; dim < DIM; ++dim)
            {
                if (bc_[dim] == BoundaryCondition::REFLECTING)
                    applyDCT2Forward(work_buffer_.data(), dim);
                else if (bc_[dim] == BoundaryCondition::ABSORBING)
                    applyDST2Forward(work_buffer_.data(), dim);
            }

            // Copy to output
            for (int i = 0; i < total_complex_grid_; ++i)
                cdata[i] = work_buffer_[i];
        }
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

//------------------------------------------------------------------------------
// Backward (real coefficient interface - all BCs)
//------------------------------------------------------------------------------
template <typename T, int DIM>
void MklFFT<T, DIM>::backward(double *cdata, T *rdata)
{
    try
    {
        if (is_all_periodic_)
        {
            // Use MKL DFTI (treat double* as complex<double>*)
            std::complex<double>* cdata_complex = reinterpret_cast<std::complex<double>*>(cdata);
            int status = DftiComputeBackward(hand_backward_, cdata_complex, rdata);
            if (status != 0)
                throw_with_line_number("MKL backward, status: " + std::to_string(status));
        }
        else
        {
            // Copy input to work buffer
            for (int i = 0; i < total_complex_grid_; ++i)
                work_buffer_[i] = cdata[i];

            // Apply inverse transforms dimension by dimension (reverse order)
            for (int dim = DIM - 1; dim >= 0; --dim)
            {
                if (bc_[dim] == BoundaryCondition::REFLECTING)
                    applyDCT3Backward(work_buffer_.data(), dim);
                else if (bc_[dim] == BoundaryCondition::ABSORBING)
                    applyDST3Backward(work_buffer_.data(), dim);
            }

            // Copy to output
            for (int i = 0; i < total_grid_; ++i)
            {
                if constexpr (std::is_same<T, double>::value)
                    rdata[i] = work_buffer_[i];
                else
                    rdata[i] = T(work_buffer_[i], 0.0);
            }
        }
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

// Explicit template instantiations
#include "TemplateInstantiations.h"
INSTANTIATE_FFT_CLASS(MklFFT);
