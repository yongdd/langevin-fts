/**
 * @file MklFFTMixedBC.cpp
 * @brief Implementation of MKL FFT with mixed boundary conditions.
 *
 * Implements DCT-II/DCT-III for reflecting (Neumann) boundary conditions
 * and DST-II/DST-III for absorbing (Dirichlet) boundary conditions.
 *
 * **Transform Definitions:**
 *
 * DCT-II (forward): X[k] = sum_{n=0}^{N-1} x[n] * cos(π*k*(n+0.5)/N)
 * DCT-III (inverse): x[n] = X[0]/N + (2/N) * sum_{k=1}^{N-1} X[k] * cos(π*k*(n+0.5)/N)
 *
 * DST-II (forward): X[k] = sum_{n=0}^{N-1} x[n] * sin(π*(k+1)*(n+0.5)/N)
 * DST-III (inverse): x[n] = (2/N) * sum_{k=0}^{N-1} X[k] * sin(π*(k+1)*(n+0.5)/N)
 *                         - x[N-1]/(2N) (with proper boundary handling)
 *
 * **Normalization:**
 *
 * The transforms are normalized so that backward(forward(x)) = x.
 *
 * @see MklFFTMixedBC.h for class documentation
 */

#include <iostream>
#include <cmath>
#include <numbers>
#include <stdexcept>

#include "MklFFTMixedBC.h"
#include "Exception.h"

//------------------------------------------------------------------------------
// Constructor
//------------------------------------------------------------------------------
template <typename T, int DIM>
MklFFTMixedBC<T, DIM>::MklFFTMixedBC(std::array<int, DIM> nx, std::array<BoundaryCondition, DIM> bc)
    : nx_(nx), bc_(bc), fft_handle_(nullptr), has_periodic_dim_(false), periodic_dim_idx_(-1)
{
    try
    {
        // Compute total grid sizes
        total_grid_ = 1;
        for (int d = 0; d < DIM; ++d)
            total_grid_ *= nx_[d];

        // Check for periodic dimensions
        for (int d = 0; d < DIM; ++d)
        {
            if (bc_[d] == BoundaryCondition::PERIODIC)
            {
                has_periodic_dim_ = true;
                periodic_dim_idx_ = d;
                break;  // For now, support only all-periodic or all-non-periodic
            }
        }

        // For simplicity, we require either all periodic or all non-periodic
        // Mixed case requires more complex handling
        if (has_periodic_dim_)
        {
            for (int d = 0; d < DIM; ++d)
            {
                if (bc_[d] != BoundaryCondition::PERIODIC)
                {
                    throw_with_line_number("Mixed periodic and non-periodic BCs not yet supported. "
                                          "Use all periodic or all non-periodic (reflecting/absorbing).");
                }
            }
        }

        // Compute complex grid size
        if (has_periodic_dim_)
        {
            // Standard r2c FFT: last dimension halved
            if (DIM == 3)
                total_complex_grid_ = nx_[0] * nx_[1] * (nx_[2] / 2 + 1);
            else if (DIM == 2)
                total_complex_grid_ = nx_[0] * (nx_[1] / 2 + 1);
            else if (DIM == 1)
                total_complex_grid_ = nx_[0] / 2 + 1;
        }
        else
        {
            // DCT/DST: real-to-real, same size
            total_complex_grid_ = total_grid_;
        }

        // Allocate work buffers
        work_buffer_.resize(total_grid_);
        temp_buffer_.resize(total_grid_);

        // Precompute trig tables for DCT/DST
        if (!has_periodic_dim_)
        {
            precomputeTrigTables();
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
MklFFTMixedBC<T, DIM>::~MklFFTMixedBC()
{
    if (fft_handle_ != nullptr)
    {
        DftiFreeDescriptor(&fft_handle_);
    }
}

//------------------------------------------------------------------------------
// Precompute trig tables
//------------------------------------------------------------------------------
template <typename T, int DIM>
void MklFFTMixedBC<T, DIM>::precomputeTrigTables()
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
void MklFFTMixedBC<T, DIM>::getStrides(int dim, int& stride, int& num_transforms) const
{
    // For row-major storage:
    // stride = product of dimensions after 'dim'
    // num_transforms = product of dimensions before 'dim'
    stride = 1;
    for (int d = dim + 1; d < DIM; ++d)
        stride *= nx_[d];

    num_transforms = 1;
    for (int d = 0; d < dim; ++d)
        num_transforms *= nx_[d];
}

//------------------------------------------------------------------------------
// DCT-II Forward (orthogonal version)
//------------------------------------------------------------------------------
template <typename T, int DIM>
void MklFFTMixedBC<T, DIM>::applyDCT2Forward(double* data, int dim)
{
    int n = nx_[dim];
    int stride, num_transforms;
    getStrides(dim, stride, num_transforms);

    const double* cos_table = cos_tables_[dim].data();

    // Process each 1D slice
    for (int batch = 0; batch < num_transforms; ++batch)
    {
        for (int s = 0; s < stride; ++s)
        {
            int offset = batch * n * stride + s;

            // Extract 1D slice
            std::vector<double> slice(n);
            for (int j = 0; j < n; ++j)
                slice[j] = data[offset + j * stride];

            // DCT-II: X[k] = sum_j x[j] * cos(π*k*(j+0.5)/n)
            for (int k = 0; k < n; ++k)
            {
                double sum = 0.0;
                for (int j = 0; j < n; ++j)
                {
                    sum += slice[j] * cos_table[k * n + j];
                }
                // Store result
                temp_buffer_[offset + k * stride] = sum;
            }
        }
    }

    // Copy back
    for (int i = 0; i < total_grid_; ++i)
        data[i] = temp_buffer_[i];
}

//------------------------------------------------------------------------------
// DCT-III Backward (orthogonal version with normalization)
//------------------------------------------------------------------------------
template <typename T, int DIM>
void MklFFTMixedBC<T, DIM>::applyDCT3Backward(double* data, int dim)
{
    int n = nx_[dim];
    int stride, num_transforms;
    getStrides(dim, stride, num_transforms);

    const double* cos_table = cos_tables_[dim].data();

    // Process each 1D slice
    for (int batch = 0; batch < num_transforms; ++batch)
    {
        for (int s = 0; s < stride; ++s)
        {
            int offset = batch * n * stride + s;

            // Extract 1D slice
            std::vector<double> slice(n);
            for (int k = 0; k < n; ++k)
                slice[k] = data[offset + k * stride];

            // DCT-III: x[j] = X[0]/n + (2/n) * sum_{k=1}^{n-1} X[k] * cos(π*k*(j+0.5)/n)
            for (int j = 0; j < n; ++j)
            {
                double sum = slice[0] / n;  // k=0 term
                for (int k = 1; k < n; ++k)
                {
                    sum += (2.0 / n) * slice[k] * cos_table[k * n + j];
                }
                temp_buffer_[offset + j * stride] = sum;
            }
        }
    }

    // Copy back
    for (int i = 0; i < total_grid_; ++i)
        data[i] = temp_buffer_[i];
}

//------------------------------------------------------------------------------
// DST-II Forward
// Matches scipy.fft.dst(x, type=2) / 2
// DST-II: X[k] = sum_{j=0}^{N-1} x[j] * sin(π*(k+1)*(2j+1)/(2N))
//------------------------------------------------------------------------------
template <typename T, int DIM>
void MklFFTMixedBC<T, DIM>::applyDST2Forward(double* data, int dim)
{
    int n = nx_[dim];
    int stride, num_transforms;
    getStrides(dim, stride, num_transforms);

    const double* sin_table = sin_tables_[dim].data();

    // Process each 1D slice
    for (int batch = 0; batch < num_transforms; ++batch)
    {
        for (int s = 0; s < stride; ++s)
        {
            int offset = batch * n * stride + s;

            // Extract 1D slice
            std::vector<double> slice(n);
            for (int j = 0; j < n; ++j)
                slice[j] = data[offset + j * stride];

            // DST-II: X[k] = sum_j x[j] * sin(π*(k+1)*(j+0.5)/n)
            for (int k = 0; k < n; ++k)
            {
                double sum = 0.0;
                for (int j = 0; j < n; ++j)
                {
                    sum += slice[j] * sin_table[k * n + j];
                }
                temp_buffer_[offset + k * stride] = sum;
            }
        }
    }

    // Copy back
    for (int i = 0; i < total_grid_; ++i)
        data[i] = temp_buffer_[i];
}

//------------------------------------------------------------------------------
// DST-III Backward (proper inverse of DST-II)
// DST-III: x[j] = (1/N) * [(-1)^j * X[N-1] + 2*sum_{k=0}^{N-2} X[k]*sin(π*(2j+1)*(k+1)/(2N))]
// This matches scipy.fft.dst(X, type=3) / (2N) which inverts dst(x, type=2)
//------------------------------------------------------------------------------
template <typename T, int DIM>
void MklFFTMixedBC<T, DIM>::applyDST3Backward(double* data, int dim)
{
    int n = nx_[dim];
    int stride, num_transforms;
    getStrides(dim, stride, num_transforms);

    const double PI = std::numbers::pi;

    // Process each 1D slice
    for (int batch = 0; batch < num_transforms; ++batch)
    {
        for (int s = 0; s < stride; ++s)
        {
            int offset = batch * n * stride + s;

            // Extract 1D slice
            std::vector<double> slice(n);
            for (int k = 0; k < n; ++k)
                slice[k] = data[offset + k * stride];

            // DST-III formula:
            // x[j] = (1/N) * [(-1)^j * X[N-1] + 2*sum_{k=0}^{N-2} X[k]*sin(π*(2j+1)*(k+1)/(2N))]
            for (int j = 0; j < n; ++j)
            {
                // Last term: (-1)^j * X[N-1]
                double sign = (j % 2 == 0) ? 1.0 : -1.0;
                double sum = sign * slice[n - 1];

                // Sum over k=0 to n-2
                for (int k = 0; k < n - 1; ++k)
                {
                    sum += 2.0 * slice[k] * std::sin(PI * (2 * j + 1) * (k + 1) / (2.0 * n));
                }

                temp_buffer_[offset + j * stride] = sum / n;
            }
        }
    }

    // Copy back
    for (int i = 0; i < total_grid_; ++i)
        data[i] = temp_buffer_[i];
}

//------------------------------------------------------------------------------
// Check if all periodic
//------------------------------------------------------------------------------
template <typename T, int DIM>
bool MklFFTMixedBC<T, DIM>::is_all_periodic() const
{
    for (int d = 0; d < DIM; ++d)
    {
        if (bc_[d] != BoundaryCondition::PERIODIC)
            return false;
    }
    return true;
}

//------------------------------------------------------------------------------
// Forward transform
//------------------------------------------------------------------------------
template <typename T, int DIM>
void MklFFTMixedBC<T, DIM>::forward(T *rdata, double *cdata)
{
    try
    {
        if (has_periodic_dim_)
        {
            throw_with_line_number("Forward transform for periodic BC should use standard MklFFT");
        }

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
            {
                applyDCT2Forward(work_buffer_.data(), dim);
            }
            else if (bc_[dim] == BoundaryCondition::ABSORBING)
            {
                applyDST2Forward(work_buffer_.data(), dim);
            }
        }

        // Copy to output
        for (int i = 0; i < total_complex_grid_; ++i)
            cdata[i] = work_buffer_[i];
    }
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

//------------------------------------------------------------------------------
// Backward transform
//------------------------------------------------------------------------------
template <typename T, int DIM>
void MklFFTMixedBC<T, DIM>::backward(double *cdata, T *rdata)
{
    try
    {
        if (has_periodic_dim_)
        {
            throw_with_line_number("Backward transform for periodic BC should use standard MklFFT");
        }

        // Copy input to work buffer
        for (int i = 0; i < total_complex_grid_; ++i)
            work_buffer_[i] = cdata[i];

        // Apply inverse transforms dimension by dimension (reverse order)
        for (int dim = DIM - 1; dim >= 0; --dim)
        {
            if (bc_[dim] == BoundaryCondition::REFLECTING)
            {
                applyDCT3Backward(work_buffer_.data(), dim);
            }
            else if (bc_[dim] == BoundaryCondition::ABSORBING)
            {
                applyDST3Backward(work_buffer_.data(), dim);
            }
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
    catch (std::exception& exc)
    {
        throw_without_line_number(exc.what());
    }
}

// Explicit template instantiations
template class MklFFTMixedBC<double, 1>;
template class MklFFTMixedBC<double, 2>;
template class MklFFTMixedBC<double, 3>;
template class MklFFTMixedBC<std::complex<double>, 1>;
template class MklFFTMixedBC<std::complex<double>, 2>;
template class MklFFTMixedBC<std::complex<double>, 3>;
