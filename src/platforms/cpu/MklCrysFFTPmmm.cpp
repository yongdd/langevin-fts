/**
 * @file MklCrysFFTPmmm.cpp
 * @brief MKL implementation of crystallographic FFT for Pmmm symmetry.
 *
 * Uses O(N log N) FFT algorithm with twiddle factors for DCT-II/III.
 */

#include <cstring>
#include <numbers>
#include "MklCrysFFTPmmm.h"

//------------------------------------------------------------------------------
// Static helper: Apply DCT-II forward along one dimension
//------------------------------------------------------------------------------
static void applyDCT2Forward1D(
    double* data, double* temp,
    int n, int stride, int num_transforms,
    DFTI_DESCRIPTOR_HANDLE fft_backward,
    const double* cos_tbl, const double* sin_tbl)
{
    int complex_size = n / 2 + 1;

    thread_local std::vector<double> slice;
    thread_local std::vector<std::complex<double>> fft_in;
    thread_local std::vector<double> fft_out;

    if (static_cast<int>(slice.size()) < n)
    {
        slice.resize(n);
        fft_in.resize(complex_size);
        fft_out.resize(n);
    }

    for (int batch = 0; batch < num_transforms; ++batch)
    {
        for (int s = 0; s < stride; ++s)
        {
            int offset = batch * n * stride + s;

            // Extract 1D slice (strided)
            for (int j = 0; j < n; ++j)
                slice[j] = data[offset + j * stride];

            // DCT-II preprocessing
            fft_in[0] = std::complex<double>(slice[0], 0.0);
            for (int k = 1; k < complex_size - 1; ++k)
            {
                double x_2k = slice[2 * k];
                double x_2k_1 = slice[2 * k - 1];
                fft_in[k] = std::complex<double>((x_2k + x_2k_1) / 2.0, -(x_2k_1 - x_2k) / 2.0);
            }
            if (n % 2 == 0)
                fft_in[n / 2] = std::complex<double>(slice[n - 1], 0.0);
            else
            {
                int k = n / 2;
                double x_2k = slice[2 * k];
                double x_2k_1 = slice[2 * k - 1];
                fft_in[k] = std::complex<double>((x_2k + x_2k_1) / 2.0, -(x_2k_1 - x_2k) / 2.0);
            }

            // Inverse FFT (c2r)
            DftiComputeBackward(fft_backward,
                                reinterpret_cast<double*>(fft_in.data()),
                                fft_out.data());

            // DCT-II postprocessing
            temp[offset] = fft_out[0];
            for (int k = 1; k <= n / 2; ++k)
            {
                double Ta = fft_out[k] + fft_out[n - k];
                double Tb = fft_out[k] - fft_out[n - k];

                double result_k = (Ta * cos_tbl[k] + Tb * sin_tbl[k]) * 0.5;
                double result_nk = (Ta * sin_tbl[k] - Tb * cos_tbl[k]) * 0.5;

                temp[offset + k * stride] = result_k;
                if (k < n - k)
                    temp[offset + (n - k) * stride] = result_nk;
            }
        }
    }

    int total = num_transforms * n * stride;
    std::memcpy(data, temp, total * sizeof(double));
}

//------------------------------------------------------------------------------
// Static helper: Apply DCT-III backward along one dimension
//------------------------------------------------------------------------------
static void applyDCT3Backward1D(
    double* data, double* temp,
    int n, int stride, int num_transforms,
    DFTI_DESCRIPTOR_HANDLE fft_forward,
    const double* cos_tbl, const double* sin_tbl)
{
    int complex_size = n / 2 + 1;

    thread_local std::vector<double> slice;
    thread_local std::vector<double> fft_in;
    thread_local std::vector<std::complex<double>> fft_out;

    if (static_cast<int>(slice.size()) < n)
    {
        slice.resize(n);
        fft_in.resize(n);
        fft_out.resize(complex_size);
    }

    for (int batch = 0; batch < num_transforms; ++batch)
    {
        for (int s = 0; s < stride; ++s)
        {
            int offset = batch * n * stride + s;

            // Extract 1D slice
            for (int j = 0; j < n; ++j)
                slice[j] = data[offset + j * stride];

            // DCT-III preprocessing
            fft_in[0] = slice[0];
            for (int k = 0; k < n / 2; ++k)
            {
                double val_k = slice[k + 1];
                double val_nk = slice[n - k - 1];

                double Ta = val_k + val_nk;
                double Tb = val_k - val_nk;

                fft_in[k + 1] = Ta * sin_tbl[k + 1] + Tb * cos_tbl[k + 1];
                fft_in[n - k - 1] = Ta * cos_tbl[k + 1] - Tb * sin_tbl[k + 1];
            }

            // Forward FFT (r2c)
            DftiComputeForward(fft_forward, fft_in.data(), fft_out.data());

            // DCT-III postprocessing
            double scale = 1.0 / n;
            temp[offset] = fft_out[0].real() * scale;
            for (int k = 1; k <= n / 2; ++k)
            {
                double re = fft_out[k].real();
                double im = fft_out[k].imag();

                temp[offset + (2 * k - 1) * stride] = (re - im) * scale;
                if (2 * k < n)
                    temp[offset + (2 * k) * stride] = (re + im) * scale;
            }
        }
    }

    int total = num_transforms * n * stride;
    std::memcpy(data, temp, total * sizeof(double));
}

//------------------------------------------------------------------------------
// Constructor
//------------------------------------------------------------------------------
MklCrysFFTPmmm::MklCrysFFTPmmm(
    std::array<int, 3> nx_logical,
    std::array<double, 6> cell_para,
    std::array<double, 9> /* trans_part */)
    : Base(nx_logical, cell_para)
{
    for (int d = 0; d < 3; ++d)
    {
        dct_fft_forward_[d] = nullptr;
        dct_fft_backward_[d] = nullptr;
    }

    // Allocate work buffers
    io_buffer_ = new double[M_physical_];
    temp_buffer_ = new double[M_physical_];

    initFFTPlans();
    precomputeTwiddleFactors();
}

//------------------------------------------------------------------------------
// Destructor
//------------------------------------------------------------------------------
MklCrysFFTPmmm::~MklCrysFFTPmmm()
{
    for (int d = 0; d < 3; ++d)
    {
        if (dct_fft_forward_[d]) DftiFreeDescriptor(&dct_fft_forward_[d]);
        if (dct_fft_backward_[d]) DftiFreeDescriptor(&dct_fft_backward_[d]);
    }

    delete[] io_buffer_;
    delete[] temp_buffer_;
}

//------------------------------------------------------------------------------
// Initialize MKL DFTI descriptors
//------------------------------------------------------------------------------
void MklCrysFFTPmmm::initFFTPlans()
{
    for (int d = 0; d < 3; ++d)
    {
        int n = nx_physical_[d];
        MKL_LONG status;

        status = DftiCreateDescriptor(&dct_fft_forward_[d], DFTI_DOUBLE, DFTI_REAL, 1, n);
        status = DftiSetValue(dct_fft_forward_[d], DFTI_PLACEMENT, DFTI_NOT_INPLACE);
        status = DftiSetValue(dct_fft_forward_[d], DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
        status = DftiCommitDescriptor(dct_fft_forward_[d]);

        status = DftiCreateDescriptor(&dct_fft_backward_[d], DFTI_DOUBLE, DFTI_REAL, 1, n);
        status = DftiSetValue(dct_fft_backward_[d], DFTI_PLACEMENT, DFTI_NOT_INPLACE);
        status = DftiSetValue(dct_fft_backward_[d], DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
        status = DftiCommitDescriptor(dct_fft_backward_[d]);

        if (status != 0)
            throw_with_line_number("MKL FFT plan creation failed for dimension " + std::to_string(d));
    }
}

//------------------------------------------------------------------------------
// Precompute twiddle factors
//------------------------------------------------------------------------------
void MklCrysFFTPmmm::precomputeTwiddleFactors()
{
    const double PI = std::numbers::pi;

    for (int d = 0; d < 3; ++d)
    {
        int n = nx_physical_[d];
        int num_twiddles = n / 2 + 1;
        cos_tables_[d].resize(num_twiddles);
        sin_tables_[d].resize(num_twiddles);

        for (int k = 0; k <= n / 2; ++k)
        {
            cos_tables_[d][k] = std::cos(k * PI / (2.0 * n));
            sin_tables_[d][k] = std::sin(k * PI / (2.0 * n));
        }
    }
}

//------------------------------------------------------------------------------
// Apply DCT-II forward along all dimensions
//------------------------------------------------------------------------------
void MklCrysFFTPmmm::dct_forward_impl(double* io, double* temp)
{
    int nx = nx_physical_[0];
    int ny = nx_physical_[1];
    int nz = nx_physical_[2];

    // Dimension 0: stride = ny*nz, num_transforms = 1
    applyDCT2Forward1D(io, temp, nx, ny * nz, 1,
                       dct_fft_backward_[0],
                       cos_tables_[0].data(), sin_tables_[0].data());

    // Dimension 1: stride = nz, num_transforms = nx
    applyDCT2Forward1D(io, temp, ny, nz, nx,
                       dct_fft_backward_[1],
                       cos_tables_[1].data(), sin_tables_[1].data());

    // Dimension 2: stride = 1, num_transforms = nx*ny
    applyDCT2Forward1D(io, temp, nz, 1, nx * ny,
                       dct_fft_backward_[2],
                       cos_tables_[2].data(), sin_tables_[2].data());
}

//------------------------------------------------------------------------------
// Apply DCT-III backward along all dimensions
//------------------------------------------------------------------------------
void MklCrysFFTPmmm::dct_backward_impl(double* io, double* temp)
{
    int nx = nx_physical_[0];
    int ny = nx_physical_[1];
    int nz = nx_physical_[2];

    // Dimension 2: stride = 1, num_transforms = nx*ny
    applyDCT3Backward1D(io, temp, nz, 1, nx * ny,
                        dct_fft_forward_[2],
                        cos_tables_[2].data(), sin_tables_[2].data());

    // Dimension 1: stride = nz, num_transforms = nx
    applyDCT3Backward1D(io, temp, ny, nz, nx,
                        dct_fft_forward_[1],
                        cos_tables_[1].data(), sin_tables_[1].data());

    // Dimension 0: stride = ny*nz, num_transforms = 1
    applyDCT3Backward1D(io, temp, nx, ny * nz, 1,
                        dct_fft_forward_[0],
                        cos_tables_[0].data(), sin_tables_[0].data());
}
