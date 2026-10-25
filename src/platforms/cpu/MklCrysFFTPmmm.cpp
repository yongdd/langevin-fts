/**
 * @file MklCrysFFTPmmm.cpp
 * @brief MKL implementation of crystallographic FFT for Pmmm symmetry.
 *
 * Uses O(N log N) FFT algorithm with twiddle factors for DCT-II/III.
 */

#include <cstring>
#include <numbers>
#include "MklCrysFFTPmmm.h"
#include "MklDctLines.h"
#include "MklStatusCheck.h"

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

    try
    {
        initFFTPlans();
        precomputeTwiddleFactors();
    }
    catch (...)
    {
        // The destructor does not run for a partly constructed object
        release();
        throw;
    }
}

//------------------------------------------------------------------------------
// Destructor
//------------------------------------------------------------------------------
MklCrysFFTPmmm::~MklCrysFFTPmmm()
{
    release();
}

void MklCrysFFTPmmm::release()
{
    for (int d = 0; d < 3; ++d)
    {
        if (dct_fft_forward_[d]) DftiFreeDescriptor(&dct_fft_forward_[d]);
        if (dct_fft_backward_[d]) DftiFreeDescriptor(&dct_fft_backward_[d]);
        dct_fft_forward_[d] = nullptr;
        dct_fft_backward_[d] = nullptr;
    }
}

//------------------------------------------------------------------------------
// Initialize MKL DFTI descriptors
//------------------------------------------------------------------------------
void MklCrysFFTPmmm::initFFTPlans()
{
    for (int d = 0; d < 3; ++d)
    {
        int n = nx_physical_[d];

        mkl_dfti_check(DftiCreateDescriptor(&dct_fft_forward_[d], DFTI_DOUBLE, DFTI_REAL, 1, n), "DftiCreateDescriptor (Pmmm forward)");
        mkl_dfti_check(DftiSetValue(dct_fft_forward_[d], DFTI_PLACEMENT, DFTI_NOT_INPLACE), "DftiSetValue(DFTI_PLACEMENT, Pmmm forward)");
        mkl_dfti_check(DftiSetValue(dct_fft_forward_[d], DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX), "DftiSetValue(DFTI_CONJUGATE_EVEN_STORAGE, Pmmm forward)");
        mkl_dfti_check(DftiCommitDescriptor(dct_fft_forward_[d]), "DftiCommitDescriptor (Pmmm forward)");

        mkl_dfti_check(DftiCreateDescriptor(&dct_fft_backward_[d], DFTI_DOUBLE, DFTI_REAL, 1, n), "DftiCreateDescriptor (Pmmm backward)");
        mkl_dfti_check(DftiSetValue(dct_fft_backward_[d], DFTI_PLACEMENT, DFTI_NOT_INPLACE), "DftiSetValue(DFTI_PLACEMENT, Pmmm backward)");
        mkl_dfti_check(DftiSetValue(dct_fft_backward_[d], DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX), "DftiSetValue(DFTI_CONJUGATE_EVEN_STORAGE, Pmmm backward)");
        mkl_dfti_check(DftiCommitDescriptor(dct_fft_backward_[d]), "DftiCommitDescriptor (Pmmm backward)");
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
//
// post_scale 1: FFTW's unnormalized REDFT10 here and REDFT01 below, as in
// FftwCrysFFTPmmm. Their round trip gains 2n per dimension, which the
// norm_factor_ = 1/M_logical of CrysFFTPmmmBase removes.
//------------------------------------------------------------------------------
void MklCrysFFTPmmm::dct_forward_impl(double* io, double* temp)
{
    int nx = nx_physical_[0];
    int ny = nx_physical_[1];
    int nz = nx_physical_[2];

    // Dimension 0: stride = ny*nz, num_transforms = 1
    mkl_dct2_lines(io, temp, nx, ny * nz, 1,
                   dct_fft_backward_[0],
                   cos_tables_[0].data(), sin_tables_[0].data(), 1.0);

    // Dimension 1: stride = nz, num_transforms = nx
    mkl_dct2_lines(io, temp, ny, nz, nx,
                   dct_fft_backward_[1],
                   cos_tables_[1].data(), sin_tables_[1].data(), 1.0);

    // Dimension 2: stride = 1, num_transforms = nx*ny
    mkl_dct2_lines(io, temp, nz, 1, nx * ny,
                   dct_fft_backward_[2],
                   cos_tables_[2].data(), sin_tables_[2].data(), 1.0);
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
    mkl_dct3_lines(io, temp, nz, 1, nx * ny,
                   dct_fft_forward_[2],
                   cos_tables_[2].data(), sin_tables_[2].data(), 1.0);

    // Dimension 1: stride = nz, num_transforms = nx
    mkl_dct3_lines(io, temp, ny, nz, nx,
                   dct_fft_forward_[1],
                   cos_tables_[1].data(), sin_tables_[1].data(), 1.0);

    // Dimension 0: stride = ny*nz, num_transforms = 1
    mkl_dct3_lines(io, temp, nx, ny * nz, 1,
                   dct_fft_forward_[0],
                   cos_tables_[0].data(), sin_tables_[0].data(), 1.0);
}
